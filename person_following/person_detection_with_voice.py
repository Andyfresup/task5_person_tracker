#!/usr/bin/env python3
# person detection + raised-hand gesture + voice call detection

import threading
import time
import queue

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped, Twist
from ultralytics import YOLO

try:
    import pyaudio
    from faster_whisper import WhisperModel
    import resampy
    from silero_vad import load_silero_vad, VADIterator
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False
    rospy.logwarn("Voice recognition dependencies not available. Voice detection disabled.")


class VoiceCallDetector(threading.Thread):
    """Background thread for continuous speech recognition and call detection."""
    def __init__(self, call_keywords=None, model_size="small", device="cpu"):
        super().__init__(daemon=True)
        self.call_keywords = call_keywords or ["waiter", "robot", "assistant"]
        self.call_queue = queue.Queue(maxsize=1)
        self.last_call_time = None
        self.model_size = model_size
        self.device = device
        self.is_running = False
        
        if not VOICE_ENABLED:
            rospy.logwarn("Voice recognition disabled: dependencies missing")
            return
            
        self.p = None
        self.stream = None
        self.whisper_model = None
        self.vad_model = None
        self.vad_iterator = None
        self._init_models()

    def _init_models(self):
        if not VOICE_ENABLED:
            return
        try:
            rospy.loginfo("Loading Whisper model (%s)...", self.model_size)
            self.whisper_model = WhisperModel(
                f"{self.model_size}.en",
                device=self.device,
                compute_type="int8"
            )
            rospy.loginfo("Whisper model loaded")
            
            rospy.loginfo("Loading Silero VAD model...")
            self.vad_model = load_silero_vad()
            self.vad_iterator = VADIterator(self.vad_model, threshold=0.5)
            rospy.loginfo("VAD model loaded")
        except Exception as e:
            rospy.logerr("Failed to load models: %s", e)
            VOICE_ENABLED = False

    def run(self):
        if not VOICE_ENABLED or self.whisper_model is None:
            return
        
        try:
            self.is_running = True
            self.p = pyaudio.PyAudio()
            
            # Find microphone
            device_index = self.p.get_default_input_device_info()["index"]
            device_info = self.p.get_device_info_by_index(device_index)
            device_rate = int(device_info.get("defaultSampleRate", 48000))
            
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=device_rate,
                input=True,
                frames_per_buffer=int(device_rate * 32 / 1000),
                input_device_index=device_index
            )
            
            rospy.loginfo("Voice detector listening...")
            target_rate = 16000
            frame_samples_16k = int(target_rate * 32 / 1000)
            buffer_16k = np.array([], dtype=np.float32)
            is_speaking = False
            
            while self.is_running:
                try:
                    data = self.stream.read(int(device_rate * 32 / 1000), exception_on_overflow=False)
                    audio_dev = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Resample to 16k
                    audio_16k = resampy.resample(audio_dev, sr_orig=device_rate, sr_new=target_rate)
                    if len(audio_16k) > frame_samples_16k:
                        audio_16k = audio_16k[:frame_samples_16k]
                    elif len(audio_16k) < frame_samples_16k:
                        audio_16k = np.concatenate([audio_16k, np.zeros(frame_samples_16k - len(audio_16k))])
                    
                    speech_dict = self.vad_iterator(audio_16k, return_seconds=False)
                    
                    if speech_dict is not None:
                        if 'start' in speech_dict:
                            is_speaking = True
                            buffer_16k = np.array([], dtype=np.float32)
                        if 'end' in speech_dict:
                            is_speaking = False
                            if len(buffer_16k) > int(target_rate * 0.1):
                                self._transcribe(buffer_16k)
                    
                    if is_speaking:
                        buffer_16k = np.concatenate([buffer_16k, audio_16k])
                        
                except Exception as e:
                    rospy.logerr("Voice recognition error: %s", e)
                    break
                    
        except Exception as e:
            rospy.logerr("Voice detector initialization failed: %s", e)
        finally:
            self._cleanup()

    def _transcribe(self, audio):
        try:
            segments, _ = self.whisper_model.transcribe(
                audio.astype(np.float32),
                beam_size=5,
                language="en",
                without_timestamps=True
            )
            text = "".join(seg.text.strip() + " " for seg in segments).strip().lower()
            if text:
                rospy.loginfo("Recognized: %s", text)
                for kw in self.call_keywords:
                    if kw in text:
                        try:
                            self.call_queue.put_nowait(True)
                        except queue.Full:
                            pass
                        self.last_call_time = rospy.Time.now().to_sec()
                        rospy.loginfo("Call detected: %s", kw)
                        break
        except Exception as e:
            rospy.logwarn("Transcription error: %s", e)

    def check_call(self, timeout=1.0):
        try:
            self.call_queue.get_nowait()
            if self.last_call_time and rospy.Time.now().to_sec() - self.last_call_time <= timeout:
                return True
        except queue.Empty:
            pass
        return False

    def stop(self):
        self.is_running = False

    def _cleanup(self):
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
        except:
            pass


class RobotController:
    def __init__(self, enabled=True):
        self.search_cmd_vel_topic = rospy.get_param("~search_cmd_vel_topic", "/person_following/search_cmd_vel")
        self.vel_pub = rospy.Publisher(self.search_cmd_vel_topic, Twist, queue_size=10)
        self.enabled = enabled
        self.is_searching = False
        self.search_start_time = None
        self.max_search_time = rospy.get_param("~max_search_time", 5.0)
        self.search_angular_speed = rospy.get_param("~search_angular_speed", 0.3)
        self.costmap_topic = rospy.get_param("~costmap_topic", "/person_following/occupancy_grid")
        self.costmap_timeout = rospy.get_param("~costmap_timeout", 1.0)
        self.costmap_obstacle_radius = rospy.get_param("~costmap_obstacle_radius", 1.4)
        self.costmap_occupied_threshold = rospy.get_param("~costmap_occupied_threshold", 55)
        self.search_direction = rospy.get_param("~search_direction", 1.0)
        self.default_search_direction = self.search_direction
        self.last_costmap = None
        self.last_costmap_time = None
        self.last_obstacle_state = False

        rospy.Subscriber(self.costmap_topic, OccupancyGrid, self.costmap_callback, queue_size=1)

    def costmap_callback(self, msg):
        self.last_costmap = msg
        self.last_costmap_time = rospy.Time.now()

    def _fresh_costmap(self):
        return (
            self.last_costmap is not None
            and self.last_costmap_time is not None
            and (rospy.Time.now() - self.last_costmap_time).to_sec() <= self.costmap_timeout
        )

    def _robot_center_index(self):
        if self.last_costmap is None:
            return None
        width = self.last_costmap.info.width
        height = self.last_costmap.info.height
        if width <= 0 or height <= 0:
            return None
        return width // 2, height // 2

    def _count_near_obstacles(self):
        if not self._fresh_costmap():
            return 0

        grid = self.last_costmap
        center = self._robot_center_index()
        if center is None:
            return 0

        cx, cy = center
        res = max(grid.info.resolution, 1e-3)
        radius_cells = max(1, int(self.costmap_obstacle_radius / res))
        width = grid.info.width
        height = grid.info.height
        occupied = 0

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                idx = ny * width + nx
                val = grid.data[idx]
                if val < 0 or val >= self.costmap_occupied_threshold:
                    occupied += 1
        return occupied

    def _obstacle_ahead(self):
        # For a rolling local grid, near occupied cells around robot imply the search spin
        # could sweep into nearby obstacles. Use a conservative occupancy count threshold.
        return self._count_near_obstacles() > 20

    def _flip_search_direction_if_needed(self):
        obstacle_ahead = self._obstacle_ahead()
        if obstacle_ahead and not self.last_obstacle_state:
            self.search_direction *= -1.0
            self.last_obstacle_state = True
            return True
        if not obstacle_ahead:
            self.last_obstacle_state = False
            self.search_direction = self.default_search_direction
        return False

    def rotate_to_search(self):
        if not self.enabled:
            return False
        if not self.is_searching:
            self.is_searching = True
            self.search_start_time = time.time()

        # If the local costmap shows nearby obstacles, spin in the opposite direction.
        self._flip_search_direction_if_needed()

        twist = Twist()
        twist.angular.z = self.search_direction * self.search_angular_speed
        self.vel_pub.publish(twist)

        if time.time() - self.search_start_time > self.max_search_time:
            self.stop_rotation()
            return False
        return True

    def stop_rotation(self):
        if not self.enabled:
            return
        twist = Twist()
        self.vel_pub.publish(twist)
        self.is_searching = False


def initialize_realsense(width=640, height=480, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()
    return pipeline, align, depth_scale, intrinsics


def get_median_depth_in_roi(depth_frame, depth_scale, x1, y1, x2, y2):
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width - 1, int(x2))
    y2 = min(height - 1, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    depth_data = np.asanyarray(depth_frame.get_data())
    roi = depth_data[y1:y2, x1:x2]
    roi_meters = roi.astype(np.float32) * depth_scale
    valid_depths = roi_meters[(roi_meters > 0.1) & (roi_meters < 8.0)]
    if len(valid_depths) == 0:
        return None
    return float(np.median(valid_depths))


def get_3d_coordinates(depth_frame, intrinsics, pixel_x, pixel_y, depth_value):
    if depth_value is None or depth_value <= 0:
        return None
    if pixel_x < 0 or pixel_y < 0 or pixel_x >= intrinsics.width or pixel_y >= intrinsics.height:
        return None
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(pixel_x), float(pixel_y)], float(depth_value))
    return np.array(point, dtype=np.float32)


def is_raised_hand(keypoints_xy, keypoints_conf, conf_th=0.35, y_margin=12.0):
    l_sh, r_sh, l_wr, r_wr = 5, 6, 9, 10

    left_ok = (
        keypoints_conf[l_sh] > conf_th
        and keypoints_conf[l_wr] > conf_th
        and keypoints_xy[l_wr][1] < (keypoints_xy[l_sh][1] - y_margin)
    )
    right_ok = (
        keypoints_conf[r_sh] > conf_th
        and keypoints_conf[r_wr] > conf_th
        and keypoints_xy[r_wr][1] < (keypoints_xy[r_sh][1] - y_margin)
    )
    return left_ok or right_ok


def body_center_from_keypoints_or_box(keypoints_xy, keypoints_conf, box_xyxy, conf_th=0.25):
    idx = [5, 6, 11, 12]
    pts = [keypoints_xy[i] for i in idx if keypoints_conf[i] > conf_th]
    if pts:
        pts = np.array(pts)
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    x1, y1, x2, y2 = box_xyxy
    return float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)


def match_track_id(tracks, cx, cy, now_sec, max_dist_px, track_timeout):
    best_id = None
    best_dist = 1e9
    for tid, t in tracks.items():
        if now_sec - t["last_seen"] > track_timeout:
            continue
        dx = cx - t["cx"]
        dy = cy - t["cy"]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < best_dist and dist <= max_dist_px:
            best_dist = dist
            best_id = tid
    return best_id


def choose_rightmost(candidates):
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["cx"])


def choose_nearest(candidates):
    """Choose person closest to camera (smallest depth)."""
    if not candidates:
        return None
    return min(candidates, key=lambda c: c["depth"])


def main():
    rospy.init_node("person_detection_with_voice")

    model_path = rospy.get_param("~model_path", "yolov8n-pose.pt")
    detect_conf = rospy.get_param("~detect_conf", 0.35)
    person_topic = rospy.get_param("~person_topic", "/person/base_link_3d_position")
    frame_id = rospy.get_param("~frame_id", "base_link")
    show_debug = rospy.get_param("~show_debug", True)
    enable_search_rotation = rospy.get_param("~enable_search_rotation", True)
    track_match_px = rospy.get_param("~track_match_px", 80.0)
    lock_match_px = rospy.get_param("~lock_match_px", 120.0)
    track_timeout = rospy.get_param("~track_timeout", 1.0)
    call_timeout = rospy.get_param("~call_timeout", 1.0)
    enable_voice = rospy.get_param("~enable_voice", True)
    whisper_model = rospy.get_param("~whisper_model", "small")

    cam_to_base_x = rospy.get_param("~cam_to_base_x", 0.0)
    cam_to_base_y = rospy.get_param("~cam_to_base_y", 0.0)
    cam_to_base_z = rospy.get_param("~cam_to_base_z", 0.0)

    pub = rospy.Publisher(person_topic, PointStamped, queue_size=1)
    controller = RobotController(enabled=enable_search_rotation)

    voice_detector = None
    if enable_voice and VOICE_ENABLED:
        voice_detector = VoiceCallDetector(model_size=whisper_model)
        voice_detector.start()
        rospy.loginfo("Voice call detector started")

    model = YOLO(model_path)
    pipeline, align, depth_scale, intrinsics = initialize_realsense()

    rate = rospy.Rate(15)
    rospy.loginfo("Person detector (raised-hand + voice call) started, model=%s", model_path)

    tracks = {}
    next_track_id = 1
    locked_track_id = None

    try:
        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                rate.sleep()
                continue

            color_img = np.asanyarray(color_frame.get_data())
            results = model.predict(color_img, conf=detect_conf, verbose=False)

            all_candidates = []
            raised_candidates = []
            now_sec = rospy.Time.now().to_sec()

            for res in results:
                if res.boxes is None or res.keypoints is None:
                    continue
                boxes = res.boxes.xyxy.cpu().numpy()
                kxy = res.keypoints.xy.cpu().numpy()
                kcf = res.keypoints.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(boxes)):
                    if cls[i] != 0:
                        continue

                    cx, cy = body_center_from_keypoints_or_box(kxy[i], kcf[i], boxes[i])
                    roi_half = 12
                    depth = get_median_depth_in_roi(
                        depth_frame,
                        depth_scale,
                        cx - roi_half,
                        cy - roi_half,
                        cx + roi_half,
                        cy + roi_half,
                    )
                    if depth is None:
                        continue

                    tid = match_track_id(tracks, cx, cy, now_sec, track_match_px, track_timeout)
                    if tid is None:
                        tid = next_track_id
                        next_track_id += 1
                        tracks[tid] = {
                            "cx": cx,
                            "cy": cy,
                            "last_seen": now_sec,
                        }
                    else:
                        tracks[tid]["cx"] = cx
                        tracks[tid]["cy"] = cy
                        tracks[tid]["last_seen"] = now_sec

                    item = {
                        "track_id": tid,
                        "cx": cx,
                        "cy": cy,
                        "box": boxes[i],
                        "depth": depth,
                    }
                    all_candidates.append(item)
                    if is_raised_hand(kxy[i], kcf[i]):
                        raised_candidates.append(item)

            stale_ids = [tid for tid, t in tracks.items() if now_sec - t["last_seen"] > track_timeout]
            for tid in stale_ids:
                del tracks[tid]

            selected = None
            if locked_track_id is None:
                voice_triggered = voice_detector and voice_detector.check_call(call_timeout)
                if raised_candidates:
                    selected = choose_rightmost(raised_candidates)
                    rospy.loginfo("Locked raised-hand target: track_id=%d", selected["track_id"])
                    locked_track_id = selected["track_id"]
                elif voice_triggered and all_candidates:
                    selected = choose_nearest(all_candidates)
                    rospy.loginfo("Locked voice-call target (nearest): track_id=%d", selected["track_id"])
                    locked_track_id = selected["track_id"]
            else:
                for c in all_candidates:
                    if c["track_id"] == locked_track_id:
                        selected = c
                        break

                if selected is None and locked_track_id in tracks:
                    last = tracks[locked_track_id]
                    best = None
                    best_d = 1e9
                    for c in all_candidates:
                        dx = c["cx"] - last["cx"]
                        dy = c["cy"] - last["cy"]
                        d = (dx * dx + dy * dy) ** 0.5
                        if d < best_d and d <= lock_match_px:
                            best = c
                            best_d = d
                    if best is not None:
                        selected = best
                        tracks[locked_track_id]["cx"] = best["cx"]
                        tracks[locked_track_id]["cy"] = best["cy"]
                        tracks[locked_track_id]["last_seen"] = now_sec

            if selected is None:
                controller.rotate_to_search()
                if show_debug:
                    cv2.imshow("person_detector", color_img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                rate.sleep()
                continue

            controller.stop_rotation()

            cx = selected["cx"]
            cy = selected["cy"]
            box = selected["box"]
            depth = selected["depth"]
            track_id = locked_track_id if locked_track_id is not None else selected["track_id"]
            p_cam = get_3d_coordinates(depth_frame, intrinsics, cx, cy, depth)
            if p_cam is None:
                rate.sleep()
                continue

            x_base = float(p_cam[2] + cam_to_base_x)
            y_base = float(-p_cam[0] + cam_to_base_y)
            z_base = float(-p_cam[1] + cam_to_base_z)

            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = frame_id
            msg.point.x = x_base
            msg.point.y = y_base
            msg.point.z = z_base
            pub.publish(msg)

            if show_debug:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(color_img, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                text = "LOCKED ID {} d={:.2f}m".format(track_id, depth)
                cv2.putText(color_img, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("person_detector", color_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            rate.sleep()

    finally:
        pipeline.stop()
        controller.stop_rotation()
        if voice_detector:
            voice_detector.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
