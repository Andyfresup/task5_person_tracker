#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Publish person-follow navigation goals with obstacle-aware target selection.

Behavior:
- Subscribe person position from /person/base_link_3d_position.
- Transform person point into global frame.
- Sample candidate goals in front sector of person and score them.
- Filter candidates with occupancy-grid collision checks.
- Publish PoseStamped to /move_base_simple/goal.
- Built-in bridge publishes PointStamped to /way_point and /goal_point.
"""

import math

import rospy
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler


class PersonGoalPublisher:
    def __init__(self):
        rospy.init_node("person_goal_publisher", anonymous=False)

        # Topics and frames.
        # Defaults below are tuned for far_planner + fastlio integrated navigation.
        self.person_topic = rospy.get_param("~person_topic", "/person/base_link_3d_position")
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.waypoint_topic = rospy.get_param("~waypoint_topic", "/way_point")
        self.goal_point_topic = rospy.get_param("~goal_point_topic", "/goal_point")
        self.map_topic = rospy.get_param("~map_topic", "/move_base/global_costmap/costmap")
        self.global_frame = rospy.get_param("~global_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        # Goal generation and publish control.
        # Equivalent to:
        # _follow_distance:=1.1
        # _robot_radius:=0.34
        # _occupancy_threshold:=50
        # _switch_score_margin:=0.35
        # _switch_score_ratio:=0.15
        self.follow_distance = rospy.get_param("~follow_distance", 1.1)
        self.goal_update_distance = rospy.get_param("~goal_update_distance", 0.22)
        self.person_timeout = rospy.get_param("~person_timeout", 0.8)
        self.min_publish_interval = rospy.get_param("~min_publish_interval", 0.3)
        self.publish_waypoint_direct = rospy.get_param("~publish_waypoint_direct", True)

        # Stop goal updates after robot reaches current goal stably to reduce jitter.
        self.stop_publish_on_reach = rospy.get_param("~stop_publish_on_reach", True)
        self.goal_reach_distance = rospy.get_param("~goal_reach_distance", 0.28)
        self.goal_reach_hold_time = rospy.get_param("~goal_reach_hold_time", 1.0)
        self.person_reacquire_distance = rospy.get_param("~person_reacquire_distance", 0.45)

        # Candidate sampling and scoring.
        self.min_candidate_radius = rospy.get_param("~min_candidate_radius", 0.9)
        self.max_candidate_radius = rospy.get_param("~max_candidate_radius", 1.6)
        self.radius_samples = rospy.get_param("~radius_samples", 5)
        self.front_angle_span_deg = rospy.get_param("~front_angle_span_deg", 90.0)
        self.angle_samples = rospy.get_param("~angle_samples", 13)
        self.weight_distance = rospy.get_param("~weight_distance", 2.4)
        self.weight_angle = rospy.get_param("~weight_angle", 1.2)
        self.weight_robot = rospy.get_param("~weight_robot", 0.8)

        # Hysteresis for goal switching: only switch when new goal is clearly better.
        self.switch_score_margin = rospy.get_param("~switch_score_margin", 0.35)
        self.switch_score_ratio = rospy.get_param("~switch_score_ratio", 0.15)

        # Collision constraints.
        self.robot_radius = rospy.get_param("~robot_radius", 0.34)
        self.path_check_step = rospy.get_param("~path_check_step", 0.05)
        self.occupancy_threshold = rospy.get_param("~occupancy_threshold", 50)
        self.unknown_is_occupied = rospy.get_param("~unknown_is_occupied", True)

        # Person heading estimation.
        self.min_heading_speed = rospy.get_param("~min_heading_speed", 0.05)

        # TF listener for frame conversion.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.goal_pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1)
        self.waypoint_pub = rospy.Publisher(self.waypoint_topic, PointStamped, queue_size=1)
        self.goal_point_pub = rospy.Publisher(self.goal_point_topic, PointStamped, queue_size=1)

        rospy.Subscriber(self.person_topic, PointStamped, self.person_callback, queue_size=1)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback, queue_size=1)

        self.last_person_time = rospy.Time(0)
        self.last_goal_x = None
        self.last_goal_y = None
        self.last_goal_theta = None
        self.last_goal_score = None
        self.last_publish_time = rospy.Time(0)
        self.goal_reach_since = None
        self.goal_publish_paused = False
        self.paused_person_x = None
        self.paused_person_y = None

        self.last_person_x = None
        self.last_person_y = None
        self.last_person_stamp = None
        self.person_heading = None

        self.grid = None

        rospy.loginfo(
            "person_goal_publisher started: person=%s goal=%s waypoint=%s goal_point=%s map=%s",
            self.person_topic,
            self.goal_topic,
            self.waypoint_topic,
            self.goal_point_topic,
            self.map_topic,
        )

    def map_callback(self, msg):
        self.grid = msg

    def _world_to_map(self, x, y):
        if self.grid is None:
            return None
        info = self.grid.info
        ox = info.origin.position.x
        oy = info.origin.position.y
        mx = int((x - ox) / info.resolution)
        my = int((y - oy) / info.resolution)
        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None
        return mx, my

    def _cell_occupied(self, mx, my):
        idx = my * self.grid.info.width + mx
        val = self.grid.data[idx]
        if val < 0:
            return self.unknown_is_occupied
        return val >= self.occupancy_threshold

    def _is_pose_collision_free(self, x, y):
        if self.grid is None:
            return True

        center = self._world_to_map(x, y)
        if center is None:
            return False

        info = self.grid.info
        rad_cells = max(1, int(self.robot_radius / info.resolution))
        cx, cy = center

        for dx in range(-rad_cells, rad_cells + 1):
            for dy in range(-rad_cells, rad_cells + 1):
                if dx * dx + dy * dy > rad_cells * rad_cells:
                    continue
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or ny < 0 or nx >= info.width or ny >= info.height:
                    return False
                if self._cell_occupied(nx, ny):
                    return False
        return True

    def _is_segment_collision_free(self, x0, y0, x1, y1):
        if self.grid is None:
            return True

        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1e-4:
            return self._is_pose_collision_free(x0, y0)

        steps = max(2, int(dist / self.path_check_step))
        for i in range(steps + 1):
            t = float(i) / float(steps)
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            if not self._is_pose_collision_free(x, y):
                return False
        return True

    def _estimate_person_front_heading(self, px, py, rx, ry, stamp):
        # Prefer person motion direction. Fallback to heading from person toward robot.
        if self.last_person_x is not None and self.last_person_y is not None and self.last_person_stamp is not None:
            dt = (stamp - self.last_person_stamp).to_sec()
            if dt > 1e-3:
                vx = (px - self.last_person_x) / dt
                vy = (py - self.last_person_y) / dt
                speed = math.hypot(vx, vy)
                if speed >= self.min_heading_speed:
                    self.person_heading = math.atan2(vy, vx)

        self.last_person_x = px
        self.last_person_y = py
        self.last_person_stamp = stamp

        if self.person_heading is not None:
            return self.person_heading

        return math.atan2(ry - py, rx - px)

    def _fallback_goal(self, px, py, rx, ry):
        # Without valid candidates, use simple follow target along robot->person ray.
        theta = math.atan2(py - ry, px - rx)
        gx = px - self.follow_distance * math.cos(theta)
        gy = py - self.follow_distance * math.sin(theta)
        return gx, gy, theta, None

    def _score_candidate(self, cx, cy, r, off, rx, ry, span):
        dist_pen = abs(r - self.follow_distance)
        angle_pen = abs(off) / max(span, 1e-3)
        robot_pen = math.hypot(cx - rx, cy - ry)
        return (
            self.weight_distance * dist_pen
            + self.weight_angle * angle_pen
            + self.weight_robot * robot_pen
        )

    def _select_goal(self, px, py, rx, ry, stamp):
        front_heading = self._estimate_person_front_heading(px, py, rx, ry, stamp)
        span = math.radians(self.front_angle_span_deg) * 0.5

        if self.radius_samples <= 1:
            radii = [self.follow_distance]
        else:
            radii = []
            for i in range(self.radius_samples):
                t = float(i) / float(self.radius_samples - 1)
                radii.append(self.min_candidate_radius + t * (self.max_candidate_radius - self.min_candidate_radius))

        if self.angle_samples <= 1:
            offsets = [0.0]
        else:
            offsets = []
            for i in range(self.angle_samples):
                t = float(i) / float(self.angle_samples - 1)
                offsets.append(-span + t * (2.0 * span))

        best = None

        for r in radii:
            for off in offsets:
                ang = front_heading + off
                cx = px + r * math.cos(ang)
                cy = py + r * math.sin(ang)

                if not self._is_pose_collision_free(cx, cy):
                    continue
                if not self._is_segment_collision_free(rx, ry, cx, cy):
                    continue
                if not self._is_segment_collision_free(px, py, cx, cy):
                    continue

                score = self._score_candidate(cx, cy, r, off, rx, ry, span)

                if best is None or score < best[0]:
                    best = (score, cx, cy, ang)

        if best is None:
            gx, gy, theta, score = self._fallback_goal(px, py, rx, ry)
            return gx, gy, theta, score, front_heading

        return best[1], best[2], best[3], best[0], front_heading

    def _evaluate_current_goal(self, px, py, rx, ry, front_heading):
        if self.last_goal_x is None or self.last_goal_y is None:
            return None

        gx = self.last_goal_x
        gy = self.last_goal_y

        if not self._is_pose_collision_free(gx, gy):
            return None
        if not self._is_segment_collision_free(rx, ry, gx, gy):
            return None
        if not self._is_segment_collision_free(px, py, gx, gy):
            return None

        vec_x = gx - px
        vec_y = gy - py
        r = math.hypot(vec_x, vec_y)
        if r < 1e-4:
            return None

        ang = math.atan2(vec_y, vec_x)
        off = math.atan2(math.sin(ang - front_heading), math.cos(ang - front_heading))
        span = math.radians(self.front_angle_span_deg) * 0.5
        score = self._score_candidate(gx, gy, r, off, rx, ry, span)
        return gx, gy, ang, score

    def _publish_goal(self, gx, gy, theta):
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = self.global_frame
        pose_goal.header.stamp = rospy.Time.now()
        pose_goal.pose.position.x = gx
        pose_goal.pose.position.y = gy
        pose_goal.pose.position.z = 0.0

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, theta)
        pose_goal.pose.orientation.x = qx
        pose_goal.pose.orientation.y = qy
        pose_goal.pose.orientation.z = qz
        pose_goal.pose.orientation.w = qw

        self.goal_pub.publish(pose_goal)

        if self.publish_waypoint_direct:
            waypoint = PointStamped()
            waypoint.header = pose_goal.header
            waypoint.point.x = gx
            waypoint.point.y = gy
            waypoint.point.z = 0.0
            self.waypoint_pub.publish(waypoint)
            self.goal_point_pub.publish(waypoint)

        self.last_goal_x = gx
        self.last_goal_y = gy
        self.last_goal_theta = theta
        self.last_goal_score = None
        self.last_publish_time = rospy.Time.now()
        self.goal_reach_since = None
        rospy.loginfo_throttle(0.5, "Published follow goal: (%.2f, %.2f)", gx, gy)

    def _update_reach_pause_state(self, rx, ry, px, py, now):
        if not self.stop_publish_on_reach:
            return False

        # Resume goal updates only when person has moved significantly.
        if self.goal_publish_paused:
            if self.paused_person_x is None or self.paused_person_y is None:
                return True
            moved = math.hypot(px - self.paused_person_x, py - self.paused_person_y)
            if moved >= self.person_reacquire_distance:
                self.goal_publish_paused = False
                self.goal_reach_since = None
                rospy.loginfo("Person moved %.2fm, resume goal publishing.", moved)
                return False
            return True

        if self.last_goal_x is None or self.last_goal_y is None:
            self.goal_reach_since = None
            return False

        dist_to_goal = math.hypot(rx - self.last_goal_x, ry - self.last_goal_y)
        if dist_to_goal <= self.goal_reach_distance:
            if self.goal_reach_since is None:
                self.goal_reach_since = now
            held = (now - self.goal_reach_since).to_sec()
            if held >= self.goal_reach_hold_time:
                self.goal_publish_paused = True
                self.paused_person_x = px
                self.paused_person_y = py
                rospy.loginfo(
                    "Goal reached (dist=%.2fm, held=%.2fs). Pause goal publishing until person moves.",
                    dist_to_goal,
                    held,
                )
                return True
        else:
            self.goal_reach_since = None

        return False

    def person_callback(self, msg):
        self.last_person_time = rospy.Time.now()

        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                msg.header.frame_id,
                rospy.Time(0),
                rospy.Duration(0.3),
            )
            person_in_global = tf2_geometry_msgs.do_transform_point(msg, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(1.0, "TF transform failed: %s", exc)
            return

        try:
            robot_tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(0.2),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(1.0, "Robot TF lookup failed: %s", exc)
            return

        rx = robot_tf.transform.translation.x
        ry = robot_tf.transform.translation.y
        px = person_in_global.point.x
        py = person_in_global.point.y

        stamp = rospy.Time.now()

        # If robot already reached a stable goal, stop publishing until person moves.
        if self._update_reach_pause_state(rx, ry, px, py, stamp):
            return

        gx, gy, theta, cand_score, front_heading = self._select_goal(px, py, rx, ry, stamp)

        # Hysteresis: keep current valid goal unless candidate is clearly better.
        current_eval = self._evaluate_current_goal(px, py, rx, ry, front_heading)
        if current_eval is not None and cand_score is not None:
            _, _, cur_theta, cur_score = current_eval
            improvement = cur_score - cand_score
            ratio = improvement / max(cur_score, 1e-3)
            if improvement < self.switch_score_margin or ratio < self.switch_score_ratio:
                gx = self.last_goal_x
                gy = self.last_goal_y
                theta = cur_theta
                cand_score = cur_score

        now = rospy.Time.now()
        if (now - self.last_publish_time).to_sec() < self.min_publish_interval:
            return

        if self.last_goal_x is not None and self.last_goal_y is not None:
            move = math.hypot(gx - self.last_goal_x, gy - self.last_goal_y)
            if move < self.goal_update_distance:
                return

        self._publish_goal(gx, gy, theta)
        self.last_goal_score = cand_score

    def run(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if (rospy.Time.now() - self.last_person_time).to_sec() > self.person_timeout:
                rospy.logwarn_throttle(1.0, "Person target timeout, waiting for fresh detections.")
            if self.grid is None:
                rospy.logwarn_throttle(2.0, "No occupancy grid yet on %s, using fallback goals.", self.map_topic)
            rate.sleep()


if __name__ == "__main__":
    node = PersonGoalPublisher()
    node.run()
