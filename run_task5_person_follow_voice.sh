#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

python3 person_following/pointcloud_to_occupancy_grid.py \
    _cloud_topic:=/cloud_registered \
    _grid_topic:=/person_following/occupancy_grid \
    _target_frame:=map \
    _base_frame:=base_link \
    _grid_width:=20.0 \
    _grid_height:=20.0 \
    _resolution:=0.10 \
    _robot_clear_radius:=0.45 \
    _inflate_radius:=0.20 \
    _publish_rate:=15.0 &
 pids+=("$!")

python3 person_following/person_goal_publisher.py \
    _global_frame:=map \
    _base_frame:=base_link \
    _map_topic:=/person_following/occupancy_grid \
    _follow_distance:=1.1 \
    _goal_update_distance:=0.22 \
    _min_publish_interval:=0.3 \
    _min_candidate_radius:=0.9 \
    _max_candidate_radius:=1.6 \
    _radius_samples:=5 \
    _front_angle_span_deg:=90.0 \
    _angle_samples:=13 \
    _weight_distance:=2.4 \
    _weight_angle:=1.2 \
    _weight_robot:=0.8 \
    _robot_radius:=0.34 \
    _path_check_step:=0.05 \
    _occupancy_threshold:=50 \
    _switch_score_margin:=0.35 \
    _switch_score_ratio:=0.15 \
    _goal_reach_hold_time:=1.0 \
    _person_reacquire_distance:=0.45 \
    _person_reacquire_forward:=0.28 \
    _person_reacquire_lateral:=0.20 \
    _person_reacquire_heading_deg:=14.0 \
    _gaze_tracking_on_pause:=true \
    _gaze_track_linear:=true \
    _gaze_track_lateral:=true \
    _gaze_yaw_deadband_deg:=4.0 \
    _gaze_max_angular:=0.45 \
    _gaze_max_forward:=0.04 \
    _gaze_max_reverse:=0.05 \
    _gaze_max_lateral:=0.03 \
    _gaze_target_distance:=1.1 \
    _gaze_distance_tolerance:=0.12 \
    _gaze_collision_probe_distance:=0.55 \
    _gaze_person_timeout:=0.6 \
    _run_rate_hz:=20.0 &
 pids+=("$!")

python3 person_following/cmd_vel_arbiter.py \
     _search_topic:=/person_following/search_cmd_vel \
     _nav_topic:=/cmd_vel_nav \
     _output_topic:=/cmd_vel \
     _search_timeout:=0.5 \
     _nav_timeout:=1.0 &
 pids+=("$!")

python3 person_following/person_detection_with_voice.py \
    _person_topic:=/person/base_link_3d_position \
    _frame_id:=base_link \
    _show_debug:=false \
    _enable_voice:=true \
    _whisper_model:=small \
    _enable_search_rotation:=true \
    _costmap_topic:=/person_following/occupancy_grid &
 pids+=("$!")

wait
