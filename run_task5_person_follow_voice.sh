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
    _publish_rate:=8.0 &
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
    _person_reacquire_distance:=0.45 &
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
