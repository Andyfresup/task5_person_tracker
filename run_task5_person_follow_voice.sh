#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_SPEECH_MODULE_FILE="$SCRIPT_DIR/../26-WrightEagle.AI-Speech/src/tts/synthesizer.py"
SPEECH_MODULE_FILE="${SPEECH_MODULE_FILE:-$DEFAULT_SPEECH_MODULE_FILE}"
USE_SPEECH_MODULE="${USE_SPEECH_MODULE:-true}"
if [[ ! -f "$SPEECH_MODULE_FILE" ]]; then
    USE_SPEECH_MODULE=false
fi

DEFAULT_SPEECH_ASR_FILE="$SCRIPT_DIR/../26-WrightEagle.AI-Speech/src/asr/vad-whisper.py"
SPEECH_ASR_FILE="${SPEECH_ASR_FILE:-$DEFAULT_SPEECH_ASR_FILE}"
USE_SPEECH_ASR_MODULE="${USE_SPEECH_ASR_MODULE:-true}"
if [[ ! -f "$SPEECH_ASR_FILE" ]]; then
    USE_SPEECH_ASR_MODULE=false
fi

if [[ -z "${DETECTION_ENABLE_VOICE:-}" ]]; then
    if [[ "${USE_SPEECH_ASR_MODULE}" == "true" ]]; then
        DETECTION_ENABLE_VOICE=false
    else
        DETECTION_ENABLE_VOICE=true
    fi
fi
FOOD_ORDER_JSON_FILE="${FOOD_ORDER_JSON_FILE:-$SCRIPT_DIR/person_following/food_orders.json}"
FOOD_SEMANTIC_ENABLED="${FOOD_SEMANTIC_ENABLED:-true}"
FOOD_SEMANTIC_BACKEND="${FOOD_SEMANTIC_BACKEND:-auto}"
FOOD_SEMANTIC_COMMAND="${FOOD_SEMANTIC_COMMAND:-}"
FOOD_SEMANTIC_COMMAND_USE_SHELL="${FOOD_SEMANTIC_COMMAND_USE_SHELL:-false}"
FOOD_SEMANTIC_MODEL_PATH="${FOOD_SEMANTIC_MODEL_PATH:-}"
FOOD_SEMANTIC_TASK="${FOOD_SEMANTIC_TASK:-text-generation}"
FOOD_SEMANTIC_TIMEOUT="${FOOD_SEMANTIC_TIMEOUT:-8.0}"
FOOD_SEMANTIC_OLLAMA_URL="${FOOD_SEMANTIC_OLLAMA_URL:-}"
FOOD_SEMANTIC_OLLAMA_MODEL="${FOOD_SEMANTIC_OLLAMA_MODEL:-llama3.2:1b}"
RETURN_ANCHOR_JSON_FILE="${RETURN_ANCHOR_JSON_FILE:-$SCRIPT_DIR/person_following/return_anchor.json}"

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
    _pause_prompt_enabled:=true \
    _pause_prompt_text:="What do you want" \
    _pause_prompt_use_speech_module:=${USE_SPEECH_MODULE} \
    _pause_prompt_speech_module_file:=${SPEECH_MODULE_FILE} \
    _pause_reply_listen_enabled:=true \
    _pause_reply_use_speech_module:=${USE_SPEECH_ASR_MODULE} \
    _pause_reply_speech_module_file:=${SPEECH_ASR_FILE} \
    _pause_reply_timeout:=6.0 \
    _pause_reply_start_delay:=1.2 \
    _pause_reply_reask_on_unrecognized:=true \
    _pause_reply_reask_text:="Can I beg you a pardon?" \
    _pause_reply_reask_max_attempts:=1 \
    _pause_reply_reask_listen_delay:=1.1 \
    _pause_reply_topic:=/person_following/pause_reply_text \
    _food_order_enabled:=true \
    _food_order_json_file:=${FOOD_ORDER_JSON_FILE} \
    _food_order_confirm_enabled:=true \
    "_food_order_confirm_template:=OK, I'll get {foods} for you" \
    _food_semantic_enabled:=${FOOD_SEMANTIC_ENABLED} \
    _food_semantic_backend:=${FOOD_SEMANTIC_BACKEND} \
    "_food_semantic_command:=${FOOD_SEMANTIC_COMMAND}" \
    _food_semantic_command_use_shell:=${FOOD_SEMANTIC_COMMAND_USE_SHELL} \
    "_food_semantic_model_path:=${FOOD_SEMANTIC_MODEL_PATH}" \
    _food_semantic_transformers_task:=${FOOD_SEMANTIC_TASK} \
    _food_semantic_timeout:=${FOOD_SEMANTIC_TIMEOUT} \
    "_food_semantic_ollama_url:=${FOOD_SEMANTIC_OLLAMA_URL}" \
    _food_semantic_ollama_model:=${FOOD_SEMANTIC_OLLAMA_MODEL} \
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
    _return_anchor_enabled:=true \
    _return_anchor_frame:=map \
    _return_anchor_base_frame:=base_link \
    _return_anchor_topic:=/person_following/return_anchor \
    _return_anchor_json_file:=${RETURN_ANCHOR_JSON_FILE} \
    _show_debug:=false \
    _enable_voice:=${DETECTION_ENABLE_VOICE} \
    _whisper_model:=small \
    _enable_search_rotation:=true \
    _costmap_topic:=/person_following/occupancy_grid &
 pids+=("$!")

wait
