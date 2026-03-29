#!/usr/bin/env bash
set -euo pipefail

# Periodically runs inference validation and captures video frames.
#
# Usage (from train_scripts/):
#   bash validate_video_watcher.bash <env_build_name> <run_id> <config_name> [eval_every_steps]
#
# Example:
#   bash validate_video_watcher.bash jack_training.x86_64 run_001 Jack_single_agent 100000
#
# Notes:
# - Detects checkpoints by scanning results/<run_id>/*/*-<steps>.onnx
# - Captures PNG sequence into: results/<run_id>/videos/step_<N>/
# - If ffmpeg is installed, also produces: results/<run_id>/videos/step_<N>.mp4

cd "$(dirname "$0")"
cd ..

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
  echo "Usage: bash train_scripts/validate_video_watcher.bash <env_build_name> <run_id> <config_name> [eval_every_steps]"
  exit 1
fi

ENV_BUILD_NAME="$1"
RUN_ID="$2"
CONFIG_NAME="$3"
EVAL_EVERY_STEPS="${4:-100000}"
FORCE_XVFB="${FORCE_XVFB:-1}"

BUILD_PATH="build_versions/${ENV_BUILD_NAME}"
RESULTS_DIR="results/${RUN_ID}"
VIDEO_ROOT="${RESULTS_DIR}/videos"
STATE_FILE="${VIDEO_ROOT}/last_recorded_target.txt"

mkdir -p "${VIDEO_ROOT}"

START_PORT=7005
CHECK_COUNT=1000
pick_free_port() {
  for i in $(seq 0 $CHECK_COUNT); do
    local p=$((START_PORT + i))
    if ! netstat -tuln | grep -q ":$p "; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

python_latest_checkpoint_step() {
  python - <<'PY' "$1"
import glob, os, re, sys
results_dir = sys.argv[1]

pattern = os.path.join(results_dir, "*", "*-*.onnx")
steps = []
for p in glob.glob(pattern):
    base = os.path.basename(p)
    m = re.search(r"-(\d+)\.onnx$", base)
    if not m:
        continue
    steps.append(int(m.group(1)))

print(str(max(steps) if steps else 0))
PY
}

read_last_recorded_step() {
  if [ -f "${STATE_FILE}" ]; then
    cat "${STATE_FILE}" 2>/dev/null || echo "0"
  else
    echo "0"
  fi
}

write_last_recorded_step() {
  local step="$1"
  echo "${step}" > "${STATE_FILE}"
}

run_eval_for_step() {
  local step="$1"
  local port
  port="$(pick_free_port)"

  local step_dir="${VIDEO_ROOT}/step_${step}"
  # Make absolute path for Unity (relative paths depend on its working directory).
  if command -v realpath >/dev/null 2>&1; then
    step_dir="$(realpath "${step_dir}")"
  fi
  mkdir -p "${step_dir}"

  # Record 2 full episodes by default. (Requires rebuilt Unity with CommandLineEvalCapture + agent hooks.)
  local quit_after_episodes="2"
  # Capture every N rendered frames to reduce IO cost.
  local capture_every="3"
  # Hard stop in seconds in case Academy doesn't step / env waits for connection.
  local quit_after_seconds="${QUIT_AFTER_SECONDS:-180}"

  echo "[watcher] validate target step=${step} on port=${port}"

  local mlagents_cmd=(mlagents-learn "custom_configs/${CONFIG_NAME}.yaml"
    --inference
    --resume
    --env="${BUILD_PATH}"
    --run-id "${RUN_ID}"
    --base-port "${port}"
    --num-envs 1
    --timeout-wait 600
    --env-args --capture-dir "${step_dir}" --capture-every "${capture_every}" --quit-after-episodes "${quit_after_episodes}" --quit-after-seconds "${quit_after_seconds}" --quit-delay-seconds 0.5
  )

  # mlagents-learn will restart workers when the env quits (even with exit code 0).
  # We bound the eval duration to avoid infinite restarts and to return control to the watcher.
  local max_eval_seconds="${MAX_EVAL_SECONDS:-600}"

  # Video capture requires a working display. In WSL this is often unreliable even when DISPLAY is set,
  # so we default to xvfb-run unless FORCE_XVFB=0.
  if [ "${FORCE_XVFB}" = "1" ] && command -v xvfb-run >/dev/null 2>&1; then
    echo "[watcher] running inference under xvfb-run (FORCE_XVFB=1)"
    timeout --signal=INT "${max_eval_seconds}" xvfb-run -a "${mlagents_cmd[@]}" || true
  else
    if [ -z "${DISPLAY:-}" ]; then
      echo "[watcher] ERROR: DISPLAY is empty and FORCE_XVFB=0. Cannot render/capture video."
      return 1
    fi
    echo "[watcher] running inference with DISPLAY=${DISPLAY} (FORCE_XVFB=0)"
    timeout --signal=INT "${max_eval_seconds}" "${mlagents_cmd[@]}" || true
  fi

  # Inference run (single env) with graphics ON, passing capture args into the Unity build.
  # We intentionally DO NOT pass --no-graphics here.
  if command -v ffmpeg >/dev/null 2>&1; then
    local out_mp4="${VIDEO_ROOT}/step_${step}.mp4"
    # 60 fps matches engine_settings.capture_frame_rate in your configs.
    ffmpeg -y -framerate 60 -i "${step_dir}/frame_%06d.png" -c:v libx264 -pix_fmt yuv420p "${out_mp4}" >/dev/null 2>&1 || true
    echo "[watcher] saved ${out_mp4}"
  else
    echo "[watcher] ffmpeg not found; kept PNGs in ${step_dir}"
  fi
}

last_recorded="$(read_last_recorded_step)"
echo "[watcher] scanning checkpoints in ${RESULTS_DIR}"
echo "[watcher] eval interval ${EVAL_EVERY_STEPS} steps (checkpoint steps may be ~, not exact)"
echo "[watcher] last recorded step: ${last_recorded}"

while true; do
  latest_step="$(python_latest_checkpoint_step "${RESULTS_DIR}")"

  if [ "${latest_step}" -gt 0 ]; then
    last_target="$(read_last_recorded_step)"
    next_target=$((last_target + EVAL_EVERY_STEPS))

    # If we haven't recorded anything yet, start from the first bucket.
    if [ "${last_target}" -le 0 ]; then
      next_target="${EVAL_EVERY_STEPS}"
    fi

    # Only trigger when the newest checkpoint has reached (or passed) our next target bucket.
    if [ "${latest_step}" -ge "${next_target}" ]; then
      run_eval_for_step "${next_target}"
      write_last_recorded_step "${next_target}"
    fi
  fi
  sleep 20
done

