#!/usr/bin/env bash
set -eu
set -o pipefail

# Periodically runs inference validation and captures video frames.
#
# Usage (from train_scripts/):
#   bash validate_video_watcher_fixed.bash <env_build_name> <run_id> <config_name> [eval_every_steps]
#
# Example:
#   bash validate_video_watcher_fixed.bash jack_training.x86_64 run_001 Jack_single_agent 100000
#
# Notes:
# - Detects checkpoints by scanning results/<run_id>/*/*-<steps>.onnx
# - Captures PNG sequences into: results/<run_id>/videos/step_<N>/ and step_<N>_jack3p/
# - If ffmpeg is installed, also produces: results/<run_id>/videos/step_<N>.mp4 and step_<N>_jack3p.mp4

cd "$(dirname "$0")"
cd ..

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
  echo "Usage: bash train_scripts/validate_video_watcher_fixed.bash <env_build_name> <run_id> <config_name> [eval_every_steps]"
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
STATE_FILE="${VIDEO_ROOT}/last_encoded_target.txt"

mkdir -p "${VIDEO_ROOT}"

START_PORT=7005
CHECK_COUNT=1000

pick_free_port() {
  for i in $(seq 0 "${CHECK_COUNT}"); do
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

python_has_any_checkpoint() {
  python - <<'PY' "$1"
import glob, os, sys
results_dir = sys.argv[1]
pattern = os.path.join(results_dir, "*", "*-*.onnx")
print("1" if glob.glob(pattern) else "0")
PY
}

read_last_encoded_step() {
  if [ -f "${STATE_FILE}" ]; then
    cat "${STATE_FILE}" 2>/dev/null || echo "0"
  else
    echo "0"
  fi
}

write_last_encoded_step() {
  local step="$1"
  echo "${step}" > "${STATE_FILE}"
}

already_encoded_target() {
  local step="$1"
  local step_di="${VIDEO_ROOT}/step_${step}"
  local mp4="${VIDEO_ROOT}/step_${step}.mp4"
  if [ -f "${mp4}" ]; then
    return 0
  fi
  if [ -f "${step_di}/capture_started.txt" ]; then
    return 0
  fi
  if [ -d "${step_di}" ] && ls "${step_di}"/frame_*.png >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

run_eval_for_step() {
  local step="$1"
  local pot
  pot="$(pick_free_port)"

  local step_di="${VIDEO_ROOT}/step_${step}"
  local step_di_b="${VIDEO_ROOT}/step_${step}_jack3p"
  # Make absolute paths for Unity (relative paths depend on its working directory).
  if command -v realpath >/dev/null 2>&1; then
    step_di="$(realpath "${step_di}")"
    step_di_b="$(realpath "${step_di_b}")"
  fi
  mkdir -p "${step_di}"
  mkdir -p "${step_di_b}"

  local quit_after_episodes="2"
  local capture_every="3"
  local quit_after_seconds="${QUIT_AFTER_SECONDS:-180}"

  echo "[watche] validate taget step=${step} on pot=${pot}"

  local mlagents_cmd=(mlagents-learn "custom_configs/${CONFIG_NAME}.yaml"
    --inference
    --resume
    --env="${BUILD_PATH}"
    --run-id "${RUN_ID}"
    --base-port "${pot}"
    --num-envs 1
    --timeout-wait 600
    --env-args --capture-dir "${step_di}" --capture-dir-b "${step_di_b}" --capture-every "${capture_every}" --quit-after-episodes "${quit_after_episodes}" --quit-after-seconds "${quit_after_seconds}" --quit-delay-seconds 0.5
  )

  local max_eval_seconds="${MAX_EVAL_SECONDS:-600}"

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

  if command -v ffmpeg >/dev/null 2>&1; then
    local out_mp4="${VIDEO_ROOT}/step_${step}.mp4"
    ffmpeg -y -framerate 60 -i "${step_di}/frame_%06d.png" -c:v libx264 -pix_fmt yuv420p "${out_mp4}" >/dev/null 2>&1 || true
    echo "[watcher] saved ${out_mp4}"

    local out_mp4_b="${VIDEO_ROOT}/step_${step}_jack3p.mp4"
    ffmpeg -y -framerate 60 -i "${step_di_b}/frame_%06d.png" -c:v libx264 -pix_fmt yuv420p "${out_mp4_b}" >/dev/null 2>&1 || true
    echo "[watcher] saved ${out_mp4_b}"
  else
    echo "[watcher] ffmpeg not found; kept PNGs in ${step_di} and ${step_di_b}"
  fi
}

echo "[watcher] scanning checkpoints in ${RESULTS_DIR}"
echo "[watcher] eval interval ${EVAL_EVERY_STEPS} steps (checkpoint steps may be ~, not exact)"

while true; do
  if [ "$(python_has_any_checkpoint "${RESULTS_DIR}")" != "1" ]; then
    sleep 20
    continue
  fi

  latest_step="$(python_latest_checkpoint_step "${RESULTS_DIR}")"
  if [ "${latest_step}" -gt 0 ]; then
    last_target="$(read_last_encoded_step)"
    if [ "${last_target}" -le 0 ]; then
      last_target="0"
    fi

    max_target=$(( (latest_step / EVAL_EVERY_STEPS) * EVAL_EVERY_STEPS ))
    target=$((last_target + EVAL_EVERY_STEPS))

    # Backfill: un fo all missing tagets up to latest_step.
    while [ "${target}" -le "${max_target}" ]; do
      if ! already_encoded_target "${target}"; then
        run_eval_for_step "${target}"
      else
        echo "[watcher] skip step_${target} (already encoded)"
      fi
      write_last_encoded_step "${target}"
      target=$((target + EVAL_EVERY_STEPS))
    done
  fi

  sleep 20
done

