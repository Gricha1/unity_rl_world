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
# - Все настройки захвата/ffmpeg — в блоке «ПАРАМЕТРЫ» ниже (не через export / окружение).
# - Captures PNG sequences into:
#     results/<run_id>/videos/step_<N>_camA/
#     results/<run_id>/videos/step_<N>_camB/
#     results/<run_id>/videos/step_<N>_jack_overhead/
# - If ffmpeg is installed, also produces 3 mp4 files with the same suffixes.

cd "$(dirname "$0")"
cd ..

# ========== ПАРАМЕТРЫ (правь здесь; не через export) ==========
FORCE_XVFB=1
CAPTURE_EVERY=1
CAPTURE_WIDTH=1920
CAPTURE_HEIGHT=1080
CAPTURE_MSAA=4
QUIT_AFTER_SECONDS=240
QUIT_AFTER_EPISODES=0
QUIT_DELAY_SECONDS=0.5
MAX_EVAL_SECONDS=600
CAPTURE_USE_META_FPS=1
CAPTURE_VIDEO_FPS=24
FFMPEG_CRF=18
FFMPEG_PRESET=medium
EVAL_EVERY_STEPS_DEFAULT=100000
# ==============================================================

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
  echo "Usage: bash train_scripts/validate_video_watcher_fixed.bash <env_build_name> <run_id> <config_name> [eval_every_steps]"
  exit 1
fi

ENV_BUILD_NAME="$1"
RUN_ID="$2"
CONFIG_NAME="$3"
EVAL_EVERY_STEPS="${4:-$EVAL_EVERY_STEPS_DEFAULT}"

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
  local mp4a="${VIDEO_ROOT}/step_${step}_camA.mp4"
  local mp4b="${VIDEO_ROOT}/step_${step}_camB.mp4"
  local mp4c="${VIDEO_ROOT}/step_${step}_jack_overhead.mp4"
  if [ -f "${mp4a}" ] && [ -f "${mp4b}" ] && [ -f "${mp4c}" ]; then
    return 0
  fi
  # If any capture started, we consider it in-progress/done for this step to avoid duplicates.
  if [ -f "${VIDEO_ROOT}/step_${step}_camA/capture_started.txt" ] || [ -f "${VIDEO_ROOT}/step_${step}_camB/capture_started.txt" ] || [ -f "${VIDEO_ROOT}/step_${step}_jack_overhead/capture_started.txt" ]; then
    return 0
  fi
  return 1
}

run_eval_for_step() {
  local step="$1"
  local pot
  pot="$(pick_free_port)"

  local step_di_a="${VIDEO_ROOT}/step_${step}_camA"
  local step_di_b="${VIDEO_ROOT}/step_${step}_camB"
  local step_di_c="${VIDEO_ROOT}/step_${step}_jack_overhead"
  # Make absolute paths for Unity (relative paths depend on its working directory).
  if command -v realpath >/dev/null 2>&1; then
    step_di_a="$(realpath "${step_di_a}")"
    step_di_b="$(realpath "${step_di_b}")"
    step_di_c="$(realpath "${step_di_c}")"
  fi
  mkdir -p "${step_di_a}"
  mkdir -p "${step_di_b}"
  mkdir -p "${step_di_c}"

  echo "[watcher] validate target step=${step} on port=${pot}"

  local env_args=(
    --capture-dir "${step_di_a}" --capture-camera-a "CamA"
    --capture-dir-b "${step_di_b}" --capture-camera-b "CamB"
    --capture-dir-c "${step_di_c}"
    --capture-every "${CAPTURE_EVERY}"
    --capture-width "${CAPTURE_WIDTH}"
    --capture-height "${CAPTURE_HEIGHT}"
    --capture-msaa "${CAPTURE_MSAA}"
    --quit-after-episodes "${QUIT_AFTER_EPISODES}"
    --quit-after-seconds "${QUIT_AFTER_SECONDS}"
    --quit-delay-seconds "${QUIT_DELAY_SECONDS}"
  )

  local mlagents_cmd=(mlagents-learn "custom_configs/${CONFIG_NAME}.yaml"
    --inference
    --resume
    --env="${BUILD_PATH}"
    --run-id "${RUN_ID}"
    --base-port "${pot}"
    --num-envs 1
    --timeout-wait 600
    --env-args "${env_args[@]}"
  )

  if [ "${FORCE_XVFB}" = "1" ] && command -v xvfb-run >/dev/null 2>&1; then
    echo "[watcher] running inference under xvfb-run (FORCE_XVFB=1)"
    timeout --signal=INT "${MAX_EVAL_SECONDS}" xvfb-run -a "${mlagents_cmd[@]}" || true
  else
    if [ -z "${DISPLAY:-}" ]; then
      echo "[watcher] ERROR: DISPLAY is empty and FORCE_XVFB=0. Cannot render/capture video."
      return 1
    fi
    echo "[watcher] running inference with DISPLAY=${DISPLAY} (FORCE_XVFB=0)"
    timeout --signal=INT "${MAX_EVAL_SECONDS}" "${mlagents_cmd[@]}" || true
  fi

  ffmpeg_fps_from_meta() {
    local cap_dir="$1"
    local meta="${cap_dir}/capture_meta.txt"
    [ -f "$meta" ] || { echo ""; return; }
    local wall frames
    wall=$(grep -E '^wall_seconds=' "$meta" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '\r')
    frames=$(grep -E '^frame_count=' "$meta" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '\r')
    [ -n "$wall" ] && [ -n "$frames" ] || { echo ""; return; }
    awk -v w="$wall" -v f="$frames" 'BEGIN {
      if (w > 0.05 && f > 0) {
        x = f / w
        if (x > 60) x = 60
        if (x < 5) x = 5
        printf "%.6f", x
      }
    }'
  }

  pick_ffmpeg_fps() {
    local cap_dir="$1"
    local fps=""
    if [ "${CAPTURE_USE_META_FPS}" = "1" ]; then
      fps="$(ffmpeg_fps_from_meta "$cap_dir")"
    fi
    if [ -z "$fps" ]; then
      fps="${CAPTURE_VIDEO_FPS}"
    fi
    echo "$fps"
  }

  if command -v ffmpeg >/dev/null 2>&1; then
    pick_frame_ext() {
      local cap_dir="$1"
      if [ -f "${cap_dir}/frame_000000.jpg" ]; then
        echo "jpg"
      else
        echo "png"
      fi
    }

    local ff_fps_a ff_fps_b ff_fps_c
    ff_fps_a="$(pick_ffmpeg_fps "${step_di_a}")"
    ff_fps_b="$(pick_ffmpeg_fps "${step_di_b}")"
    ff_fps_c="$(pick_ffmpeg_fps "${step_di_c}")"
    echo "[watcher] ffmpeg fps camA=${ff_fps_a} camB=${ff_fps_b} overhead=${ff_fps_c} (meta=${CAPTURE_USE_META_FPS}, fallback=${CAPTURE_VIDEO_FPS})"

    local ext_a ext_b ext_c
    ext_a="$(pick_frame_ext "${step_di_a}")"
    ext_b="$(pick_frame_ext "${step_di_b}")"
    ext_c="$(pick_frame_ext "${step_di_c}")"
    echo "[watcher] frame ext camA=${ext_a} camB=${ext_b} overhead=${ext_c}"

    local out_mp4_a="${VIDEO_ROOT}/step_${step}_camA.mp4"
    ffmpeg -y -framerate "${ff_fps_a}" -i "${step_di_a}/frame_%06d.${ext_a}" -c:v libx264 -pix_fmt yuv420p -crf "${FFMPEG_CRF}" -preset "${FFMPEG_PRESET}" "${out_mp4_a}" >/dev/null 2>&1 || true
    echo "[watcher] saved ${out_mp4_a}"

    local out_mp4_b="${VIDEO_ROOT}/step_${step}_camB.mp4"
    ffmpeg -y -framerate "${ff_fps_b}" -i "${step_di_b}/frame_%06d.${ext_b}" -c:v libx264 -pix_fmt yuv420p -crf "${FFMPEG_CRF}" -preset "${FFMPEG_PRESET}" "${out_mp4_b}" >/dev/null 2>&1 || true
    echo "[watcher] saved ${out_mp4_b}"

    local out_mp4_c="${VIDEO_ROOT}/step_${step}_jack_overhead.mp4"
    ffmpeg -y -framerate "${ff_fps_c}" -i "${step_di_c}/frame_%06d.${ext_c}" -c:v libx264 -pix_fmt yuv420p -crf "${FFMPEG_CRF}" -preset "${FFMPEG_PRESET}" "${out_mp4_c}" >/dev/null 2>&1 || true
    echo "[watcher] saved ${out_mp4_c}"
  else
    echo "[watcher] ffmpeg not found; kept PNGs in ${step_di_a}, ${step_di_b}, ${step_di_c}"
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

