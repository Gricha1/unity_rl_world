#!/usr/bin/env bash
set -eu
set -o pipefail

# Build a single training-progress video from per-checkpoint camA mp4 files.
#
# Inputs expected (produced by validate_video_watcher_fixed.bash):
#   results/<run_id>/videos/step_<N>_camA.mp4
#
# Usage:
#   bash train_scripts/make_training_video_camA.bash <run_id> [output_mp4]
#
# Optional speed map:
#   Create a text file: results/<run_id>/videos/speed_map_camA.txt
#   Format (one per line):
#     <step> <speed>
#   Where speed=2 means 2x faster, speed=0.5 means 2x slower.
#   Steps not listed use DEFAULT_SPEED (below).
#
# Notes:
# - Requires ffmpeg.
# - Re-encodes segments that need speed changes, then concatenates everything.

cd "$(dirname "$0")"
cd ..

if [ -z "${1:-}" ]; then
  echo "Usage: bash train_scripts/make_training_video_camA.bash <run_id> [output_mp4]"
  exit 1
fi

RUN_ID="$1"
OUT_MP4="${2:-results/${RUN_ID}/videos/video_summary/training_camA_full.mp4}"

VIDEO_DIR="results/${RUN_ID}/videos"
SUMMARY_DIR="${VIDEO_DIR}/video_summary"
SPEED_MAP_FILE="${SUMMARY_DIR}/speed_map_camA.txt"

# Default playback speed for all steps (2 = 2x faster).
DEFAULT_SPEED="6"

command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found in PATH"; exit 1; }

if [ ! -d "${VIDEO_DIR}" ]; then
  echo "ERROR: ${VIDEO_DIR} not found. Run validate_video_watcher_fixed.bash first."
  exit 1
fi

tmp_dir="${SUMMARY_DIR}/__concat_tmp_camA"
mkdir -p "${tmp_dir}"

read_speed_for_step() {
  local step="$1"
  if [ -f "${SPEED_MAP_FILE}" ]; then
    # lines: "<step> <speed>" with optional comments after '#'
    awk -v s="$step" '
      {
        gsub(/\r/, "", $0)
        sub(/#.*/, "", $0)
        if (NF >= 2 && $1 == s) { print $2; found=1; exit 0 }
      }
      END { if (!found) exit 1 }
    ' "${SPEED_MAP_FILE}" 2>/dev/null || true
  fi
}

list_steps() {
  python - <<'PY' "${VIDEO_DIR}"
import os, re, sys
d = sys.argv[1]
rx = re.compile(r"^step_(\d+)_camA\.mp4$")
steps = []
for name in os.listdir(d):
    m = rx.match(name)
    if m:
        steps.append(int(m.group(1)))
for s in sorted(steps):
    print(s)
PY
}

steps="$(list_steps)"
if [ -z "${steps}" ]; then
  echo "ERROR: No step_*_camA.mp4 found in ${VIDEO_DIR}"
  exit 1
fi

echo "[make_video] run=${RUN_ID}"
echo "[make_video] input dir: ${VIDEO_DIR}"
echo "[make_video] output: ${OUT_MP4}"
echo "[make_video] summary dir: ${SUMMARY_DIR}"
echo "[make_video] default speed: ${DEFAULT_SPEED}x"
if [ -f "${SPEED_MAP_FILE}" ]; then
  echo "[make_video] speed map: ${SPEED_MAP_FILE}"
else
  echo "[make_video] speed map: (none)"
fi

# Absolute output path (compute BEFORE any cd).
OUT_ABS="$(python - <<'PY' "${OUT_MP4}"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"

concat_list="${tmp_dir}/concat_list.txt"
: > "${concat_list}"

while read -r step; do
  [ -n "${step}" ] || continue
  in_mp4="${VIDEO_DIR}/step_${step}_camA.mp4"
  [ -f "${in_mp4}" ] || continue

  speed="$(read_speed_for_step "${step}")"
  if [ -z "${speed}" ]; then
    speed="${DEFAULT_SPEED}"
  fi

  # Normalize speeded segment (re-encode) so concat is reliable.
  # For speed factor 'v': setpts=PTS/v  (v>1 => faster, v<1 => slower)
  out_seg="${tmp_dir}/step_${step}_camA__spd_${speed}.mp4"
  if [ ! -f "${out_seg}" ]; then
    echo "[make_video] step=${step} speed=${speed}x"
    ffmpeg -y -i "${in_mp4}" \
      -filter:v "setpts=PTS/${speed}" \
      -an -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium \
      "${out_seg}" >/dev/null 2>&1
  fi

  # ffmpeg concat demuxer list file
  # Use basename because concat paths are resolved relative to concat_list directory.
  printf "file '%s'\n" "$(basename "${out_seg}")" >> "${concat_list}"
done <<< "${steps}"

mkdir -p "$(dirname "${OUT_MP4}")"

echo "[make_video] concatenating..."
(
  cd "${tmp_dir}"
  ffmpeg -y -loglevel error -f concat -safe 0 -i "concat_list.txt" \
    -an -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium \
    "${OUT_ABS}" >/dev/null 2>&1
)

echo "[make_video] done: ${OUT_MP4}"

