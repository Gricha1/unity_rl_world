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
#   bash train_scripts/make_training_video_camA.bash <run_id> <max_step>
#   bash train_scripts/make_training_video_camA.bash <run_id> <output_mp4> <max_step>
#
# If the second argument is all digits, it is max_step (only checkpoints with step <= max_step).
# Otherwise second is output path; optional third argument max_step (digits only).
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

# Python executable (WSL часто имеет только python3).
PY_BIN="python"
if ! command -v python >/dev/null 2>&1; then
  PY_BIN="python3"
fi

if [ -z "${1:-}" ]; then
  echo "Usage: bash train_scripts/make_training_video_camA.bash <run_id> [output_mp4]"
  echo "       bash train_scripts/make_training_video_camA.bash <run_id> <max_step>"
  echo "       bash train_scripts/make_training_video_camA.bash <run_id> <output_mp4> <max_step>"
  echo "  max_step: integer, only step_*_camA.mp4 with step <= max_step are joined."
  exit 1
fi

RUN_ID="$1"
MAX_STEP=""
OUT_MP4_DEFAULT="results/${RUN_ID}/videos/video_summary/training_camA_full.mp4"

if [ -z "${2:-}" ]; then
  OUT_MP4="${OUT_MP4_DEFAULT}"
elif [[ "${2}" =~ ^[0-9]+$ ]]; then
  MAX_STEP="${2}"
  OUT_MP4="${3:-${OUT_MP4_DEFAULT}}"
else
  OUT_MP4="$2"
  if [[ "${3:-}" =~ ^[0-9]+$ ]]; then
    MAX_STEP="$3"
  fi
fi

VIDEO_DIR="results/${RUN_ID}/videos"
SUMMARY_DIR="${VIDEO_DIR}/video_summary"
SPEED_MAP_FILE="${SUMMARY_DIR}/speed_map_camA.txt"

# Default playback speed for all steps (2 = 2x faster).
DEFAULT_SPEED="6"

# If 1, re-encode all segments even if cached. Можно задать извне: FORCE_REENCODE=1 bash ...
FORCE_REENCODE="${FORCE_REENCODE:-0}"

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
  "${PY_BIN}" - <<'PY' "${VIDEO_DIR}"
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

debug_list_seen_videos() {
  "${PY_BIN}" - <<'PY' "${VIDEO_DIR}"
import os, re, sys
d = sys.argv[1]
rx = re.compile(r"^step_(\d+)_camA\.mp4$")

all_mp4 = [n for n in os.listdir(d) if n.lower().endswith(".mp4")]
camA = []
other_step_mp4 = []
for n in all_mp4:
    m = rx.match(n)
    if m:
        camA.append((int(m.group(1)), n))
    elif n.startswith("step_") and "_camA" in n:
        other_step_mp4.append(n)

camA.sort()
print(f"[make_video] mp4 files in dir: {len(all_mp4)}")
print(f"[make_video] step_*_camA.mp4 matched: {len(camA)}")
for s, n in camA:
    print(f"[make_video]   seen step={s} file={n}")
if other_step_mp4:
    print(f"[make_video] step_*..._camA.mp4 that did NOT match pattern ({len(other_step_mp4)}):")
    for n in sorted(other_step_mp4)[:200]:
        print(f"[make_video]   nonmatch file={n}")
PY
}

pick_fontfile() {
  # drawtext works best with an explicit font file; fallback to ffmpeg default if not found.
  for f in \
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" \
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" \
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
  do
    if [ -f "${f}" ]; then
      echo "${f}"
      return 0
    fi
  done
  echo ""
}

format_step_label() {
  # Convert raw step number to a human-readable label (rounded to nearest 50k).
  # Examples: 49940 -> 50 000 шагов, 99804 -> 100 000 шагов
  "${PY_BIN}" - <<'PY' "$1"
import math, sys
step = int(sys.argv[1])
rounded = int(round(step / 50000.0) * 50000)
print(f"{rounded:,}".replace(",", " ") + " шагов")
PY
}

# Avoid `while read` under `set -e`: EOF from read exits with status 1 and can abort the script early.
mapfile -t STEPS < <(list_steps)
if [ "${#STEPS[@]}" -eq 0 ]; then
  echo "ERROR: No step_*_camA.mp4 found in ${VIDEO_DIR}"
  exit 1
fi

if [ -n "${MAX_STEP}" ]; then
  _filtered=()
  for _s in "${STEPS[@]}"; do
    if [ "${_s}" -le "${MAX_STEP}" ]; then
      _filtered+=("${_s}")
    fi
  done
  STEPS=("${_filtered[@]}")
fi

if [ "${#STEPS[@]}" -eq 0 ]; then
  echo "ERROR: No checkpoints left after max_step=${MAX_STEP:-unset} filter."
  exit 1
fi

debug_list_seen_videos

echo "[make_video] run=${RUN_ID}"
if [ -n "${MAX_STEP}" ]; then
  echo "[make_video] max_step=${MAX_STEP} (steps <= max_step only)"
fi
echo "[make_video] input dir: ${VIDEO_DIR}"
echo "[make_video] output: ${OUT_MP4}"
echo "[make_video] summary dir: ${SUMMARY_DIR}"
echo "[make_video] default speed: ${DEFAULT_SPEED}x"
if [ -f "${SPEED_MAP_FILE}" ]; then
  echo "[make_video] speed map: ${SPEED_MAP_FILE}"
else
  echo "[make_video] speed map: (none)"
fi

echo "[make_video] checkpoints in concat order: ${#STEPS[@]}"

# Absolute output path (compute BEFORE any cd).
OUT_ABS="$("${PY_BIN}" - <<'PY' "${OUT_MP4}"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"

concat_list="${tmp_dir}/concat_list.txt"
: > "${concat_list}"

FONTFILE="$(pick_fontfile)"
if [ -n "${FONTFILE}" ]; then
  echo "[make_video] drawtext font: ${FONTFILE}"
else
  echo "[make_video] drawtext font: (ffmpeg default)"
fi

SEG_OK=0
SEG_FAIL=0
TOTAL="${#STEPS[@]}"
i=0
for step in "${STEPS[@]}"; do
  i=$((i + 1))
  [ -n "${step}" ] || {
    echo "[make_video] FAIL ${i}/${TOTAL} step=- empty_slot"
    SEG_FAIL=$((SEG_FAIL + 1))
    continue
  }

  in_mp4="${VIDEO_DIR}/step_${step}_camA.mp4"

  if [ ! -f "${in_mp4}" ]; then
    echo "[make_video] FAIL ${i}/${TOTAL} step=${step} missing_input"
    SEG_FAIL=$((SEG_FAIL + 1))
    continue
  fi

  speed="$(read_speed_for_step "${step}")"
  if [ -z "${speed}" ]; then
    speed="${DEFAULT_SPEED}"
  fi

  label="$(format_step_label "${step}")"

  # Normalize speeded segment (re-encode) so concat is reliable.
  # For speed factor 'v': setpts=PTS/v  (v>1 => faster, v<1 => slower)
  out_seg="${tmp_dir}/step_${step}_camA__spd_${speed}.mp4"
  if [ "${FORCE_REENCODE}" = "1" ] && [ -f "${out_seg}" ]; then
    rm -f "${out_seg}" 2>/dev/null || true
  fi

  need_encode=1
  if [ -f "${out_seg}" ] && [ -s "${out_seg}" ]; then
    need_encode=0
  fi

  if [ "${need_encode}" -eq 1 ]; then
    if [ -n "${FONTFILE}" ]; then
      draw="drawtext=fontfile=${FONTFILE}:text='${label}':x=20:y=20:fontsize=48:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=12"
    else
      draw="drawtext=text='${label}':x=20:y=20:fontsize=48:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=12"
    fi
    if ! ffmpeg -y -loglevel error -i "${in_mp4}" \
      -filter:v "setpts=PTS/${speed},${draw}" \
      -an -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium \
      "${out_seg}"; then
      echo "[make_video] FAIL ${i}/${TOTAL} step=${step} ffmpeg"
      SEG_FAIL=$((SEG_FAIL + 1))
      continue
    fi
    seg_note="encode"
  else
    seg_note="cache"
  fi

  if [ ! -s "${out_seg}" ]; then
    echo "[make_video] FAIL ${i}/${TOTAL} step=${step} empty_output"
    SEG_FAIL=$((SEG_FAIL + 1))
    continue
  fi

  # ffmpeg concat demuxer list file
  # Use basename because concat paths are resolved relative to concat_list directory.
  printf "file '%s'\n" "$(basename "${out_seg}")" >> "${concat_list}"
  SEG_OK=$((SEG_OK + 1))
  echo "[make_video] OK ${i}/${TOTAL} step=${step} ${seg_note}"
done

echo "[make_video] segment summary: ok=${SEG_OK} failed=${SEG_FAIL} total_checkpoints=${TOTAL}"
if [ "${SEG_OK}" -eq 0 ]; then
  echo "[make_video] ERROR: no segments were produced; aborting concat."
  exit 1
fi
if [ "${SEG_FAIL}" -gt 0 ]; then
  echo "[make_video] WARN: skipped segments: ${SEG_FAIL}"
fi

mkdir -p "$(dirname "${OUT_MP4}")"

clist_lines="$(wc -l < "${concat_list}" | tr -d ' ')"
echo "[make_video] join ${clist_lines} segments -> $(basename "${OUT_MP4}") (full concat)"
if ! (
  cd "${tmp_dir}"
  ffmpeg -y -loglevel error -f concat -safe 0 -i "concat_list.txt" \
    -an -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium \
    "${OUT_ABS}"
); then
  echo "[make_video] FAIL concat_ffmpeg"
  exit 1
fi

if [ ! -s "${OUT_ABS}" ]; then
  echo "[make_video] FAIL output_empty"
  exit 1
fi

echo "[make_video] OK joined_video $(basename "${OUT_MP4}")"

