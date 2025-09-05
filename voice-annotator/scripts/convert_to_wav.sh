#!/usr/bin/env bash
set -euo pipefail

# convert_to_wav.sh
# Usage:
#   ./scripts/convert_to_wav.sh --input INPUT_FILE [--out OUT_FILE] [--sr 16000] [--channels 1]
#
# Notes:
# - If ffprobe is available we check input format and copy directly if it already matches.
# - If OUT is the same as INPUT we write to a temp file then atomically replace to avoid ffmpeg input/output conflict.

INPUT=""
OUT=""
SR=16000
CH=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --input) INPUT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --sr) SR="$2"; shift 2 ;;
    --channels) CH="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "Usage: $0 --input INPUT_FILE [--out OUT_FILE] [--sr 16000] [--channels 1]"
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: input file not found: $INPUT" >&2
  exit 2
fi

# default OUT if not provided: same folder, basename_conv.wav
if [[ -z "$OUT" ]]; then
  base="$(basename "$INPUT")"
  name="${base%.*}"
  OUT="$(dirname "$INPUT")/${name}_conv.wav"
fi

# Ensure ffmpeg present
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is not installed or not in PATH. Install ffmpeg and retry." >&2
  exit 3
fi

# Helper to run ffprobe (if present) to inspect input format
probe_ok=false
if command -v ffprobe >/dev/null 2>&1; then
  probe_ok=true
  # get sample_rate, channels, codec_name (each on its own line)
  probe_out=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate,channels,codec_name -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null || true)
  # ffprobe may output empty if no audio stream
  if [[ -n "$probe_out" ]]; then
    # read into array
    IFS=$'\n' read -rd '' -a _arr <<<"$probe_out" || true
    probe_sr="${_arr[0]:-}"
    probe_ch="${_arr[1]:-}"
    probe_codec="${_arr[2]:-}"
  else
    probe_sr=""
    probe_ch=""
    probe_codec=""
  fi
fi

echo "CONVERT CHECK: input='$INPUT' -> out='$OUT' (sr=$SR, channels=$CH)"
# If input already matches desired (sr,channels,codec pcm_s16le), just copy (fast)
if [[ "$probe_ok" = true && -n "$probe_sr" && -n "$probe_ch" && -n "$probe_codec" ]]; then
  if [[ "$probe_sr" == "$SR" && "$probe_ch" == "$CH" && "$probe_codec" == "pcm_s16le" ]]; then
    echo "Input audio already in desired format (sr=$probe_sr ch=$probe_ch codec=$probe_codec). Copying to out."
    cp -f "$INPUT" "$OUT"
    echo "Wrote $OUT"
    exit 0
  fi
fi

# If OUT equals INPUT, write to a temp and move over
if [[ "$(realpath "$INPUT")" == "$(realpath "$OUT")" ]]; then
  tmp_out="$(mktemp "${OUT}.tmp.XXXXXX")"
  echo "Converting (writing to temp) -> $tmp_out"
  ffmpeg -y -i "$INPUT" -ar "${SR}" -ac "${CH}" -c:a pcm_s16le "$tmp_out"
  mv -f "$tmp_out" "$OUT"
  echo "Replaced original with converted file: $OUT"
  exit 0
fi

# Normal conversion to OUT
ffmpeg -y -i "$INPUT" -ar "${SR}" -ac "${CH}" -c:a pcm_s16le "$OUT"
echo "Wrote $OUT"
