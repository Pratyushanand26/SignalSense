#!/usr/bin/env bash
#
# convert_to_wav.sh
#
# Ensure input audio is a WAV (PCM 16-bit) with target sample rate and mono channels.
# If input is already WAV and matches sample rate & channels, the file is left as-is
# (unless --force is specified, in which case it will be reconverted/overwritten).
#
# Requires: ffmpeg and ffprobe available in PATH.
#
# Usage examples:
#  ./scripts/convert_to_wav.sh --input audio/clip1.mp3 --out audio/clip1.wav --sr 16000 --force
#  ./scripts/convert_to_wav.sh --dir audio/ --sr 16000
#

set -euo pipefail

# Default parameters
TARGET_SR=16000        # target sample rate in Hz
TARGET_CHANNELS=1      # 1 = mono, 2 = stereo
FORCE=0                # if 1, overwrite/convert even if already WAV with desired params
IN_DIR=""
IN_FILE=""
OUT_FILE=""

print_usage() {
  cat <<EOF
Usage:
  $0 --input <input-file> [--out <output-file>] [--sr <sample-rate>] [--channels <1|2>] [--force]
  $0 --dir <directory> [--sr <sample-rate>] [--channels <1|2>] [--force]

Options:
  --input    Single input audio file to convert.
  --out      Output path for the converted wav (optional). If omitted, writes next to input with .wav extension.
  --dir      Batch convert all audio files in directory (non-recursive).
  --sr       Target sample rate (default: ${TARGET_SR}).
  --channels Target channels (1=mono, 2=stereo). Default: ${TARGET_CHANNELS}.
  --force    If provided, convert/overwrite even when file already matches desired format.
  --help     Show this help.
EOF
}

# parse args (simple)
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --input) IN_FILE="$2"; shift; shift ;;
    --out) OUT_FILE="$2"; shift; shift ;;
    --dir) IN_DIR="$2"; shift; shift ;;
    --sr) TARGET_SR="$2"; shift; shift ;;
    --channels) TARGET_CHANNELS="$2"; shift; shift ;;
    --force) FORCE=1; shift ;;
    --help) print_usage; exit 0 ;;
    *) echo "Unknown arg: $1"; print_usage; exit 1 ;;
  esac
done

# check ffmpeg installed
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is not installed or not in PATH. Install ffmpeg and retry." >&2
  exit 2
fi
if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ERROR: ffprobe is not installed or not in PATH. Install ffmpeg (includes ffprobe) and retry." >&2
  exit 2
fi

# helper: process one file
process_one() {
  local inpath="$1"
  local outpath="$2"

  # make output directory if needed
  mkdir -p "$(dirname "$outpath")"

  # get extension and normalize to lowercase
  ext="${inpath##*.}"
  ext_lc="$(printf '%s' "$ext" | tr '[:upper:]' '[:lower:]')"

  # gather audio properties via ffprobe: sample_rate, channels, codec_name, format_name
  # the ffprobe command below prints sample_rate and channels on separate lines
  probe=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate,channels,codec_name -of default=nokey=1:noprint_wrappers=1 "$inpath" 2>/dev/null || true)
  # probe may be empty if no audio stream
  if [[ -z "$probe" ]]; then
    echo "WARNING: '$inpath' has no audio stream or ffprobe failed. Skipping."
    return
  fi
  # read into variables
  # probe typically returns:
  # <sample_rate>
  # <channels>
  # <codec_name>
  read -r probe_sr probe_ch probe_codec <<< "$(echo "$probe" | tr '\n' ' ')"

  # Some ffprobe builds return sample rate as a float; coerce to int
  current_sr=${probe_sr%.*}
  current_ch="$probe_ch"
  codec="$probe_codec"

  # decide whether conversion is required
  needs_convert=0
  if [[ "$ext_lc" != "wav" ]]; then
    needs_convert=1
  fi
  if [[ "$current_sr" -ne "$TARGET_SR" ]]; then
    needs_convert=1
  fi
  if [[ "$current_ch" -ne "$TARGET_CHANNELS" ]]; then
    needs_convert=1
  fi
  # also convert if codec is not pcm_s16le (we want 16-bit PCM WAV)
  if [[ "$codec" != "pcm_s16le" ]]; then
    # if already wav but codec different (e.g., wavpack or float), convert
    needs_convert=1
  fi

  if [[ "$needs_convert" -eq 0 && "$FORCE" -eq 0 ]]; then
    echo "SKIP: '$inpath' is already WAV with ${current_sr}Hz, ${current_ch}ch and codec ${codec}. Output: $outpath"
    # if requested outpath differs from input path (e.g., user wants a copy), create a copy
    if [[ "$inpath" != "$outpath" ]]; then
      echo "Copying original file to output path."
      cp -n "$inpath" "$outpath" || true
    fi
    return
  fi

  # build ffmpeg command
  # -y overwrite, -n no-overwrite (we control via FORCE)
  if [[ "$FORCE" -eq 1 ]]; then
    overwrite_flag="-y"
  else
    overwrite_flag="-n"
  fi

  echo "CONVERT: '$inpath' -> '$outpath' (sr=${TARGET_SR}, ch=${TARGET_CHANNELS}, codec=pcm_s16le)"
  ffmpeg $overwrite_flag -hide_banner -loglevel error -y -i "$inpath" \
    -vn \
    -ac "$TARGET_CHANNELS" \
    -ar "$TARGET_SR" \
    -acodec pcm_s16le \
    -f wav \
    "$outpath"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "ERROR: ffmpeg failed converting '$inpath' (exit $rc)" >&2
    return 1
  fi
  echo "DONE: $outpath"
}

# if directory mode
if [[ -n "$IN_DIR" ]]; then
  if [[ ! -d "$IN_DIR" ]]; then
    echo "ERROR: directory '$IN_DIR' not found." >&2
    exit 3
  fi
  shopt -s nullglob
  # process common audio extensions (non-recursive)
  for f in "$IN_DIR"/*.{wav,mp3,m4a,flac,ogg,wma,aiff,aif,webm}; do
    # skip if literal pattern unchanged
    [[ -e "$f" ]] || continue
    out="$IN_DIR/$(basename "${f%.*}").wav"
    process_one "$f" "$out"
  done
  exit 0
fi

# single file mode
if [[ -z "$IN_FILE" ]]; then
  echo "ERROR: no input file specified. Use --input or --dir." >&2
  print_usage
  exit 4
fi

if [[ ! -f "$IN_FILE" ]]; then
  echo "ERROR: input file '$IN_FILE' not found." >&2
  exit 5
fi

# determine default out path if none provided
if [[ -z "$OUT_FILE" ]]; then
  OUT_FILE="$(dirname "$IN_FILE")/$(basename "${IN_FILE%.*}").wav"
fi

process_one "$IN_FILE" "$OUT_FILE"
