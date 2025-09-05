#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input-audio-file>"
  exit 1
fi

INPUT="$1"
BASE=$(basename "${INPUT%.*}")
AUDIO_DIR=$(dirname "$INPUT")
WAV_PATH="${AUDIO_DIR}/${BASE}.wav"

mkdir -p data/transcripts_raw data/vad data/alignments data/prosody data/annotated

echo ">>> [0] Convert to WAV (if needed)"
./scripts/convert_to_wav.sh --input "$INPUT" --out "$WAV_PATH" --sr 16000 --channels 1

echo ">>> [1] ASR (whisper quick bootstrap) -> data/transcripts_raw/${BASE}_words.csv + ${BASE}_text.txt"
python src/01_asr_whisper_simple.py --audio "$WAV_PATH" --out "data/transcripts_raw/${BASE}_words.csv" --model small

# ensure full-text transcript exists
TRANSCRIPT_TXT="data/transcripts_raw/${BASE}_text.txt"
if [[ ! -f "$TRANSCRIPT_TXT" ]]; then
  echo "Creating fallback transcript text from words CSV..."
  if [[ -f "data/transcripts_raw/${BASE}_words.csv" ]]; then
    tail -n +2 "data/transcripts_raw/${BASE}_words.csv" | awk -F, '{print $1}' | tr '\n' ' ' | sed 's/  */ /g' > "$TRANSCRIPT_TXT"
    echo "" >> "$TRANSCRIPT_TXT"
    echo "Wrote fallback transcript: $TRANSCRIPT_TXT"
  else
    echo "ERROR: ASR output missing." && exit 2
  fi
else
  echo "Found transcript: $TRANSCRIPT_TXT"
fi

echo ">>> [2] VAD -> data/vad/${BASE}_vad.csv"
python src/02_vad_webrtc.py --audio "$WAV_PATH" --out "data/vad/${BASE}_vad.csv"

echo ">>> [3] Ensure Gentle running; starting via docker-compose if not."
GENTLE_URL="http://localhost:8765"
check_gentle() {
  curl -sSf --max-time 2 "${GENTLE_URL}/" >/dev/null 2>&1
}
if check_gentle; then
  echo "Gentle reachable."
else
  echo "Starting Gentle with docker-compose..."
  docker-compose up -d gentle
  ATT=0; MAX=30
  until check_gentle || [[ $ATT -ge $MAX ]]; do
    printf "."
    sleep 1
    ATT=$((ATT+1))
  done
  echo ""
  if ! check_gentle; then
    echo "WARNING: Gentle not reachable after wait; alignment may fail."
  else
    echo "Gentle is up."
  fi
fi

echo ">>> [4] Forced alignment (Gentle) -> data/alignments/${BASE}_words.csv"
python src/03_forced_align.py --audio "$WAV_PATH" --transcript "$TRANSCRIPT_TXT" --out_json "data/alignments/${BASE}_gentle.json" --out_csv "data/alignments/${BASE}_words.csv"

echo ">>> [5] Prosody extraction -> data/prosody/${BASE}_prosody.csv"
python src/04_prosody.py --audio "$WAV_PATH" --align "data/alignments/${BASE}_words.csv" --out "data/prosody/${BASE}_prosody.csv"

echo ">>> [6] Disfluency heuristics -> data/annotated/${BASE}_events.json"
python src/05_disfluency.py --align "data/alignments/${BASE}_words.csv" --vad "data/vad/${BASE}_vad.csv" --prosody "data/prosody/${BASE}_prosody.csv" --out "data/annotated/${BASE}_events.json"

echo ">>> [7] Merge annotations -> data/annotated/${BASE}_annotated.txt"
python src/06_merger.py --align "data/alignments/${BASE}_words.csv" --prosody "data/prosody/${BASE}_prosody.csv" --vad "data/vad/${BASE}_vad.csv" --events "data/annotated/${BASE}_events.json" --out "data/annotated/${BASE}_annotated.txt"

echo ">>> [8] Create ML-ready 5-line file -> data/annotated/${BASE}_ml_5lines.txt"
python src/07_ml_prep_split.py --annot "data/annotated/${BASE}_annotated.txt" --align "data/alignments/${BASE}_words.csv" --out "data/annotated/${BASE}_ml_5lines.txt" --n 5

echo "✅ Pipeline finished. Produced:"
echo "  - transcript: $TRANSCRIPT_TXT"
echo "  - alignments: data/alignments/${BASE}_words.csv"
echo "  - prosody: data/prosody/${BASE}_prosody.csv"
echo "  - events: data/annotated/${BASE}_events.json"
echo "  - annotated text: data/annotated/${BASE}_annotated.txt"
echo "  - ML 5-lines: data/annotated/${BASE}_ml_5lines.txt"
