"""
01_asr_whisper_simple.py
Quick ASR -> produce approximate per-word timings by splitting whisper segments.

Usage:
  python src/01_asr_whisper_simple.py --audio audio/clip1.wav --out data/transcripts_raw/clip1_words.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import whisper  # openai whisper

def simple_word_timestamps_from_segments(segments):
    """
    segments: list of dicts with keys: 'start', 'end', 'text'
    Returns: list of {'word': w, 'start': s, 'end': e} approximated per word
    Strategy:
      - split segment text into words (naive whitespace split)
      - assign each word an equal share of the segment duration
      - start time of first word = segment.start, end of last = segment.end
    Note: crude but useful bootstrap for forced alignment or prosody steps.
    """
    out = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if text == "":
            continue
        words = text.split()  # naive split; later you can handle punctuation
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        seg_dur = seg_end - seg_start
        if len(words) == 0:
            continue
        # allocate equal duration per word
        per_word = seg_dur / len(words)
        for i, w in enumerate(words):
            s = seg_start + i * per_word
            e = s + per_word
            out.append({"word": w, "start": round(s, 4), "end": round(e, 4)})
    return out

def main(audio_path: Path, out_csv: Path, whisper_model: str = "small"):
    print("Loading Whisper model:", whisper_model)
    model = whisper.load_model(whisper_model)  # small/medium/large - pick depending on your machine
    print("Transcribing (this may take some time)...")
    result = model.transcribe(str(audio_path), verbose=False)  # returns segments + text
    # ensure transcripts folder exists
    text_out_dir = Path("data/transcripts_raw")
    text_out_dir.mkdir(parents=True, exist_ok=True)
    # write full transcript text to data/transcripts_raw/<audio_basename>_text.txt
    transcript_path = text_out_dir / f"{audio_path.stem}_text.txt"
    with open(transcript_path, "w", encoding="utf-8") as tf:
        tf.write(result.get("text", "").strip() + "\n")
    # get segments (list of dicts with start/end/text)
    segments = result.get("segments", [])

    print(f"Got {len(segments)} segments from whisper.")
    word_rows = simple_word_timestamps_from_segments(segments)

    df = pd.DataFrame(word_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Wrote simple word timestamps to:", out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=False, help="CSV output path")
    parser.add_argument("--model", default="small", help="whisper model name small/medium/large")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    out_csv = Path(args.out) if args.out else Path("data/transcripts_raw") / (audio_path.stem + "_words.csv")
    main(audio_path, out_csv, whisper_model=args.model)
