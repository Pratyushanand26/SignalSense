#!/usr/bin/env python3
"""
src/03_forced_align.py

POST audio + transcript to a Gentle server and save:
 - data/alignments/<base>_gentle.json (raw)
 - data/alignments/<base>_words.csv  (parsed CSV with word,start,end,case)

Usage:
  python src/03_forced_align.py --audio audio/clip1.wav --transcript data/transcripts_raw/clip1_text.txt
"""
import argparse
import json
from pathlib import Path
import time

import requests
import pandas as pd


GENTLE_URL_DEFAULT = "http://localhost:8765/transcriptions"


def call_gentle_server(audio_path: Path, transcript_path: Path, gentle_url: str = GENTLE_URL_DEFAULT,
                       out_json: Path = None, timeout=300):
    """
    POST audio + transcript to a Gentle server and save returned JSON to out_json.
    Returns tuple (parsed_json, out_json_path).
    """
    if out_json is None:
        out_json = Path("data/alignments") / f"{audio_path.stem}_gentle.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # read transcript text (gentle expects a 'transcript' form field)
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    data = {"transcript": transcript_text}

    print(f"[gentle] POSTing to {gentle_url} (audio={audio_path}, transcript={transcript_path}) ...")
    start = time.time()
    # open audio file in a context manager so it is closed properly
    with open(audio_path, "rb") as audio_file:
        files = {"audio": (audio_path.name, audio_file)}
        resp = requests.post(gentle_url, files=files, data=data, timeout=timeout)
    # raise for HTTP errors
    resp.raise_for_status()

    j = resp.json()
    elapsed = time.time() - start
    out_json.write_text(json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[gentle] Received JSON (saved to {out_json}) in {elapsed:.1f}s")
    return j, out_json


def parse_gentle_to_rows(gentle_json: dict):
    rows = []
    words = gentle_json.get("words", [])
    for w in words:
        case = w.get("case")  # e.g. 'success' or 'not-found'
        word_text = w.get("alignedWord") or w.get("word") or ""
        start = w.get("start", None)
        end = w.get("end", None)
        # cast floats when possible
        try:
            start = float(start) if start is not None else None
        except Exception:
            start = None
        try:
            end = float(end) if end is not None else None
        except Exception:
            end = None
        rows.append({"word": word_text, "start": start, "end": end, "case": case})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="Path to WAV audio")
    p.add_argument("--transcript", required=True, help="Path to full transcript text file")
    p.add_argument("--gentle_url", default=GENTLE_URL_DEFAULT, help="Gentle server transcription endpoint")
    p.add_argument("--out_json", required=False, help="Where to save Gentle raw json (optional)")
    p.add_argument("--out_csv", required=False, help="Where to save parsed CSV (optional)")
    args = p.parse_args()

    audio_path = Path(args.audio)
    transcript_path = Path(args.transcript)
    out_json = Path(args.out_json) if args.out_json else Path("data/alignments") / f"{audio_path.stem}_gentle.json"

    j, saved_json = call_gentle_server(audio_path, transcript_path, gentle_url=args.gentle_url, out_json=out_json)

    rows = parse_gentle_to_rows(j)
    df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv) if args.out_csv else Path("data/alignments") / f"{audio_path.stem}_words.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[gentle] Parsed alignment CSV written to: {out_csv}")
    # summary
    aligned = df['case'].eq('success').sum() if 'case' in df.columns else None
    print(f"[gentle] Summary: total_words={len(df)}, aligned={aligned}")


if __name__ == "__main__":
    main()
