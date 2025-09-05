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


GENTLE_URL_DEFAULT = "http://localhost:8765"

def call_gentle_server(audio_path, transcript_path, gentle_url="http://localhost:8765", out_json=None, timeout=600):
    """
    Calls Gentle forced aligner via HTTP and returns JSON + optional saved file.
    """

    print(f"[gentle] POSTing to {gentle_url}/transcriptions (audio={audio_path}, transcript={transcript_path}) ...")
    start = time.time()

    with open(audio_path, "rb") as audio_file, open(transcript_path, "rb") as transcript_file:
        files = {
            "audio": (audio_path.name, audio_file, "audio/wav"),
            "transcript": (transcript_path.name, transcript_file, "text/plain")
        }
        resp = requests.post(f"{gentle_url.rstrip('/')}/transcriptions?async=false", files=files, timeout=timeout)



    # Raise for HTTP errors (4xx/5xx)
    resp.raise_for_status()

    # Debugging: print a snippet of the response
    print("Gentle raw response (first 500 chars):")
    print(resp.text[:500])

    # Parse JSON (Gentle should return JSON, not HTML)
    try:
        j = resp.json()
    except json.JSONDecodeError as e:
        print("ERROR: Gentle did not return valid JSON. Probably sent back HTML.")
        raise

    # Save JSON if requested
    saved_json = None
    if out_json is not None:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(j, f, indent=2, ensure_ascii=False)
        saved_json = out_json
        print(f"Wrote Gentle alignment JSON -> {saved_json}")

    print(f"[gentle] Done in {time.time()-start:.1f}s")
    return j, saved_json


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
