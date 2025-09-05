#!/usr/bin/env python3
"""
src/05_disfluency.py

Run simple heuristics on alignment + vad + prosody to produce a list of events.
Output: data/annotated/<base>_events.json  (list of events)

Event format example:
  {"type":"pause","start":1.23,"duration":0.5,"token":"/pause_0.50s/","insert_after_word_idx":12}
  {"type":"filler","word_idx":7,"token":"/filler_uh/"}
  {"type":"stutter","word_idx":4,"token":"/stutter:th-th/"}
  {"type":"tremor","word_idx":9,"token":"/tremor/"}
  {"type":"repeat_word","word_idx":3,"token":"/repeat_word:the-the/"}
"""
import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict

import math


def load_alignment(csv_path: Path) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            start = None
            try:
                start = float(row["start"]) if row.get("start") not in (None, "", "None", "nan") else None
            except Exception:
                start = None
            end = None
            try:
                end = float(row["end"]) if row.get("end") not in (None, "", "None", "nan") else None
            except Exception:
                end = None
            rows.append({"word": row.get("word", ""), "start": start, "end": end})
    return rows


def load_vad(csv_path: Path):
    segments = []
    if not csv_path.exists():
        return segments
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            start = float(row["start"]); end = float(row["end"])
            is_speech = row.get("is_speech", "True").lower() in ("true", "1", "yes")
            segments.append({"start": start, "end": end, "is_speech": is_speech})
    return segments


def load_prosody(csv_path: Path):
    pros = {}
    if not csv_path.exists():
        return pros
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        idx = 0
        for row in r:
            pros[idx] = row
            idx += 1
    return pros


def detect_pauses_from_vad(vad_segments, min_pause=0.15):
    # vad_segments are alternating speech/non-speech; we look for non-speech segments of length >= min_pause
    pauses = []
    for seg in vad_segments:
        if not seg["is_speech"]:
            dur = seg["end"] - seg["start"]
            if dur >= min_pause:
                pauses.append({"start": seg["start"], "duration": dur})
    return pauses


def detect_repeat_words(alignment_rows, max_gap=0.6):
    events = []
    # naive detection: adjacent identical words (case-insensitive) with small gap
    for i in range(1, len(alignment_rows)):
        w_prev = (alignment_rows[i-1]["word"] or "").strip().lower()
        w_cur = (alignment_rows[i]["word"] or "").strip().lower()
        s_prev = alignment_rows[i-1].get("start") or 0.0
        s_cur = alignment_rows[i].get("start") or 0.0
        if w_prev != "" and w_prev == w_cur and (s_cur - s_prev) <= max_gap:
            token = f"/repeat_word:{w_prev}-{w_cur}/"
            events.append({"type": "repeat_word", "word_idx": i, "token": token})
    return events


def detect_fillers(alignment_rows):
    events = []
    filler_set = {"uh", "um", "uhh", "umm", "hmm", "mm"}
    for i, row in enumerate(alignment_rows):
        w = (row["word"] or "").strip().lower()
        if w in filler_set:
            token = f"/filler_{w}/"
            events.append({"type": "filler", "word_idx": i, "token": token})
    return events


def detect_cutoffs(alignment_rows):
    events = []
    # if a word is present but end is None, treat as possible cutoff
    for i, r in enumerate(alignment_rows):
        start = r.get("start")
        end = r.get("end")
        if end is None or start is None or (end - start) < 0.02:
            token = "/cutoff/"
            events.append({"type": "cutoff", "word_idx": i, "token": token})

    return events


def detect_stutter_by_repeated_fragment(alignment_rows):
    events = []
    # crude heuristic: if a word contains repeated characters or a hyphenated fragment it's a stutter
    for i, r in enumerate(alignment_rows):
        w = r.get("word") or ""
        # look for forms like "th-th" or "th.."
        if "-" in w:
            parts = w.split("-")
            if len(parts) >= 2 and parts[0].strip().lower() == parts[1].strip().lower():
                token = f"/stutter:{parts[0]}-{parts[1]}/"
                events.append({"type": "stutter", "word_idx": i, "token": token})
        # also detect short repeated letters like "t t" (rare)
    return events


def detect_tremor_from_prosody(prosody_map, jitter_thresh=0.015):
    events = []
    for idx, row in prosody_map.items():
        try:
            jitter = float(row.get("jitter_approx") or 0.0)
        except Exception:
            jitter = 0.0
        if jitter and jitter >= jitter_thresh:
            events.append({"type": "tremor", "word_idx": idx, "token": "/tremor/"})
    return events


def assign_pause_to_nearest_word(alignment_rows, pause_start, pause_duration):
    # find word whose start is just before pause_start (or the next word)
    idx = None
    for i, r in enumerate(alignment_rows):
        s = r.get("start")
        if s is None:
            continue
        if s <= pause_start:
            idx = i
        else:
            break
    insert_after = idx if idx is not None else 0
    token = f"/pause_{pause_duration:.2f}s/"
    return {"type": "pause", "start": pause_start, "duration": pause_duration, "token": token, "insert_after_word_idx": insert_after}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--align", required=True, help="alignment CSV path")
    parser.add_argument("--vad", required=False, help="VAD CSV path")
    parser.add_argument("--prosody", required=False, help="prosody CSV path")
    parser.add_argument("--out", required=False, help="output events json path")
    args = parser.parse_args()

    align_path = Path(args.align)
    vad_path = Path(args.vad) if args.vad else None
    prosody_path = Path(args.prosody) if args.prosody else None
    out_path = Path(args.out) if args.out else Path("data/annotated") / (Path(args.align).stem.replace("_words","") + "_events.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    alignment_rows = load_alignment(align_path)
    vad_segments = load_vad(vad_path) if vad_path else []
    prosody_map = load_prosody(prosody_path) if prosody_path else {}

    events = []
    # pauses from VAD
    pauses = detect_pauses_from_vad(vad_segments, min_pause=0.15)
    for p in pauses:
        ev = assign_pause_to_nearest_word(alignment_rows, p["start"], p["duration"])
        events.append(ev)
    # fillers
    events += detect_fillers(alignment_rows)
    # repeats
    events += detect_repeat_words(alignment_rows, max_gap=0.6)
    # stutters
    events += detect_stutter_by_repeated_fragment(alignment_rows)
    # cutoffs
    events += detect_cutoffs(alignment_rows)
    # tremor from prosody
    events += detect_tremor_from_prosody(prosody_map, jitter_thresh=0.015)

    # sort events in approximate order (pauses by start -> others by word_idx)
    def sort_key(e):
        if e.get("type") == "pause":
            return (e.get("start", 0.0), 0)
        return (alignment_rows[e.get("word_idx", 0)].get("start") if e.get("word_idx", None) is not None and alignment_rows else 0.0, 1)
    events_sorted = sorted(events, key=sort_key)

    out_path.write_text(json.dumps(events_sorted, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote events to {out_path} (count={len(events_sorted)})")


if __name__ == "__main__":
    main()
