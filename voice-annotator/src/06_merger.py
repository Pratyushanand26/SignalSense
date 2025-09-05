#!/usr/bin/env python3
"""
src/06_merger.py

Merge alignment + prosody + vad + events into a readable annotated transcript:
writes data/annotated/<base>_annotated.txt

Usage:
  python src/06_merger.py --align data/alignments/clip1_words.csv --prosody data/prosody/clip1_prosody.csv --vad data/vad/clip1_vad.csv --events data/annotated/clip1_events.json --out data/annotated/clip1_annotated.txt
"""
import argparse
import csv
import json
from pathlib import Path

def load_alignment(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            start = None
            end = None
            try:
                start = float(row["start"]) if row.get("start") not in (None, "", "None", "nan") else None
            except Exception:
                start = None
            try:
                end = float(row["end"]) if row.get("end") not in (None, "", "None", "nan") else None
            except Exception:
                end = None
            rows.append({"word": row.get("word",""), "start": start, "end": end})
    return rows

def load_prosody(csv_path):
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

def load_vad(csv_path):
    segs = []
    if not csv_path.exists():
        return segs
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            segs.append({"start": float(row["start"]), "end": float(row["end"]), "is_speech": row["is_speech"].lower() in ("true","1","yes")})
    return segs

def load_events(json_path):
    if not json_path.exists():
        return []
    return json.loads(json_path.read_text(encoding="utf-8"))

def format_time(s):
    return f"[{s:06.2f}]"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--align", required=True)
    parser.add_argument("--prosody", required=False)
    parser.add_argument("--vad", required=False)
    parser.add_argument("--events", required=False)
    parser.add_argument("--out", required=False)
    args = parser.parse_args()

    align = load_alignment(Path(args.align))
    prosody = load_prosody(Path(args.prosody)) if args.prosody else {}
    vad = load_vad(Path(args.vad)) if args.vad else []
    events = load_events(Path(args.events)) if args.events else []

    # Build map: for each word index, accumulate tokens that should be inserted before/after it
    insert_before = {i: [] for i in range(len(align))}
    insert_after = {i: [] for i in range(len(align))}

    for ev in events:
        t = ev.get("type")
        token = ev.get("token")
        if t == "pause":
            idx = ev.get("insert_after_word_idx", 0)
            insert_after.setdefault(idx, []).append(token)
        else:
            wi = ev.get("word_idx")
            if wi is None:
                continue
            # decide before/after insertion heuristics: fillers -> before, stutter -> before, tremor -> after
            if t in ("filler", "stutter", "repeat_word"):
                insert_before.setdefault(wi, []).append(token)
            else:
                insert_after.setdefault(wi, []).append(token)

    # Format per-line transcript: group words into lines by sentence-break approximations using long pauses (>0.6s)
    lines = []
    current_line = []
    current_line_start = None
    prev_end = 0.0

    for i, w in enumerate(align):
        start = w.get("start") or 0.0
        end = w.get("end") or start + 0.05
        if current_line_start is None:
            current_line_start = start
        # if there is large pause (use VAD) - break line
        large_pause = False
        for seg in vad:
            if not seg["is_speech"]:
                # if pause overlaps previous end and current start
                if seg["start"] >= prev_end - 1e-6 and seg["end"] <= start + 1e-6:
                    if (seg["end"] - seg["start"]) >= 0.6:
                        large_pause = True
                        break
        # assemble tokens for this word
        tokens = []
        if insert_before.get(i):
            tokens.extend(insert_before[i])
        tokens.append(w.get("word") or "")
        if insert_after.get(i):
            tokens.extend(insert_after[i])

        # append tokens to current line
        current_line.append(" ".join(filter(None, tokens)))
        prev_end = end

        if large_pause:
            # commit current line
            if current_line:
                header = f"[{current_line_start:06.2f}] "
                lines.append(header + " ".join(current_line))
            current_line = []
            current_line_start = None

    # commit remaining
    if current_line:
        header = f"[{current_line_start:06.2f}] " if current_line_start is not None else ""
        lines.append(header + " ".join(current_line))

    out_path = Path(args.out) if args.out else Path("data/annotated") / (Path(args.align).stem.replace("_words","") + "_annotated.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote annotated transcript: {out_path} (lines={len(lines)})")


if __name__ == "__main__":
    main()