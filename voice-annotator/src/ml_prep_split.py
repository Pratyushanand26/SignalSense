#!/usr/bin/env python3
"""
src/07_ml_prep_split.py

Create an ML-ready .txt containing N lines (default 5). Each line is the annotated tokens
covering that evenly-split time slice of the audio.

Usage:
  python src/07_ml_prep_split.py --annot data/annotated/clip1_annotated.txt --align data/alignments/clip1_words.csv --out data/annotated/clip1_ml_5lines.txt --n 5
"""
import argparse
from pathlib import Path
import csv

def load_alignment(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            start = None
            try:
                start = float(row["start"]) if row.get("start") not in (None, "", "None") else None
            except Exception:
                start = None
            rows.append({"word": row.get("word", ""), "start": start, "end": row.get("end")})
    return rows

def load_annotated_lines(annot_path):
    with open(annot_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    return lines

def build_token_to_start_map(alignment_rows):
    token_map = []
    for i, r in enumerate(alignment_rows):
        token_map.append((i, r.get("word",""), r.get("start")))
    return token_map

def split_by_time(total_duration, n):
    slice_len = float(total_duration) / float(n)
    boundaries = []
    for i in range(n):
        a = i * slice_len
        b = (i+1) * slice_len if i < n-1 else total_duration
        boundaries.append((a, b))
    return boundaries

def estimate_total_duration(alignment_rows):
    last = 0.0
    for r in reversed(alignment_rows):
        try:
            if r.get("end") is not None:
                last = float(r.get("end"))
                break
            elif r.get("start") is not None:
                last = float(r.get("start"))
                break
        except Exception:
            continue
    return max(1.0, last)

def assign_tokens_to_slices(annot_lines, token_map, boundaries):
    flat_tokens = []
    for ln in annot_lines:
        for tok in ln.split():
            flat_tokens.append(tok)

    word_iter = iter(token_map)
    try:
        cur_word_idx, cur_word, cur_start = next(word_iter)
    except StopIteration:
        cur_word_idx = None; cur_word = None; cur_start = None

    token_assignments = []
    for tok in flat_tokens:
        tok_clean = "".join(ch for ch in tok.lower() if ch.isalnum())
        cur_clean = "".join(ch for ch in (cur_word or "").lower() if ch.isalnum()) if cur_word else None

        if cur_word and tok_clean == cur_clean and cur_start is not None:
            token_assignments.append((tok, cur_start))
            try:
                cur_word_idx, cur_word, cur_start = next(word_iter)
            except StopIteration:
                cur_word_idx = None; cur_word = None; cur_start = None
        else:
            assign_time = token_assignments[-1][1] if token_assignments else (cur_start or 0.0)
            token_assignments.append((tok, assign_time))

    slices = [[] for _ in boundaries]
    for tok, t in token_assignments:
        placed = False
        for i, (a,b) in enumerate(boundaries):
            if (t >= a and (t < b or i == len(boundaries)-1)):
                slices[i].append(tok)
                placed = True
                break
        if not placed:
            slices[-1].append(tok)
    lines = [" ".join(s).strip() if len(s) > 0 else "" for s in slices]
    return lines

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annot", required=True)
    p.add_argument("--align", required=True)
    p.add_argument("--out", required=False)
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()

    a_path = Path(args.annot)
    align_path = Path(args.align)
    base = a_path.stem.replace("_annotated","")
    out_path = Path(args.out) if args.out else Path(f"data/annotated/{base}_ml_{args.n}lines.txt")

    alignment_rows = load_alignment(align_path)
    if len(alignment_rows) == 0:
        with open(out_path, "w", encoding="utf-8") as fo:
            fo.write("\n".join([""]*args.n))
        print("Wrote empty ML file:", out_path)
        return

    annot_lines = load_annotated_lines(a_path)
    token_map = build_token_to_start_map(alignment_rows)
    total_dur = estimate_total_duration(alignment_rows)
    boundaries = split_by_time(total_dur, args.n)
    ml_lines = assign_tokens_to_slices(annot_lines, token_map, boundaries)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fo:
        for ln in ml_lines:
            fo.write(ln + "\n")
    print("Wrote ML-ready file:", out_path)

if __name__ == "__main__":
    main()
