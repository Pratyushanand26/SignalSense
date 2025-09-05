#!/usr/bin/env python3
"""
src/04_prosody.py

Extract per-word prosody features (pitch F0 with pyin, jitter-approx, RMS energy).

Usage:
  python src/04_prosody.py --audio audio/clip1.wav --align data/alignments/clip1_words.csv --out data/prosody/clip1_prosody.csv
"""
import argparse
import json
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np
import pandas as pd


def load_alignment(path: Path):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        assert {"word", "start", "end"}.issubset(set(df.columns)), "CSV must have columns: word,start,end"
        rows = df[["word", "start", "end"]].to_dict(orient="records")
        # coerce numbers
        for r in rows:
            r["start"] = float(r["start"]) if not (r["start"] is None or str(r["start"]) == "nan") else None
            r["end"] = float(r["end"]) if not (r["end"] is None or str(r["end"]) == "nan") else None
        return rows
    elif path.suffix.lower() == ".json":
        items = json.loads(path.read_text(encoding="utf-8"))
        return [{"word": it["word"], "start": float(it["start"]), "end": float(it["end"])} for it in items]
    else:
        raise ValueError("Unsupported alignment format. Use CSV or JSON.")


def extract_prosody_for_interval(y, sr, start_s, end_s,
                                 fmin=librosa.note_to_hz("C2"),
                                 fmax=librosa.note_to_hz("C7"),
                                 hop_length=256,
                                 frame_length=2048):
    start_sample = int(round(start_s * sr))
    end_sample = int(round(end_s * sr))
    start_sample = max(0, start_sample)
    end_sample = min(len(y), end_sample)
    if end_sample <= start_sample:
        return {
            "duration": max(0.0, end_s - start_s),
            "f0_mean": np.nan, "f0_median": np.nan, "f0_std": np.nan,
            "voiced_fraction": 0.0, "jitter_approx": np.nan,
            "rms_mean": 0.0, "rms_std": 0.0
        }
    y_slice = y[start_sample:end_sample]

    # RMS
    try:
        rms = librosa.feature.rms(y=y_slice, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    except Exception:
        rms = np.array([])
    rms_mean = float(np.mean(rms)) if rms.size > 0 else 0.0
    rms_std = float(np.std(rms)) if rms.size > 0 else 0.0

    # F0 with pyin
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y_slice, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop_length
        )
    except Exception:
        f0 = np.array([])

    if f0.size == 0:
        f0_mean = f0_median = f0_std = np.nan
        voiced_fraction = 0.0
        jitter_approx = np.nan
    else:
        voiced_mask = ~np.isnan(f0)
        total_frames = f0.shape[0]
        num_voiced = int(np.count_nonzero(voiced_mask))
        voiced_fraction = float(num_voiced) / float(total_frames) if total_frames > 0 else 0.0
        if num_voiced >= 1:
            f0_vals = f0[voiced_mask]
            f0_mean = float(np.mean(f0_vals))
            f0_median = float(np.median(f0_vals))
            f0_std = float(np.std(f0_vals))
        else:
            f0_mean = f0_median = f0_std = np.nan
        if num_voiced >= 2:
            periods = 1.0 / f0_vals
            period_diffs = np.abs(np.diff(periods))
            mean_period = np.mean(periods)
            jitter_approx = float(np.mean(period_diffs) / mean_period) if mean_period > 0 else np.nan
        else:
            jitter_approx = np.nan

    return {
        "duration": float(end_s - start_s),
        "f0_mean": f0_mean, "f0_median": f0_median, "f0_std": f0_std,
        "voiced_fraction": voiced_fraction, "jitter_approx": jitter_approx,
        "rms_mean": rms_mean, "rms_std": rms_std
    }


def main(audio_path: Path, align_path: Path, out_path: Path,
         hop_length=256, frame_length=2048):
    print(f"Loading audio: {audio_path}")
    y, sr = sf.read(str(audio_path), always_2d=False)
    if y.ndim > 1:  # convert to mono
     y = y.mean(axis=1)
    print(f"Audio loaded: {len(y)} samples, sr={sr}")

    align = load_alignment(align_path)
    print(f"Loaded {len(align)} words from alignment.")

    rows = []
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")

    for item in align:
        w = item["word"]
        start = item["start"] if item["start"] is not None else 0.0
        end = item["end"] if item["end"] is not None else (start + 0.050)
        features = extract_prosody_for_interval(y=y, sr=sr, start_s=start, end_s=end,
                                                fmin=fmin, fmax=fmax,
                                                hop_length=hop_length, frame_length=frame_length)
        row = {"word": w, "start": start, "end": end, **features}
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote prosody CSV: {out_path} (rows={len(df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-word prosody extractor (librosa pyin)")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--align", required=True)
    parser.add_argument("--out", required=False)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--frame", type=int, default=2048)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    align_path = Path(args.align)
    out_path = Path(args.out) if args.out else Path("data/prosody") / (audio_path.stem + "_prosody.csv")
    main(audio_path, align_path, out_path, hop_length=args.hop, frame_length=args.frame)
