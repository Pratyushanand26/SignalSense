"""
02_vad_webrtc.py
Simple VAD using py-webrtcvad. Produces CSV of speech segments (start,end,is_speech).

Usage:
  python src/02_vad_webrtc.py --audio audio/clip1.wav --out data/vad/clip1_vad.csv
"""

import argparse
from pathlib import Path
import soundfile as sf
import webrtcvad
import numpy as np
import pandas as pd

def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    Yield successive frames of audio, each with (timestamp, bytes).
    webrtcvad expects 16-bit PCM byte frames.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0))
    offset = 0
    t = 0.0
    while offset + n <= len(audio):
        frame = audio[offset:offset + n]
        yield t, frame
        t += (n / sample_rate)
        offset += n

def pcm_bytes_from_float_array(arr):
    """
    Convert float32 audio [-1,1] to 16-bit PCM bytes required by webrtcvad.
    """
    ints = np.int16(arr * 32767)
    return ints.tobytes()

def main(audio_path: Path, out_csv: Path, frame_ms=30, aggressiveness=2):
    # 1) load audio file
    y, sr = sf.read(str(audio_path))  # returns float arrays or int arrays
    if sr not in (8000, 16000, 32000, 48000):
        raise ValueError(f"webrtcvad requires sample rate 8000/16000/32000/48000. Got {sr}. Convert audio first.")

    # normalize to mono float32 in [-1,1]
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if y.dtype != np.float32 and y.dtype != np.float64:
        # convert ints to float
        max_val = float(np.iinfo(y.dtype).max)
        y = y.astype(np.float32) / max_val

    vad = webrtcvad.Vad(aggressiveness)  # aggressiveness 0..3
    frames = list(frame_generator(frame_ms, y, sr))

    # classify frames
    speech_flags = []
    for t, frame in frames:
        b = pcm_bytes_from_float_array(frame)
        is_speech = vad.is_speech(b, sample_rate=sr)
        speech_flags.append((t, t + (len(frame) / sr), bool(is_speech)))

    # collapse contiguous same flags into segments
    segments = []
    if speech_flags:
        cur_flag = speech_flags[0][2]
        cur_start = speech_flags[0][0]
        cur_end = speech_flags[0][1]
        for s, e, flag in speech_flags[1:]:
            if flag == cur_flag:
                cur_end = e
            else:
                segments.append({"start": round(cur_start,3), "end": round(cur_end,3), "is_speech": cur_flag})
                cur_flag = flag
                cur_start = s
                cur_end = e
        segments.append({"start": round(cur_start,3), "end": round(cur_end,3), "is_speech": cur_flag})

    df = pd.DataFrame(segments)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Wrote VAD segments to:", out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=False)
    parser.add_argument("--frame", type=int, default=30, help="frame size in ms for VAD")
    parser.add_argument("--agg", type=int, default=2, help="webrtcvad aggressiveness 0..3")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    out_csv = Path(args.out) if args.out else Path("data/vad") / (audio_path.stem + "_vad.csv")
    main(audio_path, out_csv, frame_ms=args.frame, aggressiveness=args.agg)
