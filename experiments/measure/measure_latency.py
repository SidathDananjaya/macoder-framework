"""Real-time latency measurement (dissertation Section 6.7).

Drives the *actual* deployed pipeline (the same functions the websocket route
calls) with real RAVDESS video frames + audio and reports per-stage latency:

    Visual pipeline   -> process_frame
    Audio pipeline    -> process_audio
    Fusion + response -> fuse_results
    End-to-end        -> sum of the three per frame

Run (from the project root, in the vision_env):

    ./vision_env/Scripts/python.exe -m experiments.measure.measure_latency

Outputs a table to stdout and writes experiments/measure/latency_results.json.
Hardware is CPU-only (no GPU) - state this in the thesis.
"""

import asyncio
import base64
import glob
import json
import os
import time

import cv2
import numpy as np
import soundfile as sf
import librosa

from backend.app.services.realtime_ai_engine import process_frame, reset_engine
from backend.app.services.realtime_audio_engine import process_audio, reset_audio
from backend.app.services.realtime_fusion_engine import fuse_results

# Number of frames to time (Section 6.7 asks for N >= 100).
N_FRAMES = 120

# A RAVDESS video/audio pair to source real frames + speech from.
VIDEO_GLOB = "datasets/raw/RAVDESS/Video_Speech_Actors_01-24/Actor_01/*.mp4"
AUDIO_GLOB = "datasets/raw/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/*.wav"

OUT_PATH = "experiments/measure/latency_results.json"


def _percentile(values, pct):
    return float(np.percentile(values, pct)) if values else 0.0


def _summary(values):
    return {
        "mean_ms": round(float(np.mean(values)), 1) if values else 0.0,
        "p95_ms": round(_percentile(values, 95), 1),
        "n": len(values),
    }


def _load_frames(n):
    """Grab up to ``n`` real frames from a RAVDESS video as base64 data URLs."""

    paths = sorted(glob.glob(VIDEO_GLOB))
    if not paths:
        raise SystemExit(f"No RAVDESS video found at {VIDEO_GLOB}")

    frames = []
    cap = cv2.VideoCapture(paths[0])
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            b64 = base64.b64encode(buf).decode("ascii")
            frames.append("data:image/jpeg;base64," + b64)
    cap.release()

    # Loop the clip if it was shorter than n frames.
    if frames:
        while len(frames) < n:
            frames.append(frames[len(frames) % len(frames)])
    return frames[:n]


def _load_audio_chunks(n, chunk_ms=85):
    """Split a RAVDESS wav into ~85 ms chunks (one per frame) at native rate."""

    paths = sorted(glob.glob(AUDIO_GLOB))
    if not paths:
        return [([], 22050)] * n

    signal, sr = sf.read(paths[0])
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    signal = signal.astype(np.float32)

    step = max(1, int(sr * chunk_ms / 1000))
    chunks = [
        (signal[i:i + step].tolist(), sr)
        for i in range(0, len(signal), step)
    ]
    if not chunks:
        return [([], sr)] * n
    while len(chunks) < n:
        chunks.append(chunks[len(chunks) % len(chunks)])
    return chunks[:n]


async def main():
    print(f"Loading {N_FRAMES} real RAVDESS frames + audio chunks ...")
    frames = _load_frames(N_FRAMES)
    audio_chunks = _load_audio_chunks(N_FRAMES)

    reset_engine()
    reset_audio()

    visual_ms, audio_ms, fusion_ms, e2e_ms = [], [], [], []

    # Warm up once (first DeepFace/model call is much slower - excluded).
    await process_frame(frames[0])
    await process_audio(*audio_chunks[0])

    print("Timing ...")
    for i in range(len(frames)):
        t0 = time.perf_counter()
        visual = await process_frame(frames[i])
        t1 = time.perf_counter()
        audio = await process_audio(*audio_chunks[i])
        t2 = time.perf_counter()
        await fuse_results(visual, audio)
        t3 = time.perf_counter()

        visual_ms.append((t1 - t0) * 1000)
        audio_ms.append((t2 - t1) * 1000)
        fusion_ms.append((t3 - t2) * 1000)
        e2e_ms.append((t3 - t0) * 1000)

    results = {
        "n_frames": len(frames),
        "hardware": "CPU-only (no GPU)",
        "stages": {
            "visual_pipeline": _summary(visual_ms),
            "audio_pipeline": _summary(audio_ms),
            "fusion_and_response": _summary(fusion_ms),
            "end_to_end_per_frame": _summary(e2e_ms),
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("\n=== Real-time latency (Section 6.7) ===")
    print(f"{'Stage':<26}{'Mean (ms)':>12}{'p95 (ms)':>12}")
    for name, key in [
        ("Visual pipeline", "visual_pipeline"),
        ("Audio pipeline", "audio_pipeline"),
        ("Fusion + response", "fusion_and_response"),
        ("End-to-end per frame", "end_to_end_per_frame"),
    ]:
        s = results["stages"][key]
        print(f"{name:<26}{s['mean_ms']:>12}{s['p95_ms']:>12}")
    print(f"\nN = {len(frames)} frames, {results['hardware']}")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
