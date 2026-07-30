import numpy as np

FRAME_SEC = 0.03
HOP_SEC = 0.015
ENERGY_FRACTION = 0.15
SILENCE_RMS = 1e-4

TIMING_KEYS = (
    "response_latency",
    "pause_ratio",
    "speech_rate",
    "voiced_ratio",
    "mean_pause",
)


def _empty():
    return {k: 0.0 for k in TIMING_KEYS}


def extract_timing_features(signal, sr=22050):
    signal = np.asarray(signal, dtype=np.float32).ravel()

    if signal.size == 0:
        return _empty()

    overall_rms = float(np.sqrt(np.mean(np.square(signal))))
    if overall_rms < SILENCE_RMS:
        return _empty()

    frame = max(1, int(FRAME_SEC * sr))
    hop = max(1, int(HOP_SEC * sr))

    energies = [
        float(np.sqrt(np.mean(np.square(signal[i:i + frame]))))
        for i in range(0, max(1, len(signal) - frame + 1), hop)
    ]
    if not energies:
        return _empty()

    energies = np.asarray(energies)
    threshold = ENERGY_FRACTION * float(energies.max())
    voiced = energies >= threshold

    n = len(voiced)
    duration = n * hop / sr if n else 0.0
    if duration <= 0:
        return _empty()

    first_voiced = int(np.argmax(voiced)) if voiced.any() else n
    response_latency = first_voiced * hop / sr

    voiced_ratio = float(voiced.mean())
    pause_ratio = float(1.0 - voiced_ratio)

    onsets = int(np.sum((voiced[1:]) & (~voiced[:-1]))) + int(voiced[0])
    speech_rate = onsets / duration if duration else 0.0

    gaps = []
    run = 0
    for v in voiced:
        if not v:
            run += 1
        elif run:
            gaps.append(run)
            run = 0
    mean_pause = (float(np.mean(gaps)) * hop / sr) if gaps else 0.0

    return {
        "response_latency": round(response_latency, 4),
        "pause_ratio": round(pause_ratio, 4),
        "speech_rate": round(speech_rate, 4),
        "voiced_ratio": round(voiced_ratio, 4),
        "mean_pause": round(mean_pause, 4),
    }
