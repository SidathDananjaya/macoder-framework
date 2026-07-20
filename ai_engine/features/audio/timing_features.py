"""Timing / response-latency features (dissertation Section 5.4.3).

The third modality alongside visual and audio: behavioural *timing* cues derived
from a simple energy-based Voice Activity Detection (VAD) over the audio signal.
These are the "response latency / pause / speech-rate" features the thesis
describes, computed deterministically (no training) so they run live and can be
extracted offline for the ablation study (Section 6.3).

Features returned (all robust to silence):

* ``response_latency``  - seconds from the start of the window to the first
  voiced frame (proxy for "time to begin speaking").
* ``pause_ratio``       - fraction of the window that is unvoiced (silence).
* ``speech_rate``       - voiced onsets per second (a syllable-rate proxy).
* ``voiced_ratio``      - fraction of the window that is voiced.
* ``mean_pause``        - mean duration (s) of silent gaps between speech.
"""

import numpy as np

# VAD frame length / hop (seconds) and the energy threshold as a fraction of the
# window's peak frame energy.
FRAME_SEC = 0.03
HOP_SEC = 0.015
ENERGY_FRACTION = 0.15
# Below this absolute RMS the whole window is treated as silence (unrouted audio).
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
    """Energy-VAD timing features for a 1-D audio array. Returns a dict."""

    signal = np.asarray(signal, dtype=np.float32).ravel()

    if signal.size == 0:
        return _empty()

    overall_rms = float(np.sqrt(np.mean(np.square(signal))))
    if overall_rms < SILENCE_RMS:
        return _empty()

    frame = max(1, int(FRAME_SEC * sr))
    hop = max(1, int(HOP_SEC * sr))

    # Per-frame RMS energy.
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

    # Response latency: time to the first voiced frame.
    first_voiced = int(np.argmax(voiced)) if voiced.any() else n
    response_latency = first_voiced * hop / sr

    voiced_ratio = float(voiced.mean())
    pause_ratio = float(1.0 - voiced_ratio)

    # Voiced onsets (rising edges) -> speech-rate proxy.
    onsets = int(np.sum((voiced[1:]) & (~voiced[:-1]))) + int(voiced[0])
    speech_rate = onsets / duration if duration else 0.0

    # Mean silent-gap duration between voiced runs.
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
