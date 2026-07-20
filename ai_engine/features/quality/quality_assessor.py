# Deterministic per-frame quality scores in [0,1] (Section 5.4.2) that drive the fusion weights and the "poor signal quality" warning; no training, so they run on every frame.

import cv2
import numpy as np

# Below this score the dashboard shows the warning banner and the modality is heavily down-weighted in fusion.
QUALITY_WARNING_THRESHOLD = 0.4


def assess_video_quality(frame):
    # Video quality in [0,1] from brightness + Laplacian sharpness; returns 0.0 for a missing frame.

    if frame is None or getattr(frame, "size", 0) == 0:
        return 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Treat a frame at half brightness or above as well-exposed so only real dimming drops it below the 0.4 warning threshold.
    mean_brightness = float(np.mean(gray)) / 255.0
    brightness_score = float(np.clip(mean_brightness / 0.5, 0.0, 1.0))

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = min(1.0, laplacian_var / 250.0)

    quality = 0.5 * brightness_score + 0.5 * blur_score

    return float(np.clip(quality, 0.0, 1.0))


def assess_audio_quality(audio_chunk, sr=22050):
    # Audio quality in [0,1] from signal energy: near-silence scores ~0, clear speech ~1, so the fusion can down-weight a silent channel. Uses absolute energy because the Code Listing 2 heuristic was scale-invariant and could not drive weighting.

    signal = np.asarray(audio_chunk, dtype=np.float32)

    if signal.size == 0:
        return 0.0

    rms = float(np.sqrt(np.mean(np.square(signal))))

    if rms <= 1e-6:
        return 0.0

    # Map RMS in dB from about -60 (silence) to -10 (loud speech) onto [0, 1].
    db = 20.0 * np.log10(rms)
    quality = (db + 60.0) / 50.0

    return float(np.clip(quality, 0.0, 1.0))
