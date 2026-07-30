import cv2
import numpy as np

QUALITY_WARNING_THRESHOLD = 0.4


def assess_video_quality(frame):

    if frame is None or getattr(frame, "size", 0) == 0:
        return 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean_brightness = float(np.mean(gray)) / 255.0
    brightness_score = float(np.clip(mean_brightness / 0.5, 0.0, 1.0))

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = min(1.0, laplacian_var / 250.0)

    quality = 0.5 * brightness_score + 0.5 * blur_score

    return float(np.clip(quality, 0.0, 1.0))


def assess_audio_quality(audio_chunk, sr=22050):

    signal = np.asarray(audio_chunk, dtype=np.float32)

    if signal.size == 0:
        return 0.0

    rms = float(np.sqrt(np.mean(np.square(signal))))

    if rms <= 1e-6:
        return 0.0

    db = 20.0 * np.log10(rms)
    quality = (db + 60.0) / 50.0

    return float(np.clip(quality, 0.0, 1.0))
