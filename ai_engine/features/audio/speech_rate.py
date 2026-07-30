import librosa
import numpy as np


def estimate_speech_rate(signal, sr):

    tempo, _ = librosa.beat.beat_track(
        y=signal,
        sr=sr
    )

    if isinstance(tempo, np.ndarray):

        if len(tempo) > 0:
            return float(tempo[0])

        return 0.0

    return float(tempo)
