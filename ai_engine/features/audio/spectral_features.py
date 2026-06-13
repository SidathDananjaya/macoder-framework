import librosa
import numpy as np


def extract_spectral_centroid(signal, sr):

    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr
    )

    return np.mean(centroid)


def extract_zero_crossing_rate(signal):

    zcr = librosa.feature.zero_crossing_rate(signal)

    return np.mean(zcr)