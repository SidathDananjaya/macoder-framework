import librosa
import numpy as np
from ai_engine.configs.project_paths import RAVDESS_PATH

SAMPLE_RATE = 22050


def load_audio(file_path):

    signal, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    return signal, sr


def normalize_audio(signal):

    return librosa.util.normalize(signal)


if __name__ == "__main__":

    path = (
            RAVDESS_PATH /
            "Actor_01" /
            "03-01-01-01-01-01-01.wav"
        )

    signal, sr = load_audio(path)

    normalized = normalize_audio(signal)

    print(signal.shape)
    print(sr)
