import librosa
import numpy as np

def extract_mfcc(signal, sr, n_mfcc=13):

    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc
    )

    mfccs_mean = np.mean(mfccs.T, axis=0)

    return mfccs_mean


def extract_pitch(signal, sr):

    pitches, magnitudes = librosa.piptrack(
        y=signal,
        sr=sr
    )

    pitch_values = []

    for i in range(pitches.shape[0]):

        index = magnitudes[i].argmax()

        pitch = pitches[i, index]

        # Human speech filtering
        if 80 <= pitch <= 400:
            pitch_values.append(pitch)

    if len(pitch_values) > 0:
        return np.mean(pitch_values)

    return 0


def extract_energy(signal):

    return np.mean(signal ** 2)