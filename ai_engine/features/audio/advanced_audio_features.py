import librosa
import numpy as np


# =========================
# DELTA MFCC
# =========================

def extract_delta_mfcc(signal, sr, n_mfcc=13):

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc
    )

    delta = librosa.feature.delta(mfcc)

    return np.mean(delta.T, axis=0)


# =========================
# DELTA-DELTA MFCC
# =========================

def extract_delta2_mfcc(signal, sr, n_mfcc=13):

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc
    )

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    return np.mean(delta2.T, axis=0)


# =========================
# CHROMA FEATURES
# =========================

def extract_chroma(signal, sr):

    chroma = librosa.feature.chroma_stft(
        y=signal,
        sr=sr
    )

    return np.mean(chroma.T, axis=0)


# =========================
# SPECTRAL CONTRAST
# =========================

def extract_spectral_contrast(signal, sr):

    contrast = librosa.feature.spectral_contrast(
        y=signal,
        sr=sr
    )

    return np.mean(contrast.T, axis=0)


# =========================
# TONNETZ
# =========================

def extract_tonnetz(signal, sr):

    harmonic = librosa.effects.harmonic(signal)

    tonnetz = librosa.feature.tonnetz(
        y=harmonic,
        sr=sr
    )

    return np.mean(tonnetz.T, axis=0)


# =========================
# ENERGY VARIANCE
# =========================

def extract_energy_variance(signal):

    rms = librosa.feature.rms(y=signal)[0]

    return np.var(rms)