import os
import sys

sys.path.append(os.path.abspath("."))

from ai_engine.features.audio.audio_preprocessor import (
    load_audio,
    normalize_audio
)

from ai_engine.features.audio.advanced_audio_features import (
    extract_delta_mfcc,
    extract_delta2_mfcc,
    extract_chroma,
    extract_spectral_contrast,
    extract_tonnetz,
    extract_energy_variance
)

AUDIO_PATH = (
    "datasets/raw/RAVDESS/"
    "Audio_Speech_Actors_01-24/"
    "Actor_01/"
    "03-01-05-01-01-01-01.wav"
)

signal, sr = load_audio(AUDIO_PATH)

signal = normalize_audio(signal)

delta = extract_delta_mfcc(signal, sr)

delta2 = extract_delta2_mfcc(signal, sr)

chroma = extract_chroma(signal, sr)

contrast = extract_spectral_contrast(signal, sr)

tonnetz = extract_tonnetz(signal, sr)

energy_var = extract_energy_variance(signal)

print("Delta MFCC:", delta.shape)

print("Delta2 MFCC:", delta2.shape)

print("Chroma:", chroma.shape)

print("Spectral Contrast:", contrast.shape)

print("Tonnetz:", tonnetz.shape)

print("Energy Variance:", energy_var)