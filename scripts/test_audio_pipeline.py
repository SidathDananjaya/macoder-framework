import sys
import os

sys.path.append(os.path.abspath("."))

from ai_engine.configs.project_paths import RAVDESS_PATH

from ai_engine.features.audio.audio_preprocessor import (
    load_audio,
    normalize_audio
)

from ai_engine.features.audio.audio_feature_extractor import (
    extract_mfcc,
    extract_pitch,
    extract_energy
)

# AUDIO_PATH = "datasets/raw/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-01-05-01-01-01-01.wav"
AUDIO_PATH = (
    RAVDESS_PATH /
    "Actor_01" /
    "03-01-05-01-01-01-01.wav"
)

signal, sr = load_audio(AUDIO_PATH)

signal = normalize_audio(signal)

mfcc = extract_mfcc(signal, sr)

pitch = extract_pitch(signal, sr)

energy = extract_energy(signal)

print("MFCC Shape:", mfcc.shape)
print("Pitch:", pitch)
print("Energy:", energy)