import joblib
import pandas as pd

from ai_engine.features.audio.audio_preprocessor import (
    load_audio,
    normalize_audio
)

from ai_engine.features.audio.audio_feature_extractor import (
    extract_mfcc,
    extract_pitch,
    extract_energy
)

from ai_engine.features.audio.advanced_audio_features import (
    extract_delta_mfcc,
    extract_delta2_mfcc,
    extract_chroma,
    extract_spectral_contrast,
    extract_tonnetz,
    extract_energy_variance
)

from ai_engine.features.audio.speech_rate import (
    estimate_speech_rate
)

from ai_engine.features.audio.pause_detection import (
    calculate_silence_ratio
)

from ai_engine.features.audio.voice_stability import (
    calculate_voice_stability
)

from ai_engine.features.audio.spectral_features import (
    extract_spectral_centroid,
    extract_zero_crossing_rate
)

from ai_engine.configs.project_paths import (
    EXPERIMENTS_DIR
)

# =========================
# LOAD MODELS
# =========================

MODEL_DIR = (
    EXPERIMENTS_DIR /
    "exp_002_stress_classifier"
)

model = joblib.load(
    MODEL_DIR /
    "stress_classifier.pkl"
)

scaler = joblib.load(
    MODEL_DIR /
    "stress_scaler.pkl"
)

label_encoder = joblib.load(
    MODEL_DIR /
    "stress_label_encoder.pkl"
)

# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(audio_path):

    signal, sr = load_audio(audio_path)

    signal = normalize_audio(signal)

    row = {}

    # Basic
    mfcc = extract_mfcc(signal, sr)

    pitch = extract_pitch(signal, sr)

    energy = extract_energy(signal)

    # Advanced
    delta = extract_delta_mfcc(signal, sr)

    delta2 = extract_delta2_mfcc(signal, sr)

    chroma = extract_chroma(signal, sr)

    contrast = extract_spectral_contrast(signal, sr)

    tonnetz = extract_tonnetz(signal, sr)

    energy_var = extract_energy_variance(signal)

    # Stress
    speech_rate = estimate_speech_rate(signal, sr)

    silence_ratio = calculate_silence_ratio(signal)

    stability = calculate_voice_stability(signal)

    spectral_centroid = extract_spectral_centroid(
        signal,
        sr
    )

    zcr = extract_zero_crossing_rate(signal)

    # Row
    row["pitch"] = pitch
    row["energy"] = energy
    row["speech_rate"] = speech_rate
    row["silence_ratio"] = silence_ratio
    row["voice_stability"] = stability
    row["spectral_centroid"] = spectral_centroid
    row["zcr"] = zcr
    row["energy_variance"] = energy_var

    for i, value in enumerate(mfcc):
        row[f"mfcc_{i+1}"] = value

    for i, value in enumerate(delta):
        row[f"delta_mfcc_{i+1}"] = value

    for i, value in enumerate(delta2):
        row[f"delta2_mfcc_{i+1}"] = value

    for i, value in enumerate(chroma):
        row[f"chroma_{i+1}"] = value

    for i, value in enumerate(contrast):
        row[f"contrast_{i+1}"] = value

    for i, value in enumerate(tonnetz):
        row[f"tonnetz_{i+1}"] = value

    return pd.DataFrame([row])


# =========================
# PREDICT
# =========================

def predict_stress(audio_path):

    features = extract_features(audio_path)

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)

    stress_label = label_encoder.inverse_transform(
        prediction
    )[0]

    probabilities = model.predict_proba(scaled)[0]

    confidence = float(max(probabilities))

    return {
        "stress_level": stress_label,
        "confidence": round(confidence, 4)
    }