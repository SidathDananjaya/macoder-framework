import pandas as pd
import os

from ai_engine.data.loaders.ravdess_loader import (
    load_ravdess_files
)

from ai_engine.features.audio.audio_preprocessor import (
    load_audio,
    normalize_audio
)

# BASIC FEATURES
from ai_engine.features.audio.audio_feature_extractor import (
    extract_mfcc,
    extract_pitch,
    extract_energy
)

# ADVANCED FEATURES
from ai_engine.features.audio.advanced_audio_features import (
    extract_delta_mfcc,
    extract_delta2_mfcc,
    extract_chroma,
    extract_spectral_contrast,
    extract_tonnetz,
    extract_energy_variance
)

# STRESS FEATURES
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
    PROCESSED_DIR
)

# Emotion mapping
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def extract_emotion(file_path):

    filename = os.path.basename(file_path)

    emotion_code = filename.split("-")[2]

    return EMOTION_MAP.get(emotion_code, "unknown")


def build_dataset():

    all_files = load_ravdess_files()

    dataset_rows = []

    print(f"Processing {len(all_files)} audio files...\n")

    for index, file_path in enumerate(all_files):

        try:

            signal, sr = load_audio(file_path)

            signal = normalize_audio(signal)

            # =========================
            # BASIC FEATURES
            # =========================

            mfcc = extract_mfcc(signal, sr)

            pitch = extract_pitch(signal, sr)

            energy = extract_energy(signal)

            # =========================
            # ADVANCED FEATURES
            # =========================

            delta = extract_delta_mfcc(signal, sr)

            delta2 = extract_delta2_mfcc(signal, sr)

            chroma = extract_chroma(signal, sr)

            contrast = extract_spectral_contrast(signal, sr)

            tonnetz = extract_tonnetz(signal, sr)

            energy_var = extract_energy_variance(signal)

            # =========================
            # STRESS FEATURES
            # =========================

            speech_rate = estimate_speech_rate(signal, sr)

            silence_ratio = calculate_silence_ratio(signal)

            stability = calculate_voice_stability(signal)

            spectral_centroid = extract_spectral_centroid(
                signal,
                sr
            )

            zcr = extract_zero_crossing_rate(signal)

            # Label
            emotion = extract_emotion(file_path)

            row = {
                "file": file_path,
                "emotion": emotion,

                "pitch": pitch,
                "energy": energy,

                "speech_rate": speech_rate,
                "silence_ratio": silence_ratio,
                "voice_stability": stability,
                "spectral_centroid": spectral_centroid,
                "zcr": zcr,

                "energy_variance": energy_var
            }

            # MFCC
            for i, value in enumerate(mfcc):
                row[f"mfcc_{i+1}"] = value

            # Delta MFCC
            for i, value in enumerate(delta):
                row[f"delta_mfcc_{i+1}"] = value

            # Delta2 MFCC
            for i, value in enumerate(delta2):
                row[f"delta2_mfcc_{i+1}"] = value

            # Chroma
            for i, value in enumerate(chroma):
                row[f"chroma_{i+1}"] = value

            # Spectral Contrast
            for i, value in enumerate(contrast):
                row[f"contrast_{i+1}"] = value

            # Tonnetz
            for i, value in enumerate(tonnetz):
                row[f"tonnetz_{i+1}"] = value

            dataset_rows.append(row)

            print(f"[{index+1}/{len(all_files)}] Processed")

        except Exception as e:

            print(f"Error: {file_path}")

            print(e)

    return pd.DataFrame(dataset_rows)


if __name__ == "__main__":

    df = build_dataset()

    ADVANCED_DIR = (
        PROCESSED_DIR /
        "audio"
    )

    os.makedirs(ADVANCED_DIR, exist_ok=True)

    output_path = (
        ADVANCED_DIR /
        "advanced_audio_features.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nDataset saved:")

    print(output_path)

    print(f"\nTotal samples: {len(df)}")