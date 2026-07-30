import pandas as pd
import os

from ai_engine.data.loaders.ravdess_loader import (
    load_ravdess_files
)

from ai_engine.features.audio.audio_preprocessor import (
    load_audio,
    normalize_audio
)

from ai_engine.features.audio.audio_feature_extractor import (
    extract_mfcc,
    extract_pitch,
    extract_energy
)

from ai_engine.configs.project_paths import (
    PROCESSED_DIR
)

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


def extract_emotion_from_filename(file_path):

    filename = os.path.basename(file_path)

    emotion_code = filename.split("-")[2]

    return EMOTION_MAP.get(emotion_code, "unknown")


def build_dataset():

    all_files = load_ravdess_files()

    dataset_rows = []

    for file_path in all_files:

        try:

            signal, sr = load_audio(file_path)

            signal = normalize_audio(signal)

            mfcc = extract_mfcc(signal, sr)

            pitch = extract_pitch(signal, sr)

            energy = extract_energy(signal)

            emotion = extract_emotion_from_filename(
                file_path
            )

            row = {
                "file": file_path,
                "emotion": emotion,
                "pitch": pitch,
                "energy": energy
            }

            for i, value in enumerate(mfcc):

                row[f"mfcc_{i+1}"] = value

            dataset_rows.append(row)

        except Exception as e:

            print(f"Error processing {file_path}")
            print(e)

    return pd.DataFrame(dataset_rows)


if __name__ == "__main__":

    print("Building audio feature dataset...")

    df = build_dataset()

    print(df.head())


    AUDIO_PROCESSED_DIR = (
        PROCESSED_DIR / "audio"
    )

    os.makedirs(AUDIO_PROCESSED_DIR, exist_ok=True)

    output_path = (
        AUDIO_PROCESSED_DIR /
        "audio_features.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"\nDataset saved to:")
    print(output_path)

    print(f"\nTotal samples: {len(df)}")
