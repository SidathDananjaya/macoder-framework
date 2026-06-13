import pandas as pd

from ai_engine.configs.project_paths import (
    PROCESSED_DIR
)

# Load advanced dataset
DATASET_PATH = (
    PROCESSED_DIR /
    "audio" /
    "advanced_audio_features.csv"
)

df = pd.read_csv(DATASET_PATH)

# Emotion → Stress mapping
STRESS_MAP = {
    "angry": "high",
    "fearful": "high",

    "disgust": "medium",
    "sad": "medium",
    "surprised": "medium",

    "calm": "low",
    "happy": "low",
    "neutral": "low"
}

# Create stress label column
df["stress_level"] = df["emotion"].map(
    STRESS_MAP
)

# Save updated dataset
OUTPUT_PATH = (
    PROCESSED_DIR /
    "audio" /
    "stress_audio_features.csv"
)

df.to_csv(OUTPUT_PATH, index=False)

print("Stress dataset created.")

print(f"\nSaved to:\n{OUTPUT_PATH}")

print("\nStress Distribution:\n")

print(df["stress_level"].value_counts())