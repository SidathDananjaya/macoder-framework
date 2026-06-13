import pandas as pd
import numpy as np

from pathlib import Path


INPUT_CSV = (
    "datasets/processed/video/"
    "visual_behavior_features.csv"
)

OUTPUT_DIR = (
    "datasets/processed/temporal/"
)

SEQUENCE_LENGTH = 60


Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(INPUT_CSV)

feature_columns = [

    "avg_ear",

    "avg_yaw",

    "avg_pitch",

    "avg_roll",

    "blink_rate",

    "gaze_stability",

    "movement_score",

    "emotion_score",

    "stress_score"
]

X = []
y = []

for i in range(
    len(df) - SEQUENCE_LENGTH
):

    sequence = df.iloc[
        i:i + SEQUENCE_LENGTH
    ][feature_columns].values

    label = df.iloc[
        i + SEQUENCE_LENGTH
    ]["emotion"]

    X.append(sequence)

    y.append(label)

X = np.array(X)

y = np.array(y)

np.save(
    f"{OUTPUT_DIR}/X_sequences.npy",
    X
)

np.save(
    f"{OUTPUT_DIR}/y_labels.npy",
    y
)

print("Sequences built successfully")

print("X shape:", X.shape)

print("y shape:", y.shape)