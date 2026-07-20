import os
import pandas as pd

from ai_engine.features.audio.audio_feature_extractor import (
    AudioFeatureExtractor
)

extractor = AudioFeatureExtractor()

dataset = []

base_path = (
    "datasets/raw/RAVDESS/"
    "Audio_Speech_Actors_01-24"
)

emotion_map = {

    "01": "neutral",

    "02": "calm",

    "03": "happy",

    "04": "sad",

    "05": "angry",

    "06": "fear",

    "07": "disgust",

    "08": "surprise"
}

for actor in os.listdir(base_path):

    actor_path = os.path.join(
        base_path,
        actor
    )

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        emotion_code = file.split("-")[2]

        emotion = emotion_map.get(
            emotion_code
        )

        if emotion is None:
            continue

        filepath = os.path.join(
            actor_path,
            file
        )

        try:

            features = extractor.extract_features(
                filepath
            )

            features["emotion"] = emotion

            dataset.append(features)

            print(
                "Processed:",
                file
            )

        except Exception as e:

            print(
                "Failed:",
                file,
                e
            )

df = pd.DataFrame(dataset)

output_path = (
    "datasets/processed/audio/"
    "audio_emotion_features.csv"
)

df.to_csv(
    output_path,
    index=False
)

print(
    "\nDataset Saved:"
)

print(
    output_path
)

print(
    "\nTotal Samples:",
    len(df)
)