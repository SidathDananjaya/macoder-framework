import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    classification_report,
    accuracy_score
)

DATASET = (
    "datasets/processed/audio/"
    "audio_emotion_features.csv"
)

OUTPUT_DIR = (
    "experiments/exp_007_audio_emotion"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

df = pd.read_csv(DATASET)

X = df.drop(
    columns=["emotion"]
)

y = df["emotion"]

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y_encoded,

    test_size=0.2,

    random_state=42,

    stratify=y_encoded
)

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=15,

    random_state=42
)

model.fit(

    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

report = classification_report(

    y_test,

    predictions,

    target_names=encoder.classes_
)

print(report)

joblib.dump(

    model,

    f"{OUTPUT_DIR}/audio_emotion_model.pkl"
)

joblib.dump(

    scaler,

    f"{OUTPUT_DIR}/audio_scaler.pkl"
)

joblib.dump(

    encoder,

    f"{OUTPUT_DIR}/audio_label_encoder.pkl"
)

with open(

    f"{OUTPUT_DIR}/classification_report.txt",

    "w"

) as f:

    f.write(report)

print(
    "\nAudio Emotion Model Saved"
)