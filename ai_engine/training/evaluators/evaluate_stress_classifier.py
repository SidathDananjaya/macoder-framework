import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from ai_engine.configs.project_paths import (
    PROCESSED_DIR,
    EXPERIMENTS_DIR
)


DATASET_PATH = (
    PROCESSED_DIR /
    "audio" /
    "stress_audio_features.csv"
)

df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=[
    "file",
    "emotion",
    "stress_level"
])

y = df["stress_level"]

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


MODEL_PATH = (
    EXPERIMENTS_DIR /
    "exp_002_stress_classifier" /
    "stress_classifier.pkl"
)

model = joblib.load(MODEL_PATH)

y_pred = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nStress Accuracy: {accuracy:.4f}")

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)

EXPERIMENT_DIR = (
    EXPERIMENTS_DIR /
    "exp_002_stress_classifier"
)

metrics_path = (
    EXPERIMENT_DIR /
    "stress_metrics.json"
)

with open(metrics_path, "w") as f:

    json.dump(report, f, indent=4)


cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Stress Classification Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

confusion_path = (
    EXPERIMENT_DIR /
    "stress_confusion_matrix.png"
)

plt.savefig(confusion_path)

print("\nStress confusion matrix saved.")

plt.show()
