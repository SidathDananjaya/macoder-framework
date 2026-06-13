import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from ai_engine.configs.project_paths import (
    PROCESSED_DIR,
    EXPERIMENTS_DIR
)

# Load dataset
DATASET_PATH = (
    PROCESSED_DIR /
    "audio" /
    "audio_features.csv"
)

df = pd.read_csv(DATASET_PATH)

# Features
X = df.drop(columns=["file", "emotion"])

# Labels
y = df["emotion"]

# Encode labels
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Load trained model
MODEL_PATH = (
    EXPERIMENTS_DIR /
    "exp_001_audio_baseline" /
    "audio_baseline_model.pkl"
)

model = joblib.load(MODEL_PATH)

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)

print(f"\nAccuracy: {accuracy:.4f}")

# Create experiment directory
EXPERIMENT_DIR = (
    EXPERIMENTS_DIR /
    "exp_001_audio_baseline"
)

# Save metrics JSON
metrics_path = (
    EXPERIMENT_DIR /
    "metrics.json"
)

with open(metrics_path, "w") as f:

    json.dump(report, f, indent=4)

print(f"\nMetrics saved:")
print(metrics_path)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Audio Baseline Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

# Save figure
confusion_path = (
    EXPERIMENT_DIR /
    "confusion_matrix.png"
)

plt.savefig(confusion_path)

print(f"\nConfusion matrix saved:")
print(confusion_path)

plt.show()