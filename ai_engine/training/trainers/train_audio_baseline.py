import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

from pathlib import Path

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

print("Dataset Loaded")
print(df.head())

# Drop non-feature columns
X = df.drop(columns=["file", "emotion"])

# Labels
y = df["emotion"]

# Encode labels
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train
print("\nTraining model...")

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# Save experiment
EXPERIMENT_DIR = (
    EXPERIMENTS_DIR /
    "exp_001_audio_baseline"
)

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Save model
joblib.dump(
    model,
    EXPERIMENT_DIR / "audio_baseline_model.pkl"
)

# Save scaler
joblib.dump(
    scaler,
    EXPERIMENT_DIR / "scaler.pkl"
)

# Save label encoder
joblib.dump(
    label_encoder,
    EXPERIMENT_DIR / "label_encoder.pkl"
)

print("\nModel saved successfully.")