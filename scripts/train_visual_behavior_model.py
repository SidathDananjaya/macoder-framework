import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from ai_engine.models.visual.visual_behavior_model import VisualBehaviorModel

from sklearn.preprocessing import StandardScaler

DATASET_PATH = (
    "datasets/processed/video/"
    "visual_behavior_features.csv"
)

MODEL_OUTPUT = (
    "experiments/exp_005_visual_behavior/"
)

Path(MODEL_OUTPUT).mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(DATASET_PATH)

print(df.head())

# -----------------------------
# Features
# -----------------------------

X = df[[
    "avg_ear",
    "avg_yaw",
    "avg_pitch",
    "avg_roll",
    "left_gaze_ratio",
    "right_gaze_ratio",
    "center_gaze_ratio",
    "blink_rate",
    "gaze_stability",
    "movement_score"
]]

# -----------------------------
# Labels
# -----------------------------

y = df["emotion"]

# Encode labels
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -----------------------------
# Split dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -----------------------------
# Train model
# -----------------------------

model = VisualBehaviorModel()

model.train(
    X_train,
    y_train
)

# -----------------------------
# Predictions
# -----------------------------

predictions = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

report = classification_report(
    y_test,
    predictions
)

matrix = confusion_matrix(
    y_test,
    predictions
)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(report)

print("\nConfusion Matrix:\n")
print(matrix)

# -----------------------------
# Save model
# -----------------------------

joblib.dump(
    model,
    f"{MODEL_OUTPUT}/visual_behavior_model.pkl"
)

joblib.dump(
    scaler,
    f"{MODEL_OUTPUT}/visual_scaler.pkl"
)

joblib.dump(
    encoder,
    f"{MODEL_OUTPUT}/visual_behavior_label_encoder.pkl"
)

print("\nModel saved successfully")