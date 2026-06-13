import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from ai_engine.configs.project_paths import (
    PROCESSED_DIR,
    EXPERIMENTS_DIR
)

# =========================
# LOAD DATASET
# =========================

DATASET_PATH = (
    PROCESSED_DIR /
    "audio" /
    "stress_audio_features.csv"
)

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded")

print(df.head())

# =========================
# FEATURES
# =========================

X = df.drop(columns=[
    "file",
    "emotion",
    "stress_level"
])

# =========================
# LABELS
# =========================

y = df["stress_level"]

# Encode labels
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# =========================
# SCALING
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining samples: {len(X_train)}")

print(f"Testing samples: {len(X_test)}")

# =========================
# MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

# =========================
# TRAIN
# =========================

print("\nTraining stress classifier...")

model.fit(X_train, y_train)

# =========================
# PREDICT
# =========================

y_pred = model.predict(X_test)

# =========================
# METRICS
# =========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nStress Accuracy: {accuracy:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# =========================
# SAVE MODEL
# =========================

EXPERIMENT_DIR = (
    EXPERIMENTS_DIR /
    "exp_002_stress_classifier"
)

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    EXPERIMENT_DIR /
    "stress_classifier.pkl"
)

joblib.dump(
    scaler,
    EXPERIMENT_DIR /
    "stress_scaler.pkl"
)

joblib.dump(
    label_encoder,
    EXPERIMENT_DIR /
    "stress_label_encoder.pkl"
)

print("\nStress model saved successfully.")