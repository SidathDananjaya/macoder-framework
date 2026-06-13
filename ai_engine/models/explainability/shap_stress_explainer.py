import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

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

# =========================
# LOAD DATASET
# =========================

DATASET_PATH = (
    PROCESSED_DIR /
    "audio" /
    "stress_audio_features.csv"
)

df = pd.read_csv(DATASET_PATH)

# Features
X = df.drop(columns=[
    "file",
    "emotion",
    "stress_level"
])

# Labels
y = df["stress_level"]

# Encode
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# Scale
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

# =========================
# LOAD MODEL
# =========================

MODEL_PATH = (
    EXPERIMENTS_DIR /
    "exp_002_stress_classifier" /
    "stress_classifier.pkl"
)

model = joblib.load(MODEL_PATH)

# =========================
# SHAP EXPLAINER
# =========================

print("Building SHAP explainer...")

explainer = shap.TreeExplainer(model)

# Use smaller sample for speed
sample_data = X_test[:100]

shap_values = explainer.shap_values(sample_data)

# =========================
# SUMMARY PLOT
# =========================

print("Generating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    sample_data,
    feature_names=X.columns,
    show=False
)

OUTPUT_PATH = (
    EXPERIMENTS_DIR /
    "exp_002_stress_classifier" /
    "shap_summary.png"
)

plt.savefig(
    OUTPUT_PATH,
    bbox_inches="tight"
)

print("\nSHAP summary saved:")

print(OUTPUT_PATH)