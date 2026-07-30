import pandas as pd
import matplotlib.pyplot as plt
import joblib

from ai_engine.configs.project_paths import (
    PROCESSED_DIR,
    EXPERIMENTS_DIR
)

DATASET_PATH = (
    PROCESSED_DIR /
    "audio" /
    "audio_features.csv"
)

df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=["file", "emotion"])

MODEL_PATH = (
    EXPERIMENTS_DIR /
    "exp_001_audio_baseline" /
    "audio_baseline_model.pkl"
)

model = joblib.load(MODEL_PATH)

importance = model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print(importance_df.head(10))

plt.figure(figsize=(12, 6))

plt.bar(
    importance_df["feature"][:10],
    importance_df["importance"][:10]
)

plt.xticks(rotation=45)

plt.title("Top Audio Feature Importance")

plt.tight_layout()

output_path = (
    EXPERIMENTS_DIR /
    "exp_001_audio_baseline" /
    "feature_importance.png"
)

plt.savefig(output_path)

print(f"\nFeature importance graph saved:")
print(output_path)

plt.show()
