import os
import joblib
import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    classification_report,
    accuracy_score
)

from ai_engine.models.cognitive.cognitive_state_model import (
    CognitiveStateModel
)

DATASET = (
    "datasets/processed/cognition/"
    "cognitive_states.csv"
)

OUTPUT_DIR = (
    "experiments/exp_008_cognitive"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

df = pd.read_csv(DATASET)

# ----------------------------
# Convert categorical columns
# ----------------------------

mapping = {

    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3
}

df["stress_score"] = (
    df["stress_score"]
    .map(mapping)
)

df["cognitive_load"] = (
    df["cognitive_load"]
    .map(mapping)
)

df["deception_risk"] = (
    df["deception_risk"]
    .map(mapping)
)

X = df.drop(
    columns=["cognitive_state"]
)

y = df["cognitive_state"]

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

model = CognitiveStateModel()

model.train(
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
    predictions
)

print(report)

joblib.dump(

    model.model,

    f"{OUTPUT_DIR}/cognitive_model.pkl"
)

joblib.dump(

    scaler,

    f"{OUTPUT_DIR}/scaler.pkl"
)

joblib.dump(

    label_encoder,

    f"{OUTPUT_DIR}/label_encoder.pkl"
)

with open(

    f"{OUTPUT_DIR}/classification_report.txt",

    "w"

) as f:

    f.write(report)

print(
    "\nCognitive Model Saved"
)