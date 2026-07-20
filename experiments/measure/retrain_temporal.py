"""Honest re-training + evaluation of the temporal-emotion LSTM.

The original temporal model (exp_006) was trained on 9 features **including
`emotion_score`** - which the audit found is a 1:1 encoding of the emotion label
(feature #8) - and on *randomly shuffled overlapping sliding-window sequences*
(adjacent windows share 59/60 frames, so a random split puts near-duplicates in
both train and test). Both massively inflate its accuracy.

This script quantifies and fixes that:

* **Leaky replication** - 9 features, random split (reproduces the inflated number).
* **Honest** - 7 GENUINE behavioural features only (drop emotion_score + the
  constant stress_score) and a **blocked** (contiguous, non-shuffled) split so
  overlapping windows do not straddle train/test.

The honest model + encoder are saved for live use (7-feature input, no leaked
label), so the deployed temporal modality is real rather than an echo of the
current frame emotion.

    ./vision_env/Scripts/python.exe -m experiments.measure.retrain_temporal

Writes experiments/exp_006_temporal_emotion/temporal_emotion_model_v2.h5 (+ encoder)
and experiments/measure/temporal_eval_results.json.
"""

import json
import os

import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping

from ai_engine.models.temporal.temporal_emotion_model import TemporalEmotionModel

X_PATH = "datasets/processed/temporal/X_sequences.npy"
Y_PATH = "datasets/processed/temporal/y_labels.npy"
OUT_DIR = "experiments/exp_006_temporal_emotion"
RESULTS = "experiments/measure/temporal_eval_results.json"

# Feature order (build_temporal_sequences.py):
# 0 avg_ear 1 avg_yaw 2 avg_pitch 3 avg_roll 4 blink_rate 5 gaze_stability
# 6 movement_score 7 emotion_score(LEAKY) 8 stress_score(constant)
GENUINE_IDX = [0, 1, 2, 3, 4, 5, 6]


def _train_eval(X, y_enc, n_classes, split, epochs=20):
    if split == "random":
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(X))
    else:  # blocked / contiguous
        idx = np.arange(len(X))
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]

    model = TemporalEmotionModel().build(
        sequence_length=X.shape[1], feature_count=X.shape[2], num_classes=n_classes
    )
    model.fit(
        X[tr], y_enc[tr], validation_data=(X[te], y_enc[te]),
        epochs=epochs, batch_size=32, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
    )
    pred = model.predict(X[te], verbose=0).argmax(axis=1)
    return model, {
        "accuracy": round(float(accuracy_score(y_enc[te], pred)), 3),
        "macro_f1": round(float(f1_score(y_enc[te], pred, average="macro", zero_division=0)), 3),
        "n_test": int(len(te)),
    }


def main():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    n_classes = len(enc.classes_)

    results = {}

    print("1/2 Leaky replication (9 features, random split) ...")
    _, results["leaky_9feat_random_split"] = _train_eval(X, y_enc, n_classes, "random")

    print("2/2 Honest (7 genuine features, blocked split) ...")
    X7 = X[:, :, GENUINE_IDX]
    honest_model, results["honest_7feat_blocked_split"] = _train_eval(
        X7, y_enc, n_classes, "blocked"
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    honest_model.save(f"{OUT_DIR}/temporal_emotion_model_v2.h5")
    joblib.dump(enc, f"{OUT_DIR}/temporal_label_encoder_v2.pkl")

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("\n=== Temporal LSTM - honest vs leaky ===")
    for name, s in results.items():
        print(f"{name:<34} acc={s['accuracy']:<6} macroF1={s['macro_f1']:<6} n_test={s['n_test']}")
    print(f"\nSaved honest 7-feature model -> {OUT_DIR}/temporal_emotion_model_v2.h5")
    print("The gap between the two rows IS the leakage. Report the honest row.")


if __name__ == "__main__":
    main()
