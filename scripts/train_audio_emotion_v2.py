"""Retrain the audio-emotion model on RAVDESS with the ALIGNED richer features.

Uses the *same* ``AudioFeatureExtractor`` the live pipeline uses, so training and
inference features are identical by construction (no skew). Replaces the weak
exp_007 model (5 collapsed features, ~0.34) with the enriched feature set.

Reports both a subject-independent (by-actor) score - the honest number - and a
random-split score for comparison, then trains the final model on all data and
saves it to experiments/exp_009_audio_emotion_v2/.

    ./vision_env/Scripts/python.exe -m scripts.train_audio_emotion_v2
"""

import os

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score

from ai_engine.features.audio.audio_feature_extractor import AudioFeatureExtractor

BASE = "datasets/raw/RAVDESS/Audio_Speech_Actors_01-24"
OUT_DIR = "experiments/exp_009_audio_emotion_v2"

EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fear", "07": "disgust", "08": "surprise",
}

extractor = AudioFeatureExtractor()


def build_dataset():
    rows, labels, actors = [], [], []
    names = extractor.feature_names()

    for actor in sorted(os.listdir(BASE)):
        adir = os.path.join(BASE, actor)
        if not os.path.isdir(adir):
            continue
        for f in sorted(os.listdir(adir)):
            if not f.endswith(".wav"):
                continue
            emotion = EMOTION_MAP.get(f.split("-")[2])
            if emotion is None:
                continue
            feats = extractor.extract_features(os.path.join(adir, f))
            rows.append([feats[n] for n in names])
            labels.append(emotion)
            actors.append(actor)
        print(f"  {actor}: {len(rows)} clips")

    X = pd.DataFrame(rows, columns=names)
    return X, np.array(labels), np.array(actors)


def _model():
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=300, random_state=42),
    )


def main():
    print("Extracting aligned features from RAVDESS audio ...")
    X, y, actors = build_dataset()
    print(f"\n{len(X)} clips, {X.shape[1]} features, {len(np.unique(y))} classes\n")

    # Subject-independent (by-actor) - the honest number.
    gpred = cross_val_predict(_model(), X, y, cv=GroupKFold(5), groups=actors)
    si_acc = accuracy_score(y, gpred)
    si_f1 = f1_score(y, gpred, average="macro", zero_division=0)

    # Random stratified split - comparable to the old 0.34 reporting.
    rpred = cross_val_predict(
        _model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42)
    )
    r_acc = accuracy_score(y, rpred)
    r_f1 = f1_score(y, rpred, average="macro", zero_division=0)

    print("=== Audio-emotion (aligned enriched features) ===")
    print(f"Subject-independent (by actor): acc={si_acc:.3f}  macroF1={si_f1:.3f}")
    print(f"Random stratified split:        acc={r_acc:.3f}  macroF1={r_f1:.3f}")
    print(f"(old exp_007, 5 features: ~0.34)")

    # Final model on all data.
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(scaler.transform(X), y_enc)

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(model, f"{OUT_DIR}/audio_emotion_model.pkl")
    joblib.dump(scaler, f"{OUT_DIR}/audio_scaler.pkl")
    joblib.dump(encoder, f"{OUT_DIR}/audio_label_encoder.pkl")

    with open(f"{OUT_DIR}/metrics.txt", "w", encoding="utf-8") as fh:
        fh.write(f"features: {X.shape[1]}\n")
        fh.write(f"subject_independent_acc: {si_acc:.3f}\n")
        fh.write(f"subject_independent_macroF1: {si_f1:.3f}\n")
        fh.write(f"random_split_acc: {r_acc:.3f}\n")
        fh.write(f"random_split_macroF1: {r_f1:.3f}\n")

    print(f"\nSaved model + scaler + encoder -> {OUT_DIR}/")
    print("Classes:", list(encoder.classes_))


if __name__ == "__main__":
    main()
