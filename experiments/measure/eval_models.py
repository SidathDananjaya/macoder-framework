import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    GroupKFold,
    cross_val_predict,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

OUT_DIR = "experiments/measure"


def _model():
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, random_state=42),
    )


def _encode_categoricals(frame):
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("category").cat.codes
    return out


def _actors_from_files(files):
    actors = []
    for path in files:
        m = re.search(r"Actor_(\d+)", str(path))
        actors.append(m.group(1) if m else "unknown")
    return np.array(actors)


def _cv_scores(X, y, n_splits=5, groups=None):
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )
        split_iter = splitter.split(X, y)

    fold_acc = []
    for train_idx, test_idx in split_iter:
        mdl = _model()
        mdl.fit(X[train_idx], y[train_idx])
        pred = mdl.predict(X[test_idx])
        fold_acc.append(accuracy_score(y[test_idx], pred))

    if groups is not None:
        y_pred = cross_val_predict(
            _model(), X, y, cv=GroupKFold(n_splits=n_splits), groups=groups
        )
    else:
        y_pred = cross_val_predict(
            _model(), X, y,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        )

    return {
        "accuracy_mean": round(float(np.mean(fold_acc)), 3),
        "accuracy_std": round(float(np.std(fold_acc)), 3),
        "macro_f1": round(float(f1_score(y, y_pred, average="macro", zero_division=0)), 3),
        "macro_precision": round(float(precision_score(y, y_pred, average="macro", zero_division=0)), 3),
        "macro_recall": round(float(recall_score(y, y_pred, average="macro", zero_division=0)), 3),
        "n_samples": int(len(y)),
        "n_classes": int(len(np.unique(y))),
        "cv": f"{n_splits}-fold" + (" (subject-independent, by actor)" if groups is not None else " stratified"),
    }, y_pred


def _save_confusion(y_true, y_pred, title, filename):
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def eval_dataset(name, csv, label_col, drop_cols, note="", group_by_actor=False,
                 n_splits=5):
    df = pd.read_csv(csv)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(subset=[label_col])

    groups = None
    if group_by_actor and "file" in df.columns:
        groups = _actors_from_files(df["file"])

    y = df[label_col].astype(str).values
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X_df = X_df.select_dtypes(include=[np.number, "object"])
    X_df = _encode_categoricals(X_df).fillna(0.0)
    X = X_df.values.astype(np.float32)

    min_class = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(n_splits, min_class))

    scores, y_pred = _cv_scores(X, y, n_splits=n_splits, groups=groups)
    scores["note"] = note
    scores["features"] = X.shape[1]

    cm_path = _save_confusion(
        y, y_pred, f"{name} (CV confusion)", f"confusion_{name}.png"
    )
    scores["confusion_png"] = cm_path
    return scores


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    print("Evaluating models (this retrains per fold; ~1-2 min) ...\n")

    results["audio_emotion"] = eval_dataset(
        "audio_emotion",
        "datasets/processed/audio/audio_emotion_features.csv",
        label_col="emotion",
        drop_cols=["emotion", "file"],
        note="8-class RAVDESS emotion from audio features.",
    )

    results["audio_stress_random"] = eval_dataset(
        "audio_stress_random",
        "datasets/processed/audio/stress_audio_features.csv",
        label_col="stress_level",
        drop_cols=["file", "emotion", "stress_level"],
        note="3-class stress (proxy-labelled from emotion), random split.",
    )

    results["audio_stress_subject_independent"] = eval_dataset(
        "audio_stress_subject_independent",
        "datasets/processed/audio/stress_audio_features.csv",
        label_col="stress_level",
        drop_cols=["file", "emotion", "stress_level"],
        note="Same model, subject-independent (by actor) - the honest number.",
        group_by_actor=True,
    )

    results["visual_behavior"] = eval_dataset(
        "visual_behavior",
        "datasets/processed/video/visual_behavior_features.csv",
        label_col="emotion",
        drop_cols=["emotion", "emotion_score", "cognitive_load",
                   "deception_risk", "stress_score"],
        note="8-class emotion from genuine visual behaviour features only "
             "(leaky emotion_score/derived columns removed).",
    )

    results["cognitive_state"] = eval_dataset(
        "cognitive_state",
        "datasets/processed/cognition/cognitive_states.csv",
        label_col="cognitive_state",
        drop_cols=["cognitive_state"],
        note="WARNING: only 24 samples - NOT statistically meaningful; "
             "report as illustrative only (fixes the misleading 1.00 headline).",
    )

    with open(os.path.join(OUT_DIR, "model_eval_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"{'Model':<34}{'Acc (mean±std)':>18}{'MacroF1':>10}{'N':>7}{'K':>6}")
    for k, s in results.items():
        acc = f"{s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f}"
        print(f"{k:<34}{acc:>18}{s['macro_f1']:>10}{s['n_samples']:>7}{s['n_classes']:>6}")
    print(f"\nSaved -> {OUT_DIR}/model_eval_results.json + confusion_*.png")
    print("\nNote: cognitive_state has n=24 -> disclose as illustrative only.")


if __name__ == "__main__":
    main()
