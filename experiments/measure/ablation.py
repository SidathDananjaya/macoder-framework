import json
import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA = "datasets/processed/multimodal/aligned_features.csv"
OUT = "experiments/measure/ablation_results.json"


def _model():
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, random_state=42),
    )


def _evaluate(X, y, groups, classes, n_splits):
    gkf = GroupKFold(n_splits=n_splits)
    fold_acc, y_true_all, y_pred_all = [], [], []
    proba_all, proba_true = [], []

    for tr, te in gkf.split(X, y, groups):
        mdl = _model()
        mdl.fit(X[tr], y[tr])
        pred = mdl.predict(X[te])
        fold_acc.append(accuracy_score(y[te], pred))
        y_true_all.extend(y[te]); y_pred_all.extend(pred)

        proba = mdl.predict_proba(X[te])
        aligned = np.zeros((len(te), len(classes)))
        for j, c in enumerate(mdl.classes_):
            aligned[:, list(classes).index(c)] = proba[:, j]
        proba_all.append(aligned); proba_true.extend(y[te])

    proba_all = np.vstack(proba_all)
    y_bin = label_binarize(proba_true, classes=list(classes))
    try:
        auc = roc_auc_score(y_bin, proba_all, average="macro", multi_class="ovr")
    except ValueError:
        auc = float("nan")

    return {
        "accuracy_mean": round(float(np.mean(fold_acc)), 3),
        "accuracy_std": round(float(np.std(fold_acc)), 3),
        "macro_f1": round(float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)), 3),
        "macro_auc": round(float(auc), 3),
    }


def main():
    if not os.path.exists(DATA):
        raise SystemExit(f"{DATA} not found - run build_aligned_multimodal first.")

    df = pd.read_csv(DATA).fillna(0.0)
    y = df["emotion"].astype(str).values
    groups = df["actor"].astype(str).values
    classes = sorted(np.unique(y))
    n_splits = int(min(pd.Series(groups).nunique(), 5))

    blocks = {
        "Audio only": [c for c in df.columns if c.startswith("aud_")],
        "Video only": [c for c in df.columns if c.startswith("vis_")],
        "Timing only": [c for c in df.columns if c.startswith("tim_")],
    }
    blocks["Concatenation fusion (all)"] = (
        blocks["Audio only"] + blocks["Video only"] + blocks["Timing only"]
    )

    results = {"n_samples": int(len(df)), "n_actors": int(pd.Series(groups).nunique()),
               "n_classes": len(classes), "cv": f"{n_splits}-fold subject-independent",
               "configs": {}}

    for name, cols in blocks.items():
        X = df[cols].values.astype(np.float32)
        results["configs"][name] = _evaluate(X, y, groups, classes, n_splits)
        results["configs"][name]["n_features"] = len(cols)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"=== Cross-modal ablation (Section 6.3) ===")
    print(f"{results['n_samples']} utterances, {results['n_actors']} actors, "
          f"{results['n_classes']} emotions, {results['cv']}\n")
    print(f"{'Configuration':<30}{'Acc(mean±std)':>16}{'MacroF1':>9}{'AUC':>7}{'feats':>7}")
    for name, s in results["configs"].items():
        acc = f"{s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f}"
        print(f"{name:<30}{acc:>16}{s['macro_f1']:>9}{s['macro_auc']:>7}{s['n_features']:>7}")
    print(f"\nSaved -> {OUT}")
    best_uni = max(("Audio only", "Video only", "Timing only"),
                   key=lambda k: results["configs"][k]["macro_f1"])
    print(f"Most informative single modality: {best_uni} "
          f"(macro-F1 {results['configs'][best_uni]['macro_f1']}). "
          f"Fusion macro-F1 {results['configs']['Concatenation fusion (all)']['macro_f1']}.")


if __name__ == "__main__":
    main()
