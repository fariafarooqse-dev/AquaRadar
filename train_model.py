"""Baseline classifiers: grouped cross-validation and final model fit (Member 1 owns this file).

Usage (from project root):
    python train_model.py          # same as "cv": evaluation only, nothing saved
    python train_model.py final    # fit audio_rf on all eval rows, save model + card,
                                   # and print an abstention (uncertainty) table from
                                   # out-of-fold probabilities (primary folds)

Reads : data/processed/features_f1.npz, data/processed/manifest_v1.csv
Writes: outputs/baseline_cv_report.txt                         (cv)
        models/leak_model_v1.joblib, models/leak_model_v1_card.json,
        outputs/final_model_report.txt                         (final)

Models compared, all with the same grouped folds:
  dummy_majority   - predicts the most frequent class (floor)
  metadata_only    - uses ONLY filename metadata (material, zone, device), NO audio.
                     A shortcut probe: if this scores well, metadata confounds the labels.
  audio_logreg     - standardised logistic regression on the 39 f1 audio features
  audio_rf         - random forest on the same features
Two fold schemes:
  fold_primary     - coarse groups (strict, the headline scheme)
  fold_sensitivity - fine groups (known to leave related clips on both sides; shows inflation)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = Path("data/processed/features_f1.npz")
MANIFEST = Path("data/processed/manifest_v1.csv")
OUT_TXT = Path("outputs/baseline_cv_report.txt")
MODEL_PATH = Path("models/leak_model_v1.joblib")
CARD_PATH = Path("models/leak_model_v1_card.json")
FINAL_TXT = Path("outputs/final_model_report.txt")
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
SEED = 42
CLASSES = ["leak", "no_leak", "noise"]


def make_models() -> dict:
    return {
        "dummy_majority": ("audio", lambda: DummyClassifier(strategy="most_frequent")),
        "metadata_only": ("meta", lambda: LogisticRegression(max_iter=2000, class_weight="balanced")),
        "audio_logreg": ("audio", lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))),
        "audio_rf": ("audio", lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)),
    }


def cross_validate(X: np.ndarray, y: np.ndarray, folds: np.ndarray, make_model) -> tuple[pd.DataFrame, np.ndarray]:
    oof = np.empty(len(y), dtype=object)
    rows = []
    for k in sorted(np.unique(folds)):
        te = folds == k
        model = make_model()
        model.fit(X[~te], y[~te])
        pred = model.predict(X[te])
        oof[te] = pred
        rec = recall_score(y[te], pred, labels=CLASSES, average=None, zero_division=0)
        rows.append({"fold": int(k), "n_test": int(te.sum()),
                     "bal_acc": balanced_accuracy_score(y[te], pred),
                     "macro_f1": f1_score(y[te], pred, labels=CLASSES, average="macro", zero_division=0),
                     **{f"recall_{c}": r for c, r in zip(CLASSES, rec)}})
    return pd.DataFrame(rows), oof


def load_data():
    z = np.load(FEATURES, allow_pickle=False)
    X_audio, y, paths = z["X"], z["y"].astype(str), z["paths"].astype(str)
    meta = pd.read_csv(MANIFEST).set_index("path").loc[paths]
    X_meta = pd.get_dummies(meta[["material", "zone", "device"]].astype(str)).to_numpy(dtype=float)
    assert (meta["label"].to_numpy() == y).all(), "label mismatch between features and manifest"
    return z, X_audio, X_meta, y


def run_cv() -> None:
    z, X_audio, X_meta, y = load_data()

    out: list[str] = []
    P = out.append
    P(f"samples: {len(y)}   classes: {dict(pd.Series(y).value_counts())}")
    P(f"audio feature matrix: {X_audio.shape}   metadata matrix: {X_meta.shape}")

    summary, details = [], {}
    for scheme in ("fold_primary", "fold_sensitivity"):
        folds = z[scheme]
        for name, (kind, factory) in make_models().items():
            X = X_audio if kind == "audio" else X_meta
            per_fold, oof = cross_validate(X, y, folds, factory)
            details[(scheme, name)] = (per_fold, oof)
            miss_leak = float(np.mean(oof[y == "leak"] != "leak"))
            false_alarm = float(np.mean(oof[y == "no_leak"] == "leak"))
            summary.append({
                "scheme": scheme.replace("fold_", ""), "model": name,
                "bal_acc_mean": per_fold["bal_acc"].mean(), "bal_acc_sd": per_fold["bal_acc"].std(ddof=0),
                "bal_acc_min": per_fold["bal_acc"].min(), "bal_acc_max": per_fold["bal_acc"].max(),
                "macro_f1_mean": per_fold["macro_f1"].mean(),
                "pooled_bal_acc": balanced_accuracy_score(y, oof.astype(str)),
                "leak_missed": miss_leak, "no_leak_flagged_leak": false_alarm})
    P("\n== SUMMARY (per-fold mean/sd/min/max of balanced accuracy; pooled = all out-of-fold predictions) ==")
    P("chance balanced accuracy for 3 classes = 0.333")
    P(pd.DataFrame(summary).round(3).to_string(index=False))
    P("\nleak_missed = share of true leak clips NOT predicted leak.  "
      "no_leak_flagged_leak = share of true no_leak clips predicted leak.")

    for name in ("audio_logreg", "audio_rf", "metadata_only"):
        per_fold, oof = details[("fold_primary", name)]
        P(f"\n== PRIMARY scheme, {name}: per fold ==")
        P(per_fold.round(3).to_string(index=False))
        P(f"-- pooled out-of-fold report ({name}) --")
        P(classification_report(y, oof.astype(str), labels=CLASSES, digits=3, zero_division=0))
        cm = confusion_matrix(y, oof.astype(str), labels=CLASSES)
        P("confusion matrix (rows = true, cols = predicted; order: " + ", ".join(CLASSES) + ")")
        P(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    P("\nCAVEATS: single-site dataset; 16 leak / 18 no_leak / 15 noise groups in primary folds; "
      "noise folds test unseen sound categories; numbers are NOT field-reliability estimates.")
    text = "\n".join(out)
    print(text)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(text, encoding="utf-8")
    print(f"\n[saved {OUT_TXT.resolve()}]")


def oof_proba(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold class probabilities from audio_rf (columns in sorted class order)."""
    proba = np.zeros((len(y), len(CLASSES)))
    for k in sorted(np.unique(folds)):
        te = folds == k
        m = make_models()["audio_rf"][1]()
        m.fit(X[~te], y[~te])
        assert list(m.classes_) == CLASSES, f"unexpected class order {list(m.classes_)}"
        proba[te] = m.predict_proba(X[te])
    return proba


def abstention_table(y: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    pred = np.array(CLASSES)[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    rows = []
    for t in THRESHOLDS:
        kept = conf >= t
        leak, nl = y == "leak", y == "no_leak"
        rows.append({
            "threshold": t,
            "coverage": kept.mean(),
            "acc_on_kept": (pred[kept] == y[kept]).mean() if kept.any() else np.nan,
            "leak_silent_miss": (kept & (pred != "leak"))[leak].mean(),
            "leak_abstained": (~kept)[leak].mean(),
            "no_leak_false_alarm": (kept & (pred == "leak"))[nl].mean(),
            "noise_abstained": (~kept)[y == "noise"].mean()})
    return pd.DataFrame(rows)


def run_final() -> None:
    z, X_audio, _, y = load_data()
    folds = z["fold_primary"]
    out: list[str] = []
    P = out.append

    proba = oof_proba(X_audio, y, folds)
    oof_pred = np.array(CLASSES)[proba.argmax(axis=1)]
    cv_bal = balanced_accuracy_score(y, oof_pred)
    P(f"audio_rf out-of-fold (primary folds) pooled balanced accuracy: {cv_bal:.3f}")
    P("\n== ABSTENTION TABLE (abstain when top class score < threshold; RF scores are NOT calibrated probabilities) ==")
    P("leak_silent_miss = true leak, kept, predicted not-leak.  leak_abstained = true leak sent to 'uncertain'.")
    P(abstention_table(y, proba).round(3).to_string(index=False))

    model = make_models()["audio_rf"][1]()
    model.fit(X_audio, y)
    names = [str(n) for n in z["feature_names"]]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": list(model.classes_), "feature_version": str(z["feature_version"]),
                 "feature_names": names}, MODEL_PATH)
    card = {
        "model_version": "leak_model_v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)",
        "sklearn_version": sklearn.__version__,
        "feature_version": str(z["feature_version"]),
        "n_features": len(names),
        "classes": list(model.classes_),
        "training_rows": {k: int(v) for k, v in pd.Series(y).value_counts().items()},
        "training_data": "Wang, Q., Mei, Z., Zhan, F., & Chen, J. (2026). Acoustic data for the manuscript entitled "
                         "'Self-supervised acoustic leakage detection for water distribution systems: A real-time "
                         "diagnosis framework under data scarcity' (v1). Zenodo. "
                         "https://doi.org/10.5281/zenodo.18631450",
        "training_data_licence": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode), as stated in "
                                 "the saved Zenodo record metadata. AquaRadar extracted audio features and trained a "
                                 "classifier; this is not endorsed by the dataset authors. See "
                                 "docs/dataset_documentation.md.",
        "evaluation": f"5-fold grouped CV (coarse groups), pooled out-of-fold balanced accuracy {cv_bal:.3f}",
        "limitations": [
            "Single-site dataset; not validated on campus pipes.",
            "Screening result only: resembles labelled training clips; not proof of a leak.",
            "Noise class covers only the 15 sound categories in the dataset.",
            "Expects 1.0 s clips; longer recordings need windowing (not implemented yet).",
            "Scores are not calibrated probabilities."],
    }
    CARD_PATH.write_text(json.dumps(card, indent=2), encoding="utf-8")
    chk = joblib.load(MODEL_PATH)
    assert chk["model"].predict(X_audio[:5]).shape == (5,)
    P(f"\nsaved {MODEL_PATH.resolve()} ({MODEL_PATH.stat().st_size / 1e6:.2f} MB) and {CARD_PATH.resolve()}")
    P("reload + predict check OK")
    text = "\n".join(out)
    print(text)
    FINAL_TXT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_TXT.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="cv", choices=["cv", "final"])
    run_final() if ap.parse_args().mode == "final" else run_cv()


if __name__ == "__main__":
    main()
