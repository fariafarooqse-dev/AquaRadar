"""Extract feature set f1 for every eval row in manifest_v1.csv (no training).

Usage:
    python scripts/extract_features_v1.py data/raw/zenodo_leak_audio

Reads : data/processed/manifest_v1.csv
Writes: data/processed/features_f1.npz, outputs/features_summary.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_features import FEATURE_VERSION, extract_features, feature_names, load_audio  # noqa: E402

MANIFEST = Path("data/processed/manifest_v1.csv")
OUT_NPZ = Path("data/processed/features_f1.npz")
OUT_TXT = Path("outputs/features_summary.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    root = ap.parse_args().root.resolve()

    df = pd.read_csv(MANIFEST)
    df = df[df["is_eval_row"]].reset_index(drop=True)

    X, keep, failures = [], [], []
    for i, r in df.iterrows():
        try:
            audio, sr = load_audio(root / r["path"])
            X.append(extract_features(audio, sr))
            keep.append(i)
        except Exception as exc:  # noqa: BLE001
            failures.append((r["path"], f"{type(exc).__name__}: {exc}"))
    ok = df.loc[keep].reset_index(drop=True)
    X = np.vstack(X) if X else np.empty((0, len(feature_names())))

    out: list[str] = []
    P = out.append
    P(f"feature version: {FEATURE_VERSION}   n_features: {len(feature_names())}")
    P(f"eval rows: {len(df)}   extracted: {len(ok)}   failed: {len(failures)}")
    for path, msg in failures[:20]:
        P(f"  FAILED {path}: {msg}")
    if len(ok):
        P(f"X shape: {X.shape}   NaN/inf: {int((~np.isfinite(X)).sum())}")
        sd = X.std(axis=0)
        P(f"constant features (std < 1e-9): {[n for n, s in zip(feature_names(), sd) if s < 1e-9]}")
        P("rows per label x fold_primary:")
        P(pd.crosstab(ok["fold_primary"], ok["label"]).to_string())
        names = feature_names()
        sel = ["spectral_centroid_hz", "spectral_flatness", "zero_crossing_rate", "frame_rms_cv"]
        P("\nper-label mean of a few features (descriptive only, NOT evidence of separability):")
        tab = pd.DataFrame(X[:, [names.index(s) for s in sel]], columns=sel)
        tab["label"] = ok["label"].values
        P(tab.groupby("label").mean().round(3).to_string())
        OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
        # Plain numpy arrays only: with pandas 3, `.values.astype(str)` stays a pandas array and
        # np.savez would pickle it as an object array.
        np.savez(OUT_NPZ, X=X,
                 y=np.asarray(ok["label"].tolist(), dtype=str),
                 fold_primary=ok["fold_primary"].to_numpy(dtype=int),
                 fold_sensitivity=ok["fold_sensitivity"].to_numpy(dtype=int),
                 paths=np.asarray(ok["path"].tolist(), dtype=str),
                 feature_names=np.asarray(names, dtype=str),
                 feature_version=np.asarray(FEATURE_VERSION, dtype=str))
        chk = np.load(OUT_NPZ, allow_pickle=False)   # fails loudly if anything was pickled
        P(f"\nsaved {OUT_NPZ.resolve()}  (reload check OK: {sorted(chk.files)})")

    text = "\n".join(out)
    print(text)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
