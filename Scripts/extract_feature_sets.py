"""Extract every candidate feature set for every eval row in manifest_v1.csv (no training).

Usage (from the project root):
    python scripts/extract_feature_sets.py data/raw/zenodo_leak_audio
    python scripts/extract_feature_sets.py data/raw/zenodo_leak_audio --sets f1 f1_mfcc

Reads : data/processed/manifest_v1.csv
Writes: data/processed/feature_sets/<set>.npz   (one file per set; existing features_f1.npz is NOT touched)
        outputs/feature_sets_summary.txt

Design:
  * Each audio file is decoded once and every requested set is computed from it.
  * A clip that fails for ANY set is dropped for ALL sets, so every file has identical rows in
    identical order. The comparison in the next step is therefore like-for-like.
  * Each .npz also stores group_primary / group_sensitivity (needed for group-level confidence
    intervals later) and the fold assignments from manifest_v1.csv. No fold is recomputed here.
  * If data/processed/features_f1.npz exists, the f1 set is checked to be identical to it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import feature_sets as fs  # noqa: E402
from audio_features import FEATURE_VERSION, load_audio  # noqa: E402

MANIFEST = Path("data/processed/manifest_v1.csv")
OLD_F1 = Path("data/processed/features_f1.npz")
OUT_DIR = Path("data/processed/feature_sets")
OUT_TXT = Path("outputs/feature_sets_summary.txt")
NEEDED = ["path", "label", "is_eval_row", "fold_primary", "fold_sensitivity",
          "group_primary", "group_sensitivity"]


def extract_all(root: Path, df: pd.DataFrame, sets: list[str]):
    """Return ({set: [vectors]}, kept row indices, [(path, error)])."""
    mats: dict[str, list] = {s: [] for s in sets}
    keep: list[int] = []
    failures: list[tuple[str, str]] = []
    for i, r in df.iterrows():
        try:
            audio, sr = load_audio(root / r["path"])
            vecs = {s: fs.extract_set(audio, sr, s) for s in sets}
        except Exception as exc:  # noqa: BLE001
            failures.append((str(r["path"]), f"{type(exc).__name__}: {exc}"))
            continue
        for s in sets:
            mats[s].append(vecs[s])
        keep.append(i)
    return mats, keep, failures


def corr_with_f1(X: np.ndarray, n_f1: int = 39) -> tuple[int, int, float]:
    """For columns after the first n_f1: (n non-constant new columns, n with |r| > 0.9 to some f1 column,
    median of the per-column maximum |r|). Descriptive only; says nothing about usefulness."""
    base, new = X[:, :n_f1], X[:, n_f1:]
    bs, ns = base.std(axis=0), new.std(axis=0)
    base, new = base[:, bs > 1e-9], new[:, ns > 1e-9]
    if new.shape[1] == 0 or base.shape[1] == 0:
        return 0, 0, float("nan")
    zb = (base - base.mean(axis=0)) / base.std(axis=0)
    zn = (new - new.mean(axis=0)) / new.std(axis=0)
    r = np.abs(zn.T @ zb / len(X))
    mx = r.max(axis=1)
    return int(new.shape[1]), int((mx > 0.9).sum()), float(np.median(mx))


def save_set(path: Path, X: np.ndarray, ok: pd.DataFrame, set_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Plain numpy arrays only (pandas 3 arrays would be pickled by np.savez).
    np.savez(path, X=X,
             y=np.asarray(ok["label"].tolist(), dtype=str),
             fold_primary=ok["fold_primary"].to_numpy(dtype=int),
             fold_sensitivity=ok["fold_sensitivity"].to_numpy(dtype=int),
             group_primary=ok["group_primary"].to_numpy(dtype=int),
             group_sensitivity=ok["group_sensitivity"].to_numpy(dtype=int),
             paths=np.asarray(ok["path"].tolist(), dtype=str),
             feature_names=np.asarray(fs.feature_set_names(set_name), dtype=str),
             feature_version=np.asarray(set_name, dtype=str),
             blocks=np.asarray(",".join(fs.feature_set_blocks(set_name)), dtype=str))
    with np.load(path, allow_pickle=False) as chk:   # fails loudly if anything was pickled
        chk.files  # noqa: B018


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--sets", nargs="+", default=fs.available_sets(), choices=fs.available_sets())
    args = ap.parse_args()
    root = args.root.resolve()
    sets = list(dict.fromkeys(args.sets))

    df = pd.read_csv(MANIFEST)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise SystemExit(f"{MANIFEST} is missing columns {missing}. Re-run scripts/make_grouped_folds.py.")
    df = df[df["is_eval_row"]].reset_index(drop=True)

    mats, keep, failures = extract_all(root, df, sets)
    ok = df.loc[keep].reset_index(drop=True)

    out: list[str] = []
    P = out.append
    P(f"sets: {sets}")
    P(f"eval rows: {len(df)}   extracted for all sets: {len(ok)}   dropped: {len(failures)}")
    for p, msg in failures[:20]:
        P(f"  FAILED {p}: {msg}")
    if len(ok) == 0:
        print("\n".join(out))
        raise SystemExit("No rows extracted; nothing saved.")

    P("\nrows per label x fold_primary (identical for every set):")
    P(pd.crosstab(ok["fold_primary"], ok["label"]).to_string())

    P("\nset                n_features  constant_features  new_cols  new_cols_|r|>0.9_with_f1  median_max_|r|")
    for s in sets:
        X = np.vstack(mats[s])
        const = [n for n, sd in zip(fs.feature_set_names(s), X.std(axis=0)) if sd < 1e-9]
        if fs.feature_set_blocks(s)[0] == "f1" and X.shape[1] > 39:
            n_new, n_hi, med = corr_with_f1(X)
            corr_txt = f"{n_new:8d}  {n_hi:24d}  {med:14.3f}"
        else:
            corr_txt = f"{'-':>8}  {'-':>24}  {'-':>14}"
        P(f"{s:18s} {X.shape[1]:10d}  {len(const):17d}  {corr_txt}")
        if const:
            P(f"    constant: {const}")
        save_set(OUT_DIR / f"{s}.npz", X, ok, s)
    P(f"\nsaved {len(sets)} file(s) in {OUT_DIR.resolve()}")
    P("(correlation columns are descriptive only: they show overlap with f1, not whether a feature helps)")

    if "f1" in sets and OLD_F1.exists():
        with np.load(OLD_F1, allow_pickle=False) as zo, np.load(OUT_DIR / "f1.npz", allow_pickle=False) as zn:
            old = {k: zo[k] for k in zo.files}
            new = {k: zn[k] for k in zn.files}
        same_rows = old["paths"].shape == new["paths"].shape and bool((old["paths"] == new["paths"]).all())
        if same_rows and old["X"].shape == new["X"].shape:
            diff = float(np.abs(old["X"] - new["X"]).max())
            P(f"\ncheck vs existing {OLD_F1.name}: same rows and order; max abs difference = {diff:g}"
              f"  -> {'IDENTICAL' if diff == 0.0 else 'DIFFERENT'}")
        else:
            P(f"\ncheck vs existing {OLD_F1.name}: rows differ (old {old['paths'].shape[0]}, new "
              f"{new['paths'].shape[0]}); compare the failure list above")
        if str(old["feature_version"]) != FEATURE_VERSION:
            P(f"  note: existing file feature_version is {old['feature_version']}")

    text = "\n".join(out)
    print(text)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
