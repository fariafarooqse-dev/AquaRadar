"""Assign leakage-safe cross-validation folds from manifest_draft.csv (no audio read, no training).

Usage:
    python scripts/make_grouped_folds.py

Reads : data/processed/manifest_draft.csv
Writes: data/processed/manifest_v1.csv, outputs/folds_summary.txt

Design (all choices are printed so they can be challenged):
  * Byte-identical duplicates: only the first file of each md5 is kept for training/evaluation
    (column is_eval_row). Extras stay in the CSV and inherit their group's fold.
  * Groups are built with union-find over (grouping key) + (identical md5).
      group_primary     : COARSE key  label|material|zone|device   (noise: label|category)
      group_sensitivity : FINE   key  filename minus segment/copy (pressure/flow included)
    The coarse key is a superset of the fine key, so primary folds also respect fine groups.
  * Folds: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on label.
  * pressure/flow/segment are NOT written to manifest_v1.csv: pressure/flow are NA for every
    no_leak file in this dataset, so they would leak the label if used as features.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DRAFT = Path("data/processed/manifest_draft.csv")
OUT_CSV = Path("data/processed/manifest_v1.csv")
OUT_TXT = Path("outputs/folds_summary.txt")
N_SPLITS = 5
SEED = 42


def union_groups(df: pd.DataFrame, key_col: str) -> pd.Series:
    parent = list(range(len(df)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for col in (key_col, "md5"):
        for _, idx in df.groupby(col).indices.items():
            idx = list(idx)
            for j in idx[1:]:
                parent[find(j)] = find(idx[0])
    return pd.Series([find(i) for i in range(len(df))], index=df.index)


def coarse_key(r: pd.Series) -> str:
    if r["label"] == "noise":
        return f"noise|{r['noise_category']}"
    return f"{r['label']}|{r['material']}|{r['zone']}|{r['device']}"


def folds_for(df: pd.DataFrame, group_col: str) -> pd.Series:
    ev = df[df["is_eval_row"]]
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    g2f: dict = {}
    for k, (_, te) in enumerate(sgkf.split(ev, ev["label"], ev[group_col])):
        for g in ev.iloc[te][group_col].unique():
            g2f[g] = k
    return df[group_col].map(g2f)


def audit(df: pd.DataFrame, fold_col: str, name: str, out: list[str]) -> None:
    P = out.append
    P(f"\n--- AUDIT for {name} ({fold_col}) ---")
    P(f"rows without fold: {int(df[fold_col].isna().sum())}")
    for col in ("group_primary", "group_sensitivity", "md5"):
        n = int((df.groupby(col)[fold_col].nunique() > 1).sum())
        P(f"  {col:18s} values appearing in >1 fold: {n}   {'PASS' if n == 0 else 'FAIL'}")


def main() -> None:
    if not DRAFT.exists():
        raise SystemExit(f"Missing {DRAFT}. Run build_manifest_draft.py first.")
    df = pd.read_csv(DRAFT)
    need = {"path", "label", "md5", "recording_key", "material", "zone", "device",
            "noise_category", "sr", "dur_s", "parse_ok"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"manifest_draft.csv missing columns: {sorted(miss)}")
    if not df["parse_ok"].all():
        raise SystemExit("Some filenames failed to parse; fix before creating folds.")
    for c in ("material", "zone", "device", "noise_category"):
        df[c] = df[c].fillna("NA")
    df = df.sort_values("path").reset_index(drop=True)

    df["is_eval_row"] = ~df.duplicated("md5", keep="first")
    df["coarse_key"] = df.apply(coarse_key, axis=1)
    df["group_primary"] = union_groups(df, "coarse_key")
    df["group_sensitivity"] = union_groups(df, "recording_key")
    df["fold_primary"] = folds_for(df, "group_primary")
    df["fold_sensitivity"] = folds_for(df, "group_sensitivity")

    out: list[str] = []
    P = out.append
    ev = df[df["is_eval_row"]]
    P(f"rows: {len(df)}   eval rows (duplicates removed): {len(ev)}   extras dropped: {len(df) - len(ev)}")
    P("eval rows per label:\n" + ev["label"].value_counts().to_string())
    for col, name in (("group_primary", "PRIMARY (coarse)"), ("group_sensitivity", "SENSITIVITY (fine)")):
        gs = ev.groupby(col).agg(n=("path", "size"), label=("label", "first"))
        P(f"\n== groups, {name} ==")
        P(gs.groupby("label")["n"].agg(["count", "min", "median", "max"]).to_string())
        P("conflicting-label groups: " + str(int((ev.groupby(col)["label"].nunique() > 1).sum())))

    for fcol, name in (("fold_primary", "PRIMARY"), ("fold_sensitivity", "SENSITIVITY")):
        P(f"\n== eval rows per fold x label, {name} ==")
        ct = pd.crosstab(ev[fcol], ev["label"])
        P(ct.to_string())
        empty = [(f, l) for f in ct.index for l in ct.columns if ct.loc[f, l] == 0]
        P("WARNING: folds missing a class: " + str(empty) if empty else "every fold contains all 3 classes")

    P("\n== noise categories per PRIMARY fold ==")
    nz = ev[ev["label"] == "noise"]
    for f, sub in nz.groupby("fold_primary"):
        P(f"  fold {int(f)}: " + ", ".join(f"{k}({v})" for k, v in sub["noise_category"].value_counts().items()))
    P("\n== material x fold (PRIMARY, leak/no_leak eval rows) ==")
    pipe = ev[ev["label"] != "noise"]
    P(pd.crosstab([pipe["fold_primary"]], [pipe["label"], pipe["material"]]).to_string())

    audit(df, "fold_primary", "PRIMARY", out)
    audit(df, "fold_sensitivity", "SENSITIVITY (group_primary overlap is EXPECTED to FAIL here)", out)

    keep = ["path", "label", "is_eval_row", "fold_primary", "fold_sensitivity",
            "group_primary", "group_sensitivity", "material", "zone", "device",
            "noise_category", "md5", "sr", "dur_s"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(OUT_CSV, index=False)
    text = "\n".join(out)
    print(text)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(text, encoding="utf-8")
    print(f"\n[saved {OUT_CSV.resolve()} and {OUT_TXT.resolve()}]")


if __name__ == "__main__":
    main()
