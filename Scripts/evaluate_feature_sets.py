"""Compare candidate feature sets with grouped cross-validation and group-bootstrap intervals.

Usage (from the project root, after scripts/extract_feature_sets.py):
    python scripts/evaluate_feature_sets.py                       # 5 repeats, 1000 bootstrap draws
    python scripts/evaluate_feature_sets.py --repeats 10 --boot 2000

Reads : data/processed/feature_sets/<set>.npz
Writes: outputs/feature_set_comparison.txt, outputs/feature_set_comparison.csv   (no model is saved)

Method (fixed BEFORE any result was seen):
  * Same classifier and settings as the baseline (`train_model.make_models()["audio_rf"]`).
  * Two fold schemes, as in the baseline: "primary" (coarse groups) and "sensitivity" (fine groups).
    Repeat 0 uses the fold assignment stored in the manifest, so it reproduces the baseline.
    Repeats 1.. draw new StratifiedGroupKFold splits (5 folds) on the same groups. Every feature
    set sees identical folds, so differences are paired. Each fold is asserted to share no group
    between train and test.
  * Metrics come from pooled out-of-fold predictions, averaged over repeats: 3-class balanced
    accuracy, leak recall, leak precision, false-positive rate for "leak" (against no_leak and noise
    combined, and separately), noise recall.
  * Uncertainty: a bootstrap that resamples whole COARSE groups (the same for every set, so
    differences are paired). With about 49 groups, this is the honest sample size, not 964 clips.
  * Candidates for adoption: f1_temporal, f1_mfcc, f1_mfcc_delta. A candidate replaces f1 only if,
    in BOTH schemes: (a) the lower end of the Bonferroni-adjusted interval (alpha 0.05 / number of
    candidates) for the gain in balanced accuracy is above 0; (b) leak recall does not fall (gain >= 0);
    (c) the false-positive rate for leak does not rise by more than 0.01. Condition (c) was added in
    this step, before results. Among sets that pass, the one with the fewest features wins.
    If none passes, f1 is kept. The ablation sets are diagnostic only.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import feature_sets as fs  # noqa: E402
from train_model import CLASSES, SEED, make_models  # noqa: E402

SET_DIR = Path("data/processed/feature_sets")
OUT_TXT = Path("outputs/feature_set_comparison.txt")
OUT_CSV = Path("outputs/feature_set_comparison.csv")
CANDIDATES = ["f1_temporal", "f1_mfcc", "f1_mfcc_delta"]
SCHEMES = {"primary": ("fold_primary", "group_primary"),
           "sensitivity": ("fold_sensitivity", "group_sensitivity")}
METRICS = ["bal_acc", "leak_recall", "leak_precision", "fpr_all", "fpr_no_leak", "fpr_noise", "noise_recall"]
ALPHA = 0.05
MAX_FPR_RISE = 0.01
_L, _N, _Z = CLASSES.index("leak"), CLASSES.index("no_leak"), CLASSES.index("noise")


# ----------------------------------------------------------------------------- pure helpers
def metrics_from_conf(conf: np.ndarray) -> dict[str, np.ndarray]:
    """conf[..., true, pred] confusion counts (class order CLASSES) -> metric arrays over the leading dims."""
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        support = conf.sum(axis=-1)
        recall = np.diagonal(conf, axis1=-2, axis2=-1) / support
        pred_leak = conf[..., :, _L].sum(axis=-1)
        fp_no_leak, fp_noise = conf[..., _N, _L], conf[..., _Z, _L]
        return {
            "bal_acc": np.nanmean(recall, axis=-1),
            "leak_recall": recall[..., _L],
            "leak_precision": conf[..., _L, _L] / pred_leak,
            "fpr_all": (fp_no_leak + fp_noise) / (support[..., _N] + support[..., _Z]),
            "fpr_no_leak": fp_no_leak / support[..., _N],
            "fpr_noise": fp_noise / support[..., _Z],
            "noise_recall": recall[..., _Z],
        }


def group_confusions(preds: list[np.ndarray], y_idx: np.ndarray, g_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """Per repeat and group confusion counts: shape (n_repeats, n_groups, 3, 3)."""
    C = np.zeros((len(preds), n_groups, 3, 3))
    for r, p in enumerate(preds):
        np.add.at(C[r], (g_idx, y_idx, p), 1)
    return C


def bootstrap_multiplicities(n_groups: int, n_boot: int, seed: int = 12345) -> np.ndarray:
    """(n_boot, n_groups) counts of how often each group is drawn; every row sums to n_groups."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_boot).astype(float)


def metric_distribution(C: np.ndarray, M: np.ndarray) -> dict[str, np.ndarray]:
    """Each metric (averaged over repeats) for every bootstrap draw: arrays of shape (n_boot,)."""
    conf = np.einsum("bg,rgij->brij", M, C)
    return {k: v.mean(axis=1) for k, v in metrics_from_conf(conf).items()}


def point_metrics(C: np.ndarray) -> dict[str, np.ndarray]:
    """Metric per repeat on the real data: arrays of shape (n_repeats,)."""
    return metrics_from_conf(C.sum(axis=1))


def interval(x: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.nanpercentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def decide_candidate(per_scheme: dict[str, dict[str, float]]) -> tuple[bool, list[str]]:
    """per_scheme[scheme] has d_bal_lo_adj, d_leak, d_fpr. Returns (passes, reasons for failure)."""
    why: list[str] = []
    for scheme, d in per_scheme.items():
        if not d["d_bal_lo_adj"] > 0:
            why.append(f"{scheme}: adjusted lower bound of balanced-accuracy gain is {d['d_bal_lo_adj']:+.3f} (needs > 0)")
        if not d["d_leak"] >= 0:
            why.append(f"{scheme}: leak recall changes by {d['d_leak']:+.3f} (must not fall)")
        if not d["d_fpr"] <= MAX_FPR_RISE:
            why.append(f"{scheme}: leak false-positive rate rises by {d['d_fpr']:+.3f} (limit +{MAX_FPR_RISE})")
    return (not why), why


# ----------------------------------------------------------------------------- data and CV
def load_sets(directory: Path, wanted: list[str] | None) -> dict[str, dict]:
    names = [s for s in fs.available_sets() if (directory / f"{s}.npz").exists()]
    if wanted:
        names = [s for s in names if s in wanted]
    if "f1" not in names:
        raise SystemExit(f"f1.npz not found in {directory}. Run scripts/extract_feature_sets.py first.")
    out: dict[str, dict] = {}
    for s in names:
        with np.load(directory / f"{s}.npz", allow_pickle=False) as z:
            out[s] = {k: z[k] for k in z.files}
    ref = out["f1"]
    for s, d in out.items():
        for key in ("paths", "y", "group_primary", "group_sensitivity", "fold_primary", "fold_sensitivity"):
            if not np.array_equal(d[key], ref[key]):
                raise SystemExit(f"Set {s} has different {key} from f1; re-run extract_feature_sets.py.")
    return out


def make_folds(y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int = 5) -> np.ndarray:
    folds = np.full(len(y), -1, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        folds[te] = k
    return folds


def oof_predict(X: np.ndarray, y: np.ndarray, folds: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold class indices (into CLASSES) from the baseline random forest."""
    pred = np.full(len(y), -1, dtype=int)
    for k in sorted(np.unique(folds)):
        te = folds == k
        tr = ~te
        shared = set(groups[tr].tolist()) & set(groups[te].tolist())
        if shared:
            raise AssertionError(f"group leakage: {len(shared)} group(s) in both train and test (fold {k})")
        m = make_models()["audio_rf"][1]()
        m.fit(X[tr], y[tr])
        if list(m.classes_) != CLASSES:
            raise AssertionError(f"unexpected class order {list(m.classes_)}")
        pred[te] = [CLASSES.index(c) for c in m.predict(X[te])]
    return pred


# ----------------------------------------------------------------------------- reporting
def fmt(est: float, lo: float | None = None, hi: float | None = None) -> str:
    return f"{est:.3f}" if lo is None else f"{est:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5, help="fold repeats per scheme (repeat 0 = manifest folds)")
    ap.add_argument("--boot", type=int, default=1000, help="group-bootstrap draws")
    ap.add_argument("--sets", nargs="+", default=None, choices=fs.available_sets())
    args = ap.parse_args()
    if args.repeats < 1 or args.boot < 100:
        raise SystemExit("--repeats must be >= 1 and --boot >= 100")

    data = load_sets(SET_DIR, args.sets)
    names = [s for s in fs.available_sets() if s in data]
    cands = [s for s in CANDIDATES if s in data]
    ref = data["f1"]
    y = ref["y"]
    y_idx = np.array([CLASSES.index(v) for v in y])
    _, g_boot = np.unique(ref["group_primary"], return_inverse=True)   # bootstrap unit: coarse groups
    n_groups = int(g_boot.max()) + 1
    M = bootstrap_multiplicities(n_groups, args.boot)
    alpha_adj = ALPHA / max(1, len(cands))

    out: list[str] = []
    P = out.append
    P("FEATURE-SET COMPARISON (grouped cross-validation; group-bootstrap intervals)")
    P(f"rows: {len(y)}   coarse groups: {n_groups}   fold repeats: {args.repeats} (repeat 0 = manifest folds)   "
      f"bootstrap draws: {args.boot}")
    P(f"sets: {names}")
    P("classifier: audio_rf settings from train_model.make_models (300 trees, class_weight=balanced, seed 42)")
    P(f"intervals shown: {100 * (1 - ALPHA):.0f}% group-bootstrap. Decision uses {100 * (1 - alpha_adj):.1f}% "
      f"(Bonferroni over {len(cands)} candidate(s)).")

    rows: list[dict] = []
    verdicts: dict[str, dict[str, dict[str, float]]] = {c: {} for c in cands}
    repeat0: dict[str, dict[str, float]] = {s: {} for s in names}

    for scheme, (fold_col, group_col) in SCHEMES.items():
        fold_sets = [ref[fold_col]] + [make_folds(y, ref[group_col], SEED + r) for r in range(1, args.repeats)]
        C: dict[str, np.ndarray] = {}
        for s in names:
            t0 = time.time()
            X = data[s]["X"]
            preds = [oof_predict(X, y, fold_sets[r], ref[group_col]) for r in range(args.repeats)]
            C[s] = group_confusions(preds, y_idx, g_boot, n_groups)
            repeat0[s][scheme] = float(point_metrics(C[s][:1])["bal_acc"][0])
            print(f"[{scheme}] {s}: {args.repeats} repeat(s) done in {time.time() - t0:.0f}s", flush=True)

        dist = {s: metric_distribution(C[s], M) for s in names}
        point = {s: point_metrics(C[s]) for s in names}

        P(f"\n== {scheme.upper()} scheme: absolute metrics (mean over repeats; [95% CI]) ==")
        tab = []
        for s in names:
            pm = {k: float(v.mean()) for k, v in point[s].items()}
            ci = {k: interval(dist[s][k], ALPHA) for k in ("bal_acc", "leak_recall", "fpr_all")}
            tab.append({"set": s, "n_feat": data[s]["X"].shape[1],
                        "bal_acc": fmt(pm["bal_acc"], *ci["bal_acc"]),
                        "leak_recall": fmt(pm["leak_recall"], *ci["leak_recall"]),
                        "leak_prec": fmt(pm["leak_precision"]),
                        "fpr_all": fmt(pm["fpr_all"], *ci["fpr_all"]),
                        "fpr_no_leak": fmt(pm["fpr_no_leak"]), "fpr_noise": fmt(pm["fpr_noise"]),
                        "noise_recall": fmt(pm["noise_recall"]),
                        "bal_acc_range_over_repeats": f"{point[s]['bal_acc'].min():.3f}-{point[s]['bal_acc'].max():.3f}"})
            for k in METRICS:
                lo, hi = interval(dist[s][k], ALPHA)
                rows.append({"scheme": scheme, "set": s, "metric": k, "estimate": pm[k], "ci_lo": lo, "ci_hi": hi})
        P(pd.DataFrame(tab).to_string(index=False))

        P(f"\n== {scheme.upper()} scheme: change from f1 (paired; [95% CI]; last column = lower end of the "
          f"{100 * (1 - alpha_adj):.1f}% interval used for the decision) ==")
        dtab = []
        for s in names:
            if s == "f1":
                continue
            d = {k: dist[s][k] - dist["f1"][k] for k in METRICS}
            dp = {k: float(point[s][k].mean() - point["f1"][k].mean()) for k in METRICS}
            lo_adj = interval(d["bal_acc"], alpha_adj)[0]
            dtab.append({"set": s,
                         "d_bal_acc": fmt(dp["bal_acc"], *interval(d["bal_acc"], ALPHA)),
                         "d_leak_recall": fmt(dp["leak_recall"], *interval(d["leak_recall"], ALPHA)),
                         "d_fpr_all": fmt(dp["fpr_all"], *interval(d["fpr_all"], ALPHA)),
                         "d_noise_recall": fmt(dp["noise_recall"], *interval(d["noise_recall"], ALPHA)),
                         "d_bal_acc_lo_adj": f"{lo_adj:+.3f}"})
            for k in METRICS:
                lo, hi = interval(d[k], ALPHA)
                lo_a, hi_a = interval(d[k], alpha_adj)
                rows.append({"scheme": scheme, "set": s, "metric": f"delta_{k}", "estimate": dp[k],
                             "ci_lo": lo, "ci_hi": hi, "ci_lo_adj": lo_a, "ci_hi_adj": hi_a})
            if s in cands:
                verdicts[s][scheme] = {"d_bal_lo_adj": lo_adj, "d_leak": dp["leak_recall"], "d_fpr": dp["fpr_all"]}
        P(pd.DataFrame(dtab).to_string(index=False))

    P("\n== Repeat 0 only (manifest folds): pooled balanced accuracy, for checking against the baseline report ==")
    P(pd.DataFrame(repeat0).T.round(3).to_string())
    P("(f1 / primary here should equal audio_rf pooled_bal_acc in outputs/baseline_cv_report.txt)")

    P("\n== DECISION (rule fixed before results; see the top of scripts/evaluate_feature_sets.py) ==")
    passing = []
    for c in cands:
        ok, why = decide_candidate(verdicts[c])
        P(f"{c}: {'PASSES' if ok else 'does not pass'}")
        for w in why:
            P(f"    - {w}")
        if ok:
            passing.append(c)
    if passing:
        best = min(passing, key=lambda s: data[s]["X"].shape[1])
        P(f"\nADOPT {best} (fewest features among: {passing}).")
    else:
        P("\nKEEP f1. No candidate met the pre-declared rule. This is a valid result.")

    P("\nLIMITS: about 49 coarse groups from one dataset; groups are inferred from filenames; a pass means "
      "'separated from f1 on this dataset', not field accuracy. Several sets were compared on the same data, "
      "hence the Bonferroni adjustment. Nothing here measures performance on new sites or phone recordings.")

    text = "\n".join(out)
    print("\n" + text)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(text, encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n[saved {OUT_TXT.resolve()} and {OUT_CSV.resolve()}]")


if __name__ == "__main__":
    main()
