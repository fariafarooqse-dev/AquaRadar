import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_feature_sets.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_feature_sets_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = load_module()


def test_metrics_from_conf_known_values():
    # rows = true (leak, no_leak, noise), columns = predicted
    conf = np.array([[8.0, 1, 1], [2, 6, 2], [1, 1, 3]])
    m = ev.metrics_from_conf(conf)
    assert np.isclose(m["leak_recall"], 0.8)
    assert np.isclose(m["bal_acc"], (0.8 + 0.6 + 0.6) / 3)
    assert np.isclose(m["leak_precision"], 8 / 11)
    assert np.isclose(m["fpr_all"], 3 / 15)
    assert np.isclose(m["fpr_no_leak"], 2 / 10)
    assert np.isclose(m["fpr_noise"], 1 / 5)
    assert np.isclose(m["noise_recall"], 0.6)


def test_bootstrap_multiplicities_sum_to_group_count():
    M = ev.bootstrap_multiplicities(17, 50)
    assert M.shape == (50, 17) and np.all(M.sum(axis=1) == 17)


def test_unit_multiplicities_reproduce_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    g = rng.integers(0, 12, 200)
    preds = [rng.integers(0, 3, 200), rng.integers(0, 3, 200)]
    C = ev.group_confusions(preds, y, g, 12)
    assert C.sum() == 2 * 200
    dist = ev.metric_distribution(C, np.ones((1, 12)))
    point = ev.point_metrics(C)
    for k in ev.METRICS:
        assert np.isclose(dist[k][0], point[k].mean(), equal_nan=True), k


def test_decision_rule():
    good = {"d_bal_lo_adj": 0.02, "d_leak": 0.01, "d_fpr": 0.0}
    assert ev.decide_candidate({"primary": good, "sensitivity": good})[0]
    cases = [{"d_bal_lo_adj": -0.01, "d_leak": 0.01, "d_fpr": 0.0},    # gain not separated from zero
             {"d_bal_lo_adj": 0.02, "d_leak": -0.02, "d_fpr": 0.0},    # leak recall falls
             {"d_bal_lo_adj": 0.02, "d_leak": 0.01, "d_fpr": 0.03}]    # false positives rise
    for bad in cases:
        ok, why = ev.decide_candidate({"primary": good, "sensitivity": bad})
        assert not ok and why


def test_leakage_guard_raises_when_a_group_spans_train_and_test():
    X = np.random.default_rng(0).standard_normal((40, 3))
    y = np.array(["leak", "no_leak", "noise", "leak"] * 10)
    folds = np.array([0, 1] * 20)
    groups = np.arange(40) % 5          # group g sits on rows g, g+5, ...: both parities, so both folds
    try:
        ev.oof_predict(X, y, folds, groups)
    except AssertionError as exc:
        assert "group leakage" in str(exc)
    else:
        raise AssertionError("expected the leakage guard to raise")


def build_planted_dataset(base: Path) -> None:
    """f1 = pure noise features; f1_temporal = f1 plus one feature that carries the class."""
    rng = np.random.default_rng(1)
    labels, groups = [], []
    gid = 0
    for lab, n_groups, per in (("leak", 10, 12), ("no_leak", 10, 12), ("noise", 8, 10)):
        for _ in range(n_groups):
            labels += [lab] * per
            groups += [gid] * per
            gid += 1
    y = np.array(labels)
    g = np.array(groups)
    n = len(y)
    f1 = rng.standard_normal((n, 5))
    signal_col = np.array([["leak", "no_leak", "noise"].index(v) for v in y], dtype=float) + 0.2 * rng.standard_normal(n)
    sets = {"f1": f1, "f1_temporal": np.column_stack([f1, signal_col])}
    g_fine = g * 10 + (np.arange(n) % 2)
    out = base / "data" / "processed" / "feature_sets"
    out.mkdir(parents=True)
    for name, X in sets.items():
        np.savez(out / f"{name}.npz", X=X, y=y.astype(str),
                 fold_primary=ev.make_folds(y, g, 42), fold_sensitivity=ev.make_folds(y, g_fine, 42),
                 group_primary=g.astype(int), group_sensitivity=g_fine.astype(int),
                 paths=np.array([f"clip{i}.wav" for i in range(n)], dtype=str),
                 feature_names=np.array([f"f{i}" for i in range(X.shape[1])], dtype=str),
                 feature_version=np.asarray(name, dtype=str), blocks=np.asarray("x", dtype=str))


def test_end_to_end_with_a_planted_signal():
    old_cwd, old_argv = os.getcwd(), sys.argv
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        build_planted_dataset(base)
        try:
            os.chdir(base)
            sys.argv = ["evaluate_feature_sets.py", "--repeats", "2", "--boot", "200", "--sets", "f1", "f1_temporal"]
            ev.main()
        finally:
            os.chdir(old_cwd)
            sys.argv = old_argv
        csv = pd.read_csv(base / "outputs" / "feature_set_comparison.csv")
        txt = (base / "outputs" / "feature_set_comparison.txt").read_text(encoding="utf-8")
    gain = csv[(csv["set"] == "f1_temporal") & (csv["metric"] == "delta_bal_acc")]
    assert set(gain["scheme"]) == {"primary", "sensitivity"}
    assert (gain["estimate"] > 0.2).all()                       # the planted signal is large
    assert "ADOPT f1_temporal" in txt
    assert "Repeat 0 only" in txt
