"""AquaRadar: train YOUR local leak / no-leak model from inventory.csv + features.csv.

Usage (PowerShell, from the folder that holds this script and the two csv files):

  1) Train on everything, check with grouped cross-validation:
       python train_local_model.py inventory.csv features.csv

  2) Train on Monday, test on Tuesday (the honest test). 'tuesday' is any part of
     the Tuesday folder name you gave to make_features.py:
       python train_local_model.py inventory.csv features.csv --test-root tuesday

Writes:  models/local_leak_model_v1.joblib
         models/local_leak_model_v1_card.json
         outputs/local_model_report.txt
Needs numpy, pandas, scikit-learn, joblib (already in your project).
"""
import argparse
import datetime
import json
import os
import re
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

POS = ["leak"]                              # hole open
NEG = ["noleak_taped", "noleak_nohole"]     # hole taped / pipe without hole


def classify(file, folder):
    """Class from the file and folder names of the Monday naming style."""
    f, fo = str(file).lower(), str(folder).lower()
    if "silence audiuos" in fo:
        return "silence"
    if "machine noise" in fo and "point" not in fo:
        return "noise"
    if "leak record" in fo and "no leak" not in f:          # 'no taped leak' = hole open
        fl = "high" if " high " in fo + " " else ("low" if " low " in fo + " " else "med")
        return "leak_high_nowater" if fl == "high" else "leak"
    if "taped" in f or "tape +" in f or "tape+" in f:
        return "noleak_taped"
    if "no hole" in f:
        # taped-group folders whose names wrongly say 'no hole'
        if re.fullmatch(r"point a (low|med) record", fo):
            return "noleak_taped"
        return "noleak_nohole"
    return "unknown"


def point_of(folder):
    m = re.search(r"po[i]?u?nt ([abc])", str(folder).lower())
    return m.group(1).upper() if m else "-"


def load(inv_path, feat_path):
    inv = pd.read_csv(inv_path)
    ft = pd.read_csv(feat_path)
    feats = [c for c in ft.columns if c.startswith("band")] + \
            ["rms_db", "peak", "crest", "centroid", "flatness", "zcr", "hf_ratio"]
    if "folder" not in ft.columns:                       # older features.csv
        inv = inv.drop_duplicates("file")
        keys = ["file"]
    else:
        keys = [k for k in ("root", "folder", "file") if k in ft.columns and k in inv.columns]
        inv = inv.drop_duplicates(keys)
    inv = inv.copy()
    inv["cls"] = [classify(a, b) for a, b in zip(inv.file, inv.folder)]
    inv["point"] = [point_of(b) for b in inv.folder]
    keep = keys + ["cls", "point", "bad"]
    d = ft.drop(columns=[c for c in ("label", "bad", "pos", "cond", "flow", "day") if c in ft.columns]) \
          .merge(inv[keep], on=keys, how="left")
    if "root" not in d.columns:
        d["root"] = "all"
    d["gid"] = d.file.astype(str) + "|" + d.get("folder", pd.Series("", index=d.index)).astype(str) \
        + "|" + d.root.astype(str)
    d = d[d.bad.fillna(0) == 0]
    return d, feats


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))


def metrics(y, pred, groups):
    ba = balanced_accuracy_score(y, pred)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fv = pd.DataFrame({"g": groups, "y": y, "p": pred}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    facc = float(((fv.p >= 0.5).astype(int) == fv.y).mean())
    return ba, facc, cm, len(fv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inventory")
    ap.add_argument("features")
    ap.add_argument("--test-root", default=None)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    d, feats = load(a.inventory, a.features)
    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    say("AquaRadar local model report  (%s)" % datetime.date.today())
    say("Files per class: " + ", ".join("%s=%d" % (k, v) for k, v in d.groupby("cls").gid.nunique().items()))
    use = d[d.cls.isin(POS + NEG)].copy()
    use["y"] = use.cls.isin(POS).astype(int)
    if use.y.nunique() < 2:
        say("Need both leak and no-leak files. Check the class counts above.")
        open(os.path.join(a.out_dir, "local_model_report.txt"), "w").write("\n".join(lines))
        sys.exit(1)

    card = {"trained": str(datetime.date.today()), "classes": {"0": "noleak", "1": "leak"},
            "features": feats, "window_s": 5, "trim_s": 1, "model": "StandardScaler + LogisticRegression"}

    if a.test_root:
        t = a.test_root.lower()
        is_test = use.root.str.lower().str.contains(t)
        tr, te = use[~is_test], use[is_test]
        say("\nTRAIN on %d windows from %d files; TEST on %d windows from %d files (root contains '%s')."
            % (len(tr), tr.gid.nunique(), len(te), te.gid.nunique(), a.test_root))
        if te.y.nunique() < 2:
            say("Test set needs both leak and no-leak files; check --test-root.")
        m = make_model().fit(tr[feats].values, tr.y.values)
        pred = m.predict(te[feats].values)
        ba, facc, cm, nf = metrics(te.y.values, pred, te.gid.values)
        say("TEST balanced accuracy (windows): %.3f | file-level accuracy: %.3f (%d files)" % (ba, facc, nf))
        say("Confusion matrix [rows true noleak,leak; cols predicted]: %s" % cm.tolist())
        card["test"] = {"root": a.test_root, "bal_acc": round(ba, 3), "file_acc": round(facc, 3), "files": nf}
        final = make_model().fit(use[feats].values, use.y.values)       # final model uses all days
    else:
        say("\nGrouped 5-fold CV (a recording is never split between train and test):")
        X, y, g = use[feats].values, use.y.values, use.gid.values
        pred = np.zeros(len(use), int)
        for tr_i, te_i in GroupKFold(min(5, len(set(g)))).split(X, y, g):
            pred[te_i] = make_model().fit(X[tr_i], y[tr_i]).predict(X[te_i])
        ba, facc, cm, nf = metrics(y, pred, g)
        say("balanced accuracy (windows): %.3f | file-level accuracy: %.3f (%d files)" % (ba, facc, nf))
        say("Confusion matrix [rows true noleak,leak; cols predicted]: %s" % cm.tolist())
        card["cv"] = {"bal_acc": round(ba, 3), "file_acc": round(facc, 3), "files": nf}
        say("\nLeave one piezo point out:")
        for p in "ABC":
            tr, te = use[use.point != p], use[use.point == p]
            if len(te) and te.y.nunique() == 2 and len(tr):
                pr = make_model().fit(tr[feats].values, tr.y.values).predict(te[feats].values)
                say("  hold out %s: balanced accuracy %.3f (%d windows)" % (p, balanced_accuracy_score(te.y, pr), len(te)))
        final = make_model().fit(X, y)

    hn = d[d.cls == "leak_high_nowater"]
    if len(hn):
        r = final.predict(hn[feats].values).mean()
        say("\nHole open + high flow ('no water') windows predicted LEAK: %.0f%% (%d windows)." % (100 * r, len(hn)))
    say("\nNOT proven: other pipes/buildings, leak size, long-term stability. Pilot data from one rig.")

    card["train_files"] = {k: int(v) for k, v in use.groupby("cls").gid.nunique().items()}
    os.makedirs(os.path.join(a.out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "outputs"), exist_ok=True)
    joblib.dump({"pipeline": final, "features": feats, "card": card},
                os.path.join(a.out_dir, "models", "local_leak_model_v1.joblib"))
    with open(os.path.join(a.out_dir, "models", "local_leak_model_v1_card.json"), "w") as fh:
        json.dump(card, fh, indent=2)
    with open(os.path.join(a.out_dir, "outputs", "local_model_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nSaved models/local_leak_model_v1.joblib, models/local_leak_model_v1_card.json, outputs/local_model_report.txt")


if __name__ == "__main__":
    main()
