"""AquaRadar: train YOUR local leak / no-leak model from inventory.csv + features.csv.

Usage (PowerShell, from the folder that holds this script and the two csv files):

  1) Train on everything, check with grouped cross-validation:
       python train_local_model.py inventory.csv features.csv

  2) Same, but without the 6 doubtful files:
       python train_local_model.py inventory.csv features.csv --test-root tuesday --drop-suspect

  3) Train on Monday, test on Tuesday (the honest test). 'tuesday' is any part of
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


# Folders of the Tuesday test set (used only if inventory.csv has no 'root' column).
TUESDAY_FOLDERS = {x.lower() for x in [
    "point A high", "point a low", "point a med", "point B HIGH NOLEAK", "point B LOW NOLEAK TAPED",
    "point B med noleak", "POINT C HIGH LEAK", "point C MED NOLEAK", "point C low noleak",
    "point c high leak", "point c low leak", "point c medium leak", "point A  highl Leak recordings",
    "point A med leak recordings", "point a low leak recordings", "Point b high leak",
    "Point b medium leal", "point B LOW LEAK RECORDINGS"]}


# 6 files named '... leak ... Mon wav test' that sit in the Tuesday folders. Their sound has no
# high-frequency hiss at all (hf_ratio ~0), so they may not have been leaking when recorded.
# Dropped only when you pass --drop-suspect, and the report says so.
SUSPECT_FILES = [
    "1. 1 C low silence leak 1C Mon wav testing .wav",
    "4.1 C low  walking all that leak 4 C Mon wav test (1).wav",
    "4.1 C low  walking all that leak 4 C Mon wav test.wav",
    "1.1 C med silence leak 1C Mon wav test  (1).wav",
    "2.1C med machine noise leak 2C Mon wav test  (1).wav",
    "4.1 C med walking all that leak 4C Mon wav test   (1).wav",
]


def flow_of_folder(fo):
    return "high" if "high" in fo else ("low" if "low" in fo else "med")


def classify(file, folder):
    """Class from the file and folder names (Monday and Tuesday naming styles).
    leak_high_nowater = hole open + high flow (no water comes out), kept out of the score.
    leak_unsure       = file name says 'no taped leak' but the folder is a taped / no-leak folder."""
    f, fo = str(file).lower(), str(folder).lower()
    if "silence audiuos" in fo:
        return "silence"
    if "machine noise" in fo and "point" not in fo:
        return "noise"
    if "leak record" in fo and "no leak" not in f:          # 'no taped leak' = hole open
        return "leak_high_nowater" if flow_of_folder(fo) == "high" else "leak"
    named_leak = bool(re.search(r"no tape[d]?[\s,]+leak", f)) or \
        (re.search(r"\bleak\b", f) and not re.search(r"no\s+leak|taped|hole", f))
    if named_leak:
        if not re.search(r"(?<!no)leak", fo):               # folder is not a leak folder
            return "leak_unsure"
        return "leak_high_nowater" if flow_of_folder(fo) == "high" else "leak"
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
    if "root" not in inv.columns:                         # older inventory: split by folder name
        inv["root"] = ["tuesday" if str(b).lower() in TUESDAY_FOLDERS else "monday" for b in inv.folder]
        if "root" in ft.columns:
            ft = ft.drop(columns="root")
        print("(no 'root' column in inventory.csv: Tuesday folders recognised by name)")
    elif "root" not in ft.columns:
        ft["root"] = ft.get("folder", pd.Series("all", index=ft.index))
    if "folder" not in ft.columns:                       # older features.csv
        inv = inv.drop_duplicates("file")
        keys = ["file"]
    else:
        keys = [k for k in ("folder", "file") if k in ft.columns and k in inv.columns]
        inv = inv.drop_duplicates(keys)
    inv = inv.copy()
    inv["cls"] = [classify(a, b) for a, b in zip(inv.file, inv.folder)]
    inv["point"] = [point_of(b) for b in inv.folder]
    keep = keys + [c for c in ("cls", "point", "bad", "root") if c not in keys]
    d = ft.drop(columns=[c for c in ("label", "bad", "pos", "cond", "flow", "day", "root") if c in ft.columns]) \
          .merge(inv[keep], on=keys, how="left")
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
    ap.add_argument("--drop-suspect", action="store_true",
                    help="leave out the 6 doubtful 'Mon wav test' leak files (see SUSPECT_FILES)")
    a = ap.parse_args()

    d, feats = load(a.inventory, a.features)
    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    say("AquaRadar local model report  (%s)" % datetime.date.today())
    if a.drop_suspect:
        n_s = d[d.file.isin(SUSPECT_FILES)].gid.nunique()
        d = d[~d.file.isin(SUSPECT_FILES)]
        say("--drop-suspect: left out %d doubtful leak files (listed in SUSPECT_FILES)." % n_s)
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
        n_before = te.gid.nunique()
        te = te.drop_duplicates(subset=feats + ["win"]) if "win" in te.columns else te
        te = te[te.groupby("gid").gid.transform("size") > 0]
        say("\nTRAIN on %d windows from %d files; TEST on %d windows from %d files (root contains '%s')."
            % (len(tr), tr.gid.nunique(), len(te), te.gid.nunique(), a.test_root))
        if n_before != te.gid.nunique():
            say("(%d test files were exact copies of another test file and were dropped.)" % (n_before - te.gid.nunique()))
        say("Test files per class: " + ", ".join("%s=%d" % (k, v) for k, v in te.groupby("cls").gid.nunique().items()))
        if te.y.nunique() < 2:
            say("Test set needs both leak and no-leak files; check --test-root.")
            sys.exit(1)
        m = make_model().fit(tr[feats].values, tr.y.values)
        pred = m.predict(te[feats].values)
        ba, facc, cm, nf = metrics(te.y.values, pred, te.gid.values)
        say("TEST balanced accuracy (windows): %.3f | file-level accuracy: %.3f (%d files)" % (ba, facc, nf))
        say("Confusion matrix [rows true noleak,leak; cols predicted]: %s" % cm.tolist())
        fv = pd.DataFrame({"gid": te.gid.values, "cls": te.cls.values, "y": te.y.values, "p": pred}) \
            .groupby("gid").agg(cls=("cls", "first"), y=("y", "first"), p=("p", "mean"))
        wrong = fv[(fv.p >= 0.5).astype(int) != fv.y]
        say("Wrongly classified test files: %d" % len(wrong))
        for g, r in wrong.iterrows():
            say("   %-14s leak-prob %.2f  %s" % (r.cls, r.p, g.split("|")[0][:70]))
        for c in ("leak_high_nowater", "leak_unsure"):
            x = d[(d.cls == c) & d.root.str.lower().str.contains(t)]
            if len(x):
                say("Not in the score, model trained on non-test files says LEAK for %.0f%% of the %d windows (%d files): %s"
                    % (100 * m.predict(x[feats].values).mean(), len(x), x.gid.nunique(), c))
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
