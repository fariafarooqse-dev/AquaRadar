"""AquaRadar: turn a folder of WAV recordings into two SMALL csv files.

Usage (Windows PowerShell):
    python make_features.py "C:\\path\\to\\folder1" "C:\\path\\to\\folder2"

Outputs (next to this script):
    inventory.csv  one row per file (label, flow, position, condition, level...)
    features.csv   one row per 5 s window (about 35 numbers each)

Needs only numpy and scipy (already installed if scikit-learn works).
Reads the recordings only; it never changes or deletes them.
"""
import csv
import os
import re
import sys
import wave

import numpy as np
from scipy import signal

WIN_S = 5.0          # window length in seconds
TRIM_S = 1.0         # cut this much from the start and end of every file
N_BANDS = 24         # log-spaced frequency bands
F_LO, F_HI = 20.0, 20000.0


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError("unsupported sample width %d" % sw)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr, ch


def parse_name(fname):
    """Best-effort label, flow, position, condition, day from the file name."""
    s = fname.lower()
    bad = "bad" in s or "knock" in s
    if "noleak_taped" in s:
        label = "noleak_taped"
    elif "noleak_nohole" in s or "no_hole" in s or "nohole" in s:
        label = "noleak_nohole"
    elif "noleak" in s or "no_leak" in s:
        label = "noleak_other"
    elif "leak_" in s or s.startswith("leak"):
        label = "leak"
    elif "noise" in s:
        label = "noise"
    elif s.startswith("silence") or "silence" in s:
        label = "silence"
    else:
        label = "unknown"
    flow = "unknown"
    for key, val in (("high", "high"), ("medium", "med"), ("med", "med"), ("low", "low")):
        if re.search(key, s):
            flow = val
            break
    m = re.search(r"_([abc])_", s) or re.search(r"\d([abc])(?![a-z])", s)
    pos = m.group(1).upper() if m else "unknown"
    if "more_dis" in s or "far" in s:
        cond = "far"
    elif "machine" in s or "dis_" in s:
        cond = "machine"
    elif "walk" in s:
        cond = "walk"
    elif "quiet" in s or "silence" in s:
        cond = "quiet"
    else:
        cond = "unknown"
    day = "mon" if "mon" in s else ("tue" if "tue" in s else "unknown")
    return label, flow, pos, cond, day, bad


def window_features(x, sr):
    f, p = signal.welch(x, sr, nperseg=8192)
    hi = min(F_HI, 0.95 * sr / 2.0)
    edges = np.geomspace(F_LO, hi, N_BANDS + 1)
    bands = []
    for lo, up in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < up)
        bands.append(10.0 * np.log10(p[m].mean() + 1e-20) if m.any() else -200.0)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.abs(x).max())
    pm = p[(f >= 20) & (f < hi)]
    ff = f[(f >= 20) & (f < hi)]
    centroid = float((ff * pm).sum() / (pm.sum() + 1e-20))
    flat = float(np.exp(np.mean(np.log(pm + 1e-20))) / (pm.mean() + 1e-20))
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    hi_ratio = float(p[(f >= 1000) & (f < hi)].sum() / (p[(f >= 20) & (f < hi)].sum() + 1e-20))
    return bands + [20 * np.log10(rms + 1e-12), peak, peak / (rms + 1e-12), centroid, flat, zcr, hi_ratio]


def main():
    if len(sys.argv) < 2:
        print("Give the recordings folder, e.g.: python make_features.py \"C:\\recordings\"")
        sys.exit(1)
    files = []
    for root in sys.argv[1:]:
        for d, _, names in os.walk(root):
            for n in names:
                if n.lower().endswith(".wav"):
                    files.append(os.path.join(d, n))
    files.sort()
    print("found %d wav files" % len(files))
    out_dir = os.path.dirname(os.path.abspath(__file__))
    inv_rows, feat_rows = [], []
    for path in files:
        fname = os.path.basename(path)
        label, flow, pos, cond, day, bad = parse_name(fname)
        folder = os.path.basename(os.path.dirname(path))
        try:
            x, sr, ch = read_wav(path)
        except Exception as e:  # keep going, report at the end
            print("SKIP", fname, e)
            continue
        dur = len(x) / sr
        core = x[int(TRIM_S * sr): len(x) - int(TRIM_S * sr)] if dur > 3 * TRIM_S + WIN_S else x
        n = int(len(core) // (WIN_S * sr))
        rms = float(np.sqrt(np.mean(x ** 2)))
        inv_rows.append([fname, folder, label, flow, pos, cond, day, int(bad), sr, ch,
                         round(dur, 2), round(rms, 5), round(float(np.abs(x).max()), 4), n])
        for i in range(n):
            seg = core[int(i * WIN_S * sr): int((i + 1) * WIN_S * sr)]
            feat_rows.append([fname, i, label, flow, pos, cond, day, int(bad)] +
                             [round(float(v), 4) for v in window_features(seg, sr)])
    with open(os.path.join(out_dir, "inventory.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "folder", "label", "flow", "pos", "cond", "day", "bad", "sr", "ch",
                    "dur_s", "rms", "peak", "windows"])
        w.writerows(inv_rows)
    names = ["file", "win", "label", "flow", "pos", "cond", "day", "bad"] + \
            ["band%02d" % i for i in range(N_BANDS)] + \
            ["rms_db", "peak", "crest", "centroid", "flatness", "zcr", "hf_ratio"]
    with open(os.path.join(out_dir, "features.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        w.writerows(feat_rows)
    print("\nlabel counts (files):")
    counts = {}
    for r in inv_rows:
        counts[(r[2], r[5])] = counts.get((r[2], r[5]), 0) + 1
    for k in sorted(counts):
        print("  %-14s %-8s %d" % (k[0], k[1], counts[k]))
    unk = [r[0] for r in inv_rows if r[2] == "unknown"]
    print("\nfiles with unknown label:", len(unk))
    for u in unk[:10]:
        print("  ", u)
    print("\nwrote inventory.csv (%d rows) and features.csv (%d rows)" % (len(inv_rows), len(feat_rows)))


if __name__ == "__main__":
    main()
