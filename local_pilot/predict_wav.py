"""AquaRadar: check ONE recording with your trained local model.

Usage (PowerShell, same folder as make_features.py and models/):
    python predict_wav.py "C:\\path\\to\\recording.wav"
"""
import os
import sys

import joblib
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from make_features import read_wav, window_features, WIN_S, TRIM_S  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print('Give a wav file, e.g.: python predict_wav.py "C:\\recordings\\test.wav"')
        sys.exit(1)
    bundle = joblib.load(os.path.join(here, "models", "local_leak_model_v1.joblib"))
    x, sr, _ = read_wav(sys.argv[1])
    dur = len(x) / sr
    core = x[int(TRIM_S * sr): len(x) - int(TRIM_S * sr)] if dur > 3 * TRIM_S + WIN_S else x
    n = int(len(core) // (WIN_S * sr))
    if n == 0:
        print("Recording too short (need at least %.0f s)." % WIN_S)
        sys.exit(1)
    rows = [window_features(core[int(i * WIN_S * sr): int((i + 1) * WIN_S * sr)], sr) for i in range(n)]
    p = bundle["pipeline"].predict_proba(np.array(rows))[:, 1]
    print("windows: %d | leak probability per window: %s" % (n, ", ".join("%.2f" % v for v in p)))
    verdict = "LEAK-LIKE (open-hole sound)" if p.mean() >= 0.5 else "NO LEAK"
    print("average leak probability %.2f  ->  %s" % (p.mean(), verdict))
    print("Note: pilot model, trained on one pipe rig only.")


if __name__ == "__main__":
    main()
