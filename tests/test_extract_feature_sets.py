"""Runs scripts/extract_feature_sets.py on a tiny synthetic dataset in a temporary folder."""
import importlib.util
import os
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import pandas as pd

import audio_features as af
import feature_sets as fs

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extract_feature_sets.py"


def write_wav(path: Path, x: np.ndarray, sr: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def build_dataset(base: Path):
    """6 clips: 5 usable, 1 silent (must be dropped for every set). Returns the manifest rows."""
    rng = np.random.default_rng(0)
    rows = []
    labels = ["leak", "leak", "no_leak", "no_leak", "noise", "leak"]
    for i, lab in enumerate(labels):
        x = 0.1 * rng.standard_normal(8000) if i != 5 else np.zeros(8000)
        rel = f"{lab}/clip{i}.wav"
        write_wav(base / "raw" / rel, x)
        rows.append({"path": rel, "label": lab, "is_eval_row": True, "fold_primary": i % 2,
                     "fold_sensitivity": i % 3, "group_primary": 100 + i // 2, "group_sensitivity": 200 + i})
    # a duplicate row that is not an eval row must be ignored
    rows.append({"path": "leak/clip0.wav", "label": "leak", "is_eval_row": False, "fold_primary": 0,
                 "fold_sensitivity": 0, "group_primary": 100, "group_sensitivity": 200})
    proc = base / "data" / "processed"
    proc.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(proc / "manifest_v1.csv", index=False)


def run_script(base: Path, *extra: str) -> None:
    spec = importlib.util.spec_from_file_location("extract_feature_sets_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old_cwd, old_argv = os.getcwd(), sys.argv
    try:
        os.chdir(base)
        sys.argv = ["extract_feature_sets.py", str(base / "raw"), *extra]
        mod.main()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


def test_all_sets_have_identical_rows_and_f1_columns_match_the_existing_extractor():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        build_dataset(base)
        run_script(base)
        out = base / "data" / "processed" / "feature_sets"
        files = {}
        for p in out.glob("*.npz"):                       # close each file (Windows cannot delete open files)
            with np.load(p, allow_pickle=False) as z:
                files[p.stem] = {k: z[k] for k in z.files}
        assert set(files) == set(fs.available_sets())
        ref = files["f1"]
        assert len(ref["paths"]) == 5                      # silent clip and duplicate row dropped
        assert "leak/clip5.wav" not in set(ref["paths"])
        for name, z in files.items():
            assert (z["paths"] == ref["paths"]).all(), name
            assert (z["y"] == ref["y"]).all() and (z["group_primary"] == ref["group_primary"]).all()
            assert z["X"].shape == (5, len(fs.feature_set_names(name))), name
            assert list(z["feature_names"]) == fs.feature_set_names(name)
            assert str(z["feature_version"]) == name
            assert np.all(np.isfinite(z["X"]))
        audio, sr = af.load_audio(base / "raw" / str(ref["paths"][0]))
        assert np.array_equal(ref["X"][0], af.extract_features(audio, sr))
        assert np.array_equal(files["f1_mfcc"]["X"][:, :39], ref["X"])
        assert (base / "outputs" / "feature_sets_summary.txt").exists()


def test_sets_option_limits_output():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        build_dataset(base)
        run_script(base, "--sets", "f1", "f1_temporal")
        names = sorted(p.stem for p in (base / "data" / "processed" / "feature_sets").glob("*.npz"))
        assert names == ["f1", "f1_temporal"]
