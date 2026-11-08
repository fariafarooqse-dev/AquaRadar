"""Run the model service on audio files and print the raw AnalysisResult.

Usage: python scripts/try_model_service.py file1.wav [file2.wav ...]
NOTE: files from the training dataset were seen by the final model, so their results
check the plumbing only. They are NOT evidence of accuracy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_service import analyze_audio_bytes  # noqa: E402

for arg in sys.argv[1:]:
    p = Path(arg)
    r = analyze_audio_bytes(p.read_bytes(), p.name)
    print(f"\n{p.name}\n  status={r.status} class={r.predicted_class} score={r.confidence} model={r.model_version}")
    print(f"  message: {r.message}")
    for w in r.warnings:
        print(f"  warning: {w}")
