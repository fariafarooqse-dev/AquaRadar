"""Stable boundary between the Streamlit app and the trained model.

Member 1 owns this module. UI code should call `analyze_audio_bytes` instead of loading
model files or reproducing model logic inside app.py.

Behaviour (see docs/model_ui_contract.md):
  * Never fabricates a prediction. Any failure returns status="error" with predicted_class=None.
  * Recordings are decoded, resampled to 8 kHz mono and cut into consecutive 1-second windows,
    the same unit the model was trained on. Window scores are averaged. This averaging step was
    NOT evaluated separately (the cross-validation used single 1-second clips).
  * status="uncertain" (predicted_class=None) when the top averaged score is below
    UNCERTAIN_THRESHOLD. The threshold was read off out-of-fold results, so it is optimistic.
  * `confidence` is the top averaged random-forest vote share. It is NOT a calibrated probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import audio_features as af

MODEL_PATH = Path(__file__).resolve().parent / "models" / "leak_model_v1.joblib"
MODEL_VERSION = "leak_model_v1"
UNCERTAIN_THRESHOLD = 0.60
MAX_BYTES = 25 * 1024 * 1024
MAX_WINDOWS = 60                 # at most 60 s of audio is analysed

STANDING_WARNING = ("Screening result only: the model compares the sound with labelled examples from one "
                    "dataset and was not validated on your pipes or equipment.")

_MESSAGES = {
    "leak": "The recording resembles the dataset's leak examples. This is a screening result, not "
            "confirmation of a leak; consider an inspection.",
    "no_leak": "The recording resembles the dataset's no-leak examples. This does not rule out a leak.",
    "noise": "The recording resembles the dataset's environmental-noise examples (not pipe sound). "
             "Consider re-recording closer to the pipe.",
}

_cache: dict[str, dict] = {}


@dataclass
class AnalysisResult:
    status: str  # "completed", "uncertain", or "error"
    predicted_class: Optional[str]
    confidence: Optional[float]
    model_version: Optional[str]
    message: str
    warnings: list[str]


def _error(message: str, *warnings: str) -> AnalysisResult:
    return AnalysisResult("error", None, None, None, message,
                          [*warnings, "No model prediction was made. Do not treat this as a leak result."])


def _load_bundle() -> dict:
    key = str(MODEL_PATH)
    if key not in _cache:
        import joblib

        bundle = joblib.load(MODEL_PATH)
        if bundle.get("feature_version") != af.FEATURE_VERSION:
            raise RuntimeError("Model and feature extractor versions do not match.")
        _cache[key] = bundle
    return _cache[key]


def analyze_audio_bytes(file_bytes: bytes, filename: str) -> AnalysisResult:
    """Analyze uploaded bytes with the trained model and return an AnalysisResult."""
    if not file_bytes:
        return _error("The uploaded file is empty.")
    suffix = Path(filename or "").suffix.lower()
    if suffix not in af.SUPPORTED_SUFFIXES:
        return _error(f"Unsupported audio format: {suffix or '(none)'}. "
                      f"Supported: {', '.join(sorted(af.SUPPORTED_SUFFIXES))}.")
    if len(file_bytes) > MAX_BYTES:
        return _error(f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")

    try:
        bundle = _load_bundle()
    except Exception:  # noqa: BLE001
        return _error("The trained model is not connected yet.")

    try:
        try:
            audio, sr = af.load_audio(file_bytes)
        except ValueError:
            return _error("The file could not be read as audio. It may be corrupted or not a real audio file.")
        audio8k = af.resample_to_target(audio, sr)
        if audio8k.size < af.CLIP_SAMPLES - af.LENGTH_TOLERANCE:
            return _error("The recording is shorter than 1 second and cannot be analysed.")
        windows = af.split_windows(audio8k)
        warnings = [STANDING_WARNING]
        if len(windows) > MAX_WINDOWS:
            windows = windows[:MAX_WINDOWS]
            warnings.append(f"Only the first {MAX_WINDOWS} seconds were analysed.")
        if len(windows) > 1:
            warnings.append(f"The recording was analysed as {len(windows)} one-second windows whose scores "
                            "were averaged; this averaging was not separately evaluated.")

        feats, skipped = [], 0
        for w in windows:
            try:
                feats.append(af.extract_features(w, af.TARGET_SR))
            except ValueError:
                skipped += 1
        if not feats:
            return _error("The recording is silent or could not be analysed.")
        if skipped:
            warnings.append(f"{skipped} of {len(windows)} windows were silent or unusable and were skipped.")

        proba = bundle["model"].predict_proba(np.vstack(feats))
        classes = list(bundle["classes"])
        mean = proba.mean(axis=0)
        top = int(mean.argmax())
        score = float(mean[top])
        agree = float(np.mean(proba.argmax(axis=1) == top))
        if len(feats) > 1 and agree < 0.6:
            warnings.append("Different parts of the recording point to different classes.")

        if score < UNCERTAIN_THRESHOLD:
            return AnalysisResult("uncertain", None, score, MODEL_VERSION,
                                  "The model is not confident enough to classify this recording. "
                                  "Review it manually.", warnings)
        label = classes[top]
        return AnalysisResult("completed", label, score, MODEL_VERSION, _MESSAGES.get(label, ""), warnings)
    except ValueError as exc:
        return _error(f"The recording could not be analysed: {exc}")
    except Exception:  # noqa: BLE001
        return _error("An unexpected error occurred during analysis.")
