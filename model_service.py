"""Stable boundary between the Streamlit app and the trained model.

This version runs the LOCAL PILOT model (local_pilot/models/local_leak_model_v1.joblib), trained on
recordings from our own test rig. It uses the exact same feature code as training
(local_pilot/make_features.py), so features always match the model.

Behaviour:
  * Never fabricates a prediction. Any failure returns status="error" with predicted_class=None.
  * Only PCM WAV files are supported (the same reader logic as local_pilot/make_features.py).
  * Recordings are cut into consecutive 5-second windows (after trimming 1 s from each end of
    longer files), exactly as in training. Window probabilities are averaged.
  * The pilot model is binary: "leak" (open-hole sound) vs "no_leak".
  * status="uncertain" (predicted_class=None) when the top averaged score is below
    UNCERTAIN_THRESHOLD.
  * `confidence` is the model's averaged probability for the top class. It is NOT a calibrated
    probability that a real leak exists.

The previous public-dataset model (models/leak_model_v1.joblib) is left untouched on disk but is no
longer used by this module.
"""
from __future__ import annotations

import io
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

LOCAL_DIR = Path(__file__).resolve().parent / "local_pilot"
MODEL_PATH = LOCAL_DIR / "models" / "local_leak_model_v1.joblib"
MODEL_VERSION = "local_leak_model_v1"
UNCERTAIN_THRESHOLD = 0.60
MAX_BYTES = 25 * 1024 * 1024
MAX_WINDOWS = 24                 # at most 24 windows of 5 s (2 minutes) are analysed
N_FEATURES = 31                  # 24 frequency bands + 7 summary features (see make_features.py)
SUPPORTED_SUFFIXES = {".wav"}

STANDING_WARNING = ("Pilot model: trained on recordings from one test pipe rig only. "
                    "It has not been validated on other pipes, buildings or recording devices.")

_MESSAGES = {
    "leak": "The recording resembles the open-hole (leak-like) recordings from our test rig. "
            "This is a screening result, not confirmation of a leak; consider an inspection.",
    "no_leak": "The recording resembles the no-leak recordings from our test rig. "
               "This does not rule out a leak.",
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


def _mf():
    """Import the training-time feature code from local_pilot/make_features.py."""
    if str(LOCAL_DIR) not in sys.path:
        sys.path.insert(0, str(LOCAL_DIR))
    import make_features  # noqa: WPS433

    return make_features


def _load_bundle() -> dict:
    key = str(MODEL_PATH)
    if key not in _cache:
        import joblib

        bundle = joblib.load(MODEL_PATH)
        if "pipeline" not in bundle:
            raise RuntimeError("Unexpected model file layout.")
        _cache[key] = bundle
    return _cache[key]


def _read_wav_bytes(file_bytes: bytes):
    """Same decoding as make_features.read_wav, but from bytes. Returns (mono float array, sample rate)."""
    with wave.open(io.BytesIO(file_bytes), "rb") as w:
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
    return x, sr


def analyze_audio_bytes(file_bytes: bytes, filename: str) -> AnalysisResult:
    """Analyze uploaded bytes with the local pilot model and return an AnalysisResult."""
    if not file_bytes:
        return _error("The uploaded file is empty.")
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return _error(f"Unsupported audio format: {suffix or '(none)'}. "
                      f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}.")
    if len(file_bytes) > MAX_BYTES:
        return _error(f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")

    try:
        bundle = _load_bundle()
        mf = _mf()
    except Exception:  # noqa: BLE001
        return _error("The local pilot model could not be loaded. Check that "
                      "local_pilot/models/local_leak_model_v1.joblib exists.")

    try:
        try:
            x, sr = _read_wav_bytes(file_bytes)
        except (wave.Error, EOFError):
            return _error("The file could not be read as a PCM WAV recording. It may be corrupted, "
                          "or saved in a format this pilot does not support (use 16-bit WAV).")
        win_s, trim_s = mf.WIN_S, mf.TRIM_S
        dur = len(x) / sr
        core = x[int(trim_s * sr): len(x) - int(trim_s * sr)] if dur > 3 * trim_s + win_s else x
        n = int(len(core) // (win_s * sr))
        if n == 0:
            return _error(f"The recording is too short. At least {win_s:.0f} seconds are needed.")
        warnings = [STANDING_WARNING]
        if n > MAX_WINDOWS:
            n = MAX_WINDOWS
            warnings.append(f"Only the first {MAX_WINDOWS * int(win_s)} seconds were analysed.")
        if n > 1:
            warnings.append(f"The recording was analysed as {n} windows of {win_s:.0f} seconds whose "
                            "scores were averaged.")

        rows = [mf.window_features(core[int(i * win_s * sr): int((i + 1) * win_s * sr)], sr) for i in range(n)]
        feats = np.array(rows, dtype=float)
        if feats.shape[1] != N_FEATURES:
            return _error("The model and the feature extractor do not match. Please retrain the model.")
        if not np.all(np.isfinite(feats)):
            return _error("The recording is silent or could not be analysed.")

        p_leak = bundle["pipeline"].predict_proba(feats)[:, 1]
        mean_leak = float(p_leak.mean())
        label = "leak" if mean_leak >= 0.5 else "no_leak"
        score = mean_leak if label == "leak" else 1.0 - mean_leak
        agree = float(np.mean((p_leak >= 0.5) == (mean_leak >= 0.5)))
        if n > 1 and agree < 0.6:
            warnings.append("Different parts of the recording point to different classes.")

        if score < UNCERTAIN_THRESHOLD:
            return AnalysisResult("uncertain", None, score, MODEL_VERSION,
                                  "The model is not confident enough to classify this recording. "
                                  "Review it manually.", warnings)
        return AnalysisResult("completed", label, score, MODEL_VERSION, _MESSAGES.get(label, ""), warnings)
    except ValueError as exc:
        return _error(f"The recording could not be analysed: {exc}")
    except Exception:  # noqa: BLE001
        return _error("An unexpected error occurred during analysis.")
