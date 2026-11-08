import io
import wave
from pathlib import Path

import numpy as np
import pytest

import audio_features as af
import model_service as ms

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features_f1.npz"
RAW = ROOT / "data" / "raw" / "zenodo_leak_audio"


def wav_bytes(x: np.ndarray, sr: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def noise(seconds: float, sr: int = 8000, seed: int = 0) -> np.ndarray:
    return 0.1 * np.random.default_rng(seed).standard_normal(int(seconds * sr))


def need_model():
    if not ms.MODEL_PATH.exists():
        pytest.skip("trained model not present; run: python train_model.py final")


def assert_no_fake_prediction(r):
    if r.status in ("error", "uncertain"):
        assert r.predicted_class is None
    if r.status == "error":
        assert r.confidence is None and r.model_version is None


def test_empty_bytes_is_error():
    r = ms.analyze_audio_bytes(b"", "a.wav")
    assert r.status == "error"
    assert_no_fake_prediction(r)


def test_unsupported_suffix_is_error():
    r = ms.analyze_audio_bytes(b"abc", "notes.txt")
    assert r.status == "error"
    assert_no_fake_prediction(r)


def test_garbage_bytes_is_error():
    need_model()
    r = ms.analyze_audio_bytes(b"this is not audio" * 50, "a.wav")
    assert r.status == "error"
    assert_no_fake_prediction(r)


def test_too_short_is_error():
    need_model()
    r = ms.analyze_audio_bytes(wav_bytes(noise(0.5)), "a.wav")
    assert r.status == "error"
    assert "shorter than 1 second" in r.message


def test_silent_is_error():
    need_model()
    r = ms.analyze_audio_bytes(wav_bytes(np.zeros(16000)), "a.wav")
    assert r.status == "error"
    assert_no_fake_prediction(r)


def test_valid_clip_returns_documented_fields():
    need_model()
    r = ms.analyze_audio_bytes(wav_bytes(noise(1.0)), "a.wav")
    assert r.status in ("completed", "uncertain")
    assert r.model_version == ms.MODEL_VERSION
    assert 0.0 <= r.confidence <= 1.0
    assert r.warnings
    if r.status == "completed":
        assert r.predicted_class in ("leak", "no_leak", "noise")
    assert_no_fake_prediction(r)


def test_longer_resampled_recording_is_windowed():
    need_model()
    r = ms.analyze_audio_bytes(wav_bytes(noise(3.4, sr=16000), sr=16000), "a.wav")
    assert r.status in ("completed", "uncertain")
    assert any("one-second windows" in w for w in r.warnings)


def test_deterministic():
    need_model()
    b = wav_bytes(noise(2.0, seed=3))
    assert ms.analyze_audio_bytes(b, "a.wav") == ms.analyze_audio_bytes(b, "a.wav")


def test_low_score_gives_uncertain_without_class():
    need_model()
    old = ms.UNCERTAIN_THRESHOLD
    ms.UNCERTAIN_THRESHOLD = 1.01
    try:
        r = ms.analyze_audio_bytes(wav_bytes(noise(1.0)), "a.wav")
    finally:
        ms.UNCERTAIN_THRESHOLD = old
    assert r.status == "uncertain"
    assert r.predicted_class is None


def test_missing_model_is_error_not_fake_prediction():
    old = ms.MODEL_PATH
    ms.MODEL_PATH = ROOT / "models" / "does_not_exist.joblib"
    try:
        r = ms.analyze_audio_bytes(wav_bytes(noise(1.0)), "a.wav")
    finally:
        ms.MODEL_PATH = old
    assert r.status == "error"
    assert "not connected" in r.message
    assert_no_fake_prediction(r)


def test_inference_features_match_training_features():
    """Decoding real dataset files from bytes must reproduce the stored training features."""
    if not (FEATURES.exists() and RAW.exists()):
        pytest.skip("dataset or features file not present")
    z = np.load(FEATURES, allow_pickle=False)
    for i in np.linspace(0, len(z["paths"]) - 1, 8).astype(int):
        audio, sr = af.load_audio((RAW / str(z["paths"][i])).read_bytes())
        v = af.extract_features(af.split_windows(af.resample_to_target(audio, sr))[0], af.TARGET_SR)
        assert np.allclose(v, z["X"][i], rtol=1e-9, atol=1e-9), z["paths"][i]
