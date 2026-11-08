import numpy as np
import pytest

from audio_features import (CLIP_SAMPLES, TARGET_SR, extract_features, feature_names,
                            preprocess)


def _tone(freq=200.0, sr=TARGET_SR, seconds=1.0, amp=0.3, seed=0):
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(seed)
    return amp * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(t.size)


def test_feature_length_matches_names_and_is_finite():
    v = extract_features(_tone(), TARGET_SR)
    assert v.ndim == 1
    assert v.size == len(feature_names()) == 39
    assert np.all(np.isfinite(v))


def test_deterministic():
    a = extract_features(_tone(), TARGET_SR)
    b = extract_features(_tone(), TARGET_SR)
    assert np.array_equal(a, b)


def test_level_independent():
    quiet = extract_features(_tone(amp=0.01), TARGET_SR)
    loud = extract_features(_tone(amp=0.01) * 50, TARGET_SR)
    assert np.allclose(quiet, loud, atol=1e-6)


def test_different_sample_rate_is_resampled():
    v = extract_features(_tone(sr=16000), 16000)
    assert v.size == len(feature_names())


def test_low_vs_high_tone_changes_centroid():
    names = feature_names()
    i = names.index("spectral_centroid_hz")
    low = extract_features(_tone(freq=150.0), TARGET_SR)[i]
    high = extract_features(_tone(freq=2500.0), TARGET_SR)[i]
    assert high > low + 1000


def test_empty_raises():
    with pytest.raises(ValueError):
        extract_features(np.array([]), TARGET_SR)


def test_bad_sample_rate_raises():
    with pytest.raises(ValueError):
        extract_features(_tone(), 0)


def test_silent_raises():
    with pytest.raises(ValueError):
        extract_features(np.zeros(CLIP_SAMPLES), TARGET_SR)


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        preprocess(_tone(seconds=3.0), TARGET_SR)


def test_nan_raises():
    x = _tone()
    x[10] = np.nan
    with pytest.raises(ValueError):
        extract_features(x, TARGET_SR)
