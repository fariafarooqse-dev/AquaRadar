import numpy as np
import pytest

import audio_features as af
import feature_sets as fs

SR = af.TARGET_SR


def tone(freq=200.0, sr=SR, seconds=1.0, amp=0.3, seed=0):
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(seed)
    return amp * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(t.size)


def steady_noise(seed=1):
    return np.random.default_rng(seed).standard_normal(SR)


def gated_noise(seed=1):
    """Noise that is on for 0.2 s out of every 0.5 s: a bursty source."""
    x = np.random.default_rng(seed).standard_normal(SR)
    env = np.full(SR, 0.001)
    for start in (0.0, 0.5):
        env[int(start * SR):int((start + 0.2) * SR)] = 1.0
    return x * env


SIGNALS = {"tone": tone(), "steady": steady_noise(), "gated": gated_noise()}


def test_f1_set_is_exactly_the_existing_extractor():
    for name, x in SIGNALS.items():
        assert np.array_equal(fs.extract_set(x, SR, "f1"), af.extract_features(x, SR)), name
    x16 = tone(sr=16000)
    assert np.array_equal(fs.extract_set(x16, 16000, "f1"), af.extract_features(x16, 16000))


def test_sets_that_include_f1_keep_its_first_39_columns_unchanged():
    x = SIGNALS["steady"]
    base = af.extract_features(x, SR)
    for name in fs.available_sets():
        if fs.feature_set_blocks(name)[0] == "f1":
            assert np.array_equal(fs.extract_set(x, SR, name)[:39], base), name


def test_ablation_sets_are_exact_slices_of_f1():
    x = SIGNALS["gated"]
    base = af.extract_features(x, SR)
    assert np.array_equal(fs.extract_set(x, SR, "f1_bands_only"), base[:32])
    assert np.array_equal(fs.extract_set(x, SR, "f1_scalars_only"), base[32:])


@pytest.mark.parametrize("name", fs.available_sets())
def test_length_matches_names_and_values_are_finite(name):
    names = fs.feature_set_names(name)
    assert len(names) == len(set(names)), "duplicate feature names"
    for x in SIGNALS.values():
        v = fs.extract_set(x, SR, name)
        assert v.ndim == 1 and v.size == len(names)
        assert np.all(np.isfinite(v))


def test_expected_sizes():
    sizes = {n: len(fs.feature_set_names(n)) for n in fs.available_sets()}
    assert sizes == {"f1": 39, "f1_bands_only": 32, "f1_scalars_only": 7, "f1_temporal": 46,
                     "f1_mfcc": 63, "f1_mfcc_delta": 75}


@pytest.mark.parametrize("name", fs.available_sets())
def test_deterministic_and_level_independent(name):
    x = SIGNALS["steady"]
    a = fs.extract_set(x, SR, name)
    assert np.array_equal(a, fs.extract_set(x, SR, name))
    loud = fs.extract_set(x * 50.0, SR, name)
    assert np.allclose(a, loud, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", fs.available_sets())
def test_invalid_input_raises_like_the_existing_extractor(name):
    with pytest.raises(ValueError):
        fs.extract_set(np.array([]), SR, name)
    with pytest.raises(ValueError):
        fs.extract_set(np.zeros(af.CLIP_SAMPLES), SR, name)
    with pytest.raises(ValueError):
        fs.extract_set(tone(seconds=3.0), SR, name)
    with pytest.raises(ValueError):
        fs.extract_set(tone(), 0, name)
    bad = tone()
    bad[5] = np.nan
    with pytest.raises(ValueError):
        fs.extract_set(bad, SR, name)


def test_other_sample_rate_is_resampled_by_the_shared_preprocess():
    v = fs.extract_set(tone(sr=16000), 16000, "f1_mfcc_delta")
    assert v.size == 75 and np.all(np.isfinite(v))


def test_unknown_set_raises():
    with pytest.raises(ValueError):
        fs.extract_set(tone(), SR, "nope")


def test_frame_count():
    x = af.preprocess(tone(), SR)
    assert fs._frames(x).shape == (61, fs.FRAME)


def test_mel_filterbank_shape_and_coverage():
    fb = fs.mel_filterbank()
    assert fb.shape == (fs.N_MELS, fs.FRAME // 2 + 1)
    assert np.all(fb >= 0)
    assert np.all(fb.sum(axis=1) > 0), "a mel filter fell between FFT bins"


def test_delta_of_a_linear_ramp_is_its_slope():
    ramp = np.outer(np.arange(30.0), np.array([1.0, -2.0, 0.5]))
    d = fs._delta(ramp)
    assert np.allclose(d[3:-3], [1.0, -2.0, 0.5])


def test_temporal_block_separates_bursty_from_steady_noise():
    i = fs.BLOCK_NAMES["temporal"].index("t_frame_rms_cv32")
    steady = fs.extract_set(SIGNALS["steady"], SR, "f1_temporal")[39 + i]
    gated = fs.extract_set(SIGNALS["gated"], SR, "f1_temporal")[39 + i]
    assert gated > steady + 0.5
