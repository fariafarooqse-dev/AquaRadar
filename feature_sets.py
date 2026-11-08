"""Modular feature sets for AquaRadar experiments (Member 1 owns this module).

`audio_features.py` stays frozen as the f1 baseline: the saved model and `model_service.py` depend on it.
This module composes f1 with optional experimental blocks. Nothing here changes f1.

Why training and inference cannot drift apart:
  * Blocks that reuse f1 call `audio_features.extract_features` itself.
  * New blocks start from `audio_features.preprocess` (8 kHz mono, exactly 1 s, DC-free, unit RMS).
  * Every block is per-clip. Nothing is fitted on the dataset (no scaler, PCA or feature selection),
    so no statistic can leak from one fold into another.

Experimental blocks (all UNPROVEN for pipe acoustics; see docs and the evaluation step):
  temporal : 7 features on 32 ms frames: frame-energy variation, spectral flux, per-frame flatness
             and centroid variation. Motivation: leaks are steady, many noise sources are bursty.
  mfcc     : mean and std over frames of 12 MFCCs (c0 dropped, because c0 is level information).
  dmfcc    : std over frames of the delta (regression, width 2) of the same 12 MFCCs.

Absolute loudness is deliberately NOT a feature: it would encode recording-device gain.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.fft import dct

import audio_features as af

FRAME = 256            # 32 ms at 8 kHz
HOP = 128              # 16 ms
N_MELS = 24
N_MFCC = 12
DELTA_WIDTH = 2
_EPS = 1e-12

# Feature set name -> ordered blocks. The name doubles as the feature version stored with a model.
FEATURE_SETS: dict[str, list[str]] = {
    "f1": ["f1"],
    "f1_bands_only": ["f1_bands"],
    "f1_scalars_only": ["f1_scalars"],
    "f1_temporal": ["f1", "temporal"],
    "f1_mfcc": ["f1", "mfcc"],
    "f1_mfcc_delta": ["f1", "mfcc", "dmfcc"],
}

_F1_NAMES = af.feature_names()
_F1_SLICES = {"f1": slice(None), "f1_bands": slice(0, af.N_BANDS), "f1_scalars": slice(af.N_BANDS, None)}

BLOCK_NAMES: dict[str, list[str]] = {
    "f1": _F1_NAMES,
    "f1_bands": _F1_NAMES[:af.N_BANDS],
    "f1_scalars": _F1_NAMES[af.N_BANDS:],
    "temporal": ["t_frame_rms_cv32", "t_logrms_range_db", "t_flux_mean", "t_flux_std",
                 "t_flatness_mean", "t_flatness_std", "t_centroid_std_hz"],
    "mfcc": ([f"mfcc{k}_mean" for k in range(1, N_MFCC + 1)]
             + [f"mfcc{k}_std" for k in range(1, N_MFCC + 1)]),
    "dmfcc": [f"dmfcc{k}_std" for k in range(1, N_MFCC + 1)],
}


# ----------------------------------------------------------------------------- framing / spectra
def _frames(x: np.ndarray) -> np.ndarray:
    """Overlapping frames of a preprocessed clip: shape (61, 256) for 8000 samples."""
    return sliding_window_view(x, FRAME)[::HOP]


def _frame_power(x: np.ndarray) -> np.ndarray:
    """Hann-windowed power spectrum per frame: shape (n_frames, FRAME // 2 + 1)."""
    return np.abs(np.fft.rfft(_frames(x) * signal.get_window("hann", FRAME), axis=1)) ** 2


# ----------------------------------------------------------------------------- temporal block
def _temporal(x: np.ndarray) -> np.ndarray:
    fr = _frames(x)
    rms = np.sqrt(np.mean(fr ** 2, axis=1))
    rms_cv = rms.std() / (rms.mean() + _EPS)
    db = 20.0 * np.log10(rms + 1e-10)
    db_range = np.percentile(db, 95) - np.percentile(db, 5)

    p = _frame_power(x)
    mag = np.sqrt(p)
    s = mag / (mag.sum(axis=1, keepdims=True) + _EPS)          # level-independent spectral shape
    flux = np.sqrt((np.diff(s, axis=0) ** 2).sum(axis=1))
    flat = np.exp(np.mean(np.log(p + _EPS), axis=1)) / (p.mean(axis=1) + _EPS)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / af.TARGET_SR)
    cent = (p * freqs).sum(axis=1) / (p.sum(axis=1) + _EPS)
    return np.array([rms_cv, db_range, flux.mean(), flux.std(), flat.mean(), flat.std(), cent.std()])


# ----------------------------------------------------------------------------- MFCC blocks
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


@lru_cache(maxsize=1)
def mel_filterbank() -> np.ndarray:
    """Triangular HTK-mel filterbank, shape (N_MELS, FRAME // 2 + 1), covering 0 Hz to Nyquist."""
    nyq = af.TARGET_SR / 2.0
    pts = _mel_to_hz(np.linspace(0.0, _hz_to_mel(nyq), N_MELS + 2))
    freqs = np.fft.rfftfreq(FRAME, 1.0 / af.TARGET_SR)
    fb = np.zeros((N_MELS, freqs.size))
    for m in range(N_MELS):
        lo, centre, hi = pts[m], pts[m + 1], pts[m + 2]
        fb[m] = np.maximum(0.0, np.minimum((freqs - lo) / (centre - lo), (hi - freqs) / (hi - centre)))
    fb.flags.writeable = False
    return fb


def _mfcc_frames(x: np.ndarray) -> np.ndarray:
    """MFCCs 1..N_MFCC per frame, shape (n_frames, N_MFCC)."""
    mel = _frame_power(x) @ mel_filterbank().T
    return dct(np.log(mel + 1e-10), type=2, axis=1, norm="ortho")[:, 1:1 + N_MFCC]


def _mfcc(x: np.ndarray) -> np.ndarray:
    c = _mfcc_frames(x)
    return np.concatenate([c.mean(axis=0), c.std(axis=0)])


def _delta(c: np.ndarray, width: int = DELTA_WIDTH) -> np.ndarray:
    """Regression delta along axis 0 with edge padding (equals the slope for a linear ramp)."""
    pad = np.pad(c, ((width, width), (0, 0)), mode="edge")
    t = c.shape[0]
    d = np.zeros_like(c, dtype=float)
    for n in range(1, width + 1):
        d += n * (pad[width + n:width + n + t] - pad[width - n:width - n + t])
    return d / (2 * sum(n * n for n in range(1, width + 1)))


def _dmfcc(x: np.ndarray) -> np.ndarray:
    return _delta(_mfcc_frames(x)).std(axis=0)


_NEW_BLOCKS = {"temporal": _temporal, "mfcc": _mfcc, "dmfcc": _dmfcc}


# ----------------------------------------------------------------------------- public API
def available_sets() -> list[str]:
    return list(FEATURE_SETS)


def feature_set_blocks(set_name: str) -> list[str]:
    """Ordered block names of a feature set (to be stored with a trained model)."""
    if set_name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {set_name!r}. Available: {available_sets()}")
    return list(FEATURE_SETS[set_name])


def feature_set_names(set_name: str) -> list[str]:
    return [n for b in feature_set_blocks(set_name) for n in BLOCK_NAMES[b]]


def extract_set(audio: np.ndarray, sample_rate: int, set_name: str) -> np.ndarray:
    """Feature vector of one ~1 s clip for the named feature set.

    Raises ValueError for the same invalid input as `audio_features.extract_features`
    (empty, silent, NaN, bad sample rate, wrong length) because both go through `preprocess`.
    """
    blocks = feature_set_blocks(set_name)
    f1 = None
    x = None
    parts = []
    for b in blocks:
        if b in _F1_SLICES:
            if f1 is None:
                f1 = af.extract_features(audio, sample_rate)
            parts.append(f1[_F1_SLICES[b]])
        else:
            if x is None:
                x = af.preprocess(audio, sample_rate)
            parts.append(_NEW_BLOCKS[b](x))
    vec = np.concatenate(parts)
    if not np.all(np.isfinite(vec)):
        raise ValueError("Feature extraction produced non-finite values.")
    return vec
