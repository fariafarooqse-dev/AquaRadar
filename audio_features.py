"""Audio validation, preprocessing and feature extraction (Member 1 owns this module).

Feature set "f1" is defined for the Zenodo leak-audio clips inspected in Step 1:
1-second, mono, 8 kHz clips. Training and inference MUST both go through
`extract_features`, which applies `preprocess` internally, so they cannot drift apart.

Design choices (stated so they can be challenged):
  * Everything is resampled to 8 kHz mono.
  * DC offset is removed and the clip is RMS-normalised, so absolute recording level
    (which may differ between devices/sources) is not a feature.
  * Spectral shape is expressed as the *fraction* of power per band, again level-independent.
  * Clips must be 1.0 s (+/- 2 samples). Longer uploads need windowing, which is a later,
    separate step; this module does not silently cut or pad them.
"""
from __future__ import annotations

import io
from math import gcd
from pathlib import Path

import numpy as np
from scipy import signal

SUPPORTED_SUFFIXES = {".wav", ".flac", ".ogg"}

FEATURE_VERSION = "f1"
TARGET_SR = 8000
CLIP_SAMPLES = 8000          # 1.0 s at TARGET_SR
LENGTH_TOLERANCE = 2         # samples tolerated after resampling
MIN_RMS = 1e-4               # on a [-1, 1] float scale, after DC removal
N_BANDS = 32                 # 125 Hz wide bands from 0 to 4000 Hz
_EPS = 1e-12


def validate_audio_path(path: str | Path) -> None:
    """Raise ValueError for an unsupported or missing audio file."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ValueError("Audio file does not exist.")
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported audio format: {p.suffix}")


def load_audio(source: str | Path | bytes) -> tuple[np.ndarray, int]:
    """Decode a file path or raw bytes to (mono float32 waveform, original sample rate)."""
    import soundfile as sf  # imported lazily so the rest of the module works without it

    try:
        handle = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else str(source)
        data, sr = sf.read(handle, dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not decode audio: {exc}") from exc
    if data.size == 0:
        raise ValueError("Audio is empty.")
    return data.mean(axis=1), int(sr)


def resample_to_target(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample a mono waveform to TARGET_SR (float64). No trimming, padding or normalisation."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    x = np.asarray(audio, dtype=np.float64)
    if int(sample_rate) == TARGET_SR:
        return x
    g = gcd(TARGET_SR, int(sample_rate))
    return signal.resample_poly(x, TARGET_SR // g, int(sample_rate) // g)


def split_windows(audio_8k: np.ndarray) -> list[np.ndarray]:
    """Cut an 8 kHz waveform into consecutive 1.0 s windows; a shorter tail is dropped.

    A waveform within LENGTH_TOLERANCE samples of one clip is returned as a single window.
    """
    x = np.asarray(audio_8k)
    if abs(x.size - CLIP_SAMPLES) <= LENGTH_TOLERANCE:
        return [x]
    n = x.size // CLIP_SAMPLES
    return [x[i * CLIP_SAMPLES:(i + 1) * CLIP_SAMPLES] for i in range(n)]


def preprocess(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono waveform -> 8 kHz, exactly CLIP_SAMPLES, DC-free, unit-RMS float64 array."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Audio must be a one-dimensional mono waveform.")
    if x.size == 0:
        raise ValueError("Audio is empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Audio contains NaN or infinite values.")

    x = resample_to_target(x, sample_rate)

    if abs(x.size - CLIP_SAMPLES) > LENGTH_TOLERANCE:
        raise ValueError(
            f"Expected a {CLIP_SAMPLES / TARGET_SR:.1f} s clip ({CLIP_SAMPLES} samples at "
            f"{TARGET_SR} Hz) but got {x.size} samples after resampling."
        )
    if x.size > CLIP_SAMPLES:
        x = x[:CLIP_SAMPLES]
    elif x.size < CLIP_SAMPLES:
        x = np.pad(x, (0, CLIP_SAMPLES - x.size))

    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < MIN_RMS:
        raise ValueError("Audio is silent or almost silent.")
    return x / rms


def feature_names() -> list[str]:
    names = [f"band_{i * 125}_{(i + 1) * 125}Hz_logfrac" for i in range(N_BANDS)]
    names += ["spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_rolloff85_hz",
              "spectral_flatness", "zero_crossing_rate", "crest_factor", "frame_rms_cv"]
    return names


def extract_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return the 1-D feature vector (length len(feature_names())) for one clip.

    Args:
        audio: mono floating-point waveform of about 1 s.
        sample_rate: samples per second.

    Raises:
        ValueError: empty/invalid/silent audio, bad sample rate, or wrong clip length.
    """
    x = preprocess(audio, sample_rate)

    f, pxx = signal.welch(x, fs=TARGET_SR, window="hann", nperseg=512, noverlap=256)
    f, ps = f[:256], pxx[:256]                       # drop the Nyquist bin: 256 = 32 bands x 8 bins
    band_power = ps.reshape(N_BANDS, 8).mean(axis=1)
    frac = band_power / (band_power.sum() + _EPS)
    band_feats = np.log10(frac + 1e-10)

    tot = ps.sum() + _EPS
    centroid = float((f * ps).sum() / tot)
    bandwidth = float(np.sqrt((((f - centroid) ** 2) * ps).sum() / tot))
    rolloff = float(f[np.searchsorted(np.cumsum(ps), 0.85 * tot)])
    flatness = float(np.exp(np.mean(np.log(ps + _EPS))) / (np.mean(ps) + _EPS))
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    crest = float(np.max(np.abs(x)))                 # RMS is 1 after normalisation
    frames = x.reshape(8, 1000)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    frame_cv = float(frame_rms.std() / (frame_rms.mean() + _EPS))

    vec = np.concatenate([band_feats, [centroid, bandwidth, rolloff, flatness, zcr, crest, frame_cv]])
    if not np.all(np.isfinite(vec)):
        raise ValueError("Feature extraction produced non-finite values.")
    return vec
