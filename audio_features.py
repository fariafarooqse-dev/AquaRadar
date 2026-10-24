"""Audio validation and feature extraction starter.

Member 1 owns this module. Keep the function contract stable once agreed with the UI.
Do not invent model predictions in this module.
"""

from pathlib import Path
import numpy as np


SUPPORTED_SUFFIXES = {".wav", ".flac", ".ogg"}


def validate_audio_path(path: str | Path) -> None:
    """Raise ValueError for an unsupported or missing audio file."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ValueError("Audio file does not exist.")
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported audio format: {p.suffix}")


def extract_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """TODO: Implement and document features after inspecting the chosen dataset.

    Args:
        audio: mono floating-point waveform.
        sample_rate: samples per second.

    Returns:
        A one-dimensional numeric feature vector.

    Raises:
        ValueError: if audio is empty or sample_rate is invalid.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if audio is None or np.asarray(audio).size == 0:
        raise ValueError("Audio is empty.")
    raise NotImplementedError("Feature extraction must be implemented and tested.")
