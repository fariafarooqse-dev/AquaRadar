"""Stable boundary between the Streamlit app and the trained model.

Member 1 owns this module. UI code should call this service instead of loading
model files or reproducing model logic inside app.py.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AnalysisResult:
    status: str  # "completed", "uncertain", or "error"
    predicted_class: Optional[str]
    confidence: Optional[float]
    model_version: Optional[str]
    message: str
    warnings: list[str]


def analyze_audio_bytes(file_bytes: bytes, filename: str) -> AnalysisResult:
    """Analyze uploaded bytes with the trained model.

    TODO:
    1. Validate file and decode audio safely.
    2. Apply the exact preprocessing used during training.
    3. Extract the expected features.
    4. Load the trained model and predict.
    5. Apply documented uncertainty/quality rules.
    6. Return the actual result and warnings.

    Until implemented, return an explicit error rather than fake a prediction.
    """
    if not file_bytes:
        return AnalysisResult(
            status="error",
            predicted_class=None,
            confidence=None,
            model_version=None,
            message="The uploaded file is empty.",
            warnings=["No model inference was performed."],
        )
    return AnalysisResult(
        status="error",
        predicted_class=None,
        confidence=None,
        model_version=None,
        message="The trained model is not connected yet.",
        warnings=["Do not treat this as a leak prediction."],
    )
