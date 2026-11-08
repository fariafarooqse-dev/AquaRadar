from pathlib import Path
import pytest
from audio_features import validate_audio_path


def test_missing_audio_path_raises():
    with pytest.raises(ValueError):
        validate_audio_path(Path("definitely-not-a-real-audio-file.wav"))


def test_unsupported_suffix_raises(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("not audio")
    with pytest.raises(ValueError):
        validate_audio_path(file_path)
