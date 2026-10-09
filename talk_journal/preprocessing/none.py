"""Pass-through preprocessing for decoded mono WAV files."""
from __future__ import annotations

import shutil
from pathlib import Path


class NonePreprocessor:
    """Validate decoded audio and copy it without altering samples."""

    name = "none"

    def load(self) -> None:
        """No model or optional dependency is needed."""

    def process(
        self, source: Path, destination: Path, highpass_hz: int | None
    ) -> None:
        if highpass_hz is not None:
            raise ValueError("High-pass filtering requires DeepFilterNet3 preprocessing")
        self._validate_wav(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def close(self) -> None:
        """No resources are retained."""

    @staticmethod
    def _validate_wav(source: Path) -> None:
        if source.suffix.lower() != ".wav":
            raise ValueError(f"Expected a decoded WAV input, got: {source}")

        import soundfile as sf

        info = sf.info(source)
        if info.channels != 1:
            raise ValueError(
                "Pass-through preprocessing requires mono WAV input; "
                f"got {info.samplerate} Hz with {info.channels} channel(s)."
            )
