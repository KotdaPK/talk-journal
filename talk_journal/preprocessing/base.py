"""Preprocessors operate on one decoded WAV at a time; original media is untouched."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Preprocessor(Protocol):
    name: str

    def load(self) -> None: ...

    def process(self, source: Path, destination: Path, highpass_hz: int | None) -> None: ...

    def close(self) -> None: ...
