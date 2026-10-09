"""ASR adapter contract; segmentation belongs to the implementing backend."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from talk_journal.models import TranscriptResult


class ASRBackend(Protocol):
    model_id: str
    sample_rate: int
    supports_hotwords: bool

    def load(self) -> None: ...

    def transcribe(self, audio: Path, hotwords: list[str]) -> TranscriptResult: ...

    def close(self) -> None: ...
