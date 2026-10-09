"""Stable, provider-independent transcription results. None means unsupported/unknown."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Word:
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None


@dataclass
class Segment:
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptResult:
    text: str
    model_id: str
    revision: str | None
    duration_seconds: float
    segments: list[Segment] = field(default_factory=list)
    timestamp_precision: str = "none"  # none | segment | word
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
