"""Moondream Parakeet Ultra adapter using the Photon speech API."""
from __future__ import annotations

from collections.abc import Mapping
import importlib
from pathlib import Path
from typing import Any
import wave

from talk_journal.models import Segment, TranscriptResult, Word


class ParakeetBackend:
    """Transcribe normalized 16 kHz WAV files with Parakeet Ultra on CUDA."""

    model_id = "moondream/parakeet-ultra"
    sample_rate = 16_000
    supports_hotwords = False

    def __init__(self) -> None:
        self._context: Any | None = None
        self._speech: Any | None = None

    def load(self) -> None:
        """Load the Photon model once, explicitly on the first CUDA device."""
        if self._speech is not None:
            return
        import torch
        if not torch.cuda.is_available() or torch.version.cuda is None:
            raise RuntimeError("Parakeet Ultra requires CUDA PyTorch and a visible NVIDIA GPU")
        print(f"Photon CUDA device: {torch.cuda.get_device_name(0)}")
        moondream: Any = importlib.import_module("moondream")
        context = moondream.photon(self.model_id, device="cuda:0")
        self._context = context
        try:
            self._speech = context.__enter__()
        except BaseException:
            self._context = None
            raise

    def transcribe(self, audio: Path, hotwords: list[str]) -> TranscriptResult:
        """Return the provider result normalized into the application models."""
        if hotwords:
            raise ValueError("Parakeet Ultra does not support keyword prompting")
        duration_seconds = self._validated_wav_duration(audio)
        self.load()
        assert self._speech is not None

        result = self._speech.transcribe(audio=audio, timestamps="word")
        segments = [self._normalize_segment(segment) for segment in self._field(result, "segments", [])]
        return TranscriptResult(
            text=str(self._field(result, "text", "")),
            model_id=self.model_id,
            revision=None,
            duration_seconds=duration_seconds,
            segments=segments,
            timestamp_precision="word" if any(s.words for s in segments) else "segment" if segments else "none",
            metadata={
                "input_sample_rate": self.sample_rate,
                "hotwords_supported": self.supports_hotwords,
                "device": "cuda:0",
                "segmentation": "photon-internal-vad",
            },
        )

    def close(self) -> None:
        """Release the Photon context after a batch."""
        if self._context is None:
            return

        context, self._context, self._speech = self._context, None, None
        context.__exit__(None, None, None)

    @classmethod
    def _validated_wav_duration(cls, audio: Path) -> float:
        if audio.suffix.lower() != ".wav":
            raise ValueError("Parakeet requires a 16 kHz WAV input")
        try:
            with wave.open(str(audio), "rb") as input_wav:
                sample_rate = input_wav.getframerate()
                duration_seconds = input_wav.getnframes() / sample_rate
        except (FileNotFoundError, wave.Error, ZeroDivisionError) as error:
            raise ValueError("Parakeet requires a readable 16 kHz WAV input") from error
        if sample_rate != cls.sample_rate:
            raise ValueError("Parakeet requires a 16 kHz WAV input")
        return duration_seconds

    @classmethod
    def _normalize_segment(cls, segment: Any) -> Segment:
        return Segment(
            text=str(cls._field(segment, "text", "")),
            start=cls._timestamp(segment, "start"),
            end=cls._timestamp(segment, "end"),
            words=[cls._normalize_word(word) for word in cls._field(segment, "words", [])],
        )

    @classmethod
    def _normalize_word(cls, word: Any) -> Word:
        return Word(
            text=str(cls._field(word, "word", cls._field(word, "text", ""))),
            start=cls._timestamp(word, "start"),
            end=cls._timestamp(word, "end"),
        )

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _timestamp(cls, value: Any, name: str) -> float | None:
        timestamp = cls._field(value, name)
        return None if timestamp is None else float(timestamp)
