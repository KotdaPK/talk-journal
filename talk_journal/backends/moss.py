"""CUDA-only native MOSS transcription and diarization backend."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from talk_journal.models import Segment, TranscriptResult


MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
MODEL_REVISION = "704aa4a9c304e8520be88901e0d1960158ef5b15"
DEFAULT_PROMPT = (
    "Transcribe the audio. For each segment, start with the timestamp "
    "and speaker ID ([S01], [S02], [S03], ...), then the spoken text, "
    "and end with the segment timestamp."
)


class MOSSBackend:
    """Use MOSS's own Transformers helpers without CPU fallback.

    A recording is first submitted as one native long-form request.  Only a CUDA
    OOM or a generation-limit result retries as overlapping WAV chunks; chunk
    speaker identifiers are deliberately scoped to the chunk because independent
    requests cannot establish a reliable global speaker identity.
    """

    model_id = MODEL_ID
    revision = MODEL_REVISION
    sample_rate = 16_000
    supports_hotwords = True

    def __init__(
        self,
        *,
        max_new_tokens: int = 8_192,
        chunk_seconds: float = 10 * 60,
        overlap_seconds: float = 30,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if chunk_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
            raise ValueError("require 0 <= overlap_seconds < chunk_seconds")
        self.max_new_tokens = max_new_tokens
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self._torch: Any | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device: Any | None = None
        self._dtype: Any | None = None
        self._build_messages: Any | None = None
        self._generate: Any | None = None
        self._parse: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("MOSS transcription requires CUDA; CPU fallback is disabled.")

        from moss_transcribe_diarize import parse_transcript
        from moss_transcribe_diarize.inference_utils import (
            build_transcription_messages,
            generate_transcription,
            resolve_device,
        )
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = resolve_device("cuda")
        if device.type != "cuda":
            raise RuntimeError("MOSS transcription requires CUDA; CUDA device resolution failed.")
        dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            revision=self.revision,
            dtype="auto",
            attn_implementation="sdpa",
        )
        model.to(dtype=dtype)
        model.to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, revision=self.revision)

        self._torch = torch
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype
        self._build_messages = build_transcription_messages
        self._generate = generate_transcription
        self._parse = parse_transcript

    def transcribe(self, audio: Path, hotwords: list[str]) -> TranscriptResult:
        self._require_loaded()
        source = Path(audio)
        prompt = self._prompt(hotwords)
        try:
            payload = self._transcribe_one(source, prompt)
        except RuntimeError as exc:
            if not self._is_cuda_oom(exc):
                raise
            self._torch.cuda.empty_cache()
            return self._adaptive_chunks(source, prompt, cause="cuda-oom")

        if self._is_truncated(payload):
            return self._adaptive_chunks(source, prompt, cause="generation-limit")
        try:
            native = self._result_from_payload(payload, offset=0.0)
        except RuntimeError as exc:
            if "parse_transcript" not in str(exc):
                raise
            return self._adaptive_chunks(source, prompt, cause="invalid-native-output")
        import soundfile as sf
        source_duration = sf.info(source).duration
        if source_duration > 120 and (not native.segments or native.segments[-1].end is None
                                      or source_duration - native.segments[-1].end > 90):
            return self._adaptive_chunks(source, prompt, cause="incomplete-native-coverage")
        native.duration_seconds = source_duration
        return native

    def _adaptive_chunks(self, source: Path, prompt: str, *, cause: str) -> TranscriptResult:
        while True:
            try:
                return self._transcribe_chunks(source, prompt, cause=cause)
            except RuntimeError as exc:
                recoverable = ("out of CUDA memory even after overlap" in str(exc)
                               or "chunk output reached max_new_tokens" in str(exc))
                if not recoverable or self.chunk_seconds <= 60:
                    raise
                self._torch.cuda.empty_cache()
                self.chunk_seconds = max(60, self.chunk_seconds / 2)
                print(f"MOSS fallback retry with {self.chunk_seconds:g}-second windows")

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None
        self._build_messages = None
        self._generate = None
        self._parse = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None

    def _transcribe_one(self, audio: Path, prompt: str) -> dict[str, Any]:
        assert self._build_messages is not None
        assert self._generate is not None
        messages = self._build_messages(str(audio), prompt=prompt)
        return self._generate(
            self._model,
            self._processor,
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            device=self._device,
            dtype=self._dtype,
        )

    def _transcribe_chunks(self, source: Path, prompt: str, *, cause: str) -> TranscriptResult:
        segments: list[Segment] = []
        raw_parts: list[str] = []
        with TemporaryDirectory(prefix="talk-journal-moss-") as temporary:
            chunks = self._decode_chunks(source, Path(temporary))
            if not chunks:
                raise RuntimeError("MOSS fallback could not decode any audio chunks.")
            for chunk_index, (chunk_path, offset, chunk_end) in enumerate(chunks):
                try:
                    payload = self._transcribe_one(chunk_path, prompt)
                except RuntimeError as exc:
                    if self._is_cuda_oom(exc):
                        raise RuntimeError(
                            "MOSS ran out of CUDA memory even after overlap chunk fallback."
                        ) from exc
                    raise
                if self._is_truncated(payload):
                    raise RuntimeError(
                        "MOSS chunk output reached max_new_tokens; refusing a silently truncated transcript."
                    )
                chunk_result = self._result_from_payload(payload, offset=offset)
                if not chunk_result.segments:
                    raise RuntimeError("MOSS produced no parseable timestamped segments for a fallback chunk.")
                owned_start = offset if chunk_index == 0 else offset + self.overlap_seconds / 2
                owned_end = chunk_end if chunk_index == len(chunks) - 1 else chunk_end - self.overlap_seconds / 2
                for segment in chunk_result.segments:
                    midpoint = (segment.start + segment.end) / 2
                    if owned_start <= midpoint < owned_end or (chunk_index == len(chunks) - 1 and midpoint == owned_end):
                        segment.speaker = f"chunk-{chunk_index:03d}:{segment.speaker}" if segment.speaker else None
                        segments.append(segment)
                raw_parts.append(str(payload.get("text", "")))

        duration = chunks[-1][2]
        return TranscriptResult(
            text="\n".join(segment.text for segment in segments),
            model_id=self.model_id,
            revision=self.revision,
            duration_seconds=duration,
            segments=segments,
            timestamp_precision="segment",
            metadata={"fallback": "overlap-chunks", "fallback_cause": cause, "diarization_scope": "chunk-local", "raw_transcript": "\n".join(raw_parts)},
        )

    def _result_from_payload(self, payload: dict[str, Any], *, offset: float) -> TranscriptResult:
        assert self._parse is not None
        raw_text = str(payload.get("text", ""))
        parsed = self._parse(raw_text)
        segments = [
            Segment(
                text=str(item.text).strip(),
                start=float(item.start) + offset,
                end=float(item.end) + offset,
                speaker=str(item.speaker),
            )
            for item in parsed
            if str(item.text).strip()
        ]
        if any(segment.start is None or segment.end is None or segment.start < 0
               or segment.end < segment.start for segment in segments):
            raise RuntimeError("MOSS returned invalid segment timestamps")
        if raw_text.strip() and not segments:
            raise RuntimeError("MOSS returned text that parse_transcript could not normalize into segments.")
        duration = max((segment.end or 0.0 for segment in segments), default=0.0)
        return TranscriptResult(
            text="\n".join(segment.text for segment in segments),
            model_id=self.model_id,
            revision=self.revision,
            duration_seconds=duration,
            segments=segments,
            timestamp_precision="segment" if segments else "none",
            metadata={"raw_transcript": raw_text, "generated_tokens": payload.get("generated_tokens")},
        )

    def _decode_chunks(self, source: Path, temporary: Path) -> list[tuple[Path, float, float]]:
        """Decode only for fallback and emit overlapping mono WAV chunks."""
        import soundfile as sf
        info = sf.info(source)
        if info.samplerate != self.sample_rate or info.channels != 1:
            raise RuntimeError("MOSS fallback requires decoded 16 kHz mono WAV")
        total_samples = info.frames
        chunk_samples = int(self.chunk_seconds * self.sample_rate)
        step_samples = int((self.chunk_seconds - self.overlap_seconds) * self.sample_rate)
        chunks: list[tuple[Path, float, float]] = []
        for index, start in enumerate(range(0, total_samples, step_samples)):
            end = min(start + chunk_samples, total_samples)
            destination = temporary / f"chunk-{index:03d}.wav"
            with sf.SoundFile(source) as input_audio:
                input_audio.seek(start)
                chunk = input_audio.read(end - start, dtype="float32")
            sf.write(destination, chunk, self.sample_rate)
            chunks.append((destination, start / self.sample_rate, end / self.sample_rate))
            if end == total_samples:
                break
        return chunks

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("MOSS backend is not loaded; call load() before transcribe().")

    def _is_truncated(self, payload: dict[str, Any]) -> bool:
        generated = payload.get("generated_tokens")
        return isinstance(generated, int) and generated >= self.max_new_tokens

    @staticmethod
    def _is_cuda_oom(error: RuntimeError) -> bool:
        message = str(error).lower()
        return "out of memory" in message and ("cuda" in message or "cudnn" in message)

    @staticmethod
    def _prompt(hotwords: list[str]) -> str:
        words = [word.strip() for word in hotwords if word and word.strip()]
        return DEFAULT_PROMPT if not words else f"{DEFAULT_PROMPT} Hotwords: {', '.join(words)}."
