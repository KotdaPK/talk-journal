"""Legacy Granite speech backend, retaining Silero VAD and prompting."""
from __future__ import annotations

from pathlib import Path

import soundfile as sf

from talk_journal.models import Segment, TranscriptResult


MODEL_ID = "ibm-granite/granite-4.0-1b-speech"
MODEL_REVISION = "bd87ab862416353633ea431fe49b1614003623c5"
SAMPLE_RATE = 16_000
MAX_SPEECH_SECONDS = 55


def build_prompt(tokenizer, keywords: list[str]) -> str:
    user_prompt = "<|audio|>can you transcribe the speech into a written format?"
    if keywords:
        user_prompt += "\nKeywords: " + ", ".join(keywords)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True,
    )


def transcribe_segment(segment, model, processor, prompt: str, device: str) -> str:
    import torch
    model_inputs = processor(prompt, segment, device=device, return_tensors="pt").to(device)
    with torch.inference_mode():
        model_outputs = model.generate(
            **model_inputs, max_new_tokens=512, do_sample=False, num_beams=1,
        )
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
    return processor.tokenizer.batch_decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True,
    )[0].strip()


class GraniteBackend:
    model_id = MODEL_ID
    revision = MODEL_REVISION
    sample_rate = SAMPLE_RATE
    supports_hotwords = True

    def __init__(self):
        self.model = self.processor = self.vad_model = None
        self.device = "cuda"

    def load(self) -> None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Granite requires CUDA; install CUDA PyTorch and check the NVIDIA driver")
        try:
            from silero_vad import load_silero_vad
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Granite dependencies missing; rerun installer with --models parakeet,granite") from exc
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=self.revision)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, revision=self.revision, device_map=self.device, dtype=torch.bfloat16,
        )
        self.vad_model = load_silero_vad()

    def transcribe(self, audio: Path, hotwords: list[str]) -> TranscriptResult:
        import torch
        from silero_vad import get_speech_timestamps
        if self.model is None or self.processor is None or self.vad_model is None:
            raise RuntimeError("Granite is not loaded")
        wav, sample_rate = sf.read(audio, dtype="float32")
        if sample_rate != SAMPLE_RATE or wav.ndim != 1:
            raise RuntimeError("Granite needs 16 kHz mono audio")
        mono = torch.from_numpy(wav)
        timestamps = get_speech_timestamps(
            mono, self.vad_model, sampling_rate=SAMPLE_RATE,
            max_speech_duration_s=MAX_SPEECH_SECONDS, return_seconds=False,
        )
        prompt = build_prompt(self.processor.tokenizer, hotwords)
        segments = []
        for timestamp in timestamps:
            start, end = timestamp["start"], timestamp["end"]
            text = transcribe_segment(mono[start:end].unsqueeze(0), self.model,
                                      self.processor, prompt, self.device)
            if text:
                segments.append(Segment(text=text, start=start / SAMPLE_RATE,
                                        end=end / SAMPLE_RATE))
        return TranscriptResult(
            text="\n\n".join(segment.text for segment in segments),
            model_id=MODEL_ID, revision=self.revision,
            duration_seconds=len(mono) / SAMPLE_RATE, segments=segments,
            timestamp_precision="segment" if segments else "none",
            metadata={"segmentation": "silero-vad", "device": self.device},
        )

    def close(self) -> None:
        self.model = self.processor = self.vad_model = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
