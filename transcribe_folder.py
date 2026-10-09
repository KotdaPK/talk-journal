from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


MODEL_NAME = "ibm-granite/granite-4.0-1b-speech"
SAMPLE_RATE = 16_000
MAX_SPEECH_SECONDS = 55

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe every media file in a folder with Granite Speech."
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Audio/video file or folder containing media.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="Optional transcript folder (default: beside a file or in folder/transcripts).",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also process files inside subdirectories.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-transcribe files even when the transcript already exists.",
    )

    parser.add_argument(
        "--keywords",
        nargs="*",
        default=[],
        help="Optional Granite keyword-biasing terms.",
    )

    return parser.parse_args()


def find_media_files(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")

    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def transcript_path(
    source: Path,
    input_dir: Path,
    output_dir: Path,
    recursive: bool,
) -> Path:
    if recursive:
        relative = source.relative_to(input_dir)
        return output_dir / relative.with_suffix(".txt")

    return output_dir / f"{source.stem}.txt"


def ffmpeg_executable() -> str:
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

def convert_to_wav(source: Path, destination: Path):
    command = [
        ffmpeg_executable(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]

    subprocess.run(command, check=True)


def build_prompt(tokenizer, keywords: list[str]) -> str:
    user_prompt = (
        "<|audio|>can you transcribe the speech into a written format?"
    )

    if keywords:
        user_prompt += "\nKeywords: " + ", ".join(keywords)

    chat = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )


def transcribe_segment(
    segment: torch.Tensor,
    model,
    processor,
    prompt: str,
    device: str,
) -> str:
    model_inputs = processor(
        prompt,
        segment,
        device=device,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        model_outputs = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
        )

    num_input_tokens = model_inputs["input_ids"].shape[-1]

    new_tokens = model_outputs[
        0,
        num_input_tokens:
    ].unsqueeze(0)

    text = processor.tokenizer.batch_decode(
        new_tokens,
        add_special_tokens=False,
        skip_special_tokens=True,
    )[0]

    return text.strip()


def transcribe_file(
    source: Path,
    model,
    processor,
    vad_model,
    prompt: str,
    device: str,
) -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"

        convert_to_wav(source, wav_path)

        wav, sample_rate = sf.read(wav_path, dtype="float32")

        if sample_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"Unexpected sample rate: {sample_rate}"
            )

        # Silero expects a 1-D waveform.
        mono = torch.from_numpy(wav)

        timestamps = get_speech_timestamps(
            mono,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            max_speech_duration_s=MAX_SPEECH_SECONDS,
            return_seconds=False,
        )

        if not timestamps:
            return ""

        transcript_parts = []

        for index, timestamp in enumerate(timestamps, start=1):
            start = timestamp["start"]
            end = timestamp["end"]

            # Granite's example uses [channels, samples].
            segment = mono[start:end].unsqueeze(0)

            duration = (end - start) / SAMPLE_RATE

            print(
                f"    segment {index}/{len(timestamps)} "
                f"({duration:.1f}s)"
            )

            text = transcribe_segment(
                segment=segment,
                model=model,
                processor=processor,
                prompt=prompt,
                device=device,
            )

            if text:
                transcript_parts.append(text)

    return "\n\n".join(transcript_parts)


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    temporary_path.write_text(
        text,
        encoding="utf-8",
    )

    # Atomic replacement on the same filesystem.
    os.replace(temporary_path, path)


def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        device_map=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    print("Loading Silero VAD...")
    vad_model = load_silero_vad()
    return model, processor, vad_model, device


def main():
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise SystemExit(f"Unsupported media file: {input_path}")
        output_dir = (args.output_dir or input_path.parent).expanduser().resolve()
        input_dir = input_path.parent
        files = [input_path]
        recursive = False
    elif input_path.is_dir():
        input_dir = input_path
        output_dir = (args.output_dir or input_dir / "transcripts").expanduser().resolve()
        files = find_media_files(input_dir, args.recursive)
        recursive = args.recursive
    else:
        raise SystemExit(f"Input does not exist: {input_path}")

    if not files:
        print("No supported media files found.")
        return

    destinations = {}
    for source in files:
        destination = transcript_path(source, input_dir, output_dir, recursive)
        if destination in destinations:
            raise SystemExit(
                f"Transcript collision: {destination} ({destinations[destination]} and {source})"
            )
        destinations[destination] = source

    jobs = []
    for source in files:
        destination = transcript_path(source, input_dir, output_dir, recursive)
        if destination.exists() and not args.overwrite:
            print(f"SKIP: {source.name} -> {destination}")
            continue
        jobs.append((source, destination))
    if not jobs:
        print("Nothing to do. All files are already transcribed.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor, vad_model, device = load_models()
    prompt = build_prompt(
        processor.tokenizer,
        args.keywords,
    )

    completed = 0
    failed = 0

    for index, (source, destination) in enumerate(
        jobs,
        start=1,
    ):
        print(
            f"\n[{index}/{len(jobs)}] "
            f"{source}"
        )

        try:
            transcript = transcribe_file(
                source=source,
                model=model,
                processor=processor,
                vad_model=vad_model,
                prompt=prompt,
                device=device,
            )

            atomic_write(
                destination,
                transcript,
            )

            print(
                f"  DONE -> {destination}"
            )

            completed += 1

        except Exception as error:
            # No final .txt is created on failure,
            # so rerunning will retry this file.
            print(
                f"  FAILED: {error}"
            )

            failed += 1

    print("\nFinished")
    print(f"  Transcribed: {completed}")
    print(f"  Failed:      {failed}")
    print(f"  Skipped:     {len(files) - len(jobs)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
