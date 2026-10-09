"""One-file-at-a-time local transcription pipeline."""
from __future__ import annotations

import tempfile
from importlib import metadata
from pathlib import Path

from talk_journal import audio, output
from talk_journal.config import BACKENDS, PREPROCESSORS, make


def media_files(source: Path, recursive: bool) -> list[tuple[Path, Path]]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in audio.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported media file: {source}")
        return [(source, Path(source.name))]
    if not source.is_dir():
        raise ValueError(f"Input does not exist: {source}")
    iterator = source.rglob("*") if recursive else source.glob("*")
    return [(file, file.relative_to(source)) for file in sorted(iterator)
            if file.is_file() and file.suffix.lower() in audio.SUPPORTED_EXTENSIONS]


def process(input_path: Path, output_dir: Path | None = None, *,
            model: str = "parakeet", preprocess: str = "deepfilternet",
            highpass_hz: int | None = None, keywords: list[str] | None = None,
            recursive: bool = False, overwrite: bool = False) -> tuple[int, int, int]:
    if model not in BACKENDS or preprocess not in PREPROCESSORS:
        raise ValueError(f"Unknown model/preprocessor: {model}/{preprocess}")
    if highpass_hz is not None and (preprocess != "deepfilternet" or not 20 <= highpass_hz <= 500):
        raise ValueError("--highpass-hz requires deepfilternet and a frequency from 20 to 500 Hz")
    keywords = keywords or []
    backend = make(BACKENDS, model)
    if keywords and not backend.supports_hotwords:
        raise ValueError(f"{model} does not support --keywords")
    if output_dir is None:
        output_dir = input_path.parent if input_path.is_file() else input_path / "transcripts"
    output_dir = output_dir.expanduser().resolve()
    def package_version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    config = {
        "model": model, "model_id": backend.model_id,
        "model_revision": getattr(backend, "revision", None),
        "preprocess": preprocess, "highpass_hz": highpass_hz,
        "keywords": keywords, "schema_version": output.SCHEMA_VERSION,
        "backend_runtime_version": package_version({"parakeet": "moondream", "moss": "moss-transcribe-diarize", "granite": "transformers"}[model]),
        "preprocessor_runtime_version": package_version("deepfilternet") if preprocess == "deepfilternet" else None,
    }
    jobs = []
    skipped = 0
    for source, relative in media_files(input_path, recursive):
        destination = output.paths(output_dir, relative, config)
        identity = {"path": str(source), "relative_path": relative.as_posix(),
                    "sha256": output.digest_file(source)}
        if not overwrite and output.manifest_matches(destination, identity, config):
            print(f"SKIP {source} -> {destination}")
            skipped += 1
        else:
            jobs.append((source, destination, identity))
    if not jobs:
        print("Nothing to do." if skipped else "No supported media files found.")
        return 0, 0, skipped
    # A selected overwrite or stale source/config is no longer a completed job,
    # including when model loading or the process is interrupted before save().
    for _, destination, _ in jobs:
        (destination / "metadata.json").unlink(missing_ok=True)
    enhancer = make(PREPROCESSORS, preprocess)
    completed = failed = 0
    print(f"model={model} preprocess={preprocess} highpass_hz={highpass_hz} jobs={len(jobs)}")
    try:
        backend.load()
        enhancer.load()
        for index, (source, destination, identity) in enumerate(jobs, 1):
            try:
                print(f"[{index}/{len(jobs)}] {source} -> {destination}")
                with tempfile.TemporaryDirectory() as tmp:
                    decoded = Path(tmp) / "decoded.wav"
                    prepared = Path(tmp) / "prepared.wav"
                    if preprocess == "deepfilternet":
                        audio.decode(source, decoded, 48_000)
                        enhancer.process(decoded, prepared, highpass_hz)
                    else:
                        audio.decode(source, decoded, backend.sample_rate)
                        enhancer.process(decoded, prepared, None)
                    if preprocess == "deepfilternet" and backend.sample_rate != 48_000:
                        resampled = Path(tmp) / "asr.wav"
                        audio.decode(prepared, resampled, backend.sample_rate)
                        prepared = resampled
                    result = backend.transcribe(prepared, keywords)
                if result.model_id != backend.model_id:
                    raise RuntimeError("Backend returned a different model identifier")
                if not result.text.strip():
                    raise RuntimeError("Backend returned an empty transcript; no completion recorded. Try --preprocess none and inspect the source audio.")
                output.save(destination, identity, config, result)
                completed += 1
            except Exception as exc:
                (destination / "metadata.json").unlink(missing_ok=True)
                print(f"FAILED {source}: {exc}")
                failed += 1
    finally:
        enhancer.close()
        backend.close()
    return completed, failed, skipped
