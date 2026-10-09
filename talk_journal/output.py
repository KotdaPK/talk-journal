"""Configuration-aware result layout and atomic completion manifests."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

from talk_journal.models import TranscriptResult

SCHEMA_VERSION = 1


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def paths(output_root: Path, relative: Path, config: dict) -> Path:
    # Keep the complete filename, including extension, to distinguish a.wav from a.m4a.
    return output_root / relative / f"{config['model']}-{config['preprocess']}-{signature(config)[:16]}"


def manifest_matches(directory: Path, identity: dict, config: dict) -> bool:
    marker = directory / "metadata.json"
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
        if manifest.get("state") != "complete" or manifest.get("source") != identity or manifest.get("configuration") != config or manifest.get("schema_version") != SCHEMA_VERSION:
            return False
        if not (directory / "transcript.txt").read_text(encoding="utf-8").strip():
            return False
        return all(
            (directory / name).is_file() and digest_file(directory / name) == manifest["artifacts"][name]
            for name in ("transcript.txt", "transcript.json")
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save(directory: Path, identity: dict, config: dict, result: TranscriptResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "metadata.json"
    marker.unlink(missing_ok=True)  # prior completion is invalid during any rewrite
    atomic_text(directory / "transcript.txt", result.text)
    atomic_text(directory / "transcript.json", json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete", "source": identity, "configuration": config,
        "model_id": result.model_id, "model_revision": result.revision,
        "timestamp_precision": result.timestamp_precision,
        "artifacts": {name: digest_file(directory / name) for name in ("transcript.txt", "transcript.json")},
    }
    atomic_text(marker, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
