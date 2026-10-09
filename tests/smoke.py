"""Opt-in real CUDA inference smoke: python tests/smoke.py MEDIA --model parakeet."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from talk_journal.pipeline import process  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real inference on one local media file")
    parser.add_argument("media", type=Path)
    parser.add_argument("--model", choices=("parakeet", "moss", "granite"), default="parakeet")
    parser.add_argument("--preprocess", choices=("none", "deepfilternet"), default="deepfilternet")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="talk-journal-smoke-") as temp:
        completed, failed, _ = process(args.media, Path(temp), model=args.model, preprocess=args.preprocess)
        if completed != 1 or failed:
            return 1
        transcripts = list(Path(temp).rglob("transcript.json"))
        if len(transcripts) != 1:
            raise AssertionError(f"Expected one structured transcript, found {len(transcripts)}")
        transcript = json.loads(transcripts[0].read_text(encoding="utf-8"))
        marker = json.loads((transcripts[0].parent / "metadata.json").read_text(encoding="utf-8"))
        assert marker["state"] == "complete" and transcript["model_id"] == marker["model_id"]
        assert transcript["duration_seconds"] > 0
        print(f"PASS {args.model}+{args.preprocess}: {transcript['duration_seconds']:.1f}s, "
              f"{len(transcript['segments'])} segments; text={transcript['text'][:120]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
