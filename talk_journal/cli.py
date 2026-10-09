"""Talk Journal command line."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from talk_journal.config import BACKENDS, PREPROCESSORS
from talk_journal.pipeline import process


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local, GPU-only speech transcription")
    parser.add_argument("input_path", type=Path, nargs="?", help="Media file or directory")
    parser.add_argument("output_dir", type=Path, nargs="?", help="Output root")
    parser.add_argument("--model", choices=BACKENDS, default="parakeet")
    parser.add_argument("--preprocess", choices=PREPROCESSORS, default="deepfilternet")
    parser.add_argument("--highpass-hz", type=int, default=None)
    parser.add_argument("--keywords", nargs="*", default=[])
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-preprocessors", action="store_true")
    args = parser.parse_args(argv)
    if args.list_models:
        print("\n".join(BACKENDS))
        return 0
    if args.list_preprocessors:
        print("\n".join(PREPROCESSORS))
        return 0
    if args.input_path is None:
        parser.error("input_path is required unless listing backends")
    if args.model == "moss":
        suffix = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        moss_env = ".venv-moss"
        moss_python = Path(sys.prefix).resolve().parent / moss_env / suffix
        if Path(sys.prefix).resolve() != moss_python.parent.parent.resolve():
            if not moss_python.is_file():
                parser.exit(1, "MOSS is not installed; rerun install with --models parakeet,moss\n")
            return subprocess.run([str(moss_python), "-m", "talk_journal.cli", *(argv if argv is not None else sys.argv[1:])], check=False).returncode
    try:
        _, failures, _ = process(args.input_path, args.output_dir,
                                  model=args.model, preprocess=args.preprocess,
                                  highpass_hz=args.highpass_hz,
                                  keywords=args.keywords, recursive=args.recursive,
                                  overwrite=args.overwrite)
    except (ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Talk Journal: {exc}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
