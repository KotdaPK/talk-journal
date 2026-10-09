"""FFmpeg decode and bounded one-file-at-a-time sample-rate conversion."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mov", ".mkv", ".webm", ".avi"}


def decode(source: Path, destination: Path, sample_rate: int) -> None:
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
                    "-vn", "-ac", "1", "-ar", str(sample_rate),
                    "-c:a", "pcm_s16le", str(destination)], check=True)
