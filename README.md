# Talk Journal

Offline-inference speech-to-text for local audio and video, using [IBM Granite 4.0 1B Speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) and Silero VAD. Files stay on your machine; the model is fetched from Hugging Face on first use. Transcripts are plain UTF-8 text.

## Windows (PowerShell)

Clone or download this repository, then from its root run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

The installer sets up a local `.venv` without administrator privileges, creates `recordings/` and `raw-text/`, and transcribes any supported media already in `recordings/` to matching `.txt` files in `raw-text/`. Put new audio/video files in `recordings/`, then run:

```powershell
.\run.ps1
```

If your execution policy blocks direct `.ps1` invocation, use `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1`. Pass `--recursive` to include subfolders or `--overwrite` to regenerate transcripts. Both scripts resolve folders relative to their own location, not the shell's working directory. The first use downloads the Granite model (about 4.6 GB); CPU inference can be slow. The scripts require an Internet connection for setup and initial model download. Neither recordings nor transcripts are tracked by Git.

## Install (Linux x86-64)

```bash
curl -fsSL https://raw.githubusercontent.com/KotdaPK/talk-journal/main/install.sh | bash
```

This installs `uv` if needed, a Python 3.12 environment, CPU PyTorch, a bundled ffmpeg, and `talk-journal` under `~/.local/bin`. No root privileges required. Requires `curl`, `tar`, Internet access for installation and the first model download, and roughly 8 GB free for installation/model cache. Review `install.sh` before piping it to bash if you prefer. Ensure `~/.local/bin` is on your PATH; otherwise use `~/.local/bin/talk-journal`.

## Run

```bash
talk-journal "/path/to/recording.m4a"
```

A single file creates `/path/to/recording.txt` beside the original. A folder is one command too:

```bash
talk-journal "/path/to/recordings"
```

This creates `.txt` files under `/path/to/recordings/transcripts/`. To choose an output folder, pass it as the second argument. Add `--recursive` to traverse nested folders (their layout is preserved); use `--overwrite` to reprocess existing transcripts. `--keywords PostgreSQL FastAPI` adds recognition hints.

Existing transcripts are skipped *before* the model loads. An output-name collision, e.g. `meeting.mp3` and `meeting.m4a` in the same folder, is rejected without overwriting anything. Individual failures return a nonzero exit status while allowing the rest of a folder batch to finish. Audio-only source data is never part of this repository.

## Development

```bash
python -m unittest discover -s tests -v
```

The installer currently targets CPU inference. To use a GPU, install a matching CUDA build of PyTorch in the installed virtual environment and check that `torch.cuda.is_available()` returns true. The original script remains callable as `python transcribe_folder.py INPUT [OUTPUT]`.
