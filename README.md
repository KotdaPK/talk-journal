# Talk Journal

Local GPU-only speech transcription for audio/video journals. **Default:** FFmpeg → 48 kHz mono → DeepFilterNet3 → 16 kHz mono → [Moondream Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) via official Photon (built-in VAD and pause-aware long-form segmentation). High-pass filtering is off by default. No API key or cloud inference; installation and the first model download require Internet access. Later inference works offline with cached weights. Original recordings remain untouched.

## Install

Windows 11 PowerShell, from the repository checkout:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 --models parakeet,moss,granite
.\run.ps1
.\run.ps1 --model moss --preprocess none --keywords Kotda
```

The installer sets up user-scoped `uv`, Python 3.11 in `.venv-default`, matched CUDA 12.8 PyTorch wheels, Parakeet and DeepFilterNet3; adds Granite to that environment if selected; and sets up a separate Python 3.11 `.venv-moss` for MOSS if selected. It retains any historical `.venv`. The installer runs `run.ps1` once, processing `recordings/` into `raw-text/`; subsequent invocations can use `run.ps1` with CLI flags. The first run may download model weights. Both installers check package compatibility and require a working CUDA/NVIDIA driver; CPU-only PyTorch is rejected. The optional MOSS source comes from the official upstream GitHub repository and its Hugging Face checkpoint uses remote Python code; review and pin its revision before relying on it in a high-trust environment.

Linux/WSL2:

```bash
curl -fsSL https://raw.githubusercontent.com/KotdaPK/talk-journal/main/install.sh | bash
# From a checkout instead: TALK_JOURNAL_SOURCE_DIR="$PWD" ./install.sh --models parakeet,moss,granite
```

This installs under `~/.local/share/talk-journal`, linking `~/.local/bin/talk-journal`. No root needed. Requires `curl`, `tar`, Internet for initial setup, sufficient disk, and an NVIDIA GPU with compatible driver. For a local checkout, prefer the second command while developing unpublished changes. On Windows and Linux MOSS gets its own environment because it needs Transformers 5.x while Granite needs 4.x. Installing additional backends does not remove existing default weights or saved results.

## Run and compare

```bash
talk-journal --list-models
talk-journal --list-preprocessors
talk-journal --help
talk-journal recordings                          # Parakeet + DeepFilterNet3
talk-journal recordings --model parakeet --preprocess none
talk-journal recordings --model parakeet --preprocess deepfilternet
talk-journal recordings --model moss --preprocess none
talk-journal recordings --model moss --preprocess deepfilternet
talk-journal recordings --model granite --preprocess none --keywords Kotda
talk-journal recordings --highpass-hz 70         # only with deepfilternet
talk-journal recording.m4a custom-output --overwrite
talk-journal recordings custom-output --recursive
python transcribe_folder.py recordings
```

`--keywords` is supported by Granite and MOSS, not Parakeet. No unsupported hint is silently discarded. `--highpass-hz` selects an optional gentle low-frequency high-pass **before** enhancement; otherwise no extra gate/filter is applied. `none` decodes directly into the backend's sample rate. Each backend owns its own segmentation; Parakeet uses Photon VAD, Granite uses Silero, and MOSS uses native long-form processing with conservative chunk fallback when necessary. MOSS speaker labels identify anonymous speakers only within a request; fallback chunk labels are namespaced and must not be read as global identities.

Each source gets distinct results under `OUTPUT_ROOT/<relative/full.filename>/<model>-<preprocessor>-<config digest>/` with `transcript.txt`, `transcript.json` (normalized text, segments, words if available, speakers if available, timing precision, model/revision, metadata), and `metadata.json` (completion marker: source path/content SHA-256, configuration, pipeline schema, artifact SHA-256 hashes). Different model/preprocessing settings coexist for later WER comparison. Changing the source, high-pass setting, or keywords invalidates the matching completion; an unchanged second run skips enhancement and inference **before model load**. `--overwrite` redoes only the selected combination. A failed or interrupted write has no valid completion marker. Recursive paths and source extensions remain distinct.

Legacy root-level `.txt` transcripts are left untouched and never relabeled as Parakeet. Migrating them requires deliberate manual inspection; the new structured outputs are separate. Recording and generated transcript directories remain Git-ignored. Word timing precision exists only when the backend supplies it; Granite and MOSS expose segment timing, not invented word timestamps. Exit code is nonzero for failed jobs; a bad file does not block later files in a batch. One recording is decoded at a time. An empty transcript is a failure, not a completed recording: short/quiet audio can lose intelligibility under DeepFilterNet3; compare `--preprocess none` and inspect the original instead of trusting an empty result. Genuine silence also needs explicit review.

## Requirements and limitations

Parakeet Ultra/Photon requires NVIDIA CUDA and supports long encoded recordings with internal pause-aware segmentation. DeepFilterNet3 is a 48 kHz mono enhancement model; version 0.5.6 uses Python 3.11 wheels (Linux and Windows), so the default environment is Python 3.11. DeepFilterNet 0.5.6 imports a torchaudio API removed in 2.11; this application supplies a narrow in-process `df.io` adapter for already-decoded WAV files using `soundfile` rather than modifying installed vendor code. MOSS native long-form on a 12 GB RTX 4080 Laptop GPU is **not established** by its published 90-minute context; processing time, true 30–45 minute VRAM behavior, and accurate coverage must be measured on the target recording. In case of OOM/incomplete native output the chunk fallback limits context and speaker identity to each chunk; inspect metadata. For long/problematic audio, use Parakeet as default. Granite remains available with its historical prompts/VAD when its extra has been installed.

MOSS and Granite model checkpoints are pinned to exact Hugging Face revisions; MOSS's helper source is pinned to a Git commit. Photon does not expose an enforceable Parakeet revision selector in this integration: Parakeet's `revision: null` means **unknown**, not pinned. Runtime package versions participate in configuration-aware idempotence. Model/license references: [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) CC-BY-4.0 (attribution required), [MOSS](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) Apache-2.0, [Granite](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) Apache-2.0, [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) MIT OR Apache-2.0 source; pretrained weights' redistribution status is not confirmed, so do not bundle/redistribute them without clarification. Consult each model card before redistribution.

## Extend and test

`talk_journal/backends/base.py` defines `ASRBackend`; return `TranscriptResult` in `models.py` with truthful timestamp precision. Register your adapter lazily in `config.py`; implement `load`, `transcribe`, and `close` plus its audio sample rate and hotword capability. `preprocessing/base.py` has a separate contract. The central `pipeline.py` determines file layout, skip-before-load, per-file temporary audio, and manifest finalization. Tests mock providers so unit tests need no downloaded weights:

```bash
python -m unittest discover -s tests -v
./.venv-default/bin/python tests/smoke.py recordings/example.wav
./.venv-moss/bin/python tests/smoke.py recordings/example.wav --model moss --preprocess none
```

For optional backend failures, ensure the matching `--models` list was installed and run `uv pip check --python <venv-python>`. Check `torch.cuda.is_available()` in the corresponding environment. The normal CLI invokes `.venv-moss` automatically for MOSS; this is local process isolation, not a service.
