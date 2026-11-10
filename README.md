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

## Offline editorial handoff (synthetic acceptance boundary)

The existing CLI can consume a reviewed, externally authored JSON editorial packet.
It does not call Codex, embed Hermes, infer prose, or download anything. An authorized
native Codex session reads the source and current full pages, writes a dossier and
complete replacement pages, and hands that packet to this deterministic consumer.
The `author` field is a declaration, not cryptographic proof of a provider response.
Tests' small inline packets are fixtures, not captured responses. The two packets
in `tests/fixtures/editorial/` were authored by the native
`openai-codex/gpt-6.1-sol-900k` engineering session from the accompanying fictional
conversations; their prose, attribution and uncertainty are editorial work, while
hashing, provenance footers, indexing and diff rendering are deterministic formatting.
No real conversation, provider trace or private vault is included.

From this checkout, using the already installed environment:

```bash
python -m talk_journal.cli editorial tests/fixtures/editorial/source-1.txt tests/fixtures/editorial/packet-1.json /your/private/scratch/wiki-1
# Exact replay prints NOOP and changes no files or timestamps:
python -m talk_journal.cli editorial tests/fixtures/editorial/source-1.txt tests/fixtures/editorial/packet-1.json /your/private/scratch/wiki-1
# Overlapping new source creates a separate complete snapshot; wiki-1 is untouched:
python -m talk_journal.cli editorial tests/fixtures/editorial/source-2.txt tests/fixtures/editorial/packet-2.json /your/private/scratch/wiki-2 --base /your/private/scratch/wiki-1
```

`talk-journal editorial ...` is the equivalent installed entry point after publication
and normal installation; merely editing this checkout does not update installations.
For a media directory literally named `editorial`, use `./editorial` with the original
transcription command. No manifest dependency or original transcription output format
has changed. The text handoff records exact transcript byte identity; ASR identity,
speaker identity and timing not supplied by that text are not invented. Keep the
original transcript JSON/completion metadata alongside the original text when needed.

Packet schema version 1 has exactly these fields (see the executable examples):

- `source_sha256`: SHA-256 of the exact UTF-8 transcript bytes.
- `author`: `openai-codex/gpt-6.1-sol-900k` declaration.
- `dossier`: attributed editorial narrative, not an extraction template.
- `uncertainty`: nonempty list of unresolved claims, or an explicit reviewable statement
  that none was identified. The consumer cannot verify that judgement.
- `quotes`: nonempty verbatim source quotations supporting the edits. Presence is
  checked, but entailment, completeness and correct speaker interpretation need review.
- `pages`: nonempty edits with `path`, `before_sha256`, and complete `markdown`.
  The before-image is the hash of the entire rendered previous page, including its
  provenance footer; use null only for a new page. Retain relevant earlier meaning,
  attribution and uncertainty when writing the replacement; overlap is editorial,
  not a keyword merge. Unedited pages are carried forward unchanged.

Paths are lowercase ASCII basenames such as `rooftop-trial.md`; `index.md` and
`review.md` are reserved. Supported authored Markdown is deliberately narrow: ordinary
text, headings, lists and `[label](existing-page.md)` links to included pages.
HTML, embeds/images, external/reference/wiki links, frontmatter, backslashes and
control characters are refused. Dossier/uncertainty/quotes are plain text without
links. This conservative subset may reject legitimate source quotations; choose a
safe verbatim excerpt or stop for review, never alter the raw source to satisfy it.

Open `index.md` to find pages, follow each page's dossiers to immutable source text,
and open `review.md` for the current dossier, uncertainty, complete before/after pages
and unified diffs. Prior snapshots retain their previous review. New snapshots carry
all historical dossiers and source bytes, and each edited page links every contributing
source. Same transcript bytes plus the same packet are a no-op; a different packet
for an already consumed source is a conflict, not silent re-ingestion. This content
identity intentionally deduplicates identical bytes under different filenames, not
paraphrases. No automatic semantic duplicate detector is claimed.

Destinations must be task-owned private scratch, never a live vault. Updates require
a new destination plus `--base`; there is no in-place apply or automatic vault sync.
The base must have an intact completion manifest and exact tracked artifact inventory;
changed pages, symlinks, unexpected files, unsafe paths and stale before-images fail
closed. A completion manifest is written last. An interrupted write can leave an
incomplete destination, which is not a valid base; preserve it for diagnosis and use
a fresh path. Single-writer ownership is required: filesystem races, concurrent
hostile writers and authenticated provenance are not solved by these hashes.

Before accepting ingestion, review full pages plus diffs against all contributing
sources: fidelity, attribution, unresolved facts, contradictions, bounded approvals,
and retrieval/navigation. The fictional overlap example deliberately preserves an
unsettled budget/currency and weather condition while updating Bo's approval; it
does not turn a greenhouse into approved scope. Unit and scratch CLI checks prove
source/synthetic behavior only. Ordinary Obsidian opening/navigation, private-data
permission and human source-fidelity acceptance remain separate and unperformed.

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
