#!/usr/bin/env bash
set -euo pipefail
# User-scoped CUDA installation. --models extends an existing installation.
models=parakeet
while (($#)); do
    case "$1" in
        --models) models="${2:?missing model list}"; shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case ",$models," in *,,*) printf 'Empty model name\n' >&2; exit 2;; esac
IFS=, read -ra choices <<< "$models"
for selected in "${choices[@]}"; do
    case "$selected" in parakeet|moss|granite) ;; *) printf 'Unknown model: %s\n' "$selected" >&2; exit 2;; esac
done
repo=KotdaPK/talk-journal
root="${TALK_JOURNAL_INSTALL_ROOT:-$HOME/.local/share/talk-journal}"
bin_dir="${TALK_JOURNAL_BIN_DIR:-$HOME/.local/bin}"
for cmd in curl tar; do command -v "$cmd" >/dev/null || { printf 'Missing %s\n' "$cmd" >&2; exit 1; }; done
if command -v uv >/dev/null; then uv_bin="$(command -v uv)";
elif [ -x "$HOME/.local/bin/uv" ]; then uv_bin="$HOME/.local/bin/uv";
else curl -LsSf https://astral.sh/uv/install.sh | sh; uv_bin="$HOME/.local/bin/uv"; fi
"$uv_bin" python install 3.11
mkdir -p "$root" "$bin_dir"
source_dir="${TALK_JOURNAL_SOURCE_DIR:-}"
if [ -z "$source_dir" ]; then
    scratch="$(mktemp -d)"
    trap 'rm -rf "$scratch"' EXIT
    mkdir -p "$scratch/source"
    curl -fsSL "https://github.com/$repo/archive/refs/heads/main.tar.gz" | tar -xz -C "$scratch/source" --strip-components=1
    source_dir="$scratch/source"
fi
if [ ! -x "$root/venv-default/bin/python" ]; then "$uv_bin" venv --python 3.11 "$root/venv-default"; fi
python="$root/venv-default/bin/python"
"$uv_bin" pip install --python "$python" 'torch==2.11.0+cu128' 'torchaudio==2.11.0+cu128' --index https://download.pytorch.org/whl/cu128
"$uv_bin" pip install --python "$python" "$source_dir"
for selected in "${choices[@]}"; do
    case "$selected" in
        granite) "$uv_bin" pip install --python "$python" "$source_dir[granite]" ;;
        moss)
            if [ ! -x "$root/.venv-moss/bin/python" ]; then "$uv_bin" venv --python 3.11 "$root/.venv-moss"; fi
            moss="$root/.venv-moss/bin/python"
            "$uv_bin" pip install --python "$moss" 'torch==2.11.0+cu128' 'torchaudio==2.11.0+cu128' --index https://download.pytorch.org/whl/cu128
            "$uv_bin" pip install --python "$moss" "$source_dir"
            "$uv_bin" pip install --python "$moss" "$source_dir[moss]"
            "$uv_bin" pip check --python "$moss"
            "$moss" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.version.cuda else 1)' || { printf 'MOSS CUDA is unavailable\n' >&2; exit 1; }
            ;;
    esac
done
"$uv_bin" pip check --python "$python"
"$python" -c 'import torch,sys; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"); sys.exit(0 if torch.cuda.is_available() and torch.version.cuda else 1)'
ln -sfn "$root/venv-default/bin/talk-journal" "$bin_dir/talk-journal"
printf 'Installed: %s/talk-journal (models=%s)\n' "$bin_dir" "$models"
