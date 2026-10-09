#!/usr/bin/env bash
set -euo pipefail

# Installs into user-owned directories; no sudo or system Python modifications.
repo="KotdaPK/granite-folder-transcriber"
install_root="${GRANITE_INSTALL_ROOT:-$HOME/.local/share/granite-transcriber}"
bin_dir="${GRANITE_BIN_DIR:-$HOME/.local/bin}"

for command in curl tar; do
    command -v "$command" >/dev/null || { printf 'Missing required command: %s\n' "$command" >&2; exit 1; }
done

if command -v uv >/dev/null; then
    uv_bin="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
    uv_bin="$HOME/.local/bin/uv"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv_bin="$HOME/.local/bin/uv"
fi

"$uv_bin" python install 3.12
mkdir -p "$install_root" "$bin_dir"
if [ ! -x "$install_root/venv/bin/python" ]; then
    "$uv_bin" venv --python 3.12 "$install_root/venv"
fi
python="$install_root/venv/bin/python"
# Avoid pulling gigabytes of CUDA wheels on machines using CPU inference.
"$uv_bin" pip install --python "$python" 'torch==2.14.1+cpu' 'torchaudio==2.11.0+cpu' --index https://download.pytorch.org/whl/cpu

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/source"
if [ -n "${GRANITE_SOURCE_DIR:-}" ]; then
    cp -a "$GRANITE_SOURCE_DIR/." "$scratch/source/"
else
    curl -fsSL "https://github.com/$repo/archive/refs/heads/main.tar.gz" | tar -xz -C "$scratch/source" --strip-components=1
fi
"$uv_bin" pip install --python "$python" "$scratch/source"
"$uv_bin" pip check --python "$python"
ln -sfn "$install_root/venv/bin/granite-transcribe" "$bin_dir/granite-transcribe"

printf 'Installed: %s/granite-transcribe\n' "$bin_dir"
if [[ ":$PATH:" != *":$bin_dir:"* ]]; then
    printf 'Add %s to PATH (e.g. export PATH="%s:$PATH") or use its full path.\n' "$bin_dir" "$bin_dir"
fi
printf 'First transcription downloads the Granite model (about 4.6 GB).\n'
