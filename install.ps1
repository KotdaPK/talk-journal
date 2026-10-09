# Run from any working directory: .\install.ps1
# Installs in this checkout and transcribes recordings/ to raw-text/.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uv = $uvCommand.Source
} else {
    $uv = Join-Path $HOME '.local\bin\uv.exe'
    if (-not (Test-Path -LiteralPath $uv -PathType Leaf)) {
        Write-Host 'Installing uv for the current user...'
        Invoke-RestMethod 'https://astral.sh/uv/install.ps1' | Invoke-Expression
    }
    if (-not (Test-Path -LiteralPath $uv -PathType Leaf)) {
        throw "uv installation failed: $uv was not found."
    }
}

$venv = Join-Path $PSScriptRoot '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
& $uv python install 3.12
if ($LASTEXITCODE -ne 0) { throw 'Could not install Python 3.12.' }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    & $uv venv --python 3.12 $venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
}

# Install matched CUDA wheels; never fall back to a CPU-only PyTorch build.
& $uv pip install --python $python 'torch==2.11.0+cu128' 'torchaudio==2.11.0+cu128' --index 'https://download.pytorch.org/whl/cu128'
if ($LASTEXITCODE -ne 0) { throw 'Could not install CUDA PyTorch.' }
& $uv pip install --python $python $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not install Talk Journal.' }
& $uv pip check --python $python
if ($LASTEXITCODE -ne 0) { throw 'Installed packages are incompatible.' }
& $python -c 'import torch,sys;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None);sys.exit(0 if torch.cuda.is_available() else 1)'
if ($LASTEXITCODE -ne 0) { throw 'CUDA is unavailable. Talk Journal requires a working NVIDIA GPU and driver.' }

Write-Host 'Talk Journal installed. Transcribing recordings/ into raw-text/...'
& (Join-Path $PSScriptRoot 'run.ps1')
if ($LASTEXITCODE -ne 0) { throw "Transcription failed (exit code $LASTEXITCODE)." }
