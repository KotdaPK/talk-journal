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

# Explicit CPU wheels avoid downloading CUDA packages on ordinary Windows PCs.
& $uv pip install --python $python 'torch==2.14.1+cpu' 'torchaudio==2.11.0+cpu' --index 'https://download.pytorch.org/whl/cpu'
if ($LASTEXITCODE -ne 0) { throw 'Could not install CPU PyTorch.' }
& $uv pip install --python $python $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not install Talk Journal.' }
& $uv pip check --python $python
if ($LASTEXITCODE -ne 0) { throw 'Installed packages are incompatible.' }

Write-Host 'Talk Journal installed. Transcribing recordings/ into raw-text/...'
& (Join-Path $PSScriptRoot 'run.ps1')
if ($LASTEXITCODE -ne 0) { throw "Transcription failed (exit code $LASTEXITCODE)." }
