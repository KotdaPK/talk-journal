# User-scoped CUDA installer. Example: .\install.ps1 --models parakeet,moss,granite
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$models = 'parakeet'
if ($args.Count -gt 0) {
    if ($args.Count -ne 2 -or $args[0] -ne '--models') { throw 'Usage: .\install.ps1 [--models parakeet,moss,granite]' }
    $models = [string]$args[1]
}
$selected = $models.Split(',')
foreach ($name in $selected) {
    if ($name -notin @('parakeet', 'moss', 'granite')) { throw "Unknown model: $name" }
}
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) { $uv = $uvCommand.Source }
else {
    $uv = Join-Path $HOME '.local\bin\uv.exe'
    if (-not (Test-Path -LiteralPath $uv -PathType Leaf)) {
        Write-Host 'Installing uv for the current user...'
        Invoke-RestMethod 'https://astral.sh/uv/install.ps1' | Invoke-Expression
    }
    if (-not (Test-Path -LiteralPath $uv -PathType Leaf)) { throw 'uv installation failed.' }
}
& $uv python install 3.11
if ($LASTEXITCODE -ne 0) { throw 'Could not install Python 3.11.' }
$venv = Join-Path $PSScriptRoot '.venv-default'
$python = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    & $uv venv --python 3.11 $venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create virtual environment.' }
}
& $uv pip install --python $python 'torch==2.11.0+cu128' 'torchaudio==2.11.0+cu128' --index 'https://download.pytorch.org/whl/cu128'
if ($LASTEXITCODE -ne 0) { throw 'Could not install CUDA PyTorch.' }
& $uv pip install --python $python $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not install Talk Journal.' }
foreach ($name in $selected) {
    switch ($name) {
        'granite' {
            & $uv pip install --python $python "${PSScriptRoot}[granite]"
            if ($LASTEXITCODE -ne 0) { throw 'Could not install Granite dependencies.' }
        }
        'moss' {
            $mossVenv = Join-Path $PSScriptRoot '.venv-moss'
            $mossPython = Join-Path $mossVenv 'Scripts\python.exe'
            if (-not (Test-Path -LiteralPath $mossPython -PathType Leaf)) {
                & $uv venv --python 3.11 $mossVenv
                if ($LASTEXITCODE -ne 0) { throw 'Could not create MOSS virtual environment.' }
            }
            & $uv pip install --python $mossPython 'torch==2.11.0+cu128' 'torchaudio==2.11.0+cu128' --index 'https://download.pytorch.org/whl/cu128'
            if ($LASTEXITCODE -ne 0) { throw 'Could not install MOSS CUDA PyTorch.' }
            & $uv pip install --python $mossPython $PSScriptRoot
            if ($LASTEXITCODE -ne 0) { throw 'Could not install Talk Journal in MOSS environment.' }
            & $uv pip install --python $mossPython "${PSScriptRoot}[moss]"
            if ($LASTEXITCODE -ne 0) { throw 'Could not install MOSS dependencies.' }
            & $uv pip check --python $mossPython
            if ($LASTEXITCODE -ne 0) { throw 'MOSS dependencies conflict.' }
            & $mossPython -c 'import torch,sys;sys.exit(0 if torch.cuda.is_available() and torch.version.cuda else 1)'
            if ($LASTEXITCODE -ne 0) { throw 'MOSS CUDA unavailable.' }
        }
    }
}
& $uv pip check --python $python
if ($LASTEXITCODE -ne 0) { throw 'Installed packages are incompatible.' }
& $python -c 'import torch,sys;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None);sys.exit(0 if torch.cuda.is_available() and torch.version.cuda else 1)'
if ($LASTEXITCODE -ne 0) { throw 'CUDA unavailable. NVIDIA GPU and CUDA PyTorch required.' }
Write-Host "Talk Journal installed: $models. Processing recordings/..."
& (Join-Path $PSScriptRoot 'run.ps1')
if ($LASTEXITCODE -ne 0) { throw "Transcription failed (exit $LASTEXITCODE)." }
