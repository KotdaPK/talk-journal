# Run after install.ps1: .\run.ps1 [--overwrite] [--recursive]
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$command = Join-Path $PSScriptRoot '.venv-default\Scripts\talk-journal.exe'
if (-not (Test-Path -LiteralPath $command -PathType Leaf)) {
    throw 'Talk Journal is not installed. Run .\install.ps1 first.'
}
$recordings = Join-Path $PSScriptRoot 'recordings'
$rawText = Join-Path $PSScriptRoot 'raw-text'
New-Item -ItemType Directory -Force -Path $recordings, $rawText | Out-Null
& $command $recordings $rawText @args
if ($LASTEXITCODE -ne 0) { throw "Transcription failed (exit code $LASTEXITCODE)." }
