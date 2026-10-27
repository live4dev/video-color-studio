$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")
uv sync --frozen --extra dev --extra build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv run --no-sync python tools/build_desktop.py
exit $LASTEXITCODE
