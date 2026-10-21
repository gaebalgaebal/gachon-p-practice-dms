$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.10-3.12 is required.' }
& ./.venv/Scripts/python.exe -m pip install -e '.[vision]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& ./.venv/Scripts/python.exe scripts/download_model.py
if ($LASTEXITCODE -ne 0) { throw 'Model download/verification failed.' }
Write-Host 'Setup complete. Run: ./run_demo.ps1'
