param([int]$Camera = 0, [string]$Video = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath './.venv/Scripts/python.exe')) {
    throw 'Run ./setup_demo.ps1 first.'
}
if ($Video) {
    & ./.venv/Scripts/python.exe -m dms.live --video $Video
} else {
    & ./.venv/Scripts/python.exe -m dms.live --camera $Camera
}
if ($LASTEXITCODE -ne 0) { throw 'Demo exited with an error. See the message above.' }
