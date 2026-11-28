param([switch]$SkipInstaller)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$Python = './.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Build environment is not prepared.' }
foreach ($Model in @('models/face_landmarker.task', 'models/open_closed_eye.onnx')) {
    if (-not (Test-Path -LiteralPath $Model)) { throw "Required model not found: $Model" }
}

& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
& $Python -m PyInstaller --noconfirm --clean packaging/dms_desktop.spec
if ($LASTEXITCODE -ne 0) { throw 'Application build failed.' }
& $Python scripts/audit_release.py dist/GachonDMS
if ($LASTEXITCODE -ne 0) { throw 'Release audit failed.' }

if (-not $SkipInstaller) {
    $Compiler = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if (-not $Compiler) { throw 'Inno Setup 6 compiler was not found.' }
    & $Compiler.Source packaging/installer.iss
    if ($LASTEXITCODE -ne 0) { throw 'Installer build failed.' }
    & $Python scripts/create_release_manifest.py release
} else {
    & $Python scripts/create_release_manifest.py dist/GachonDMS
}
Write-Host 'Release build completed.' -ForegroundColor Green
