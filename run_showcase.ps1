param(
    [int]$Camera = 0,
    [string]$Video = '',
    [string]$ActionCheckpoint = 'models/aihub_abnormal_yolo11n.pt',
    [string]$ActionDevice = 'auto',
    [double]$ActionThreshold = 0.75,
    [switch]$Record,
    [switch]$Sound,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$Python = './.venv/Scripts/python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Run ./setup_showcase.ps1 first.'
}
foreach ($Required in @('models/face_landmarker.task', 'models/open_closed_eye.onnx', $ActionCheckpoint)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required model not found: $Required"
    }
}

$Arguments = @(
    '-m', 'dms.live',
    '--model', 'models/face_landmarker.task',
    '--eye-model', 'models/open_closed_eye.onnx',
    '--action-checkpoint', $ActionCheckpoint,
    '--action-backend', 'yolo11',
    '--action-device', $ActionDevice,
    '--action-threshold', $ActionThreshold,
    '--action-interval', '5',
    '--camera-fps', '30'
)
if ($Video) { $Arguments += @('--video', $Video) } else { $Arguments += @('--camera', $Camera) }
if ($Record) { $Arguments += @('--record', '--record-fps', '15') }
if ($Sound) { $Arguments += '--sound' }
if ($Output) { $Arguments += @('--output', $Output) }

Write-Host 'Starting DMS showcase. Look forward with eyes open until Calibration: ready.' -ForegroundColor Cyan
Write-Host 'Keys: C recalibrate, 0-4 manual labels, U unknown, Q/Esc exit.'
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw 'Showcase exited with an error. See the message above.' }
