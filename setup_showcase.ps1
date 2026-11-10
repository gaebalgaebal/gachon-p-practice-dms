param([switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.10-3.12 64-bit is required.' }
$Python = './.venv/Scripts/python.exe'

& $Python -m pip install --upgrade pip
& $Python -m pip install -e '.[vision,eye,action]'
if ($LASTEXITCODE -ne 0) { throw 'Core dependency installation failed.' }

# Ultralytics installs opencv-python. Both OpenCV wheels share the cv2 package,
# so restore the project-pinned contrib build after installing Ultralytics.
& $Python -m pip install 'ultralytics==8.4.198'
if ($LASTEXITCODE -ne 0) { throw 'Ultralytics installation failed.' }
& $Python -m pip uninstall -y opencv-python opencv-python-headless
& $Python -m pip install --force-reinstall 'numpy==2.2.6' 'opencv-contrib-python==4.13.0.92'
if ($LASTEXITCODE -ne 0) { throw 'OpenCV contrib restoration failed.' }

& $Python scripts/download_model.py
if ($LASTEXITCODE -ne 0) { throw 'MediaPipe model download/verification failed.' }
& $Python scripts/download_eye_model.py
if ($LASTEXITCODE -ne 0) { throw 'Eye model download/verification failed.' }

& $Python -c "import cv2, mediapipe, openvino, torch, ultralytics; print('Windows showcase dependencies OK'); print('CUDA available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'Dependency import check failed.' }

Write-Host ''
Write-Host 'Setup complete.' -ForegroundColor Green
Write-Host 'Copy the approved checkpoint to: models/aihub_abnormal_yolo11n.pt'
Write-Host 'Then run: ./run_showcase.ps1'
if ($CpuOnly) {
    Write-Host 'CPU-only mode requested. Use ./run_showcase.ps1 -ActionDevice cpu'
}
