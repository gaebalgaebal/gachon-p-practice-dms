@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo setup_windows.bat을 먼저 실행하세요.
  pause
  exit /b 1
)
.venv\Scripts\python.exe label_yawdd_folder.py --member MEMBER_PLACEHOLDER
pause
