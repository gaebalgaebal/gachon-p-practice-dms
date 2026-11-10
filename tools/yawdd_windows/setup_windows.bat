@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [1/3] Python 확인
py -3 --version || goto :no_python
echo [2/3] 가상환경 생성
if not exist .venv py -3 -m venv .venv || goto :failed
echo [3/3] OpenCV 설치
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :failed
.venv\Scripts\python.exe label_yawdd_folder.py --member MEMBER_PLACEHOLDER --check || goto :failed
echo.
echo 설치와 영상 검사가 완료되었습니다. run_labeler.bat을 실행하세요.
pause
exit /b 0

:no_python
echo Python이 없습니다. https://www.python.org/downloads/ 에서 Python 3을 설치하세요.
echo 설치 화면에서 Add python.exe to PATH를 선택해야 합니다.
pause
exit /b 1

:failed
echo 설치 또는 검사에 실패했습니다. 오류 화면을 담당자에게 보내주세요.
pause
exit /b 1
