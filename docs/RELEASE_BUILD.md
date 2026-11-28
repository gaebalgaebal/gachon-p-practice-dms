# Windows 오프라인 배포 빌드

## 빌드 환경

- Windows 10/11 x64, Python 3.10~3.12 x64, Inno Setup 6
- `models/face_landmarker.task`, `models/open_closed_eye.onnx`

모델 파일은 `configs/model_manifest.json`의 SHA-256과 일치해야 한다. 기본 설치본에는 운영 기능에 필요한 얼굴·눈 모델만 포함한다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e '.[vision,eye,desktop,build]'
./build_release.ps1
```

인터넷이 차단된 빌드 PC에서는 사전에 승인된 wheel 저장소로 같은 버전을 설치한다. 빌드는 전체 시험, PyInstaller 앱 생성, 배포 파일 검사, Inno Setup 설치 파일, SHA-256 manifest 순서로 진행된다.

결과는 `release/GachonDMS-Setup-0.4.0.exe`와 `release/release-manifest.json`에 생성된다. `-SkipInstaller`를 사용하면 앱 폴더까지만 빌드한다.

## 배포 전 확인

- Python이 설치되지 않은 오프라인 Windows PC
- 관리자 권한 없는 설치·실행·제거
- 한글 사용자명과 공백 경로
- 카메라 선택·보정·세션 종료
- 30분 연속 실행과 메모리 확인
- 모델 누락·손상 오류 안내

Windows 실기 결과는 설치본 SHA-256, PC 사양, 카메라명과 함께 기록한다.
