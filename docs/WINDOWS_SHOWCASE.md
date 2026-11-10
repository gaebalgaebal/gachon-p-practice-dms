# Windows 통합 데모 시연

이 데모는 Windows 10/11, Python 3.10~3.12 64비트에서 지금까지 구현한 MediaPipe·OpenVINO·YOLO11 통합 화면을 시연하기 위한 실행 경로다. 기본 `setup_demo.ps1`/`run_demo.ps1`은 얼굴·눈 기능만 빠르게 확인하는 안정 경로이고, 이 문서의 showcase 스크립트는 행동 모델까지 포함한다.

## 보여주는 기능

- 얼굴·눈·입 랜드마크, EAR·MAR, yaw·pitch, 홍채 상대 위치
- 개인별 정면·눈 뜬 기준 보정
- 눈 감김, 하품, 고개·시선 이탈 상태와 경고
- OpenVINO 눈 개폐와 EAR 교차 판정
- YOLO11 휴대폰 행동 박스·점수와 시간 투표
- 최근 10초 눈 개방도 그래프
- FPS, coverage, 상태 패널, 위험 시 빨간 테두리
- 프레임·상태 변화·경고·자원 사용량 JSON 로그
- 선택적 주석 영상 저장과 Windows 경고음
- `0~4` 키 수동 정답 라벨

## 준비물

1. Windows 10/11 PC와 웹캠
2. Python 3.10~3.12 64비트
3. 저장소 `middle-01` 브랜치
4. 승인된 `models/aihub_abnormal_yolo11n.pt` 체크포인트

가중치는 데이터 재배포 조건과 Git 용량 때문에 저장소에 포함하지 않는다. Linux 개발 PC의 같은 파일을 USB 또는 내부 공유 경로로 Windows PC의 `models` 폴더에 복사한다. MediaPipe와 OpenVINO 모델은 설치 스크립트가 공식 배포처에서 내려받는다.

## 설치

PowerShell에서 저장소 루트로 이동한다. 실행 정책을 영구 변경하지 않으려면 현재 프로세스에만 허용한다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
./setup_showcase.ps1
```

설치 마지막에 `Windows showcase dependencies OK`가 출력되어야 한다. `CUDA available: False`여도 CPU로 실행할 수 있다. NVIDIA GPU를 사용하려면 Windows 드라이버와 CUDA 지원 PyTorch 설치가 별도로 필요하다.

## 실행

일반 시연:

```powershell
./run_showcase.ps1 -Camera 0
```

CPU 강제 실행:

```powershell
./run_showcase.ps1 -Camera 0 -ActionDevice cpu
```

주석 영상과 경고음 포함:

```powershell
./run_showcase.ps1 -Camera 0 -Record -Sound -Output 'runs/windows-showcase'
```

카메라가 여러 개라면 `-Camera 1`을 시도한다. 실제 영상 파일로 재현하려면 다음과 같이 실행한다.

```powershell
./run_showcase.ps1 -Video 'C:\demo\driver.mp4' -Record -Output 'runs/windows-video'
```

## 권장 시연 순서

1. 카메라에 얼굴과 상반신이 들어오게 앉고 정면·눈 뜬 상태를 유지한다.
2. `Calibration: ready`를 확인한다.
3. 짧은 깜박임 후 2초가량 눈을 감아 `Drowsiness: suspected`와 경고를 확인한다.
4. 입을 크게 벌린 상태를 유지해 `Yawn: yawning`을 확인한다.
5. 고개를 좌우 또는 아래로 돌려 `Attention: off_forward`를 확인한다.
6. 얼굴 일부를 가려 `unknown` 안전 처리를 확인한다.
7. 휴대폰은 박스 출력만 기술 시연한다. 현재 모델은 운영 정확도 기준 미달이다.
8. `C`로 재보정, `Q` 또는 Esc로 종료한다.

## 중요한 제한

- 휴대폰 모델은 confidence 0.75에서 미탐이 크고 0.25에서 오경보가 과도하다. 기본 showcase는 0.75를 사용하며 운영 기능으로 설명하지 않는다.
- 음료 행동은 State Farm 데이터 기준선만 있고 현재 YOLO 체크포인트의 클래스가 아니다.
- `--camera-exposure 120`은 Linux 시험 카메라 전용이므로 Windows 스크립트에서는 사용하지 않는다.
- Windows PC 성능이 다르므로 15 FPS는 해당 PC에서 다시 측정해야 한다.
- 얼굴 영상과 `runs` 출력은 Git에 올리지 않는다.

## 발표 시 설명

“MediaPipe로 얼굴 기하 특징, OpenVINO로 눈 개폐, YOLO11로 행동 후보를 동시에 처리합니다. 눈 감김·하품·주시 이탈은 실제 웹캠 이벤트 평가까지 완료했고, 휴대폰은 기술 검출은 되지만 오경보 문제 때문에 연구용으로만 표시합니다.”
