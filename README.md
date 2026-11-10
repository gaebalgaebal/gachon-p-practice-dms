# 가천대 P-실무 | 실시간 운전자 모니터링 데모

**가천대학교 AI2분반 5조 · 먼작귀**

개발의 단일 기준은 [PPT 기준 확정 구현 계획](docs/IMPLEMENTATION_PLAN.md)입니다. 목표는 MediaPipe Face Landmarker, OpenVINO 눈 개폐 모델, YOLO11을 함께 사용하는 PC·웹캠 실시간 DMS입니다.

현재 저장소에는 RGB 입력·MediaPipe·보정·상태 관리·로그 베이스라인과 OpenVINO 눈 개폐 교차 판정, MAR 하품 상태, 눈 개방도 그래프가 구현되어 있습니다. 초기 데이터셋 6종도 로컬에 준비되어 데이터 변환과 정량 평가 단계로 진입했습니다.

## 현재 진행 상황

| 영역 | 상태 |
| --- | --- |
| MediaPipe EAR·MAR·Head Pose·보정 | 완료 |
| OpenVINO 좌우 눈 개폐·EAR 교차 판정 | 완료 |
| 실시간 UI·그래프·경고·로그 | 완료 |
| MRL Eye | 84,898장 준비, subject 분할 manifest 완료 |
| AI Hub 졸음운전 | 원천·라벨 준비, 837명·450,738행 archive manifest 완료 |
| YawDD | AVI 349개 준비, 구간 라벨링 대기 |
| UTA-RLDD | Fold 1~4 즉시 사용 가능, Fold 5 ZIP 검증 완료 |
| State Farm | train/test·운전자 CSV 준비, 분류 학습 대기 |
| AI Hub 이상행동 | 44,564장 YOLO 변환과 영상 단위 proxy 분할 완료 |
| YOLO11 | YOLO11n 학습 완료, test mAP50-95 29.51% |
| 세 모델 통합 | MediaPipe·OpenVINO·YOLO11 동시 추론 완료, 파일 입력 36.6 FPS |

현재 마일스톤과 작업 순서는 [로드맵](docs/ROADMAP.md), 상세 데이터 상태는 [로컬 데이터셋 현황](docs/DATASET_INVENTORY.md)에서 관리합니다.

## 지금 실행되는 기능

| 기능 | 구현 상태 |
| --- | --- |
| 웹캠 / 영상 파일 입력 | 구현. 로컬에서는 영상 파일 실행 검증, 물리 웹캠은 장치 접근 불가로 미검증 |
| 얼굴 검출·랜드마크 | 공식 MediaPipe Face Landmarker 사전학습 모델 사용 |
| 눈 개폐·졸음 징후 | 좌우 EAR, 개인별 눈 뜬 기준 보정, 연속 감김 1.5초 경고 |
| 하품 징후 | MediaPipe MAR, 히스테리시스, 지속시간 기반 yawning 상태 |
| 얼굴 방향·미주시 징후 | 정면 기준 보정 후 yaw/pitch와 지속시간으로 판단 |
| 눈 방향 보조 정보 | 개인별 보정 홍채 상대 위치를 얼굴 방향과 결합. 실제 3차원 시선 벡터는 아님 |
| 화면·경고·저장 | 눈·입 포인트, 좌우 EAR 10초 그래프, 상태 패널·시각 경고, JSONL 로그 |
| 예외 처리 | 얼굴 미검출·여러 얼굴·너무 작은 얼굴·과도한 회전 → 판단 불가 |
| 시간창 통계 | 시간 가중 눈 감김 비율 및 관측 coverage. 임상적 PERCLOS 아님 |
| 행동 인식 | 기존 MobileNetV3 실험 코드는 참고용. 확정 목표는 State Farm·AI Hub 기반 YOLO11 |
| 평가 | 프레임 지표·unknown coverage·EAR Baseline과 이벤트 Recall·오경보/분·탐지 지연 |

현재 베이스라인의 졸음 출력은 normal/suspected/unknown입니다. 전방 주시는 **보정된 얼굴 방향과 홍채 상대 위치를 결합한 프록시**이며 실제 시선 벡터가 아닙니다. YOLO11 행동 인식은 휴대폰 사용·통화 행동 영역을 `phone` 상태로 통합하며 시간 투표 후 경고합니다.

## Windows 빠른 실행

Python **3.10~3.12 64비트** 권장. 저장소 루트에서:

```powershell
./setup_demo.ps1
./run_demo.ps1
```

카메라가 여러 개라면 `./run_demo.ps1 -Camera 1`, 영상 파일은 `./run_demo.ps1 -Video 'C:/path/demo.mp4'`로 실행합니다. PowerShell에서 스크립트 실행이 제한되면 아래 Python 명령을 직접 사용하면 됩니다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e '.[vision,eye]'
./.venv/Scripts/python.exe scripts/download_model.py
./.venv/Scripts/python.exe scripts/download_eye_model.py
./.venv/Scripts/python.exe -m dms.live --camera 0 --eye-model models/open_closed_eye.onnx
```

Linux는 `python3 -m venv .venv`, 이후 `.venv/bin/python`을 사용합니다. 모델은 공식 제공처에서 로컬로 내려받고 SHA256을 검사합니다. **얼굴 영상은 외부 추론 서비스로 전송하지 않습니다.**

### 시연 순서

1. 카메라를 고정하고 한 사람의 얼굴과 상반신이 보이도록 앉습니다.
2. **정면을 보고 눈을 뜬 채 2초 이상** 유지합니다. `Calibration: ready`를 확인합니다.
3. 짧은 깜박임과 1.5초 이상 눈 감김을 비교합니다.
4. 고개를 좌우 또는 아래로 돌리고 1.5초 유지한 뒤 복귀합니다.
5. 얼굴을 가리거나 화면 밖으로 이동해 unknown 처리를 확인합니다.
6. `C`로 기준을 다시 잡고 `Q`/Esc로 종료합니다.

카메라·사람·앉는 위치가 바뀌면 재보정합니다. 기준 보정은 사용자가 정면·눈 뜬 상태를 유지한다는 가정이며 눈을 감은 채 보정하면 제대로 판단할 수 없습니다.

## 영상 입력·기록

```sh
python -m dms.live --video data/local_demo.mp4 --headless --output runs/video-test
python -m dms.live --camera 0 --roi 0.1,0,0.8,1 --record --sound
```

`--record`는 로컬 주석 영상 저장, `--sound`는 Windows 경고음에 명시적으로 동의하는 옵션입니다. 기본은 영상 저장·소리 모두 끔입니다. 새 실행 폴더에 frames.jsonl / events.jsonl / summary.json / calibration.json이 저장됩니다. 기존 결과 폴더는 덮어쓰지 않습니다. 입력 영상 재생은 처리 속도로 진행되며 실제 시간 재생을 보장하지 않습니다.

## 테스트와 데이터 manifest

```sh
python -m unittest discover -s tests -v
python scripts/integration_smoke.py
python -m dms --input examples/synthetic_features.csv --output runs/synthetic.jsonl
python scripts/validate_manifest.py examples/synthetic_dms_manifest.csv --profile dms
```

통합 검사는 공식 예제 얼굴 이미지로 만든 짧은 영상과 검은 프레임으로 실제 모델·보정·얼굴 유실·복귀를 확인합니다. 이미지/영상은 Git에 포함하지 않습니다. 이 검사는 운전자 인식 정확도 평가가 아닙니다.

확정 데이터셋과 과제 조합은 [데이터 관리](docs/DATA.md)와 `configs/datasets.json`에서 관리합니다. 실제 데이터는 이용 승인을 확인한 뒤 운전자 단위로 분할합니다.

## 문서

- [현재 단계 최종 검증·모델 승인 상태](docs/FINAL_VALIDATION.md)
- [팀원 인계 — 남은 현장 측정](docs/TEAM_HANDOFF.md)
- [실행·오류 해결·행동 학습·평가](docs/DEMO_GUIDE.md)
- [확정 기획안과 구현 대응](docs/DEVELOPMENT.md)
- [PPT 기준 확정 구현 계획](docs/IMPLEMENTATION_PLAN.md)
- [단계별 개발 기록](docs/DEVELOPMENT_LOG.md)
- [실제 아키텍처·계산 방식](docs/ARCHITECTURE.md)
- [데이터·학습·추론·CI 파이프라인](docs/PIPELINES.md)
- [검증 기록과 한계](docs/VALIDATION.md)
- [평가 목표·프로토콜](docs/EVALUATION.md)
- [확정 계획 기준 작업 순서·남은 작업](docs/ROADMAP.md)
- [데이터 관리](docs/DATA.md) · [참고 자료](docs/REFERENCES.md)
- [로컬 데이터셋 준비 현황](docs/DATASET_INVENTORY.md)
- [협업](CONTRIBUTING.md) · [실험 기록 양식](docs/EXPERIMENT_TEMPLATE.md)

연구·교육용 프로토타입이며 실차 안전 제품이 아닙니다. 원본 기획서의 학번·연락처·비공개 멘토링 원문, 얼굴 영상, 데이터셋 원본 및 학습 가중치는 커밋하지 않습니다. 코드 재사용 라이선스는 팀 협의 후 결정합니다.

# 의견 제시
--> (함태은 10/27) : 졸음 운전 파악 방법에 하품이나 고개 변환 각도를 통해 조는 것을 파악하는 것도 좋을 것 같습니다.
