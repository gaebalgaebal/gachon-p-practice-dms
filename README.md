# 가천대 P-실무 | 실시간 운전자 모니터링 데모

**가천대학교 AI2분반 5조 · 먼작귀**

확정 과제 계획서를 바탕으로 개발한 **실제 RGB 영상 입력용 1차 데모(v0.2)**입니다. Python·OpenCV·MediaPipe Face Landmarker·NumPy로 얼굴을 분석하고, 눈 감김과 얼굴 방향을 시간에 따라 판단합니다.

## 지금 실행되는 기능

| 기능 | 구현 상태 |
| --- | --- |
| 웹캠 / 영상 파일 입력 | 구현. 로컬에서는 영상 파일 실행 검증, 물리 웹캠은 장치 접근 불가로 미검증 |
| 얼굴 검출·랜드마크 | 공식 MediaPipe Face Landmarker 사전학습 모델 사용 |
| 눈 개폐·졸음 징후 | 좌우 EAR, 개인별 눈 뜬 기준 보정, 연속 감김 1.5초 경고 |
| 얼굴 방향·미주시 징후 | 정면 기준 보정 후 yaw/pitch와 지속시간으로 판단 |
| 눈 방향 보조 정보 | 홍채의 상대 위치를 진단 로그에 기록. 실제 시선 분류에는 사용하지 않음 |
| 화면·경고·저장 | 눈 포인트·상태 패널·시각 경고, JSONL 이벤트·프레임 로그, 실행 요약 |
| 예외 처리 | 얼굴 미검출·여러 얼굴·너무 작은 얼굴·과도한 회전 → 판단 불가 |
| 시간창 통계 | 시간 가중 눈 감김 비율 및 관측 coverage. 임상적 PERCLOS 아님 |
| 행동 인식 | MobileNetV3-Small 학습·체크포인트 연결 코드 제공. 실제 행동 데이터/가중치 미확보로 기본 비활성 |
| 평가 | 프레임 라벨 기반 Accuracy·Macro F1·Precision/Recall·혼동행렬·unknown coverage, EAR Baseline 비교 |

졸음 출력은 normal/suspected/unknown이며 3상태 임상 졸음 분류가 아닙니다. 전방 주시 출력은 **얼굴 방향 기반 프록시**입니다. 행동 모델을 연결하지 않으면 action은 unknown입니다. 데이터셋 정확도 목표는 아직 측정·달성하지 않았습니다.

## Windows 빠른 실행

Python **3.10~3.12 64비트** 권장. 저장소 루트에서:

```powershell
./setup_demo.ps1
./run_demo.ps1
```

카메라가 여러 개라면 `./run_demo.ps1 -Camera 1`, 영상 파일은 `./run_demo.ps1 -Video 'C:/path/demo.mp4'`로 실행합니다. PowerShell에서 스크립트 실행이 제한되면 아래 Python 명령을 직접 사용하면 됩니다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e '.[vision]'
./.venv/Scripts/python.exe scripts/download_model.py
./.venv/Scripts/python.exe -m dms.live --camera 0
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

## 테스트와 선택적 행동 학습

```sh
python -m unittest discover -s tests -v
python scripts/integration_smoke.py
python -m dms --input examples/synthetic_features.csv --output runs/synthetic.jsonl
```

통합 검사는 공식 예제 얼굴 이미지로 만든 짧은 영상과 검은 프레임으로 실제 모델·보정·얼굴 유실·복귀를 확인합니다. 이미지/영상은 Git에 포함하지 않습니다. 이 검사는 운전자 인식 정확도 평가가 아닙니다.

행동 인식은 `.[action]` 설치 후 승인된 데이터의 manifest로 학습해 연결합니다. [실행·학습·평가 안내](docs/DEMO_GUIDE.md)에 명령과 스키마를 정리했습니다. ImageNet 기본 가중치를 통화·음료 모델로 오인하지 않습니다.

## 문서

- [실행·오류 해결·행동 학습·평가](docs/DEMO_GUIDE.md)
- [확정 기획안과 구현 대응](docs/DEVELOPMENT.md)
- [실제 아키텍처·계산 방식](docs/ARCHITECTURE.md)
- [데이터·학습·추론·CI 파이프라인](docs/PIPELINES.md)
- [검증 기록과 한계](docs/VALIDATION.md)
- [평가 목표·프로토콜](docs/EVALUATION.md)
- [확정 계획 기준 일정·남은 작업](docs/ROADMAP.md)
- [데이터 관리](docs/DATA.md) · [참고 자료](docs/REFERENCES.md)
- [협업](CONTRIBUTING.md) · [실험 기록 양식](docs/EXPERIMENT_TEMPLATE.md)

연구·교육용 프로토타입이며 실차 안전 제품이 아닙니다. 원본 기획서의 학번·연락처·비공개 멘토링 원문, 얼굴 영상, 데이터셋 원본 및 학습 가중치는 커밋하지 않습니다. 코드 재사용 라이선스는 팀 협의 후 결정합니다.
