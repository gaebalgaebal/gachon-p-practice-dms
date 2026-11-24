# DMS 현재 단계 최종 검증 요약

기준일: 2026-11-24
범위: 계획서 중심 초기 DMS — 눈 감김·하품·주시 이탈·휴대폰 행동, 종합 DMS는 후반 고려

## 완료 판정

| 영역 | 결과 | 판정 |
|---|---|---|
| 데이터 준비 | MRL Eye, AI Hub 졸음·이상행동, YawDD, UTA-RLDD, State Farm 모두 준비 | 완료 |
| 눈 개폐 | MRL Eye test Accuracy 96.82%, macro F1 93.51% | 기준선 완료 |
| 하품 | 실제 웹캠 이벤트 Recall/Precision 100%, 평균 지연 1.26초 | 기능 완료 |
| 주시 이탈 | 실제 웹캠 Recall 100%, Precision 50%, 평균 지연 1.14초 | 기능 완료, 오경보 개선 필요 |
| 장시간 눈 감김 | 실제 웹캠 Recall 100%, Precision 50%, 평균 지연 1.80초 | 기능 완료, 오경보 개선 필요 |
| 휴대폰 | 0.25에서 Recall 100%·Precision 16.67%·오경보 10.84회/분 | 운영 불승인 |
| 통합 성능 | 얼굴 활성 640×480 정상상태 15.011 FPS | 목표 통과 |
| 자원 | CPU 69.38%, RSS 최대 1.69GB, CUDA 최대 allocated 59.71MB | 측정 완료 |
| 환경 조건 | Windows 9조건·2,700프레임; 역광·좌우 unknown 100% 취약점 확인 | 1차 완료 |
| 회귀 테스트 | 90개 통과 | 완료 |

## 모델 승인 상태

- MediaPipe Face Landmarker: 승인. 정면 보정과 yaw·pitch·iris 특징에 사용한다.
- OpenVINO open-closed-eye-0001: 승인. EAR과 교차 판정하며 RGB 웹캠 임계값은 0.70을 유지한다.
- AI Hub YOLO11n 행동 모델: 연구용 승인, 운영 경고 불승인. confidence 0.75는 미탐이 크고 0.25는 오경보가 과도하다.
- State Farm MobileNetV3: 데이터셋 비교 기준으로 유지. 실제 웹캠 운영 모델로 승격하지 않는다.
- 정상 강화 YOLO v2와 State Farm·YOLO AND 앙상블: 기존 기준선보다 개선되지 않아 불승인한다.

## 재현 명령

실제 카메라 통합 데모:

```bash
.venv/bin/python -m dms.live --camera 0 --camera-fps 30 \
  --model models/face_landmarker.task \
  --eye-model models/open_closed_eye.onnx \
  --action-checkpoint models/aihub_abnormal_yolo11n.pt \
  --action-backend yolo11 --action-device 0 \
  --action-threshold 0.75 --action-interval 5
```

자동 시간표 라벨·15 FPS 녹화:

```bash
.venv/bin/python -m dms.live --camera 0 \
  --model models/face_landmarker.task --eye-model models/open_closed_eye.onnx \
  --action-checkpoint models/aihub_abnormal_yolo11n.pt --action-backend yolo11 \
  --label-schedule configs/phone_webcam_scenario.json \
  --record --record-fps 15 --max-frames 405 --output runs/phone-check
```

이벤트 평가:

```bash
.venv/bin/python scripts/evaluate_events.py \
  --predictions runs/phone-check/frames.jsonl --embedded-labels \
  --output runs/phone-check/event-evaluation.json
```

## 남은 작업

1. 구현된 `--record-raw`와 `prepare_webcam_action.py`로 최소 3명 이상의 근거리 normal/phone 데이터를 수집하고 행동 모델을 재학습한다.
2. YOLO ONNX/TensorRT 변환은 운영 후보 모델이 기준을 통과한 뒤 수행한다. 현재 부정확한 모델을 최적화하지 않는다.
3. Windows 새 환경에서 설치·카메라·경고음·녹화까지 최종 smoke test를 수행한다.
4. 환경 수치를 제품 일반화 지표로 사용할 경우 피험자와 반복 횟수를 늘린다.
5. UTA-RLDD·AI Hub 졸음 전체 시간 모델과 DMD·Drive&Act·L2CS-Net·6DRepNet은 종합 DMS 후반 단계로 유지한다.

현재 초기 DMS 데모는 눈 감김·하품·주시 이탈, 로깅, 라벨링, 이벤트 평가, 15 FPS 성능 목표까지 재현 가능하다. 휴대폰 경고는 화면에 기술적으로 연결돼 있지만 운영 기능으로 간주하지 않는다.
