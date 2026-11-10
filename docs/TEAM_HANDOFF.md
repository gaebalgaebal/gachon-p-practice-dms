# 팀원 인계 — 남은 현장 측정

기준일: 2026-11-10

초기 DMS 구현, 데이터 변환, 모델 기준선, 실제 웹캠 통합, 자동 라벨과 이벤트 평가는 완료했다. 현장 장비에서 반복해야 하는 환경 조건 측정은 이 문서를 기준으로 팀원이 이어서 수행한다. 원본 영상은 명시적으로 필요한 경우에만 `--record`로 저장한다.

## 현재 기준선

- 640×480 얼굴 활성 정상상태: 15.011 FPS
- 환경 비교 baseline: 얼굴 검출 100%, any-unknown 31%, 평균 밝기 98.28
- 입·코 하단 가림: 얼굴 검출 75.67%, any-unknown 53%, 평균 밝기 104.72
- 가림 전후 정상상태 FPS: 약 14.994
- 기준 결과: `docs/results/ENVIRONMENT_CONDITIONS.md`

## 남은 조건

각 조건은 같은 사람·카메라·거리·해상도에서 300프레임씩 측정한다.

1. glasses: 안경 착용
2. backlight: 얼굴 뒤쪽에 강한 조명 또는 밝은 창
3. low_light: 얼굴 조도를 낮춘 상태
4. camera_high / camera_low: 눈높이보다 위·아래
5. camera_left / camera_right: 정면 기준 좌·우 이동

보정이 필요하므로 처음에는 정면을 보고 눈을 자연스럽게 뜬다. 측정 중 의도하지 않은 동작은 하지 않는다. 조건을 만들 수 없으면 임의로 흉내 내지 말고 미측정으로 남긴다.

## 실행 명령

`CONDITION`과 출력 폴더 이름을 같은 값으로 바꾼다.

```bash
.venv/bin/python -m dms.live --camera 0 --camera-fps 30 \
  --model models/face_landmarker.task \
  --eye-model models/open_closed_eye.onnx \
  --action-checkpoint models/aihub_abnormal_yolo11n.pt \
  --action-backend yolo11 --action-device 0 \
  --action-threshold 0.75 --action-interval 5 \
  --condition CONDITION --headless --max-frames 300 \
  --output runs/condition_CONDITION_YYYYMMDD
```

실행 전 얼굴이 잡히는지 빠르게 확인하고 싶으면 GUI에서 `--headless`를 빼고 확인한 뒤 새 출력 폴더로 본 측정을 실행한다. 출력 폴더는 비어 있어야 한다.

## 집계

모든 조건 폴더를 인수로 전달한다.

```bash
.venv/bin/python scripts/summarize_condition_runs.py \
  runs/condition_baseline_YYYYMMDD \
  runs/condition_glasses_YYYYMMDD \
  runs/condition_backlight_YYYYMMDD \
  runs/condition_low_light_YYYYMMDD \
  --output runs/condition_comparison_YYYYMMDD.json
```

다음 항목을 `docs/results/ENVIRONMENT_CONDITIONS.md` 표에 추가한다.

- face_valid_fraction
- calibrated_fraction
- any_unknown_fraction
- brightness_mean
- steady_state_fps
- face_failure_reasons

## 판정과 Git 규칙

- 최초 보정 구간도 unknown에 포함되므로 300프레임 비율은 장기 운행 coverage와 동일하지 않다.
- baseline보다 얼굴 검출률이 낮거나 unknown이 높으면 조건 실패 영향을 수치로 기록한다.
- `runs/`, `data/`, 모델 가중치, 원본 PPT, 실제 얼굴 영상은 Git에 올리지 않는다.
- Git에는 집계 수치 JSON의 필요한 요약과 `docs/results/ENVIRONMENT_CONDITIONS.md` 변경만 올린다.
- 휴대폰 YOLO는 운영 불승인 상태다. confidence 0.25로 임의 변경하지 않는다.
