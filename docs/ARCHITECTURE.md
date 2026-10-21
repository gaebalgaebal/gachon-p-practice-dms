# 아키텍처와 모듈 계약

## 입력과 처리 경계

```text
Camera/VideoSource [예정]
  → Frame(timestamp_s, RGB, driver_roi) [예정]
  ├→ FaceAnalyzer [계약만 제공]
  │    → Features(timestamp_s, ear, yaw_degrees, face_valid)
  │    → StateEngine [초기 구현]
  └→ ActionClassifier [계약만 제공]
       → 클래스별 점수 [예정]
  → DriverState(drowsiness, attention, action)
  → 화면/경고 [예정] + JSONL [특징 재생 구현]
```

모듈 계약은 `src/dms/contracts.py`, 시간 기반 규칙은 `src/dms/state.py`, 특징 재생은 `src/dms/__main__.py`에 있습니다. 실제 이미지 처리·학습 함수는 아직 구현하지 않았습니다.

## 특징 스키마

| 필드 | 단위·정의 | 누락 처리 |
| --- | --- | --- |
| timestamp_s | 시작 기준 초, 유한·0 이상·엄격한 증가 | 잘못된 순서면 입력 오류 |
| ear | 좌우 눈 EAR 요약 값, 0~1 범위의 입력 계약 | None/잘못된 값 → 졸음 unknown |
| yaw_degrees | 보정된 정면 대비 좌우 얼굴 각도, -180~180° | None/잘못된 값 → 주시 unknown |
| face_valid | 드라이버 얼굴의 유효 여부 | false → 두 상태 unknown 및 이력 초기화 |

현재 EAR 입력 범위는 데모 검증 계약입니다. 실제 랜드마크에서 산출할 때 좌우 눈·투영·안경의 영향을 검증해야 합니다. 한쪽 지표가 누락되어도 다른 유효 상태 출력은 독립적으로 유지합니다.

## 출력 스키마

```json
{"timestamp_s": 2.0, "drowsiness": "suspected", "attention": "forward", "action": "unknown", "source": "feature_rules_v0"}
```

- drowsiness: normal / suspected / unknown. normal은 현재 규칙의 경고 미발생 상태이며 운전자의 각성을 보증하지 않습니다.
- attention: forward / off_forward / unknown. 얼굴 방향 프록시로 실제 눈의 주시 위치를 의미하지 않습니다. 회전 누적 대기 중에는 unknown을 출력합니다.
- action: 초기 구현은 항상 unknown. 미래 행동 모델의 출력은 졸음·주시와 독립적으로 유지합니다.
- source: 추론 출처 식별자. 실제 실험에는 모델·설정 해시·세션 ID를 추가할 예정입니다.

## 시간 기반 규칙

설정은 `configs/default.json`에 있습니다. 값은 **소프트웨어 동작 확인용 가정**이며 사용자별 검증 없이 실제 운전에 적용하지 않습니다.

1. EAR < 0.20이면 눈 감김 시작, EAR ≥ 0.24이면 회복. 중간 구간은 이전 눈 상태를 유지하되 최초 상태는 unknown.
2. 눈 감김이 1.5초 이상 연속되면 suspected. 짧은 깜박임은 경고하지 않음.
3. |yaw| > 25°가 1.5초 이상 연속되면 off_forward. 정면 복귀 시 forward.
4. 입력 간격이 0.75초를 넘거나 얼굴이 유효하지 않으면 누적 이력을 초기화.
5. missing/NaN/Inf 특징은 해당 상태 unknown 처리. 타임스탬프 NaN/Inf·역순·중복은 오류로 거부.

현재는 단일 세션·주 운전자 1명 기준입니다. 새 영상/새 운전자마다 엔진을 새로 생성해야 합니다. PERCLOS, pitch, 실제 gaze, confidence, 다중 얼굴 추적, 행동 스케줄링, 알림 쿨다운은 후속 작업입니다.
