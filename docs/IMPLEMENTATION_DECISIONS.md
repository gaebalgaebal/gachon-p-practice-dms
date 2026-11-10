# 확정 구현 기준

모델·데이터셋·구현 순서는 [DMS 확정 구현 계획](IMPLEMENTATION_PLAN.md)으로 고정한다. 이전 데이터셋과 모델 제안은 더 이상 개발 기준으로 사용하지 않는다.

## 확정 모델

| 역할 | 모델 | 데이터·검증 |
| --- | --- | --- |
| 얼굴·눈·입·자세 | MediaPipe Face Landmarker | AI Hub 졸음운전, UTA-RLDD Fold 1~4, YawDD, 자체 웹캠 |
| 눈 개폐 교차 확인 | OpenVINO open-closed-eye-0001 | MRL Eye, AI Hub 졸음운전 |
| 휴대폰·음료 | YOLO11 | State Farm 분류, AI Hub 이상행동 검출 |

6DRepNet과 L2CS-Net은 후반 비교 후보이며 DMD·Drive&Act와 함께 초기 완료 조건에 포함하지 않는다.

## 확정 출력

- `Eye`: open / closed / uncertain
- `Drowsiness`: normal / warning / drowsy / unknown
- `Yawn`: normal / yawning / unknown
- `Attention`: forward / off_forward / unknown
- `Action`: normal / phone / drink / unknown

판단 불가능한 입력은 정상으로 바꾸지 않는다. 모델 불일치와 낮은 점수는 `uncertain` 또는 `unknown`으로 기록한다.

## 성능 개선 허용 범위

- 개인별 정면·눈 뜬 기준 보정
- 히스테리시스와 지속시간 판정
- 최근 추론 결과 투표
- ROI·모델별 실행 주기 조절
- ONNX Runtime 최적화
- 조건별 실패율과 unknown coverage 보고

## 완료 판정

완료는 세 모델이 같은 실시간 프로그램에서 실행되고, 목표 UI·경고·로그가 동작하며, 운전자 단위 test 결과와 640×480 기준 15 FPS 측정값이 있을 때만 인정한다.
