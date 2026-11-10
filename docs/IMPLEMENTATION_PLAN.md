# DMS 확정 구현 계획

이 문서는 `운전자감시카메라_프로젝트_4x3_20261030_1119.pptx`를 기준으로 확정한 구현 계획이다. 데이터셋과 모델 선택은 계획서 내용을 우선하며, 개인별 보정·시간 안정화·선택적 추론·ONNX 최적화만 성능 개선 범위로 추가한다.

기존 문서와 선택이 다를 경우 이 문서의 데이터셋·모델·구현 순서를 우선한다. 접근이 복잡한 NTHU-DDD는 2026-10-31 결정에 따라 이미 확보한 UTA-RLDD Fold 1~4 부분집합으로 대체한다.

## 1. 초기 구현 범위

- 눈 감김 및 시간창 눈 감김 비율
- 하품
- 고개 숙임·좌우 돌림과 전방 미주시
- 휴대폰·음료 사용
- 위험 상태의 화면·소리 경고
- 프레임·이벤트·모델 점수 로그
- 눈 개방도 그래프와 얼굴 분석 결과가 포함된 실시간 화면

DMD와 Drive&Act를 사용하는 종합 DMS·세부 행동 인식은 후반 검토 범위로 둔다.

## 2. 데이터셋

| 과제 | 주 데이터셋 | 보조 데이터셋 | 목적 |
| --- | --- | --- | --- |
| 눈 뜸·감김 | MRL Eye | AI Hub 졸음운전 | 눈 상태 분류와 EAR 임계값 검증 |
| 하품 | YawDD | AI Hub 졸음운전 | MAR 임계값과 말하기·하품 구분 |
| 졸음 상태 | AI Hub 졸음운전 | UTA-RLDD Fold 1~4 | 실제 도로와 영상 단위 졸음 단계 평가 |
| 고개 숙임·돌림 | AI Hub 졸음운전 | 자체 웹캠 검증 | Head Pose와 지속시간 평가 |
| 휴대폰·음료 | AI Hub 이상행동 | State Farm | 객체 검출과 행동 분류 |
| 시선 이탈 | MediaPipe 기반 자체 웹캠 검증 | DMD는 후반 검토 | 전방 미주시 판정 |
| 종합 DMS | DMD, Drive&Act | 없음 | 후반 확장·종합 평가 |

모든 분할은 운전자 단위로 수행한다. 데이터 이용 승인, 발표 이미지 공개 범위, 원본·가중치 재배포 조건을 데이터셋별로 기록한다.

## 3. 동시 실행 모델

1. **MediaPipe Face Landmarker**
   - 얼굴·눈·입 랜드마크
   - 좌우 EAR, MAR
   - Head Pose와 홍채 상대 위치
   - 얼굴·눈·입 시각화
2. **OpenVINO open-closed-eye-0001**
   - 좌우 눈 개폐 분류
   - EAR 결과 교차 확인
3. **YOLO11**
   - 휴대폰·컵 검출 또는 행동 분류
   - State Farm 분류 베이스라인과 AI Hub 검출 모델 비교 후 최종 체크포인트 선택

6DRepNet은 Head Pose 비교 후보, L2CS-Net은 DMD를 도입하는 후반 gaze 확장 후보로 둔다.

## 4. 공통 데이터 계약

데이터셋 변환 결과는 CSV manifest로 관리한다. DMS manifest의 필수 컬럼은 다음과 같다.

```text
sample_id,dataset,subject_id,session_id,file_path,frame_index,timestamp_s,task,label,split
```

- `dataset`: `mrl_eye`, `aihub_drowsiness`, `uta_rldd`, `yawdd`, `state_farm`, `aihub_abnormal`, `dmd`, `drive_act`
- `task`: `eye_state`, `yawn`, `drowsiness`, `head_pose`, `attention`, `action`
- `split`: `train`, `val`, `test`
- 이미지도 `frame_index=0`, `timestamp_s=0`으로 기록한다.
- `subject_id`와 `session_id`는 분할 간 겹치지 않아야 한다.
- 원본 영상·이미지는 Git에 포함하지 않는다.

## 5. 구현 순서

1. 데이터 이용 신청과 라이선스 기록
2. 공통 manifest 검증기와 데이터셋별 변환기
3. MediaPipe EAR·MAR·Head Pose 완성
4. OpenVINO 눈 개폐 모델 연결
5. 눈 개방도 그래프와 실시간 UI
6. MRL Eye·AI Hub를 이용한 눈 감김 평가
7. YawDD 하품 구간 라벨과 MAR 평가
8. UTA-RLDD Fold 1~4 졸음 단계 평가
9. State Farm 행동 분류 베이스라인
10. AI Hub 이상행동 기반 YOLO11 파인튜닝
11. 세 모델 통합과 경고 로직
12. ONNX 변환·선택적 추론·15 FPS 최적화
13. 조건별 웹캠 평가와 데모·보고서 작성

## 6. 상태와 경고

모델 출력은 독립적으로 유지한다.

```text
Eye: open / closed / uncertain
Drowsiness: normal / warning / drowsy / unknown
Yawn: normal / yawning / unknown
Attention: forward / off_forward / unknown
Action: normal / phone / drink / unknown
```

얼굴 미검출이나 모델 불일치는 정상으로 바꾸지 않고 `unknown` 또는 `uncertain`으로 기록한다. 단일 프레임 결과는 즉시 경고로 사용하지 않고 지속시간·히스테리시스·최근 추론 투표를 적용한다.

## 7. 목표 화면

- 웹캠 영상 위 얼굴 박스, 눈·입 랜드마크, 머리 방향
- 휴대폰·컵 검출 박스와 점수
- 좌우 EAR와 눈 개폐 상태의 최근 10초 그래프
- 졸음·주시·하품·행동 상태 패널
- EAR, MAR, 눈 감김 비율, yaw, pitch, FPS, 지연시간
- 위험 시 빨간 테두리, 원인 문구, 선택적 경고음
- 분석 영상과 프레임·이벤트 로그 저장

## 8. 평가와 완료 기준

| 대상 | 지표 |
| --- | --- |
| 눈 감김 | Accuracy, F1 |
| 하품 | 검출률, 오경보율 |
| 졸음 | F1, 이벤트 Recall |
| 고개 방향 | 각도 오차, 정확도 |
| 휴대폰·음료 | mAP, Macro F1 |
| 경고 | 오경보/분, 미탐, 탐지 지연 |
| 시스템 | FPS, 평균·p95 지연, 프레임 드롭률 |

초기 완료 조건은 MediaPipe·OpenVINO eye·YOLO11의 동시 실행, 640×480 기준 15 FPS 이상, 목표 UI와 경고 동작, 데이터셋별 정량 평가, 재현 가능한 실행 문서다.

## 9. 성능 개선 범위

계획서 범위를 바꾸지 않는 선에서 다음만 추가한다.

- 개인별 정면·눈 뜬 기준 보정
- 경고 진입·해제 히스테리시스
- 시간 기반 상태 안정화와 행동 추론 투표
- 얼굴·눈·상반신 ROI 기반 선택적 추론
- ONNX Runtime과 모델별 실행 주기 조절
- `unknown` coverage와 조건별 실패율 보고
