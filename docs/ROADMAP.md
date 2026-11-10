# 확정 계획 기준 작업 순서와 남은 작업

구현의 기준은 [DMS 확정 구현 계획](IMPLEMENTATION_PLAN.md)이다. 주차가 아니라 선행 조건과 검증 결과에 따라 다음 순서로 진행한다.

## v0.3에 추가한 개발 결과

- [x] 실제 MediaPipe 얼굴 모델과 OpenCV 영상 입력 연결
- [x] 개인별 정면·눈 뜬 기준 보정, EAR·yaw/pitch 연속시간 판정
- [x] 얼굴 유실·여러 얼굴·프레임 간격 처리
- [x] 상태 화면·시각 경고·JSONL·실행 요약·선택적 녹화
- [x] 행동 학습·체크포인트 추론 경로와 데이터 누수 검사
- [x] 프레임 지표·EAR Baseline 비교·통합 테스트

## 데이터셋 준비 현황

- [x] MRL Eye 84,898장 압축 해제, 37명 subject 분할 manifest 생성
- [x] AI Hub 졸음운전 원천 ZIP 6개·라벨 ZIP 6개 확보 및 환경별 개수 대응 확인
- [x] YawDD AVI 349개 압축 해제
- [x] NTHU-DDD를 UTA-RLDD로 대체
- [x] UTA-RLDD Fold 1~4의 48명·145영상 압축 해제
- [x] UTA-RLDD Fold 5의 12명·37영상 ZIP 다운로드·무결성 확인
- [x] State Farm train 22,424장·test 79,726장 및 운전자 CSV 압축 해제
- [x] AI Hub 이상행동 TS4/TL4·VS1/VL1 부분집합 확보 및 휴대폰·통화 라벨 확인

데이터 원본은 모두 Git 제외 경로에 보관한다. 최신 파일 수와 부분집합 한계는 [로컬 데이터셋 현황](DATASET_INVENTORY.md)에 기록한다.

## 현재 마일스톤 — 데이터 변환과 정량 평가

1. [x] 공통 DMS manifest 검증기
2. [x] MRL Eye 변환기와 subject 단위 train/val/test 분할
3. [x] MRL Eye validation 임계값 `0.50` 선정과 test 최종 평가
4. [x] AI Hub 졸음운전 원천·JSON archive 변환기와 450,738행 manifest
5. [x] YawDD 349개 영상 manifest와 하품 시작·종료 라벨링 도구
   - [x] 전달 라벨 41개 영상·68개 구간 통합 및 MAR validation/test 평가
6. [x] UTA-RLDD Fold 1~4 영상 단계 manifest와 3-class·이진 평가기
7. [x] State Farm 운전자 단위 manifest와 MobileNetV3 분류 베이스라인
8. [x] AI Hub 이상행동 TAR 선별 변환과 YOLO 형식 생성

## 후속 마일스톤

### 모델 통합

- [x] MediaPipe EAR·MAR·Head Pose
- [x] OpenVINO open-closed-eye-0001과 EAR 교차 판정
- [x] State Farm 분류 또는 AI Hub 검출 기반 YOLO11 체크포인트
- [x] MediaPipe·OpenVINO eye·YOLO11 동시 추론

### 평가·최적화·완료

- [ ] 데이터셋별 Accuracy·F1·mAP·이벤트 지표 확정
- [ ] 안경·가림·역광·야간·카메라 위치별 실패율과 unknown coverage (baseline·하단 가림 완료, 나머지는 팀원 현장 측정 인계)
- [x] 640×480 기준 FPS, 평균·p95 지연, 프레임 드롭률 측정
- [ ] ONNX 변환·모델 실행 주기·ROI 선택 추론 최적화
- [x] 실제 웹캠 시나리오와 최종 데모·보고서

## 다음 구현

- [x] 실제 웹캠 장치에서 입력·정면 보정·장시간 감김·고개 회전 시연 (눈 감김·주시 이탈 warning 현장 확인)
- [x] 라이브 단축키 정답 라벨과 이벤트 Precision·Recall·오경보·지연 자동 평가 연결
- [x] AI Hub 졸음운전·이상행동 데이터 확보
- [x] MRL Eye·YawDD·State Farm 압축 해제
- [x] NTHU-DDD를 확보된 UTA-RLDD Fold 1~4 부분집합으로 대체
- [x] 준비된 데이터셋별 manifest 변환기 작성 (MRL Eye·AI Hub 졸음·YawDD·UTA-RLDD·State Farm·AI Hub 이상행동)
- [x] MediaPipe MAR·하품 상태와 눈 개방도 그래프 구현
- [x] OpenVINO open-closed-eye-0001 연결과 EAR 교차 판정
- [x] MRL Eye에서 OpenVINO 정확도와 F1 평가 (test Accuracy 96.82%, macro F1 93.51%)
- [x] State Farm 행동 분류 베이스라인과 AI Hub YOLO11 모델 학습
- [x] MediaPipe·OpenVINO eye·YOLO11 통합 추론
- [ ] 안경·가림·역광·야간·카메라 위치별 실패율·unknown 비율 (팀원 인계: `docs/TEAM_HANDOFF.md`)
- [x] 이벤트 오경보/분·미탐·탐지 지연 평가 도구 구현
- [x] 실제 라벨 영상으로 이벤트 오경보/분·미탐·탐지 지연 실험 (단일 통제 반복, 4개 이벤트)
- [x] 동일 하드웨어에서 640×480, 15 FPS와 CPU/메모리 측정 (얼굴 활성 실제 웹캠 정상상태 15.011 FPS, CPU·RSS·CUDA 측정 완료)
- [x] DMD·Drive&Act·L2CS-Net·6DRepNet 후반 도입 여부 결정 (현재 범위에서는 보류)

## 수동 실험 반영 고도화 순서

1. 눈이 충분히 열린 프레임에서만 홍채 특징을 주시 판정에 사용하고, 수직 홍채값은 안정성이 확인될 때까지 진단 지표로 분리합니다.
2. 얼굴 검출과 주시 상태에 히스테리시스·유예 시간을 적용해 짧은 누락과 경계값 진동으로 인한 unknown 전환을 줄입니다.
3. 실험 중 정답 구간을 남길 수 있는 단축키 또는 라벨 입력 도구를 추가하고 같은 시나리오를 반복 측정합니다.
4. 공개 데이터셋의 사용 조건을 확정한 뒤 운전자 단위로 train/validation/test를 분리하고, 통화·음료 행동 모델을 전이학습합니다.
5. 프레임 분류 지표와 함께 이벤트 재현율·오경보/분·탐지 지연을 비교해 임계값과 최종 모델을 선정합니다.
