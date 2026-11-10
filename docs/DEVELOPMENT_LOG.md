# 개발 단계 기록

이 문서는 [DMS 확정 구현 계획](IMPLEMENTATION_PLAN.md)에 따라 실제로 구현한 내용과 검증 결과를 누적한다. 새 기능을 완료할 때마다 변경 파일, 동작, 테스트, 남은 한계를 기록한다.

## 단계 0. 계획 기준 전환

**상태:** 완료  
**기록일:** 2026-10-31

### 구현

- PPT 데이터셋과 모델 조합을 프로젝트의 단일 기준으로 확정
- 초기 데이터셋을 MRL Eye, AI Hub 졸음운전·이상행동, YawDD, NTHU-DDD, State Farm으로 지정
- 초기 모델을 MediaPipe Face Landmarker, OpenVINO open-closed-eye-0001, YOLO11로 지정
- DMD, Drive&Act, L2CS-Net, 6DRepNet은 후반 검토 범위로 분리

### 검증

- README, 데이터, 평가, 로드맵, 구현 결정 문서를 새 기준으로 정리
- 기존 원본 데이터는 삭제하지 않고 새 파이프라인에서 제외

## 단계 1. 공통 DMS manifest

**상태:** 완료  
**기록일:** 2026-10-31

### 구현

- `configs/datasets.json` 데이터셋 레지스트리 추가
- 공통 CSV 필드와 허용 데이터셋·과제 조합 정의
- `scripts/validate_manifest.py --profile dms` 추가
- sample 중복, 운전자·세션 분할 누수, 데이터셋·과제 불일치, 잘못된 프레임·시간값 검사

### 검증

- 기존 manifest 프로필과 DMS 프로필 예제 통과
- 정상·오류 manifest 단위 테스트 추가

### 남은 작업

- 실제 데이터 확보 후 데이터셋별 변환기 구현

## 단계 2. MAR·하품·눈 개방도 그래프

**상태:** 완료  
**기록일:** 2026-10-31

### 구현

- MediaPipe 입술 8점 기반 MAR 계산
- 하품 진입·복귀 히스테리시스와 지속시간 판정
- 최근 10초 좌우 EAR 그래프
- 입술 포인트, MAR, Yawn 상태, 경고·로그 추가

### 검증

- 전체 단위 테스트 통과
- MediaPipe 실제 모델 비전 테스트 통과
- 120프레임 통합 스모크 테스트 통과
- 통합 스모크 처리 속도 40.4 FPS. 정식 성능 지표로 사용하지 않음

### 남은 작업

- YawDD에서 MAR 임계값과 말하기 오경보 검증

## 단계 3. OpenVINO 눈 개폐 교차 판정

**상태:** 완료  
**기록일:** 2026-10-31

### 구현

- 공식 `open-closed-eye-0001` ONNX 다운로드와 SHA-256 검증
- MediaPipe 눈 랜드마크 기반 좌우 눈 crop
- 32×32 BGR OpenVINO 추론
- EAR와 모델 결과가 일치할 때만 open/closed 확정
- 불일치·낮은 점수는 uncertain 처리
- 좌우 모델 점수와 통합 점수를 UI·JSONL에 기록
- Windows 기본 설치·실행 스크립트에 OpenVINO와 눈 모델 추가

### 검증

- 공식 ONNX SHA-256 검증 및 입력 `[1,3,32,32]`, 출력 `[1,2,1,1]` 확인
- 공식 Open Model Zoo 예제 3장으로 배포 ONNX의 실제 클래스 순서 확인
- 공식 인물 샘플 120프레임에서 눈 점수 60프레임 생성
- 얼굴 유실 20프레임에서 이전 눈 점수가 남지 않음을 확인
- MediaPipe·OpenVINO 동시 통합 스모크 50.0 FPS. 정식 성능 지표로 사용하지 않음
- OpenVINO 비활성 상태의 단위·통합 회귀 테스트 통과

### 남은 작업

- MRL Eye에서 Accuracy·F1 측정
- RGB 웹캠, 안경, 역광, 측면 얼굴 조건별 임계값 검증
- 현재 `0.70` 신뢰도 임계값은 검증 전 초기값

## 단계 4. MRL Eye 데이터 변환·평가 기반

**상태:** 완료  
**기록일:** 2026-10-31

### 구현

- 공식 8개 파일명 속성(subject, image, gender, glasses, eye state, reflections, lighting, sensor) 파서
- `0=closed`, `1=open` 라벨 변환과 잘못된 속성값 검사
- 운전자 단위 고정 시드 train/val/test 분리 및 공통 DMS manifest 생성
- 성별·안경·반사·조명·센서 속성을 manifest에 보존
- 잘린 눈 영상을 직접 받는 OpenVINO 공개 추론 API
- 전체 Accuracy·macro F1·coverage 및 조건별 동일 지표 평가 스크립트
- 낮은 신뢰도 결과를 강제 분류하지 않고 `unknown`으로 집계

### 검증

- 파일명 파싱, 오류 거부, 분할 재현성·누수 방지, manifest 변환 단위 테스트 추가
- 합성 파일 구조로 DMS manifest 검증기까지 연결 확인

### 실행

```bash
python scripts/prepare_mrl_eye.py /path/to/mrlEyes_2018_01 \
  --output data/manifests/mrl_eye.csv
python scripts/evaluate_mrl_eye.py data/manifests/mrl_eye.csv \
  --model models/open_closed_eye.onnx \
  --predictions artifacts/mrl_eye_predictions.csv \
  --output artifacts/mrl_eye_report.json
```

### 남은 작업

- MRL Eye는 눈 crop 데이터라 MediaPipe EAR 계산 불가. EAR 단독·교차 판정 비교는 전체 얼굴 데이터에서 수행

### 실제 데이터 평가 완료 — 2026-11-01

- validation 4,970장에서 `0.50~0.95` 임계값 탐색
- `0.50` 선택: Accuracy 95.67%, macro F1 95.51%, coverage 100%
- test 9,377장 1회 평가: Accuracy 96.82%, macro F1 93.51%, coverage 100%
- 안경 test macro F1 84.52%, 큰 반사 83.52%, 나쁜 조명 89.70%로 취약 조건 확인
- 상세 결과: `docs/results/MRL_EYE_EVALUATION.md`
- MRL은 IR crop이므로 실시간 RGB용 `0.70`은 AI Hub·웹캠 검증 전까지 유지

## 단계 5. 데이터셋 선행 준비 체계

**상태:** 준비 체계 완료, 계정·승인·원본 다운로드 대기  
**기록일:** 2026-10-31

### 구현

- 계획서 초기 데이터셋 6종의 공식 출처·접근 방식·로컬 위치를 기계 판독 가능한 설정으로 등록
- 원본·전처리·manifest·승인 기록·평가 결과 디렉터리 초기화
- 예상 파일 구조 기반 준비 상태 점검 스크립트 추가
- AI Hub, IEEE DataPort, NTHU, Kaggle의 수동 계정·승인 절차와 반입 순서 문서화
- 데이터셋별 버전·허용 범위·SHA-256·발표 이미지 권한 기록 양식 추가

### 검증

- 현재 저장 공간 및 기존 원본을 변경하지 않고 초기 디렉터리 생성 확인
- 당시 기준으로 UTA-RLDD를 제외했으나, 이후 단계 6에서 NTHU-DDD 대체 데이터로 결정 변경

## 단계 6. 다운로드 현황 검증과 NTHU 대체

**상태:** 완료  
**기록일:** 2026-10-31

### 결정

- 접근이 복잡한 NTHU-DDD를 기존 UTA-RLDD Fold 1~4 부분집합으로 대체
- UTA-RLDD는 졸음 단계 평가에만 사용하고, 하품·Head Pose의 세부 라벨 데이터로 과대 해석하지 않음

### 검증

- MRL Eye ZIP: 무결성 통과, 84,898 이미지·37명
- State Farm ZIP: 무결성 통과, train 22,424장·test 79,726장·10개 클래스·운전자 CSV 포함
- YawDD: 외부 gzip 무결성 통과, 내부 RAR 압축 해제 필요
- UTA-RLDD: Fold 1~4, 48명·145영상 압축 해제. Fold 5의 12명·37영상 ZIP도 무결성 통과
- AI Hub 졸음운전: 학습·검증 원천 ZIP 6개와 라벨 ZIP 6개 확보, 환경별 개수 일치

### 반입 완료

- MRL Eye를 `data/raw/mrl_eye/`에 압축 해제
- State Farm을 `data/raw/state_farm/`에 압축 해제
- YawDD 이중 압축을 검증하고 AVI 349개를 `data/raw/yawdd/`에 압축 해제
- 상세 현황을 `docs/DATASET_INVENTORY.md`에 고정 기록
- AI Hub 이상행동 TS4/TL4·VS1/VL1 부분집합 확보 및 휴대폰조작·통화 라벨 확인

### Git 일정 반영

- 저장소 로드맵에 데이터셋 6종의 준비 상태를 완료 항목으로 반영
- 현재 마일스톤을 데이터 변환·정량 평가로 지정
- 후속 작업을 모델 통합과 평가·최적화 마일스톤으로 분리
- README에 데이터 준비 상태와 다음 개발 범위를 요약

## 단계 7. AI Hub 졸음운전 manifest 변환

**상태:** 완료  
**기록일:** 2026-11-01

### 구현

- 실제도로·통제환경 원천/라벨 ZIP 직접 pairing
- 원본 Training/Validation을 합치고 환경별 운전자 ID로 70/15/15 재분할
- 좌우 눈 가시성·상태 일치 표본의 eye state 생성
- 통제환경 파일명에서 normal/drowsy와 주시 방향 7종 생성
- ZIP을 풀지 않고 archive와 member 경로를 manifest에 보존
- 원천 없는 라벨 제외와 AI Hub 파일명 불일치 fallback 및 품질 플래그 기록

### 검증

- 837명, 450,738 task 행 생성
- eye state 300,400, drowsiness 75,169, head pose 75,169
- subject/session 분할 누수 검사 통과
- 고아 라벨 800개 제외, accessory fallback 40개 기록
- 상세 결과: `docs/results/AIHUB_DROWSINESS_MANIFEST.md`

## 단계 8. 적용 예시 보존과 YawDD 영상 manifest·구간 라벨링 도구

**상태:** 기반 구현 완료, 수동 구간 라벨링 진행 필요  
**기록일:** 2026-11-01

### 기록 자산

- AI Hub JSON의 얼굴·좌우 눈·입 박스와 생성 라벨을 한 화면에 표시한 적용 예시 저장
- 한글 `head_pose=계기판`이 깨지지 않도록 Noto Sans CJK 글꼴로 수정
- `docs/results/AIHUB_DROWSINESS_MANIFEST.md`에서 이미지와 변환 근거를 함께 참조
- 자산: `docs/assets/aihub_drowsiness_manifest_example.jpg`

### YawDD 구현

- 실제 349개 AVI 파일명의 카메라·성별·피험자·안경 조건·활동 파서
- `.avi.avi`, 공백, 대소문자가 섞인 원본 파일명 처리
- 성별+번호 기준 90명 피험자 단위 train/val/test 고정 분할
- FPS·프레임 수·길이·해상도와 약한 영상 라벨을 포함한 DMS manifest 생성
- OpenCV 키보드 조작으로 하품 시작·종료 구간을 누적 CSV에 저장하는 도구 구현
- 잘못된 구간과 영상 길이를 넘어선 구간 거부

### 실제 데이터 결과

- 349개 영상, Mirror 320개·Dash 29개
- train 246개·validation 50개·test 53개, 피험자/세션 누수 없음
- normal 205개·yawn 포함 115개·unknown sequence 29개
- Yawning/Talking&Yawning과 Dash를 합친 144개 영상을 구간 검수 대상으로 지정
- 상세 결과: `docs/results/YAWDD_MANIFEST.md`

### 검증

- YawDD 파서·피험자 식별·재현 가능한 분할·구간 저장 단위 테스트 5개 통과
- 공통 DMS manifest 검증기에서 349행과 train/val/test 누수 검사 통과

### 남은 작업

- 144개 대상 영상의 하품 시작·종료 구간 수동 라벨링
- 구간 라벨을 이용한 MAR 임계값, 검출률, 오경보율 평가

## 단계 9. UTA-RLDD 영상 manifest와 졸음 단계 평가기

**상태:** 완료  
**기록일:** 2026-11-01

### 구현

- 공식 라벨 정의 `0=alert`, `5=low_vigilance`, `10=drowsy` 파서
- 대소문자가 섞인 MP4/MOV/M4V와 분할 영상 `10_1`, `10_2` 처리
- 48명 피험자 단위 고정 seed train/val/test 분할
- FPS·프레임 수·영상 길이·해상도·Fold·segment를 포함한 DMS manifest 생성
- 모델 독립적인 `sample_id,prediction` 기반 3-class 평가기
- alert 대 fatigued(low_vigilance+drowsy) 보조 이진 평가
- `unknown`을 오답으로 유지하고 coverage를 별도 보고
- prediction 누락·추가·중복·잘못된 라벨 검사와 입력 템플릿 생성

### 실제 데이터 결과

- Fold 1~4의 48명, 145개 영상, 약 25.02시간
- alert 48개·low_vigilance 48개·drowsy 49개
- train 103개·validation 21개·test 21개, 피험자/세션 누수 없음
- validation/test는 각각 세 클래스 7개씩
- 상세 결과: `docs/results/UTA_RLDD_MANIFEST.md`

### 검증

- 경로 파싱·분할 segment·재현 가능한 피험자 분할·이진 매핑·평가기 단위 테스트 5개 통과
- 공통 DMS manifest 검증기에서 실제 145행과 train/val/test 누수 검사 통과
- test 21행 prediction 템플릿 생성 및 전부 unknown인 입력에서 coverage 0% 확인

### 남은 작업

- DMS 시계열 출력 또는 별도 졸음 분류 모델의 video-level prediction 생성
- validation에서 시간창·임계값을 선택한 뒤 test를 한 번만 최종 평가

## 단계 10. YawDD 전달 라벨 통합과 MAR 정량 평가

**상태:** 완료  
**기록일:** 2026-11-03

### 라벨 반입·품질 검증

- Downloads의 annotator A CSV 35구간과 annotator B JSON 34구간 반입
- manifest 기준 video ID·파일명·split·영상 길이 범위 교차 검증
- 서로 겹치지 않는 validation/test 41개 영상 확인
- `yawdd:video:017`의 거의 동일한 중복 구간 1쌍을 자동 병합
- 원본 69구간을 평가용 68구간으로 정규화하고 품질 플래그·원본 ID 보존
- 통합 하품 구간 길이 264.733초

### MAR 평가 구현

- 라벨 영상만 5 FPS로 샘플링하고 MediaPipe MAR를 재사용 가능한 CSV로 캐시
- 얼굴 미검출·유효하지 않은 MAR를 정상으로 바꾸지 않고 unknown 처리
- 진입·복귀 히스테리시스와 최소 지속시간 기반 이벤트 생성
- validation에서 MAR threshold와 지속시간 45개 조합 탐색
- validation event F1, frame macro F1, 오경보율 순으로 설정 선택
- 선택 설정을 변경하지 않고 test에 한 번 적용

### 결과

- MAR 추출: 6,005프레임 중 5,913 유효, coverage 98.47%
- 선택값: 진입 `0.35`, 복귀 `0.25`, 지속 `0.8초`
- Validation 이벤트 Precision 96.67%, Recall 90.63%, F1 93.55%
- Test 이벤트 Precision 96.43%, Recall 75.00%, F1 84.37%
- Test 오경보 0.093회/분, 프레임 Accuracy 85.19%, macro F1 73.22%
- 런타임 기본 MAR 설정을 validation 선택값으로 갱신
- 상세 결과: `docs/results/YAWDD_MANIFEST.md`

### 검증

- 라벨 중복 병합·MAR 지속시간·히스테리시스·unknown coverage 단위 테스트 추가
- 전체 회귀 테스트 72개 통과

### 후속 개선

- test 하품 프레임 recall 38.45%는 이벤트 포착 후 라벨 구간 전체를 짧게 덮는 한계
- test 재튜닝 없이 validation에서 입 벌림 상승·하강 또는 시간창 특징을 검토

## 단계 11. State Farm 운전자 분할과 행동 분류 베이스라인

**상태:** 완료  
**기록일:** 2026-11-03

### 데이터 변환

- 공식 c0~c9 클래스 파서와 초기 DMS `normal/phone/drink` 매핑
- c0을 normal, c1~c4를 phone, c6을 drink로 통합
- 초기 범위 밖 c5·c7·c8·c9는 제외 수량을 summary에 보존
- 26명 운전자 단위 고정 seed train/validation/test 분할
- 14,070장 DMS manifest 생성과 운전자·세션 누수 검사

### 학습 개선

- 선택 의존성 PyTorch 2.9.1+cu128·torchvision 0.24.1 설치
- RTX 5080 자동 선택과 CPU fallback 구현
- DataLoader worker·pinned memory 지원
- phone 클래스 편향을 줄이기 위한 train 역빈도 class weight 적용
- GPU 체크포인트를 CPU tensor로 저장해 런타임 이식성 유지

### 정식 베이스라인

- ImageNet 사전학습 MobileNetV3-Small feature 고정, classifier 10 epoch 학습
- validation macro F1 기준 epoch 7 체크포인트 선택
- 최고 validation Accuracy 85.57%, macro F1 80.62%
- 고정 test Accuracy 72.05%, macro F1 65.87%
- test F1: drink 63.99%, normal 55.17%, phone 78.45%
- 라이브 어댑터에서 체크포인트 로드와 c0/c1/c6 실제 이미지 추론 확인
- 상세 결과: `docs/results/STATE_FARM_BASELINE.md`

### 해석

- 엄격한 운전자 분할에서 validation-test 격차가 커 운전자·차량 배경 일반화가 약함
- State Farm은 공식 설명상 통제 촬영이므로 실제 도로 성능으로 과대 해석하지 않음
- 최종 행동 모델은 AI Hub 이상행동 YOLO11과 비교한 뒤 선정

## 단계 12. AI Hub 이상행동 변환과 YOLO11 기준선

**상태:** 완료  
**기록일:** 2026-11-03

### 데이터 변환

- TL4/VL1 JSON과 TS4/VS1 이미지 TAR를 스트리밍 대응해 필요한 프레임만 추출
- 장면별 중앙 프레임을 선택해 인접 프레임 중복을 줄임
- 차량제어를 empty-label 정상 샘플, 휴대폰조작·통화를 2개 행동 클래스로 변환
- 44,564장 생성: train 21,111, validation 9,674, test 13,779
- 원본에 운전자 ID가 없어 video_id를 proxy로 사용하고 split 누수 없음 확인
- 잘못된 박스 1건과 원천 이미지 누락 507건 제외 및 summary 보존

### YOLO11n 학습과 평가

- Ultralytics 8.4.198, PyTorch 2.9.1+cu128, RTX 5080에서 640·batch 32 학습
- 최대 30 epoch 중 patience 10으로 18 epoch 조기 종료, 최고 validation epoch 8
- validation 최고 mAP50-95 45.35%
- 고정 test: Precision 30.90%, Recall 88.40%, mAP50 40.80%, mAP50-95 29.51%
- phone_use mAP50-95 23.50%, phone_call 35.60%
- 체크포인트와 JSON 지표 저장, 재현 가능한 전용 학습 스크립트 추가

### 해석과 다음 단계

- 높은 recall과 낮은 precision 때문에 단일 프레임 경고 대신 confidence·시간 투표 필요
- 라벨은 휴대폰 물체가 아닌 운전자 몸/행동 박스이므로 행동 영역 검출로 정의
- 실제 운전자 ID 분리가 아니므로 사람 일반화 평가로 과대 해석하지 않음
- 다음 단계는 YOLO11 런타임 어댑터와 MediaPipe·OpenVINO 동시 추론
- 상세 결과: `docs/results/AIHUB_ABNORMAL_YOLO.md`

## 단계 13. MediaPipe·OpenVINO·YOLO11 동시 추론

**상태:** 완료  
**기록일:** 2026-11-03

### 구현

- 공통 행동 인터페이스에 YOLO11 어댑터와 backend 자동/명시 선택 추가
- phone_use·phone_call을 공통 `phone` 상태로 집계하고 원 클래스 점수 보존
- 얼굴이 가려져도 행동을 감지하도록 YOLO는 얼굴 유효성과 독립적으로 전체 ROI 추론
- 검출 박스·클래스·신뢰도를 화면과 frames.jsonl에 기록
- phone 시간 투표 확정 시 화면 경고, state_change, warning 이벤트 연결
- MobileNetV3 기존 체크포인트 경로와 얼굴 유실 시 초기화 동작 유지

### 통합 검증

- YawDD 실제 영상 300프레임에서 MediaPipe·OpenVINO·YOLO11 동시 실행
- 얼굴 유효 300프레임, 정면 보정 후 OpenVINO 눈 출력 166프레임
- YOLO 행동 점수 300프레임, 검출 박스 누적 63개
- 처리 36.64 FPS, 평균 루프 27.29ms, p95 42.87ms
- 전체 회귀 테스트 81개 통과

파일을 가능한 속도로 처리한 기술 성능이며 카메라 센서부터 화면까지의 실제 지연은 아니다. 다음 단계에서 실제 웹캠 640×480 성능과 행동 confidence·시간 투표를 validation 영상으로 조정한다.

## 단계 14. YOLO 행동 confidence·시간 투표 validation 검증

**상태:** 기준 미달 확인, 재학습 필요  
**기록일:** 2026-11-03

- 고정 validation 9,674장에서 confidence 0.25·0.35·0.45·0.55·0.65·0.75 비교
- recall 60% 이상 후보 중 F1 최대 규칙을 사전에 고정
- 0.25 선택: precision 46.31%, recall 74.16%, F1 57.01%
- 선택 결과 JSON과 재현 가능한 threshold sweep 스크립트 저장
- 정지 프레임 후보 0.25를 YOLO11 비교 기준으로 반영, MobileNet 기존 0.70 유지
- test는 임계값 선택에 재사용하지 않음
- 원본 validation의 5프레임 장면 600개·3,000프레임으로 시간 투표 공동 탐색
- recall 60% 이상·정상 클립 오경보 20% 이하를 동시에 만족한 조합 없음
- confidence 0.25 최고 F1 조합은 recall 94.25%지만 정상 클립 오경보 100%
- 오경보 제약만 만족한 최고 F1 조합은 confidence 0.75·3프레임/2표: recall 19.75%, 오경보 9.5%
- 현재 YOLO를 운영 확정하지 않고 정상 차량제어 프레임 강화 재학습으로 전환

## 단계 15. 정상 강화 YOLO v2 재학습

**상태:** 실험 완료, 운영 승격 안 함  
**기록일:** 2026-11-03

### 데이터·학습

- 차량제어 장면 5프레임 전체를 음성 샘플로 사용
- 휴대폰조작·통화를 단일 phone_action 클래스로 통합
- 총 97,244장: 정상 65,850, phone 31,394, train 43,643
- 기존 v1과 원본을 보존하고 별도 20GB v2 데이터셋 생성
- YOLO11n 9 epoch 조기 종료, 고정 test mAP50-95 20.50%

### 동일 시퀀스 비교

- v1 최고 F1 77.18%, 정상 클립 오경보 100%
- v2 최고 F1 76.88%, 정상 클립 오경보 99.5%
- 오경보 20% 이하 최고 후보: v1 F1 31.73%·recall 19.75%; v2 F1 28.63%·recall 17.75%
- 사전 기준 recall 60% 이상·정상 오경보 20% 이하를 두 모델 모두 충족하지 못함

정상 프레임 증량만으로는 몸 전체 행동 박스가 차량제어 동작과 휴대폰 동작을 혼동하는 문제가 해결되지 않았다. v1을 비교 기준으로 유지하고 v2는 운영 모델로 승격하지 않는다.

## 단계 16. State Farm·YOLO 교차 확인 앙상블

**상태:** 실험 완료, 운영 반영 안 함  
**기록일:** 2026-11-05

- State Farm MobileNet phone 확률과 AI Hub YOLO 검출을 AND 조건으로 결합
- YOLO confidence 7개·분류 confidence 4개·시간 투표 6개 조합을 validation에서 탐색
- 동일 600클립·3,000프레임과 recall 60%·정상 오경보 20% 기준 사용
- 기준 만족 조합 없음
- 오경보 제약 내 최고: precision 65.52%, recall 19.00%, F1 29.46%, 오경보 20%
- YOLO 단독 안전 후보 F1 31.73%보다 낮아 런타임에 반영하지 않음

두 모델의 데이터 촬영 환경 차이 때문에 교차 확인이 유효한 오경보 필터가 되지 못했다. 휴대폰 물체 박스 데이터가 추가되기 전에는 v1을 연구용 비교 기준으로 유지한다.

## 단계 17. 실제 웹캠 입력 최적화와 성능 측정

**상태:** 정상상태 FPS 목표 통과, 얼굴 시나리오 재측정 필요  
**기록일:** 2026-11-05

- `/dev/video0` 640×480 실제 카메라 300프레임 측정
- 자동 노출·직렬 입력에서 14.30 FPS로 목표 미달 확인
- YOLO 실행 주기 3→5 최적화만으로는 카메라 병목이 개선되지 않음
- 장치 전용 수동 노출 120에서 원천 입력 약 28 FPS 확인
- 최신 프레임 비동기 캡처 구현과 건너뛴 프레임 수 기록 추가
- 최초 30프레임 warm-up을 분리한 정상상태 지표를 summary에 추가
- 최종 전체 14.33 FPS, 정상상태 15.008 FPS, 정상상태 p95 80.83ms
- 건너뛴 프레임 11개, 녹화·이미지 저장 없음
- 전체 회귀 테스트 82개 통과

촬영 구간에 유효 얼굴이 없어 OpenVINO 눈 출력과 상태 coverage는 평가하지 못했다. 실제 운전자가 정면에 있는 조건에서 세 모델 활성 상태의 FPS와 coverage를 다시 측정해야 한다. 상세 결과: `docs/results/WEBCAM_BENCHMARK.md`.

## 단계 18. 얼굴 활성 3모델 자원 계측

**상태:** 영상 기반 검증 완료, 실제 웹캠 얼굴 시나리오 남음  
**기록일:** 2026-11-10

- 실행 요약에 프로세스 CPU, RSS 시작·종료·표본 최대, CUDA 현재·최대 메모리 기록 추가
- `psutil`이 없는 환경에서는 자원 필드만 `null`로 남겨 기존 데모 실행 호환성 유지
- YawDD 얼굴 유효 영상 300프레임에서 MediaPipe·OpenVINO·YOLO11n 동시 실행
- 얼굴 유효 300프레임, 보정 이후 눈 모델 166프레임, YOLO 행동 점수 300프레임 확인
- 전체 39.42 FPS, 정상상태 42.83 FPS, 정상상태 p95 39.83ms
- CPU 단일 코어 환산 131.66%, RSS 표본 최대 1,710.11MB
- CUDA 최대 allocated 59.71MB, reserved 68.00MB
- 전체 회귀 테스트 82개 통과

얼굴이 활성화된 세 모델 연결과 자원 계측은 완료했다. 파일 기반 수치이므로 실제 웹캠에서 운전자가 정면에 있는 조건의 센서 지연·coverage 확인은 최종 현장 검증으로 유지한다. 상세 결과: `docs/results/RESOURCE_BENCHMARK.md`.

## 단계 19. 얼굴 활성 실제 웹캠 현장 검증

**상태:** 완료  
**기록일:** 2026-11-10

- 동일 `/dev/video0`, 640×480, 수동 노출 120에서 운전자 정면 상태 300프레임 측정
- 얼굴 검출 300/300, 보정 완료 및 OpenVINO 눈 출력 263프레임
- YOLO 행동 점수 300프레임 출력, 세 모델 동시 활성 확인
- 전체 14.36 FPS, 최초 30프레임 제외 정상상태 15.011 FPS로 목표 통과
- CPU 단일 코어 환산 69.38%, RSS 표본 최대 1,687.95MB
- CUDA 최대 allocated 59.71MB, reserved 68.00MB
- 녹화·이미지 저장 없이 JSON 수치 로그만 보존

실제 웹캠의 얼굴 활성 조건에서 성능과 자원 사용량 측정을 완료했다. 다음 단계는 정상 자세뿐 아니라 눈 감기·고개 돌림·하품·휴대폰 행동을 짧은 통제 시나리오로 실행해 화면 상태 전이와 경고 발생을 확인하는 것이다. 상세 결과: `docs/results/WEBCAM_BENCHMARK.md`.

## 단계 20. 실제 웹캠 통제 동작 시험

**상태:** 눈 감기·하품 성공, 고개 돌림·휴대폰 개선 필요  
**기록일:** 2026-11-10

- 정면→눈 감기→회복→고개 돌림→하품→휴대폰 순서로 660프레임 현장 시험
- 눈 감기에서 `suspected` 16프레임과 warning 이벤트 확인
- 하품에서 `yawning` 19프레임과 warning 이벤트 확인
- 고개 돌림은 얼굴 랜드마크 유실로 `unknown`이 증가해 지속 `off_forward` 검출 실패
- 휴대폰 confidence 0.75에서는 후보 박스가 검출되지 않음
- confidence 0.25 집중 재시험에서도 최대 phone 점수 0.154, 상태 확정 0프레임
- 휴대폰 재시험 얼굴 유효 324/330, 보정 298프레임으로 얼굴 파이프라인 이상은 배제
- 영상·이미지는 저장하지 않고 JSON 수치 로그만 보존

눈 감기와 하품의 상태 전이·경고 연결은 실제 환경에서 확인했다. 고개 돌림은 얼굴 유실 대응을 개선해야 하며, 휴대폰 YOLO는 데이터 도메인 차이로 실제 웹캠 기준을 충족하지 못해 운영 모델로 승격하지 않는다. 상세 결과: `docs/results/CONTROLLED_WEBCAM_TEST.md`.

## 단계 21. 고개 돌림 단기 추적 개선

**상태:** 완료, 좌우 비대칭 후속 튜닝 필요  
**기록일:** 2026-11-10

- 유실 직전 yaw·pitch·gaze가 임계값을 넘었을 때만 attention 타이머를 최대 2초 유지
- 정면에서 발생한 얼굴 유실은 추론하지 않고 기존처럼 `unknown` 유지
- 얼굴 유실 시 눈 감김·하품 타이머는 즉시 초기화해 가림 시간의 오검출 방지
- 장시간 얼굴 유실과 프레임 시간 간격 초과 시 추적 상태 초기화
- 실제 통제 시험 660프레임 로그 재생에서 `off_forward` 0→81프레임 개선
- 추적·비추적·만료 조건 단위 테스트 추가, 전체 85개 테스트 통과
- 실시간 재시험은 수동·자동 노출 모두 입력 얼굴 0/390으로 판정 대상에서 제외

로직 개선 효과는 얼굴이 정상 입력된 실제 로그에서 확인했다. 카메라가 다시 사람을 향한 상태에서 좌·우 고개 돌림 현장 시험을 한 차례 더 수행하면 이 단계를 최종 완료할 수 있다.

### 최종 현장 재시험

- `/dev/video0`, 390프레임에서 얼굴 유효 313프레임, 보정 상태 342프레임
- 정상상태 15.000 FPS 유지
- `off_forward` warning 이벤트 2회 발생
- 얼굴 미검출 중 직전 회전 추적으로 `off_forward` 18프레임 유지
- 좌측 회전은 검출, 우측 시험 구간은 대부분 `forward`로 판정

얼굴 유실 추적의 현장 동작을 확인해 단계를 완료했다. 우측 회전 민감도는 카메라 설치 각도와 개인별 중립 자세의 영향을 함께 확인하는 후속 개선 항목으로 유지한다.

## 단계 22. 라이브 정답 라벨과 이벤트 평가 연결

**상태:** 구현 완료  
**기록일:** 2026-11-10

- 라이브 화면에 `0~4`, `U` 수동 정답 라벨 단축키 추가
- normal·eyes_closed·away·yawn·phone·unknown 라벨을 프레임 JSONL에 기록
- 라벨 전환 시점은 별도 `manual_labels.jsonl`에 기록
- 눈 감김→drowsiness/suspected, 고개 이탈→attention/off_forward, 하품→yawn/yawning, 휴대폰→action/phone 매핑
- 이벤트 평가기에 `--embedded-labels` 옵션 추가
- 내장 라벨에서 이벤트 Precision·Recall·미탐·오경보/분·탐지 지연 자동 산출
- 10프레임 headless 호환성 smoke test와 전체 회귀 테스트 87개 통과

이제 동일한 웹캠 시나리오에서 사용자가 동작 시작·종료 키만 누르면 별도 CSV 작성 없이 이벤트 성능을 재현 가능하게 계산할 수 있다. 다음 단계는 GUI 상태에서 한 번의 라벨 시험을 수행하고 생성된 평가 JSON을 최종 결과에 반영하는 것이다.

## 단계 23. 라벨 웹캠 이벤트 평가와 영상 저장

**상태:** 완료  
**기록일:** 2026-11-10

- 실제 웹캠 660프레임·44.66초에서 4개 동작 수동 라벨 입력
- 눈 감김, 고개 이탈, 하품 이벤트 Recall 100%; 휴대폰 Recall 0%
- 하품 Precision 100%, 눈 감김·고개 이탈 Precision 50%
- 눈 감김 평균 탐지 지연 1.80초, 고개 이탈 1.14초, 하품 1.26초
- 주석 영상 660프레임과 상태·라벨·평가 JSON 저장
- 카메라 30 FPS 메타데이터로 인한 2배속 재생을 확인하고 14.777 FPS 실시간 보정본 생성
- 향후 녹화 재생 시간을 맞추기 위한 `--record-fps` 옵션 추가

단일 반복 기능 시험에서 눈·주의·하품 이벤트의 end-to-end 연결을 정량 확인했다. 휴대폰 미탐과 눈·주의 추가 이벤트는 최종 제한사항으로 보고하며, 여러 사람·조명 조건 반복 전에는 일반화 성능으로 해석하지 않는다. 상세 결과: `docs/results/LABELED_WEBCAM_EVENTS.md`.

## 단계 24. 휴대폰 저임계값 집중 재측정

**상태:** 기술 검출 확인, 정답 구간 재측정 필요  
**기록일:** 2026-11-10

- YOLO confidence 0.25, 실행 주기 3프레임으로 405프레임·27초 측정
- phone 상태 138프레임 확정, 최대 점수 0.560
- 검출 박스 132프레임·누적 222개로 실제 웹캠에서도 기술 검출 확인
- 405프레임·15 FPS 주석 영상 저장
- 수동 `4` 라벨 입력이 기록되지 않아 이벤트 Recall·지연 평가는 제외
- 1.13초부터 phone 예측이 시작되어 낮은 임계값의 오경보 위험 확인

confidence 0.25는 휴대폰 민감도를 높이지만 기존 validation 오경보 문제와 이번 조기 예측이 함께 확인됐다. 운영 임계값은 변경하지 않으며, 다음 반복에서는 키보드 포커스에 의존하지 않는 사전 시간표 라벨 기능으로 정답 구간을 보장한다.

## 단계 25. 휴대폰 자동 시간표 정답 평가

**상태:** 완료, 운영 기준 미달 확정  
**기록일:** 2026-11-10

- JSON 구간으로 ground truth를 자동 기록하는 `--label-schedule` 구현
- 겹침·잘못된 라벨·시간 범위를 검증하고 시간표 SHA-256을 summary에 기록
- normal 6초→phone 15초→normal 6초 시나리오를 키 입력 없이 측정
- 휴대폰 이벤트 Recall 100%, Precision 16.67%
- 오경보 5회·10.84회/분, 미탐 0회
- phone 정답 225프레임 중 162프레임 검출
- normal 정답 170프레임 중 134프레임을 phone으로 오판
- 405프레임·15 FPS·27초 주석 영상과 평가 JSON 저장
- 시간표 단위 테스트 추가, 전체 회귀 테스트 89개 통과

confidence 0.25의 높은 recall과 높은 오경보가 실제 웹캠에서도 함께 재현됐다. 현 AI Hub YOLO는 휴대폰 운영 경고 모델로 승격하지 않으며, 휴대폰 물체 중심 데이터 또는 근거리 웹캠 도메인 데이터가 추가되기 전까지 연구용으로 유지한다.

## 단계 26. 최종 검증 정리와 승인 상태 고정

**상태:** 완료  
**기록일:** 2026-11-10

- 6개 준비 데이터셋의 로컬 readiness 재확인
- 완료된 manifest·MRL Eye 평가·640×480 성능·현장 데모 항목을 로드맵에 반영
- MediaPipe·OpenVINO 운영 승인, 행동 모델 연구용 한정 상태 명시
- 눈 감김·하품·주시 이탈 현장 이벤트 결과와 휴대폰 운영 기준 미달을 한 문서로 통합
- 실행·자동 라벨·이벤트 평가 재현 명령 정리
- DMD·Drive&Act·L2CS-Net·6DRepNet은 종합 DMS 후반까지 도입 보류

초기 DMS의 구현·측정·재현 문서는 완료했다. 남은 핵심은 환경 조건 반복 평가와 실제 웹캠 도메인의 휴대폰 데이터 추가이며, 부정확한 행동 모델의 ONNX 최적화는 재학습 이후로 미룬다. 상세 결과: `docs/FINAL_VALIDATION.md`.

## 단계 27. 환경 조건 집계와 하단 가림 시험

**상태:** baseline·가림 완료, 안경·역광·야간·카메라 위치 남음  
**기록일:** 2026-11-10

- live 프레임에 평균 밝기, summary에 조건명 기록 추가
- 여러 실행의 얼굴 검출률·보정률·unknown coverage·밝기·FPS 자동 집계 도구 구현
- 유효 baseline 300프레임에서 얼굴 검출 100%, any-unknown 31%, 정상상태 14.994 FPS
- 입·코 하단 가림 300프레임에서 얼굴 검출 75.67%, any-unknown 53%
- 가림 조건에서 no_face 63프레임, multiple_faces 10프레임
- 가림 전후 정상상태 FPS는 동일해 성능 병목이 아닌 얼굴 가시성 문제로 확인
- 집계 단위 테스트 추가, 전체 회귀 테스트 90개 통과

환경 실패율 측정 파이프라인을 완성하고 첫 비교를 수행했다. 안경·역광·저조도·카메라 위치 조건은 동일 프로토콜로 이어서 추가한다. 상세 결과: `docs/results/ENVIRONMENT_CONDITIONS.md`.

## 단계 28. 현장 측정 팀원 인계와 Git 정리

**상태:** 인계 준비 완료  
**기록일:** 2026-11-10

- 추가 현장 측정은 팀원이 같은 300프레임 프로토콜로 수행하도록 범위 전환
- 안경·역광·저조도·카메라 상하좌우 조건, 실행 명령, 집계 명령, 판정 항목 문서화
- 실행 로그·얼굴 영상·데이터·가중치·원본 PPT의 Git 제외 원칙 명시
- 휴대폰 모델 운영 불승인과 confidence 0.25 변경 금지 사항 인계
- README와 ROADMAP에서 인계 문서 연결

코드·결과 요약·재현 문서만 Git에 게시하고 로컬 데이터와 실제 얼굴 산출물은 제외한다. 상세 인계: `docs/TEAM_HANDOFF.md`.

## 단계 29. Windows 발표용 통합 데모 패키지

**상태:** 스크립트·문서 완료, Windows PC 실기 확인 필요  
**기록일:** 2026-11-10

- 기존 얼굴·눈 기본 데모와 분리된 `setup_showcase.ps1` 추가
- MediaPipe·OpenVINO·PyTorch·Ultralytics 설치와 공식 얼굴·눈 모델 다운로드 자동화
- Ultralytics 설치 후 OpenCV contrib 빌드를 복구해 `cv2` 배포판 충돌 방지
- `run_showcase.ps1`에서 3모델, 경고음, 녹화, 카메라·영상 입력, CPU/GPU 장치 선택 지원
- 승인된 YOLO 체크포인트 존재 여부를 실행 전에 검사
- 발표 순서, 기능 목록, 휴대폰 모델 한계, 가중치 전달 방법 문서화
- Linux 회귀 테스트 90개 통과

현재 저장소만으로 얼굴·눈 기본 Windows 데모는 실행할 수 있다. 전체 행동 쇼케이스는 Git 제외된 `aihub_abnormal_yolo11n.pt`를 Windows의 `models` 폴더에 별도 복사해야 한다. Linux 환경에는 PowerShell 실행기가 없어 구문·실기 최종 확인은 Windows PC에서 수행한다. 상세 절차: `docs/WINDOWS_SHOWCASE.md`.
