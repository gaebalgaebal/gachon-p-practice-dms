# 실제 파이프라인

## 추론

OpenCV 입력 → 시간 정보 → 크기 조정·ROI → MediaPipe Face Landmarker → 얼굴 유효성 검사 → EAR·얼굴 각도·홍채 진단값 → 정면/눈 뜬 보정 → 시간 기반 졸음·주시 판정 → 선택적 행동 CNN → 화면·경고·프레임/이벤트 JSONL.

파일은 영상 시간으로 판정하며 가능한 속도로 처리합니다. 카메라는 monotonic capture 시간으로 판정합니다. 프레임 지연·로그 I/O·화면까지 포함하는 루프 시간과 전체 처리 FPS를 저장하며 카메라 센서부터 화면까지의 실제 지연을 측정한 값은 아닙니다. 버퍼 크기를 요청하지만 백엔드에 따라 준수 여부는 다릅니다.

## 데이터와 학습

이용 조건 확인 → 로컬 이미지/영상 확보 → 라벨 매핑 → 운전자/촬영 세션 기준 분할 → manifest 검증 → train 전용 증강 → MobileNetV3-Small 특징망 고정·분류기 학습 → val Macro F1로 모델 선택 → test 1회 평가 → 가중치·클래스·전처리·manifest 해시 저장 → 선택적 데모 연결.

`scripts/train_action.py`는 실제 로컬 이미지 manifest가 필요합니다. 사용 조건을 대신 확인하거나 데이터를 자동 수집하지 않습니다. 이미지 해시가 split 간 같으면 학습을 거부합니다. 영상 프레임 간 유사성·실제 운전자 ID 오류는 별도 점검해야 합니다.

기본 전이학습은 ImageNet 사전학습 특징을 사용하되 행동 클래스 분류기를 새로 학습합니다. 수평 반전은 운전자 좌우/차량 배치 의미를 바꿀 수 있어 기본 증강에서 제외합니다. 학습·추론 모두 RGB, resize 256, crop 224, ImageNet 정규화를 맞춥니다. 데모 ROI도 학습 이미지의 운전자 상반신 구도와 일치시켜야 합니다.

## 평가

프레임 로그 + 별도 수동 정답 CSV → 프레임 ID 정합 → task별 Accuracy/Macro F1/Precision/Recall/혼동행렬/coverage → 단일 EAR Baseline 비교. unknown은 정답 누락(FN)으로 계산하며 조용히 제외하지 않습니다. 이벤트 오경보/분·탐지 지연·3상태 졸음 평가 도구는 후속 작업입니다.

## 협업·CI

이슈 → 기능 브랜치 → 로컬 검사 → PR/팀 리뷰 → main 반영.

- core: Windows/Linux × Python 3.10/3.12, 기본 패키지·규칙·분할·합성 CSV 검사.
- vision: Windows/Linux × Python 3.12, 실제 의존성·모델 해시·기하 테스트·공식 예제 기반 영상 통합 검사.
- action: Linux CPU, 선택적 PyTorch 환경·무작위 이미지 학습 경로 검사. 생성 가중치는 실제 데모 사용 금지로 표시합니다.

CI에는 실제 운전자 영상·비공개 데이터·카메라 접근·완성 모델의 인식 성능 평가를 넣지 않습니다. 공식 모델과 테스트 이미지는 CI에서 로컬로 다운로드하고 재배포하지 않습니다.

설치/실행법은 [데모 안내](DEMO_GUIDE.md), 검사 근거는 [검증 기록](VALIDATION.md)에 있습니다. CI 액션은 [checkout](https://github.com/actions/checkout), [setup-python](https://github.com/actions/setup-python) 공식 문서의 사용 방법을 따릅니다.
