# 참고 자료와 출처

범위·역할·일정의 기준은 팀이 제공한 확정 「AI P-실무프로젝트 과제 계획서 양식_미완1」입니다. 이번 v0.2가 해당 확정 내용을 우선 반영합니다. 원본 문서와 학번·연락처·비공개 멘토링 자료는 저장소에 넣지 않습니다.

## 실제 구현에 사용

- [MediaPipe Face Landmarker Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python): Tasks VIDEO API, 타임스탬프, 모델 옵션.
- [Face Landmarker 개요·모델](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/index): 얼굴 랜드마크·변환 행렬·다인 설정. 이 출력은 실제 gaze 정확도 보증이 아님.
- [공식 version 1 모델](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task): 로컬 다운로드, 해시 검증; Git 재배포 제외.
- [공식 MediaPipe 테스트 이미지](https://storage.googleapis.com/mediapipe-assets/portrait.jpg): 통합 검사 입력, Git 재배포 제외. 운전자 데이터셋이 아님.
- [Torchvision MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html): 선택적 전이학습·추론 아키텍처. ImageNet 가중치는 행동 인식 완성 모델이 아님.
- [actions/checkout](https://github.com/actions/checkout), [actions/setup-python](https://github.com/actions/setup-python): CI 구성.

## 확정안의 데이터·연구 후보

접근·사용 허가를 완료한 목록이 아닙니다.

- [UTA-RLDD](https://sites.google.com/view/utarldd/home)
- [A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection](https://arxiv.org/abs/1904.07312)
- [State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data)
- [Driver Monitoring Dataset](https://dmd.vicomtech.org/)
