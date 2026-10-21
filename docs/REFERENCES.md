# 참고 자료

프로젝트 범위·일정·제안 목표의 출처는 팀이 제공한 「AI P-실무프로젝트(캡스톤디자인) 과제 계획서」 초안입니다. 원본 문서, 개인 연락처 및 비공개 멘토 회신 원문은 업로드하지 않습니다. 저장소의 초기 구현·파일 구조·CI는 해당 계획을 바탕으로 추가한 개발용 구성입니다.

## 기술 문서

- [MediaPipe Face Landmarker Python 가이드](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python): 얼굴 랜드마크·표정 관련 출력과 영상 처리 모드. 실제 gaze 벡터 정확성을 보증하는 근거로 사용하지 않음.
- [Torchvision MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html): 행동 분류의 제안 후보, 구현 전에 현재 API·모델 사용 조건 확인.
- [actions/checkout](https://github.com/actions/checkout), [actions/setup-python](https://github.com/actions/setup-python): CI 구성 참고.

## 데이터·연구 후보

아래는 계획서에 기재된 참고 링크이며 접근성·라이선스를 승인한 목록이 아닙니다.

- [UTA-RLDD](https://sites.google.com/view/utarldd/home)
- [A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection](https://arxiv.org/abs/1904.07312)
- [State Farm Distracted Driver Detection 데이터](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data)
- [Driver Monitoring Dataset](https://dmd.vicomtech.org/)
