# 참고 자료와 출처

범위·데이터셋·모델의 기준은 `운전자감시카메라_프로젝트_4x3_20261030_1119.pptx`와 [DMS 확정 구현 계획](IMPLEMENTATION_PLAN.md)이다. 아래 링크는 접근·사용 허가가 완료되었다는 의미가 아니다.

## 모델

- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker): 얼굴·눈·입 랜드마크와 변환 행렬
- [OpenVINO open-closed-eye-0001](https://docs.openvino.ai/nightly/omz_models_model_open_closed_eye_0001.html): 좌우 눈 개폐 분류
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11): 휴대폰·컵 검출과 행동 분류
- [6DRepNet](https://github.com/thohemp/6DRepNet): 후반 Head Pose 비교 후보
- [L2CS-Net](https://github.com/Ahmednull/L2CS-Net): 후반 gaze 확장 후보

## 초기 데이터셋

- [AI Hub](https://aihub.or.kr): 졸음운전 및 운전자·탑승자 이상행동 데이터
- [NTHU-DDD 소개 논문](https://arxiv.org/abs/1910.09722): IR 졸음운전 영상
- [YawDD](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset): 하품 영상
- [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset): 눈 뜸·감김 이미지
- [State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data): 주의분산 이미지

## 후반 데이터셋

- [Driver Monitoring Dataset](https://dmd.vicomtech.org/): 종합 DMS
- [Drive&Act](https://openaccess.thecvf.com/content_ICCV_2019/html/Martin_DriveAct_A_Multi-Modal_Dataset_for_Fine-Grained_Driver_Behavior_Recognition_in_ICCV_2019_paper.html): 다중 시점 세부 행동
