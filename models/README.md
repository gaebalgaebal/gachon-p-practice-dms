# 로컬 모델

`python scripts/download_model.py`로 공식 MediaPipe Face Landmarker float16 version 1을 내려받습니다. 모델 SHA256은 다운로드 도구와 실행 summary.json에서 확인합니다. 모델 파일은 Git에서 제외합니다.

선택적 행동 모델은 `scripts/train_action.py`로 승인된 데이터에서 학습하고 `--action-checkpoint models/actions.pt`로 연결합니다. 저장소에 통화·음료 인식 가중치는 없습니다. ImageNet 기본 모델이나 smoke_only 테스트 가중치를 완성된 행동 모델로 사용하지 않습니다. 사용·재배포 조건과 학습 데이터 접근 권한은 별도로 관리합니다.
