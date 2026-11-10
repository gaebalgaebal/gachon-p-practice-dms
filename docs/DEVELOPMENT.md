# 확정 계획과 현재 구현 대응

개발 기준은 [DMS 확정 구현 계획](IMPLEMENTATION_PLAN.md)이다. 기존 코드는 폐기 대상이 아니라 새 계획에서 재사용할 베이스라인으로만 취급한다.

| 확정 항목 | 현재 기반 | 다음 구현 |
| --- | --- | --- |
| Python·OpenCV RGB 입력 | 웹캠·영상·ROI·화면 출력 | 물리 카메라별 연결과 640×480 성능 측정 |
| MediaPipe 얼굴 분석 | 랜드마크·좌우 EAR·Head Pose·홍채 상대 위치 | MAR·입 랜드마크·눈 개방도 그래프 |
| OpenVINO 눈 개폐 | 미구현 | 모델 다운로드·좌우 눈 crop·EAR 교차 판정 |
| YOLO11 행동 인식 | 기존 MobileNetV3 실험 경로만 존재 | State Farm 분류와 AI Hub 이상행동 검출 파이프라인 |
| 계획서 데이터셋 | 부분 확보 | MRL Eye·YawDD·State Farm 다운로드 완료, NTHU-DDD는 UTA-RLDD Fold 1~4로 대체 |
| 독립 상태·경고 | 졸음·주시·행동 상태, JSONL 로그 | Eye·Yawn 상태와 모델 점수, 원인별 경고 |
| 목표 UI | 영상·눈 포인트·오른쪽 상태 패널 | 좌우 눈 그래프·입 포인트·행동 박스·모델별 점수 |
| 평가 | 프레임·이벤트 평가 도구 | 데이터셋별 F1·mAP와 15 FPS 통합 측정 |

## 첫 구현 단위

1. `configs/datasets.json`을 데이터셋·과제 조합의 단일 기준으로 사용한다.
2. `scripts/validate_manifest.py --profile dms`로 공통 manifest를 검사한다.
3. 실제 데이터 확보 후 데이터셋별 변환기를 하나씩 추가한다.
4. 모델 구현은 MediaPipe 기능 보강, OpenVINO 눈 개폐, YOLO11 순으로 진행한다.

## 완료의 의미

단위 테스트나 예제 영상 실행만으로 기능 완료를 선언하지 않는다. 운전자 단위 test 결과, 실제 웹캠 동작, 모델 3개 동시 실행, 640×480에서 15 FPS, 목표 UI·경고·로그를 모두 확인해야 초기 구현 완료로 기록한다.
