# 얼굴 활성 3모델 자원 성능

## 조건

- YawDD 하품 영상 300프레임, 640×480 처리, headless
- MediaPipe Face Landmarker + OpenVINO eye + AI Hub YOLO11n 동시 실행
- YOLO confidence 0.75, 실행 주기 5프레임, CUDA 사용
- 녹화·이미지 저장 없음
- 결과: `runs/resource_benchmark_face_active_20261110/summary.json`

## 결과

| 지표 | 결과 |
|---|---:|
| 얼굴 유효 프레임 | 300 / 300 |
| 보정 완료 이후 눈 모델 출력 | 166 / 300 |
| YOLO 행동 점수 출력 | 300 / 300 |
| 전체 처리 FPS | 39.42 |
| 최초 30프레임 제외 정상상태 FPS | 42.83 |
| 전체 평균 루프 | 25.30 ms |
| 정상상태 p95 루프 | 39.83 ms |
| 프로세스 CPU 사용량(단일 코어 환산) | 131.66% |
| 프로세스 RSS 시작 / 종료 / 표본 최대 | 97.79 / 1,689.65 / 1,710.11 MB |
| CUDA 현재 allocated / reserved | 42.00 / 68.00 MB |
| CUDA 최대 allocated / reserved | 59.71 / 68.00 MB |

CPU 백분율은 전체 논리 코어 합계가 아니라 단일 코어 100% 기준이라 멀티스레드 실행에서 100%를 넘을 수 있다. RSS에는 Python, MediaPipe, OpenVINO, PyTorch·YOLO 런타임과 모델 메모리가 모두 포함된다. CUDA 수치는 모델 로드 이후 추론 구간의 PyTorch 메모리다.

이 결과로 얼굴 활성 상태에서 세 모델의 연결과 15 FPS 이상 처리 여유를 확인했다. 다만 파일을 가능한 속도로 처리한 기술 벤치마크이므로 실제 카메라 캡처 지연을 포함하지 않는다. 실제 웹캠에서 운전자가 정면에 있는 조건의 최종 측정은 별도로 남아 있다.
