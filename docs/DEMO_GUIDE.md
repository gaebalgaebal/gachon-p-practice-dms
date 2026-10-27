# 실행·학습·평가 안내

## 설치

Python 3.10~3.12 64비트를 권장합니다. Windows PowerShell에서 `./setup_demo.ps1`, `./run_demo.ps1` 순서로 실행하거나 README의 Python 명령을 직접 사용합니다. 스크립트는 시스템 실행 정책을 바꾸지 않습니다.

Ubuntu처럼 최소 설치된 Linux에서는 MediaPipe CPU 실행에도 EGL/GLES 공유 라이브러리가 필요할 수 있습니다. `libGLESv2.so.2` 또는 EGL 로딩 오류가 나면 `sudo apt-get install libegl1 libgles2 libgl1`로 시스템 의존성을 준비합니다. GUI 화면을 쓰려면 별도 데스크톱 세션도 필요하고 서버에서는 `--headless`를 사용합니다. GitHub의 Linux 영상 검사에도 이 준비 단계를 포함합니다.

`pip install -e '.[vision]'`은 MediaPipe 0.10.35, OpenCV-contrib 4.13.0.92, NumPy 2.2.6을 설치합니다. OpenCV-contrib에는 일반 cv2 API가 포함됩니다. opencv-python이나 headless 배포판을 같은 환경에 추가로 설치하지 마세요. 주요 직접 의존성은 고정했고 전이 의존성 전체 잠금은 아직 없습니다. 재현 실험 때 `python -m pip freeze`도 기록하세요.

```sh
python scripts/download_model.py
python scripts/check_camera.py --camera 0 --frames 60
python -m dms.live --camera 0
python -m dms.live --camera 1 --roi 0.1,0,0.8,1
python -m dms.live --video data/my_video.mp4 --headless --output runs/my-run
```

모델 다운로드는 인터넷이 필요하고, 이후 얼굴 추론은 로컬에서 실행됩니다. 저장소 루트를 현재 작업 폴더로 사용합니다. 웹캠 0번이 실패하면 다른 카메라 앱을 닫고 OS 개인정보 설정과 장치 번호를 확인하세요. Windows는 DirectShow 후 기본 백엔드로 재시도합니다. 카메라 입력 실패는 정상 종료로 처리하지 않습니다.

## 보정·화면·오류

- 한 사람의 얼굴과 상반신이 잘 보이도록 ROI를 설정합니다. ROI 좌표는 원본 비율을 유지해 축소한 유효 영상 영역 기준 0~1의 x,y,w,h입니다.
- 정면·눈 뜬 상태를 2초 이상 유지하고 Calibration: ready를 기다립니다. 조명이 좋고 안경 반사가 적은 환경에서 시작합니다.
- 카메라·앉은 자세·사람이 바뀌면 C로 재보정합니다. 고개를 돌리거나 눈을 감은 채 보정하면 기준을 잘못 잡을 수 있습니다.
- Q/Esc 종료. 얼굴 유실·다인 입력·작은 얼굴·큰 회전은 unknown입니다. 얼굴 가림/안경의 모든 오류를 자동 탐지하는 것은 아닙니다.
- 화면 표시는 OpenCV 기본 글꼴을 사용해 영어입니다. 코드와 개발 문서는 한국어로 안내합니다.
- Head attention은 보정된 얼굴 방향과 사용 가능한 경우 홍채 상대 위치를 결합합니다. 홍채가 불안정하면 Head Pose Baseline으로 폴백합니다. 이 값은 실제 3차원 시선 벡터가 아닙니다.
- `--sound`는 Windows 경고음, `--record`는 주석 영상 저장을 켭니다. 기본 영상 저장은 꺼져 있습니다.
- 파일은 처리 가능한 속도로 실행합니다. 영상 FPS 메타데이터가 없으면 `--fps 30`처럼 명시해야 합니다.
- headless 카메라는 `--max-frames 300` 같은 종료 조건이 필요합니다.
- 결과 폴더가 비어 있지 않으면 덮어쓰지 않습니다. `--output`에 새 경로를 지정하세요.

## 실행 결과

| 파일 | 내용 |
| --- | --- |
| frames.jsonl | 프레임 ID·시간·상태·EAR·원시/보정 얼굴 각도·홍채 상대 위치·유효성·단일 EAR Baseline·처리 시간 |
| events.jsonl | 상태 변경·경고(3초 간격 제한)·재보정 이벤트 |
| calibration.json | 마지막 정면·눈 뜬 기준과 유효 임계값 |
| summary.json | 프레임 수·처리 FPS·루프 지연·unknown 비율·버전·모델 해시·ROI·설정 |
| annotated.mp4 | --record를 켠 경우만 저장 |

로그에는 이미지나 랜드마크 전체를 저장하지 않습니다. 그래도 실제 사람의 행동·상태 기록이므로 저장·공유 범위를 관리하세요. 녹화는 처리 프레임을 소스 FPS로 기록하며 카메라에서 프레임 드롭이 있으면 실제 경과 시간과 재생 시간이 달라질 수 있습니다. 루프 시간은 센서/디스플레이 지연 측정이 아닙니다.

## 선택적 행동 모델

승인된 데이터에서 통화·음료 이미지를 준비한 후 실행합니다. 기본 데모에 해당 데이터나 완성 가중치는 없습니다.

```sh
python -m pip install 'torch==2.9.1' 'torchvision==0.24.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[action]'
python scripts/train_action.py --manifest data/manifests/actions.csv --output models/actions.pt --epochs 10
python -m dms.live --camera 0 --action-checkpoint models/actions.pt --action-threshold 0.7 \
  --action-interval 3 --action-window 5 --action-min-samples 3 --action-consensus 0.6
```

학습 manifest 필수 컬럼:

```csv
sample_id,subject_id,session_id,split,label,file_path
sample_001,driver_01,session_01,train,normal,../processed/sample_001.jpg
```

전체 manifest에는 train/val/test와 각 클래스가 모두 있어야 합니다. 운전자·세션은 split 간 겹치면 안 됩니다. file_path는 manifest 파일의 폴더 기준 상대 경로 또는 절대 경로입니다. 얼굴만 크롭하지 말고 실제 추론과 일치하는 운전자 상반신 구도를 사용합니다.

기본 클래스는 normal/phone/drink이며 데이터가 확보되면 eat/smoke를 추가할 수 있습니다. State Farm 원본 c0/c2/c4/c6 등의 매핑은 실제 제공 라벨과 규정을 확인해서 팀에서 명시적으로 변환해야 합니다. train split에 없는 클래스는 val/test에 둘 수 없습니다.

MobileNetV3-Small ImageNet 특징망을 고정하고 분류기를 학습합니다. 밝기·대비·crop 증강은 train에만 적용합니다. seed·분할 해시·가중치 해시와 클래스별 test 지표를 기록합니다. 임계값은 val에서 정하고 test 결과를 보고 조정하지 않습니다. softmax 점수는 보정된 확률이 아닙니다. 현재는 매 프레임 fresh inference이므로 모델 연결 후 FPS를 재측정해야 합니다.

행동 CNN은 기본적으로 유효 프레임 3개마다 한 번 실행하고 최근 5개 fresh 예측의 다수결을 사용합니다. 얼굴 또는 ROI가 무효가 되면 과거 행동을 즉시 버리고 unknown으로 돌아갑니다. 이 설정은 실제 val 영상과 목표 PC에서 조정하고 test 결과를 본 뒤 바꾸지 않습니다.

`python scripts/action_smoke.py`는 무작위 픽셀로 학습 코드의 동작만 검사합니다. 생성 가중치는 smoke_only로 표시해 live에서 거부하며 실제 행동 인식 성능으로 보고하지 않습니다.

## 프레임 평가

수동 정답 CSV를 별도로 만들고 영상 프레임 번호를 맞춥니다.

```csv
frame_index,drowsiness,attention
100,normal,forward
101,suspected,
```

```sh
python scripts/evaluate_states.py --predictions runs/my-run/frames.jsonl --labels data/labels.csv --output runs/evaluation.json
```

졸음 라벨은 normal/suspected, 주시는 forward/off_forward입니다. 빈 task 라벨은 미평가이고, 예측 unknown은 FN으로 반영합니다. 수동 라벨 자체가 실제 졸음 진단을 뜻하지 않습니다. 각성/저주의/졸음 3상태와의 매핑에는 별도 학습이 필요합니다.

## 이벤트 평가

정답 이벤트는 다음 스키마로 기록합니다.

```csv
task,label,start_s,end_s
drowsiness,suspected,12.0,16.5
attention,off_forward,25.0,28.0
```

```sh
python scripts/evaluate_events.py \
  --predictions runs/my-run/frames.jsonl \
  --labels data/event_labels.csv \
  --output runs/event-evaluation.json
```

결과에는 이벤트 Precision/Recall, 미탐 수, 오경보 수와 분당 오경보, 탐지 지연이 포함됩니다. 기본 0.5초 허용 범위는 평가 전에 고정하고 test 결과를 본 뒤 변경하지 않습니다.
