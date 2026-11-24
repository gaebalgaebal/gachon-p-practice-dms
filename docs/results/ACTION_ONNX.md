# 행동 분류 ONNX·OpenVINO 검증

검증일: 2026-11-24

## 대상과 목적

State Farm MobileNetV3-Small 기준선을 ONNX opset 17로 변환하고 OpenVINO CPU에서 실행했다. 이 모델은 변환 경로와 실행 성능 확인용이며, 근거리 웹캠의 운영 휴대폰 모델로 승인된 것은 아니다.

- 원본 체크포인트 SHA-256: `fc5bc0cf38332c688140c1d1a589cf198641020501d1f4647ffcd0451253fe0a`
- ONNX SHA-256: `ebdbed5654068019439f491fe9f5e4886f6315238cac6c874092f9f432743020`
- 입력: RGB, ImageNet 정규화, resize 256 후 center crop 224
- 출력 클래스: drink / normal / phone
- 장치: OpenVINO CPU

## 변환 동일성 및 속도

State Farm의 보지 않은 test 운전자 이미지 100장을 PyTorch와 OpenVINO에 동일하게 입력했다.

| 항목 | 결과 |
|---|---:|
| 최종 클래스 일치율 | 100% |
| 최대 확률 절대 오차 | 0.00000906 |
| PyTorch 단일 이미지 평균 | 3.91ms |
| OpenVINO 단일 이미지 평균 | 2.67ms |
| 평균 시간 감소 | 31.7% |

측정값은 이미지 읽기 후 전처리와 추론을 포함한 순차 실행 평균이며 전체 라이브 파이프라인 FPS와 동일한 지표가 아니다.

별도 통합 smoke에서는 MediaPipe 얼굴 모델, OpenVINO 눈 모델, OpenVINO 행동 ONNX를 640×480 영상 30프레임에 함께 연결해 32.85 processing FPS와 정상 종료를 확인했다. 이는 파일 입력 단기 연결 시험이며 실제 카메라 성능 수치로 사용하지 않는다.

## 재현

```bash
python scripts/export_action_onnx.py \
  models/state_farm_mobilenet_v3.pt models/state_farm_mobilenet_v3.onnx

python scripts/validate_action_onnx.py \
  models/state_farm_mobilenet_v3.pt models/state_farm_mobilenet_v3.onnx \
  data/manifests/state_farm_actions.csv --samples 100
```

라이브에서는 `.onnx` 확장자를 자동으로 OpenVINO backend로 선택하며 같은 이름의 `.onnx.metadata.json` 파일이 필요하다. 향후 팀원 데이터로 승인 모델이 만들어지면 동일 절차로 체크포인트만 교체한다.
