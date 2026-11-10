# State Farm 행동 분류 베이스라인

변환·학습일: 2026-11-03  
원본: State Farm Distracted Driver Detection train 22,424장·26명

## 초기 DMS 라벨 변환

프로젝트 초기 범위인 normal/phone/drink에 맞춰 공식 클래스를 다음처럼 변환했다.

| 원본 | 공식 의미 | 프로젝트 라벨 |
| --- | --- | --- |
| c0 | normal driving | normal |
| c1, c3 | texting right/left | phone |
| c2, c4 | talking on phone right/left | phone |
| c6 | drinking | drink |
| c5, c7, c8, c9 | radio/reaching/hair-makeup/passenger | 초기 학습 제외 |

공식 클래스와 데이터 설명: <https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data>

원본은 대회 규칙 적용 대상이며 Git이나 모델 배포물에 포함하지 않는다. 또한 공식 설명상
통제 환경에서 촬영된 데이터이므로 실제 운전 환경 일반화 성능으로 해석하지 않는다.

## Manifest

- 전체 사용 이미지: 14,070장
- 피험자: 26명
- train: 18명·9,355장
- validation: 4명·2,425장
- test: 4명·2,290장
- 라벨 분포: normal 2,489 / phone 9,256 / drink 2,325
- 운전자·세션 분할 누수 없음

phone이 다수 클래스이므로 train split의 역빈도 class weight를 사용했다. 무작위 이미지
분할은 같은 운전자 누수를 일으키므로 사용하지 않았다.

## MobileNetV3-Small 베이스라인

- ImageNet 사전학습 MobileNetV3-Small
- feature extractor 고정, classifier만 10 epoch 학습
- 224×224 crop, 밝기·대비 augmentation
- AdamW, learning rate 0.001, batch 128
- validation macro F1 기준 최고 체크포인트 선택
- RTX 5080, PyTorch 2.9.1+cu128

최고 validation은 epoch 7의 Accuracy 85.57%, macro F1 80.62%였다. 이 설정을
변경하지 않고 test 운전자 4명에 한 번 평가했다.

| Test 지표 | 결과 |
| --- | ---: |
| Accuracy | 72.05% |
| Macro F1 | 65.87% |
| Coverage | 100% |

| 클래스 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| drink | 50.99% | 85.90% | 63.99% |
| normal | 70.00% | 45.53% | 55.17% |
| phone | 82.34% | 74.92% | 78.45% |

운전자별 외형·차량 배경 차이로 validation과 test 간 격차가 크다. 이 모델은 분류
베이스라인으로 사용하며, 최종 행동 모델은 AI Hub 이상행동 객체 검출 결과와 비교한다.

## 출력

- `data/manifests/state_farm_actions.csv`
- `models/state_farm_mobilenet_v3.pt`
- `models/state_farm_mobilenet_v3.metrics.json`

데이터 manifest와 모델 산출물은 Git 제외 경로다. 코드와 결과 요약 문서만 추적한다.
