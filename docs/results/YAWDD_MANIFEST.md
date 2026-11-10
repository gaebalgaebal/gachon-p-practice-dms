# YawDD 영상 manifest 및 구간 라벨링 기반

변환일: 2026-11-01  
입력: `data/raw/yawdd/YawDD dataset`의 AVI 349개

## 변환 정책

- 성별과 번호를 함께 사용해 90명의 고유 피험자를 구성한다.
- 고정 seed 42, 피험자 단위 70/15/15 분할로 영상 누수를 방지한다.
- Mirror 파일명의 Normal/Talking은 `normal`, Yawning/Talking&Yawning은
  `yawn` 약한 라벨로 보존한다.
- Dash 영상은 여러 행동이 이어지는 sequence이므로 `unknown`으로 두고 수동 구간
  라벨링 대상으로 지정한다.
- 영상 단위 yawn 라벨은 하품 전체 구간을 뜻하지 않는다. 정확한 MAR 평가는
  시작·종료 구간 CSV를 별도로 작성한 뒤 수행한다.

## 영상 manifest 결과

| 항목 | 수량 |
| --- | ---: |
| 전체 영상 | 349 |
| 피험자 | 90 |
| Mirror / Dash | 320 / 29 |
| Train / Validation / Test 영상 | 246 / 50 / 53 |
| normal 약한 라벨 | 205 |
| yawn 포함 약한 라벨 | 115 |
| unknown sequence | 29 |
| 구간 검수가 필요한 영상 | 144 |

## 전달받은 구간 라벨 통합

2026-11-03에 Downloads의 A annotator CSV와 B annotator JSON을 가져와 manifest와
대조했다.

| 항목 | 수량 |
| --- | ---: |
| 원본 구간 | 69 |
| 평가용 정규화 구간 | 68 |
| 라벨이 있는 영상 | 41 (validation 19 / test 22) |
| 총 하품 구간 길이 | 264.733초 |

- 모든 video ID, 원본 파일명, split, 영상 길이 범위를 검증했다.
- B의 `yawdd:video:017`에 사실상 동일한 구간 두 개가 중복되어 있었다.
- IoU 80% 이상인 같은 annotator·같은 원본의 중복만 1개로 병합했다.
- 통합 CSV에는 `quality_flag=merged_overlap`과 두 source interval ID를 보존했다.
- 통합 결과: `data/annotations/yawdd_yawn_intervals.csv` (Git 제외)

## MAR 정량 평가

MediaPipe Face Landmarker로 5 FPS 샘플링한 6,005프레임 중 5,913프레임에서 MAR를
얻어 추출 coverage는 98.47%였다. test를 보지 않고 validation의 45개 설정 조합에서
event F1, frame macro F1 순서로 설정을 선택했다.

선택 설정:

- 하품 진입 MAR: `0.35`
- 정상 복귀 MAR: `0.25`
- 최소 지속시간: `0.8초`

| 지표 | Validation | Test |
| --- | ---: | ---: |
| 이벤트 Precision | 96.67% | 96.43% |
| 이벤트 Recall | 90.63% | 75.00% |
| 이벤트 F1 | 93.55% | 84.37% |
| 오경보/분 | 0.110 | 0.093 |
| 프레임 Accuracy | 89.57% | 85.19% |
| 프레임 Macro F1 | 83.48% | 73.22% |
| 프레임 Coverage | 98.87% | 98.13% |

Test 하품 프레임 recall은 38.45%로 이벤트 recall보다 낮다. 현재 방식은 하품을
대부분 한 번 이상 포착하지만 라벨 구간 전체를 덮지 못한다는 뜻이다. 후속 개선에서는
진입 임계값을 다시 test에 맞추지 않고, 입 벌림의 상승·하강 형태 또는 시간창 특징을
validation에서만 추가 검토한다.

평가 산출물:

- `artifacts/yawdd_mar_samples.csv`
- `artifacts/yawdd_mar_report.json`

## 실행

```bash
.venv/bin/python scripts/prepare_yawdd.py \
  "data/raw/yawdd/YawDD dataset" \
  --output data/manifests/yawdd_videos.csv

.venv/bin/python scripts/label_yawdd.py \
  data/manifests/yawdd_videos.csv yawdd:video:001 \
  --output data/annotations/yawdd_yawn_intervals.csv \
  --annotator NAME
```

라벨링 창에서 `Space`는 일시정지, `[`는 하품 시작, `]`는 하품 종료 및 저장,
`A/D`는 1초 이동, `Q`는 종료다. 출력 manifest와 구간 CSV는 원본 데이터와 함께
Git 제외 경로에 저장한다.

전달 라벨 통합과 MAR 평가는 다음과 같이 재현한다.

```bash
.venv/bin/python scripts/import_yawdd_labels.py \
  data/manifests/yawdd_videos.csv A_INTERVALS.csv B_LABELS.json \
  --output data/annotations/yawdd_yawn_intervals.csv

.venv/bin/python scripts/evaluate_yawdd_mar.py \
  data/manifests/yawdd_videos.csv data/annotations/yawdd_yawn_intervals.csv
```
