# UTA-RLDD 영상 manifest 및 평가기

변환일: 2026-11-01  
입력: 로컬 Fold 1~4 (`data/raw/uta_rldd`)

## 라벨 정책

UTA-RLDD 공식 정의에 따라 파일명 숫자를 다음 3단계로 보존한다.

| 파일명 | 평가 라벨 |
| --- | --- |
| `0` | `alert` |
| `5` | `low_vigilance` |
| `10` | `drowsy` |

기본 평가는 세 단계를 합치지 않는다. 실제 경고 관점의 보조 결과로만
`alert` 대 `fatigued(low_vigilance+drowsy)` 이진 지표를 함께 산출한다.

- 공식 설명: <https://sites.google.com/view/utarldd/home>
- 원 논문: <https://openaccess.thecvf.com/content_CVPRW_2019/html/AMFG/Ghoddoosian_A_Realistic_Dataset_and_Baseline_Temporal_Model_for_Early_Drowsiness_CVPRW_2019_paper.html>

## 변환 결과

| 항목 | 수량 |
| --- | ---: |
| 피험자 | 48 |
| 영상 | 145 |
| 총 영상 길이 | 약 25.02시간 |
| alert | 48 |
| low_vigilance | 48 |
| drowsy | 49 |
| Train / Validation / Test | 103 / 21 / 21 |

32번 피험자의 drowsy 영상이 `10_1`, `10_2` 두 부분으로 나뉘어 있어 다른
피험자보다 한 개 많다. 두 파일은 동일한 session으로 유지하고 segment만 구분한다.
Validation과 test는 각각 세 클래스가 7개씩이다.

## 평가기

평가 입력은 다음 CSV 형식이다.

```csv
sample_id,prediction
uta_rldd:01:level_0:segment_1,alert
```

허용 prediction은 `alert`, `low_vigilance`, `drowsy`, `unknown`이다. 누락·추가·중복
ID와 정의되지 않은 라벨을 거부하며, `unknown`은 정답에서 제외하지 않고 오답으로
계산하면서 coverage를 별도로 보고한다.

```bash
.venv/bin/python scripts/prepare_uta_rldd.py data/raw/uta_rldd \
  --output data/manifests/uta_rldd_videos.csv

.venv/bin/python scripts/evaluate_uta_rldd.py \
  data/manifests/uta_rldd_videos.csv \
  --split test --write-template artifacts/uta_rldd_test_predictions.csv

.venv/bin/python scripts/evaluate_uta_rldd.py \
  data/manifests/uta_rldd_videos.csv artifacts/uta_rldd_test_predictions.csv \
  --split test --output artifacts/uta_rldd_test_report.json
```

평가기 출력은 3-class와 alert-vs-fatigued 각각의 Accuracy, macro F1, coverage,
클래스별 precision/recall/F1, 혼동행렬을 포함한다.
