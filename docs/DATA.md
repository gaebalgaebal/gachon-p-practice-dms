# 데이터 관리

실제 확보 절차와 계정·승인 체크리스트는 [데이터셋 선행 준비 체크리스트](DATASET_SETUP.md)를 따른다. 준비 상태는 `python scripts/check_dataset_readiness.py`로 확인한다.

데이터셋과 모델 선택의 기준은 [DMS 확정 구현 계획](IMPLEMENTATION_PLAN.md)이다. NTHU-DDD는 접근 문제로 UTA-RLDD Fold 1~4 부분집합으로 대체했다.

## 초기 데이터셋

| 우선순위 | 데이터셋 | 적용 기능 | 현재 상태 | 다음 조치 |
| --- | --- | --- | --- | --- |
| 1 | MRL Eye | 눈 뜸·감김 | 다운로드 완료 | 압축 해제, subject 단위 manifest 변환 |
| 2 | AI Hub 졸음운전 | 눈 상태·졸음·Head Pose | 원천·라벨 확보, manifest 완료 | archive member 기반 학습·평가 loader 연결 |
| 3 | YawDD | MAR·하품 | 다운로드 완료, 이중 압축 | 압축 해제, 하품 구간 직접 라벨링 |
| 4 | UTA-RLDD Fold 1~4 | 영상 단위 졸음 단계 | 48명·145영상 확보 | manifest 변환, Fold 5 없이 부분집합임을 명시 |
| 5 | State Farm | 주의분산 이미지 분류 | 다운로드 완료 | 압축 해제, 운전자 단위 분할 |
| 6 | AI Hub 이상행동 | 휴대폰·통화·음료 검출 | 미수집 | 이용 신청, JSON 박스를 YOLO 형식으로 변환 |

DMD와 Drive&Act는 종합 DMS를 검토하는 후반 범위다.

## 데이터셋별 사용 원칙

- MRL Eye의 IR 눈 이미지 성능을 RGB 웹캠 성능으로 그대로 해석하지 않는다.
- YawDD는 하품 구간 라벨이 없으므로 시작·종료 기준과 라벨 작성자를 기록한다.
- UTA-RLDD는 0/5/10 영상 단위 단계 라벨이며 프레임별 졸음 정답이 아님을 결과에 명시한다.
- State Farm은 `driver_imgs_list.csv`를 이용해 운전자 단위로 분리한다.
- AI Hub 데이터는 승인 범위·재배포 금지·발표 이미지 사용 가능 여부를 기록한다.
- 데이터셋의 원래 train/test 구성이 운전자 분리를 보장하는지 다시 검사한다.

## DMS manifest 계약

필수 CSV 컬럼은 다음과 같다.

```text
sample_id,dataset,subject_id,session_id,file_path,frame_index,timestamp_s,task,label,split
```

허용 데이터셋과 과제 조합은 `configs/datasets.json`이 단일 기준이다. `split`은 `train`, `val`, `test`만 허용한다. `sample_id`는 중복되면 안 되며 `subject_id`와 `session_id`는 분할 간 겹치면 안 된다. 이미지도 `frame_index=0`, `timestamp_s=0`으로 기록한다.

검증 명령:

```sh
python scripts/validate_manifest.py data/manifests/<dataset>.csv --profile dms
python scripts/validate_manifest.py examples/synthetic_dms_manifest.csv --profile dms
```

## 등록 양식

```text
dataset_id / version:
공식 출처 / 접근 확인 날짜:
이용 목적과 허가 증빙의 내부 위치:
다운로드·학습·결과 공개·가중치 재배포 허용 범위:
라벨 정의 / 운전자·세션 ID 존재 여부:
로컬 보관 위치 / 접근 담당 / 보관·삭제 시점:
분할 정책 / seed / 분할 해시:
카메라·조도·RGB/IR 도메인 차이 / 제한점:
```

## 저장·공유 원칙

- 원본은 `data/raw/`, 전처리는 `data/processed/`, manifest는 `data/manifests/`에 둔다.
- 원본 얼굴 이미지·영상·개인정보·재배포 불가 가중치는 Git에 커밋하지 않는다.
- 자체 촬영은 동의서, 목적, 보관 기간, 접근 통제, 삭제 절차를 마련한 뒤 진행한다.
- 얼굴 영상을 외부 추론 서비스로 보내지 않는다.
- 발표용 예시와 실패 사례에도 원본 데이터의 공개 조건을 적용한다.
