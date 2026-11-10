# 데이터셋 선행 준비 체크리스트

이 문서는 추가 기능 구현 전에 초기 데이터셋을 확보하기 위한 실행 기준이다. 원본은 모두 `data/raw/<dataset_id>/`에 두며 Git에 포함하지 않는다. 계정 토큰, 승인 메일, 서명한 계약서는 저장소에 커밋하지 않고 `data/licenses/` 또는 별도 접근 제한 위치에 보관한다.

## 1. 한 번만 실행할 준비

```bash
python scripts/check_dataset_readiness.py --initialize
python scripts/check_dataset_readiness.py
```

생성되는 주요 위치:

```text
data/raw/<dataset_id>/     원본
data/processed/            변환·추출 결과
data/manifests/            공통 CSV manifest
data/licenses/             로컬 승인·약관 기록(커밋 금지)
artifacts/                 평가 리포트·예측 결과
```

현재 디스크 여유 공간은 약 967GB다. 모든 대용량 데이터를 무조건 받기보다 초기 기능에 필요한 클래스·모달리티부터 받는다.

## 2. 확보 순서

### 1순위 — MRL Eye

- 공식 공개 데이터로 별도 계정이 필요 없다.
- 공식 ZIP을 `data/raw/mrl_eye/` 아래에 풀어 84,898개 이미지와 37개 subject가 있는지 확인한다.
- 공식 서버가 응답하지 않을 때는 시간을 두고 재시도한다. 출처와 분할이 바뀐 Kaggle 재가공본을 원본 대신 사용하지 않는다.
- 준비 후 `prepare_mrl_eye.py`로 운전자 단위 manifest를 만든다.

### 2순위 — AI Hub 졸음운전

- AI Hub에서 정확한 데이터명 `졸음운전 예방을 위한 운전자 상태 정보 영상`을 검색한다.
- 회원가입, 이용 신청·승인 후 먼저 샘플 데이터로 JSON 구조를 확인한다.
- 실제 도로·준통제·통제 환경과 눈 상태·졸음·머리 자세 라벨을 우선 확보한다.
- 승인 범위와 발표 이미지 사용 가능 여부를 `data/licenses/aihub_drowsiness.md`에 기록한다.

### 3순위 — YawDD

- 공식 IEEE DataPort 공개 페이지에서 무료 IEEE 계정으로 다운로드한다.
- 원본에는 하품의 프레임 시작·종료 라벨이 없으므로 영상 확보와 별개로 구간 라벨링 작업이 필요하다.
- 먼저 백미러 카메라 세트를 사용하고, 대시보드 시점은 외부 조건 검증으로 둔다.

### 4순위 — UTA-RLDD Fold 1~4

- NTHU-DDD의 복잡한 접근 절차를 피하기 위해 이미 확보한 UTA-RLDD로 대체한다.
- 현재 `data/raw/uta_rldd/`에 48명·145개 영상이 있으며 Fold 5의 12명·36개 영상은 없다.
- 대부분 대상자는 `0`, `5`, `10` 세 단계 영상이 있고, 32번 대상자의 `10` 영상은 2개 파일로 분할돼 있다.
- 결과에는 전체 60명이 아닌 Fold 1~4 부분집합임을 명시한다.

### 5순위 — State Farm

- Kaggle 대회 페이지에서 Competition Rules에 동의한다.
- Kaggle CLI 인증을 설정한 뒤 competition 데이터를 `data/raw/state_farm/`에 받는다.
- `driver_imgs_list.csv`가 반드시 있어야 하며, 이미지 랜덤 분할 대신 운전자 단위 분할에 사용한다.

```bash
kaggle competitions download -c state-farm-distracted-driver-detection \
  -p data/raw/state_farm
```

### 6순위 — AI Hub 이상행동

- AI Hub 데이터셋 651 `운전자 및 탑승자 상태 및 이상행동 모니터링`에 이용 신청한다.
- 전체 데이터보다 휴대폰 조작·통화·음료 행동의 JPG·JSON 바운딩박스를 먼저 받는다.
- 현재 공식 페이지의 공개 버전은 1.2이며, 로컬 기록에 다운로드 버전과 날짜를 남긴다.

## 3. 각 데이터셋 반입 시 기록

`data/licenses/<dataset_id>.md`에 다음 항목을 작성한다.

```text
dataset_id / version:
official URL / downloaded date:
account owner / approval date:
allowed: download / training / result publication / weight redistribution
forbidden or uncertain:
local archive SHA-256:
extracted root:
subject/session identifiers:
publication-image permission:
notes:
```

## 4. 준비 완료 판정

```bash
python scripts/check_dataset_readiness.py
```

`READY`는 예상 파일 구조가 존재한다는 뜻이며, 라이선스 준수나 데이터 완전성을 자동 보증하지 않는다. 반입 직후 압축 파일 SHA-256, 파일 개수, subject 수, 라벨 분포를 별도 검증하고 그 결과를 개발 기록에 추가한다.
