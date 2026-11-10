# 로컬 데이터셋 현황

점검일: 2026-10-31

| 데이터셋 | 원본/압축 상태 | 로컬 상태 | 판정 |
| --- | --- | --- | --- |
| MRL Eye | `Downloads/archive.zip` 무결성 통과 | 84,898장, 37명 압축 해제 | READY |
| AI Hub 졸음운전 | 원천 ZIP 6개·라벨 ZIP 6개 구조 확인 | 환경별 원천·라벨 파일 수 일치 | READY (archive) |
| YawDD | `YawDD.rar.gz` gzip 및 내부 RAR 무결성 통과 | AVI 349개 압축 해제 | READY |
| UTA-RLDD | Fold 1~4 압축 해제, Fold 5 ZIP 무결성 통과 | 48명·145영상 즉시 사용 가능, 전체 60명 확장 가능 | READY (subset/full archive) |
| State Farm | competition ZIP 무결성 통과 | train 22,424장, test 79,726장, 10클래스, `driver_imgs_list.csv` 압축 해제 | READY |
| AI Hub 이상행동 | TS4/TL4·VS1/VL1 대응 TAR 확보 | 휴대폰조작·통화 포함 부분집합 | READY (subset archive) |

## UTA-RLDD 주의사항

- 전체 60명 데이터가 아니라 Fold 1~4의 48명 부분집합이다.
- Fold 5의 12명·37개 영상 ZIP은 다운로드·무결성 검증을 마쳤고 아직 압축 해제하지 않았다.
- 대부분 대상자는 `0`, `5`, `10` 3개 단계 영상으로 구성된다.
- 32번 대상자의 `10` 단계 영상은 2개 파일로 나뉘어 총 영상 수가 145개다.
- `.zip.part`, `.crdownload`, quota HTML은 완료 데이터로 계산하지 않는다.

## AI Hub 졸음운전 주의사항

원천 ZIP 6개와 라벨 ZIP 6개가 모두 있으며 환경별 항목 수가 일치한다. 아직 압축 해제·manifest 변환 전이다.

보유 라벨:

- Training: 실제도로 bbox, 통제환경 bbox, 준통제환경 keypoint
- Validation: 실제도로 bbox, 통제환경 bbox, 준통제환경 keypoint

환경별 원천/라벨 쌍을 스트리밍 또는 선택 압축 해제 방식으로 변환해야 한다.

## AI Hub 이상행동 부분집합

- Training: `TS4.tar` + `TL4.tar`
- Validation: `VS1.tar` + `VL1.tar`
- TL4 라벨에는 휴대폰조작 7,377장면, 통화 8,609장면이 포함된다.
- VL1 라벨에는 휴대폰조작 6,129장면, 통화 9,787장면이 포함된다.
- 음료 행동은 State Farm `c6`를 주 데이터로 사용한다.

## 다운로드 폴더의 관련 파일

- `archive.zip`: MRL Eye, 정상
- `YawDD.rar.gz`: YawDD 보존 원본, 정상
- `YawDD.rar`: 압축 해제를 위해 만든 중간 파일
- `state-farm-distracted-driver-detection.zip`: State Farm, 정상
- `Participants Information.zip`: 내용 확인이 필요한 보조 자료
- `Fold*.zip`: 완료된 UTA-RLDD 압축 파일
- `Fold*.zip.part`, `Unconfirmed *.crdownload`: 미완료 다운로드
