# 협업 안내

## 작업 흐름

1. 백로그의 DMS ID를 참고해 이슈에 목적·완료 조건·검증 방법을 작성합니다.
2. `feature/<topic>`, `fix/<topic>`, `docs/<topic>` 브랜치에서 작업합니다.
3. 로컬 테스트와 관련 시나리오를 확인하고 PR에 동작·제약·증거를 기록합니다.
4. 팀원 리뷰 후 main에 반영합니다. 실험 결과는 설정·분할·커밋과 함께 기록합니다.

## 로컬 검사

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python -m dms --input examples/synthetic_features.csv --output runs/demo.jsonl
python scripts/validate_manifest.py examples/synthetic_manifest.csv
```

초기 패키지는 표준 라이브러리만 사용합니다. 실제 영상·학습 환경을 추가할 때 호환 버전과 실행 검증 결과를 기록하고 재현 환경을 고정합니다. 테스트 성공을 AI 인식 성능으로 보고하지 않습니다.

데이터 원본·모델 가중치·개인 얼굴·학번·연락처·비밀 키를 올리지 않습니다. 모델이나 데이터의 사용 조건 변경은 관련 문서에 반영합니다. 생성형 AI로 작성한 코드도 팀원이 흐름·한계·검증 방법을 설명하고 수정할 수 있어야 합니다.
