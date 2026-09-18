# Winner 공개 CatBoost baseline

Konstantin Yakovlev의 `ieee-catboost-baseline-with-groupkfold-cv`를 실제 출처로 사용한다.
원본 CSV→저자의 공개 minification/FE(`data/winner_fe.py`)→범주형 복원/결측 그룹/지배값/UID 복원→월별 6-fold CatBoost 학습 경로다.
공개 baseline이며 최종 우승 CAT의 전체 FE/모델과 동일하다고 확인된 코드는 아니다.

기본 GPU, depth 8, lr .07, 5,000 trees다. `IEEE_WINNER_CAT_DEVICE=CPU`로 변경할 수 있다.
공통 `IEEE_WINNER_NROWS`, `IEEE_WINNER_FOLDS`, `IEEE_WINNER_ROUNDS`, `IEEE_WINNER_TAG` 설정을 지원한다.
float32 유지, fold-local target mean, 명시적 범주형 API 변경을 출처 문서와 노트북에 기록했다.
출력은 `outputs/<TAG>/`의 metrics/로그/확률 OOF/원문 min-max OOF/test 예측/중요도이며 전체 실행만 제출 파일을 만든다.

저장된 실행은 월별 표본 train 30,000/test 30,002행, 6-fold, 100 trees, GPU다. 720개 피처, 확률 OOF AUC 0.857846, 42.8초였다.
전체 학습이나 최종 우승 점수 재현 결과가 아니다.
