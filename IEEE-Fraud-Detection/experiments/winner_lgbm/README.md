# Winner 공개 LightGBM baseline

Konstantin Yakovlev의 `ieee-lgbm-with-groupkfold-cv`를 실제 출처로 사용한다.
원본 CSV→저자의 공개 minification/FE(`data/winner_fe.py`)→월별 6-fold LightGBM 학습 경로다.
공개 baseline이며 최종 우승 LGB의 전체 FE/모델과 동일하다고 확인된 코드는 아니다.

기본 lr .007, 256 leaves, feature fraction .5, bagging .7, 10,000 rounds, early stopping 100, CPU 8 threads다.
공통 `IEEE_WINNER_NROWS`, `IEEE_WINNER_FOLDS`, `IEEE_WINNER_ROUNDS`, `IEEE_WINNER_TAG` 설정을 지원한다.
float32 유지, fold-local target mean, 최신 callback 변경을 기록했다.
출력은 `outputs/<TAG>/`의 metrics/로그/확률 OOF/원문 min-max OOF/test 예측/중요도이며 전체 실행만 제출 파일을 만든다.

저장된 실행은 월별 표본 train 30,000/test 30,002행, 6-fold, 100 rounds다. 772개 피처, 확률 OOF AUC 0.830459, 40.3초였다.
전체 학습이나 최종 우승 점수 재현 결과가 아니다.
