# Winner XGBoost

Chris Deotte의 공개 XGB95/96을 원본 transaction/identity CSV부터 실행한다. 기존 choco Model2 baseline과 비교할 별도 실험이다.
피처와 학습은 `winner_xgb.ipynb`에 직접 작성했다. 출처/추가 UID 입력/수정 사항은 `../../references/winner/README.md`를 참고한다.

기본은 XGB96, CUDA, 시간순 75/25 holdout 2,000 rounds, 월별 6-fold 5,000 rounds다. `BUILD95=True`이면 UID 전 모델도 실행한다.
`IEEE_WINNER_DEVICE=cpu`로 CPU를 선택할 수 있다. `IEEE_WINNER_NROWS`를 지정하면 각 월에서 비율 표본을 뽑으며 `outputs/smoke/`에 저장한다.
`IEEE_WINNER_FOLDS`, `IEEE_WINNER_ROUNDS`, `IEEE_WINNER_LOCAL_ROUNDS`, `IEEE_WINNER_TAG`로 실행을 조정한다.

출력은 `metrics.json`, `run.log`, 모델별 OOF/test numpy·CSV, 피처 중요도다.
전체 실행만 raw/PP submission을 만들고, 축소 실행은 `sample_predictions.csv`만 만든다. 전체 데이터 학습은 아직 검증하지 않았다.

저장된 실행은 월별 표본 train 30,000/test 30,002행, 6-fold, CV/holdout 각 최대 100 rounds, CUDA다.
XGB96 263개 피처, 확률 OOF AUC 0.862225, 시간순 holdout AUC 0.860963, 17.7초였다.
전체 학습이나 원문 LB 점수 재현 결과가 아니다.
