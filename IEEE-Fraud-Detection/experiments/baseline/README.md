# Baseline — Model2

[choco9966/IEEE-CIS-Fraud-Detection의 Model2.ipynb](https://github.com/choco9966/IEEE-CIS-Fraud-Detection/blob/main/Model2.ipynb)를 원본 CSV부터 실행하도록 정리한 노트북이다.
이 실험이 이후 피처 ablation의 비교 기준이 된다. 피처 생성과 학습 코드는 `baseline.ipynb` 안에 있다.

- 입력: 원본 transaction/identity CSV 4개, sample_submission.csv.
- 모델: 원본의 LightGBM GBDT, 최대 15,000 rounds, early stopping 100.
- 검증: 최빈 ProductCD 그룹과 나머지 그룹을 나눠 각각 StratifiedKFold 5-fold, seed 42.
- 출력: `outputs/<TAG>/metrics.json`, `run.log`, OOF/test 예측, feature importance, submission.
- 원본의 `goss` 파일명과 달리 실제 학습 파라미터는 GBDT다. 이 동작을 유지한다.

원본 대비 수정 사항과 빠른 실행 방법은 대회 루트 README에 기록한다.
