# Instant Gratification

Kaggle **Instant Gratification** 대회 스터디 저장소.
원본 스터디 노트북(`1_Instant_Gratification.ipynb`)을 baseline 으로 삼아
EDA 리포트와 재현 가능한 실험 스크립트로 정리했다.

- 대회: <https://www.kaggle.com/competitions/original-instant-gratification> (community re-run)
- 참고 solution: <https://www.kaggle.com/code/yeonmin/solution>

---

## 1. 대회 소개

| 항목 | 내용 |
|---|---|
| 문제 유형 | 이진 분류 (binary classification) |
| 평가지표 | **ROC AUC** |
| train | 262,144 행 × 256 피처 (+ `id`, `target`) |
| test (public) | 262,144 행 × 256 피처 |
| 결측치 | 없음 |
| 클래스 균형 | 50.00% (131,079 / 131,065) |

2019년에 열린 Kaggle "Instant Gratification" 은 **합성 데이터** 대회다.
피처 이름이 `wheezy-copper-turtle-magic` 처럼 무작위 단어 조합이고,
전체 데이터만 보면 어떤 피처도 target 과 상관이 없다(단일 피처 AUC ≈ 0.50).

이 대회의 전부라고 할 수 있는 발견은 **`wheezy-copper-turtle-magic` 컬럼**이다.
이 컬럼만 유일하게 0~511 의 정수값(512개 unique)을 갖고,
262,144 = 512 × 512 로 **데이터가 512개의 독립적인 하위 데이터셋으로 쪼개져 있다**.
각 하위 데이터셋은 `sklearn.datasets.make_classification` 으로 따로 생성된 것처럼 동작하므로,
**magic 값별로 모델을 따로 학습**해야 비로소 신호가 잡힌다.

> 참고: 원 대회는 이 구조를 알아낸 뒤 pseudo labeling 까지 붙이면 AUC 0.975 부근에서
> 리더보드가 포화되는 것으로 끝났다. 이 저장소의 baseline 은 pseudo labeling 이전 단계다.

---

## 2. 저장소 구조

```
Instant-Gratification/
├── README.md                      # 이 문서
├── 1_Instant_Gratification.ipynb  # 원본 스터디 노트북 (baseline 의 출처)
├── data/                          # 데이터 다운로드 · 경로 해석
│   ├── download.py                #   python data/download.py 로 kagglehub 다운로드
│   └── loader.py                  #   get_data_dir() / load_data()
├── eda/
│   └── 01_eda_report.ipynb        # EDA 리포트 (실행 결과 포함)
└── experiments/
    └── baseline/
        ├── baseline.py            # baseline 실험 코드 (.py)
        └── outputs/               # 실행 결과가 여기에 생성됨
            ├── baseline_metrics.json
            ├── baseline_submission.csv
            ├── baseline_run.log
            └── baseline_oof_*.npy
```

데이터는 train/test 각 ~1.3GB 라 저장소에 포함하지 않는다.
`data/loader.py` 가 `IG_DATA_DIR` 환경변수 → `data/raw/` → `data/data_path.txt` → kagglehub 캐시
순으로 경로를 찾는다.

---

## 3. 실행 방법

```bash
# 1) 환경
python -m venv .venv
.venv/Scripts/python -m pip install pandas numpy scikit-learn matplotlib seaborn lightgbm tqdm kagglehub jupyter
```

```bash
# 2) 데이터 다운로드 (Kaggle API 토큰 필요, 대회 참가 후 ~/.kaggle/kaggle.json)
python data/download.py
```

```bash
# 3) EDA 리포트 실행
jupyter lab eda/01_eda_report.ipynb
```

```bash
# 4) baseline 실험 (전체 512개 magic 그룹)
python experiments/baseline/baseline.py
```

```bash
# 4-1) 빠른 스모크 테스트 (magic 그룹 4개만, 1분 내외)
python experiments/baseline/baseline.py --magic-limit 4 --meta-seeds 1 --meta-folds 3 --tag smoke
```

주요 옵션: `--magic-limit` `--n-splits` `--meta-seeds` `--meta-folds` `--meta-boost-rounds`
`--skip-meta` `--tag` `--output-dir` `--quiet`

---

## 4. EDA 요약

자세한 내용은 [`eda/01_eda_report.ipynb`](eda/01_eda_report.ipynb). 핵심만 옮기면:

| 관찰 | 수치 |
|---|---|
| 결측치 없음, target 50:50 균형, 행 순서 편향 없음 | missing 0, positive ratio 0.5000 |
| `wheezy-copper-turtle-magic` 만 이산값 | unique 512개, 그룹당 511~513행 |
| 전체 데이터 기준 단일 피처는 무력 | 단일 피처 AUC 0.4955 ~ 0.5040 |
| **그룹별 std 가 이봉 분포** | std ≈ 1.0 (노이즈) vs std ≈ 3.4~4.1 (신호) |
| 그룹당 유효 피처는 일부뿐 | 255개 중 평균 **39.7개** (33 ~ 47개) |
| 유효 피처 집합은 그룹마다 다름 | magic 0 ∩ magic 1 = 7개 / 피처당 평균 79.7개 그룹에서만 유효 |
| 그룹 안에서는 신호가 잡힘 (magic=0) | 유효 피처 평균 \|AUC−0.5\| = 0.047 vs 노이즈 0.020 |
| 클래스 차이는 평균보다 **분산 구조** | mean_diff 평균 0.66 (≈0.2σ), 클래스별 std 비 0.82~1.20 |
| 유효 피처끼리 거의 무상관 | \|corr\| 평균 0.06 |

→ 그래서 **그룹별 피처 선택 + 가우시안 계열 모델(QDA/NuSVC/GMM)** 이라는 baseline 설계가 나온다.

---

## 5. Baseline

### 5.1 Feature engineering

`wheezy-copper-turtle-magic` 값 하나(그룹 크기 ≈ 512행)마다 다음을 **독립적으로** 수행한다.

1. **그룹별 피처 선택** — 그 그룹 안에서 `std > 2` 인 컬럼만 사용 (255개 → 평균 40개).
   나머지는 std ≈ 1 인 순수 노이즈.
2. **KernelPCA (cosine, `n_components = len(cols)`)** — train + test 를 합쳐 적합.
   차원은 그대로 두고 좌표계만 정리한다.
3. **GMM 파생 피처** — `GaussianMixture(n_components=5, max_iter=1000)` 을 train+test 에 적합해
   - `predict_proba` (5열) — 군집 소속 확률, **5번 반복**해서 붙임 (가중치 부여 목적)
   - `score_samples` (1열) — 로그 우도, **3번 반복**해서 붙임
   - `predict` — CV 분할 시 stratify 기준으로 사용
4. **히스토그램 밀도 피처** — 변수별 50-bin 히스토그램 높이의 평균 (1열).
5. **StandardScaler** 로 전체 스케일 정리 후 다시 train/test 로 분리.

최종 입력 차원 = `len(cols)` + 5×5 + 1 + 3×1 ≈ **40 + 29 ≈ 69열**.

### 5.2 Modeling — 3-level stacking

**Level 1** — 그룹별로 6개 분류기의 5-fold OOF 예측을 생성.

| 모델 | 설정 |
|---|---|
| `NuSVC` | poly, degree 4, nu 0.4, coef0 0.08 |
| `NuSVC` | poly, degree 2, nu 0.4, coef0 0.08 |
| `QDA` | reg_param 0.111 |
| `SVC` | poly, degree 4 |
| `KNeighborsClassifier` | n_neighbors 16 |
| `LogisticRegression` | liblinear, L1, C 0.05 |

CV 는 target 이 아니라 **GMM 군집 라벨로 stratify** 한다.
그리고 위 과정 전체를 `(gmm_init_params, random_state)` = `(kmeans,1) (kmeans,2) (random,1) (random,2)`
**4가지 조합으로 반복해 평균**낸다 (그룹당 512행뿐이라 분산이 크기 때문).

**Level 2** — level-1 OOF 6열을 입력으로 메타 모델을 학습 (seed 4개 × 5-fold).

| 모델 | 설정 |
|---|---|
| `LGBMClassifier` (`lgbm.train`) | num_leaves 18, lr 0.025, feature_fraction 0.45, bagging 0.343, L1 7.961 / L2 7.781 |
| `MLPClassifier` | hidden (16,), relu, lbfgs, tol 1e-6 |

**Level 3** — level-1 평균 6열 + LGBM 메타 1열 + MLP 메타 1열 = **8열의 단순 평균**이 최종 예측.

---

## 6. Metric

평가지표는 **ROC AUC** 이고, 아래 수치는 전체 학습 데이터 262,144행에 대한
**OOF(out-of-fold) AUC** 다. 원본 수치는 [`experiments/baseline/outputs/baseline_metrics.json`](experiments/baseline/outputs/baseline_metrics.json).

실행 조건: `python experiments/baseline/baseline.py` (magic 512개 전체 · level-1 5-fold ·
메타 seed 4 × 5-fold), Windows 11 / Python 3.14 / scikit-learn 1.9 / LightGBM 4.7, **총 2,616.7초 (약 44분)**.

### Level 1 — magic 그룹별 모델 (OOF AUC)

| 모델 | kmeans / seed1 | kmeans / seed2 | random / seed1 | random / seed2 |
|---|---|---|---|---|
| NuSVC (degree 4) | 0.94903 | 0.94880 | 0.94839 | 0.94829 |
| NuSVC (degree 2) | 0.94832 | 0.94794 | 0.94791 | 0.94759 |
| **QDA** | **0.94906** | **0.94894** | **0.94834** | **0.94853** |
| SVC (degree 4) | 0.94900 | 0.94880 | 0.94835 | 0.94818 |
| KNN (k=16) | 0.94893 | 0.94887 | 0.94810 | 0.94801 |
| LogisticRegression (L1) | 0.93839 | 0.93864 | 0.93790 | 0.93772 |
| *6개 모델 평균* | *0.94934* | *0.94929* | *0.94850* | *0.94838* |
| *소요 시간* | *530s* | *535s* | *634s* | *634s* |

### Level 2 / 3 — 스태킹

| 단계 | 구성 | OOF AUC |
|---|---|---|
| Level 1 앙상블 | 4개 config × 6개 모델 평균 (6열) | 0.94960 |
| Level 2 — LightGBM meta | level-1 6열 입력, seed 4 × 5-fold | 0.94956 |
| Level 2 — MLP meta | level-1 6열 입력, seed 4 × 5-fold | **0.94996** |
| **Level 3 (최종 제출)** | level-1 6열 + LGBM + MLP = 8열 단순 평균 | **0.94960** |

제출 파일: `experiments/baseline/outputs/baseline_submission.csv` (262,144행, 예측값 0.039 ~ 0.960)

### 해석

- QDA · NuSVC · SVC · KNN 이 모두 0.948 대에 몰려 있고 **LogisticRegression 만 0.938 로 뚜렷하게 뒤처진다.**
  "클래스를 가르는 것은 평균 위치가 아니라 분산/군집 구조" 라는 EDA 관찰과 일치한다.
- GMM 초기화는 `kmeans` 가 `random` 보다 일관되게 약 0.0009 높다.
- **스태킹으로 얻은 이득은 사실상 없다** (level-1 평균 0.949604 → 최종 0.949605).
  MLP 메타 단독이 0.94996 으로 가장 높은데, 최종 예측이 8열 **단순** 평균이라
  메타 모델의 이점이 level-1 6열에 희석된다. 가중 앙상블 여지가 남아 있다.
- 원 대회 상위권(≈0.975)과의 격차는 대부분 **pseudo labeling 부재**에서 온다.
  그룹당 학습 표본이 512행뿐이라, test 를 라벨링해 표본을 늘리는 것이 다음 단계의 핵심이다.

---

## 7. 원본 노트북에서 수정한 오류

`1_Instant_Gratification.ipynb` 의 학습 파트는 최신 라이브러리에서 그대로 실행되지 않는다.
`experiments/baseline/baseline.py` 는 **로직은 그대로 두고 아래 오류만** 고친 버전이다.

| # | 원본 | 문제 | 수정 |
|---|---|---|---|
| 1 | `run_model(model_list, train, test, 1)` | `train` / `test` 라는 이름은 정의된 적이 없음 (`train_df` / `test_df`) → `NameError` | 스크립트 전체에서 `train_df` / `test_df` 사용 |
| 2 | `StratifiedKFold(n_splits=5, random_state=random_state)` | `shuffle=False` 인데 `random_state` 를 주면 최신 scikit-learn 에서 `ValueError` | `shuffle=True` 추가 |
| 3 | GMM 군집 라벨로 stratify | 군집 크기가 fold 수보다 작거나 특정 fold 학습셋이 한 클래스만 갖게 되면 학습 실패 (방어적 수정) | `_make_folds()` 에서 검증 후 문제가 있을 때만 target stratify 로 폴백 |
| 4 | `lgbm.Dataset(..., silent=True)` | LightGBM 4.x 에서 제거된 인자 → `TypeError` | 인자 제거 |
| 5 | `lgbm.train(..., verbose_eval=False, early_stopping_rounds=100)` | LightGBM 4.x 에서 제거된 인자 → `TypeError` | `callbacks=[lgbm.early_stopping(100), lgbm.log_evaluation(0)]` 로 대체 |
| 6 | `sns.distplot(...)` | seaborn 0.14 에서 제거 예정(현재 deprecated) | `sns.kdeplot(...)` (EDA 노트북) |
| 7 | `Styler.set_precision(2)` | 최신 pandas 에서 제거 → `AttributeError` (원본에도 주석으로 기록됨) | `.format("{:.2f}")` |
| 8 | `tqdm_notebook` | deprecated | `tqdm` |
| 9 | `submission["target"] = oof_test_third.mean(1)` | 인덱스 정렬에 의존 — 순서가 다르면 조용히 잘못된 제출 파일 생성 | `sample_submission` 의 `id` 와 대조하고, 다르면 `id` 기준으로 매핑 |
| 10 | `train_second.mean(1)` | pandas 3.x 에서 위치 인자 deprecated | `mean(axis=1)` |
| 11 | 6개 모델 인스턴스를 fold/그룹 간 공유하며 재적합 | 상태가 누적될 여지가 있고 재현성이 떨어짐 | `sklearn.base.clone()` 으로 fold 마다 새 인스턴스 |
| 12 | `cols` 가 빈 리스트일 때 `KernelPCA(n_components=0)` | 그룹에 유효 피처가 없으면 예외 (방어적 수정 — 실제 데이터에서는 그룹당 최소 33개) | 해당 그룹 건너뜀 |
| 13 | `hist_model.predict` 의 이중 파이썬 루프 | 512 그룹 × 4회 실행에서 현실적인 시간 안에 끝나지 않음 | `np.searchsorted` 로 벡터화 (**결과값은 원본과 동일**) |

그 외 동작은 원본과 동일하게 유지했다.
단, `lgbm.train` 의 부스팅 라운드는 원본이 LightGBM 기본값(100)에 의존했으므로 그대로 100 으로 두고
`--meta-boost-rounds` 옵션으로 노출했다 (값을 키우면 원본이 의도했던 early stopping 이 실제로 동작한다).

---

## 8. 향후 계획

- **Pseudo labeling** — test 예측을 라벨로 되먹여 그룹당 표본 수를 늘리기 (원 대회 상위 솔루션의 핵심)
- GMM `n_components`(현재 5) 및 유효 피처 임계값(현재 `std > 2`) 민감도 분석
- 단순 평균 대신 모델별 가중 앙상블 / 최적 가중치 탐색
- 그룹별 표본이 적은 상황에서의 CV 분산 축소 (반복 CV, seed 확대)
