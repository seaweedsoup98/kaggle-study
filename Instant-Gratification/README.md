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
├── eda/                           # 모든 노트북은 실행 결과가 포함된 상태로 커밋됨
│   ├── 01_eda_report.ipynb        #   기초 EDA — 데이터 개요와 magic 컬럼 발견
│   ├── 02_magic_group_structure.ipynb  # 그룹 구조 심층 — 생성 메커니즘 역추적
│   ├── 03_feature_engineering.ipynb    # 피처 엔지니어링 실험 — 구성요소별 ablation
│   └── 04_model_diagnostics.ipynb      # 모델·앙상블 진단 — 실행 결과 OOF 분석
└── experiments/                   # 실험마다 폴더 하나, 결과물은 각자 outputs/ 에
    ├── baseline/
    │   ├── baseline.py            #   baseline 실험 코드 (.py)
    │   └── outputs/               #   실행 결과가 여기에 생성됨
    │       ├── baseline_metrics.json
    │       ├── baseline_submission.csv
    │       ├── baseline_run.log
    │       └── baseline_oof_*.npy
    ├── weighted_ensemble/         # 추가 실험 1 — 가중 앙상블 (7.1절)
    │   ├── weighted_ensemble.py
    │   └── outputs/
    └── pseudo_labeling/           # 추가 실험 2 — pseudo labeling (7.2절)
        ├── pseudo_labeling.py
        └── outputs/
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
# 3) EDA 노트북 실행 (01 → 02 → 03 → 04 순서)
jupyter lab eda/
```

> `04_model_diagnostics.ipynb` 는 baseline 실행 산출물(`outputs/baseline_oof_*.npy`)을 읽으므로
> 아래 4)를 먼저 완주해야 한다.

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

```bash
# 5) 추가 실험 — 가중 앙상블 (baseline 의 OOF .npy 재사용, 재학습 없음)
python experiments/weighted_ensemble/weighted_ensemble.py
```

```bash
# 6) 추가 실험 — pseudo labeling (전체 실행 약 76분)
python experiments/pseudo_labeling/pseudo_labeling.py
```

> 5) 와 6) 은 모두 `experiments/baseline/outputs/` 의 산출물을 읽으므로 4) 를 먼저 완주해야 한다.

---

## 4. EDA

노트북 4개로 나뉘어 있고, 모두 **실행 결과가 포함된 상태**로 커밋되어 있다.

이 노트북들은 **점수를 짜내는 대회 노트북이 아니라 학습용 자료**로 작성했다.
각 절이 `무엇을 하나 → 왜 하나 → 원리 → 기대 결과 → 관찰` 순서로 구성되어 있고,
코드에도 "왜 이렇게 쓰는지"를 주석으로 달았다.
**기대와 다른 결과가 나온 곳(가설 기각)도 지우지 않고 그대로 남겨 두었다** — 02 노트북에 두 개 있다.

읽는 순서는 01 → 02 → 03 → 04 이며, 각 노트북 끝에 `직접 해보기` 연습문제가 있다.

### 4.1 [`01_eda_report.ipynb`](eda/01_eda_report.ipynb) — 기초 EDA

| 관찰 | 수치 |
|---|---|
| 결측치 없음, target 50:50 균형, 행 순서 편향 없음 | missing 0, positive ratio 0.5000 |
| `wheezy-copper-turtle-magic` 만 이산값 | unique 512개, 그룹당 511~513행 |
| 전체 데이터 기준 단일 피처는 무력 | 단일 피처 AUC 0.4955 ~ 0.5040 |
| **그룹별 std 가 이봉 분포** | std ≈ 1.0 (노이즈) vs std ≈ 3.4~4.1 (신호) |
| 그룹당 유효 피처는 일부뿐 | 255개 중 평균 **39.7개** (33 ~ 47개) |
| 그룹 안에서는 신호가 잡힘 (magic=0) | 유효 피처 평균 \|AUC−0.5\| = 0.047 vs 노이즈 0.020 |
| 유효 피처끼리 거의 무상관 | \|corr\| 평균 0.06 |

### 4.2 [`02_magic_group_structure.ipynb`](eda/02_magic_group_structure.ipynb) — 그룹 구조 심층

512개 하위 데이터셋이 **어떻게 만들어졌는지**를 역으로 추적한다.

| 질문 | 결론 |
|---|---|
| train/test 구조가 같은가 | 유효 피처 집합 Jaccard = 1.0 (**512/512 그룹 완전 일치**) |
| 그룹 간 공통 구조가 있는가 | 그룹 쌍의 공유 피처 수 6.17 = 무작위 추출 기댓값 6.17 — **없음** |
| 중복(redundant) 피처가 있는가 | 공분산 full rank, 고유값 계단 없음 — **유효 피처 ≈ informative 차원** |
| 그룹 난이도는 무엇이 결정하는가 | 유효 피처 수와 AUC 상관 **+0.09 (무관)**. 그룹별 AUC 편차(0.0124) 중 측정 오차가 0.0113 — **진짜 그룹 차이는 분산의 17%뿐** |
| 클래스당 군집 수 | BIC 는 full/diag 모두 **k=1** 선택 — 256개 표본으로는 다봉 구조 식별 불가 |
| GMM 군집 ↔ target | 80개 그룹 중 **78개(97.5%)에서 카이제곱 p < 0.05**, Cramér's V 0.327 |

### 4.3 [`03_feature_engineering.ipynb`](eda/03_feature_engineering.ipynb) — 피처 엔지니어링 실험

40개 그룹 · QDA · 5-fold · **대응 비교(paired, 95% CI)** 로 구성요소를 하나씩 켜고 끈다.
기준선(원본 설정) pooled AUC = **0.94826**.

> ⚠️ 이 표의 마지막 줄(pseudo labeling)은 파일럿 결과이며, **7.2절에서 전체 규모로 확장했을 때 재현되지 않았다.** 노트북에도 그 전말을 후속 상자로 남겨 두었다 — 축소 실험의 한계를 보여 주는 사례다.

| 실험 | 결과 | 판정 |
|---|---|---|
| **std 임계값** | 1.5 / 2.0 / 2.5 가 **완전히 동일**. 1.0 → 0.726 붕괴, 3.5 → 0.935 | 2.0 은 안전한 구간 한가운데 |
| **피처 선택 제거** | 255개 피처 = 284차원 vs 클래스당 209행 → **QDA 자체가 성립 불가** | 필수 |
| **차원 축소** | ratio 0.75 → −0.0082, 0.5 → −0.0503, 0.25 → −0.1837 (모두 유의) | 02의 full-rank 결론 확인 |
| **KernelPCA** | cosine 이 최고지만 투영 없음 대비 **+0.0008 (CI가 0을 포함)** | 효과 미확인 |
| 커널 종류 | rbf 0.611, poly 0.814 — **치명적** | cosine 외 금지 |
| **GMM n_components** | k=5 가 최적. k=0 −0.0120, k=3 −0.0046, k=10 −0.0138 (모두 유의) | **BIC(k=1)와 정반대 — AUC 기준으로는 k=5** |
| **GMM proba ×5 복사** | ×1 로 줄여도 **+0.0004 (CI가 0을 포함)** | 효과 없음 |
| GMM score / hist 피처 | 제거해도 각각 +0.0001 / +0.0000 | 효과 없음 |
| GMM 라벨로 stratify | 제거해도 +0.0004 | 효과 없음 |
| StandardScaler | 제거 시 −0.0066 (유의) | 필요 |
| **transductive (train+test)** | train 만 쓰면 **−0.0169**, **40/40 그룹에서 열세** | 가장 큰 단일 요인 |
| **pseudo labeling** | p>0.9 에서 **+0.0011** (fold 당 ~504행 추가) | 파일럿 한정 — 512그룹·6모델에서 **재현 실패** (7.2절) |

### 4.4 [`04_model_diagnostics.ipynb`](eda/04_model_diagnostics.ipynb) — 모델·앙상블 진단

512개 그룹 전체의 실제 OOF 예측(262,144행 × 8열)을 연다.

| 질문 | 결론 |
|---|---|
| 모델 다양성 | level-1 6개 모델 간 순위상관 **0.89 ~ 0.99** — 대체로 같은 실수를 한다 |
| **pooled AUC ≠ 그룹별 AUC 평균** | 그룹 안에서 순위로 정규화하면 모든 level-1 모델이 **떨어진다**. 그룹 간 확률 스케일 자체가 정보 → **순위 평균(rank averaging) 은 쓰면 안 된다** |
| 의외의 결과 | `logreg_l1` 은 그룹별 AUC 최하위(0.94636)인데 **pooled 로는 2위(0.94977)** — 확률이 좁은 구간에 압축돼 그룹 간 비교 가능성이 높기 때문 |
| 그룹별 모델 선택 | 정답을 보고 고르는 **오라클조차 상한이 +0.0034** — 실전에서는 잡음 추종 |
| 가중 앙상블 | 홀드아웃 평가 0.949575 vs 단순 평균 0.949603 — 차이 −0.00003 으로 이득 없음. 두 fold 가중치 상관 **−0.115** 라 가중치 자체가 재현되지 않음 |

→ 종합하면 baseline 설계의 **핵심은 ① 그룹별 피처 선택 ② 차원 유지 ③ GMM(k=5) 파생 피처
④ transductive 적합** 네 가지이고, 나머지(hist 피처, 반복 복사, GMM stratify, 가중 앙상블)는
측정 가능한 기여가 없다.

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
  메타 모델의 이점이 level-1 6열에 희석된다.
  다만 [04_model_diagnostics](eda/04_model_diagnostics.ipynb) 에서 확인한 바로는
  **가중 앙상블로도 이득이 관측되지 않는다** — 절반으로 가중치를 학습해 나머지 절반에서 평가하면
  0.949575 로 단순 평균(0.949603)과 사실상 동률(−0.00003)이고,
  더 결정적으로 두 fold 에서 학습한 가중치의 상관이 −0.115 라 가중치가 재현되지 않는다.
  level-1 모델들의 순위상관이 0.89~0.99 로 너무 높은 것이 근본 원인이다.
- 원 대회 상위권(≈0.975)과의 격차를 처음에는 **pseudo labeling 부재**로 봤으나,
  512개 그룹 전체에서 실제로 측정해 보니 **이득이 없었다**(7.2절). 격차의 원인은 다른 곳에 있다.

---

## 7. 추가 실험

baseline 위에서 개선 여지가 있다고 본 두 가지를 **512개 그룹 전체 규모로** 측정했다.
둘 다 결론은 **측정 가능한 이득 없음**이다. 두 실험 모두 baseline 과 같은 조건(512그룹 · 5-fold ·
4개 config · 6개 모델)에서 돌렸으므로 최종 숫자를 0.949605 와 직접 비교할 수 있다.

### 7.1 가중 앙상블 — [`experiments/weighted_ensemble/`](experiments/weighted_ensemble/)

최종 예측이 8열 **단순** 평균이라 가장 좋은 mlp 메타(0.949963)가 희석된다는 관찰에서 출발했다.
21가지 결합 방법을 **5-fold 로 가중치는 학습 폴드에서만 적합하고 홀드아웃에서 평가**해 비교했다.

| 방법 | CV AUC | in-sample AUC | 과적합 폭 |
|---|---|---|---|
| **mlp_meta 단독** | **0.949968** | 0.949963 | −0.000005 |
| AUC 직접 최적화 (Nelder-Mead, 8열) | 0.949943 | 0.950561 | **+0.000619** |
| 로지스틱 스태킹 (8열) | 0.949880 | 0.950084 | +0.000203 |
| 8열 단순 평균 (현재 baseline) | 0.949604 | 0.949605 | ≈0 |
| rank 평균 (8열) | 0.949427 | 0.949427 | ≈0 |

- **최적해가 퇴화한다.** 부분집합 탐색 · simplex 그리드 · 비음수 최소제곱이 **전부 독립적으로
  "mlp_meta 하나만 쓰기"로 수렴**했다. 나머지 7열에 0 이 아닌 가중치를 주는 진짜 혼합은 없다.
- **개선은 유의하지 않다.** mlp 단독 − 단순 평균 = +0.00038, 300회 대응 부트스트랩
  95% CI **[−0.00038, +0.00114]**, P(차이 ≤ 0) = 0.17. 폴드별 차이가 −0.00105 ~ +0.00122 로 요동친다.
- AUC 를 직접 최적화하는 방법이 과적합이 가장 심하다. AUC 는 계단 함수라 최적화기가
  폴드별 순위 뒤집힘을 쫓아간다(전체 적합 시 `nusvc_deg2` −0.228 같은 큰 음수 가중치가 나온다).
- **rank 평균과 logit 평균은 확률 평균보다 나쁘다.** 8열이 이미 4 seed × 5 fold 평균이라 캘리브레이션이
  거의 같은데, 순위로 바꾸면 확신도 간격 정보만 버리게 된다. [04](eda/04_model_diagnostics.ipynb) 의
  "그룹 간 확률 스케일 자체가 정보" 라는 결론과 같은 이야기다.
- 실용적 함의: mlp 단독 제출은 기존 제출과 Spearman **0.709** 밖에 안 된다.
  유의하지도 않은 +0.0004 를 위해 제출을 크게 바꾸는 셈이라, 단순 평균을 유지하는 편이 낫다.

> [04_model_diagnostics](eda/04_model_diagnostics.ipynb) 5절이 절반 홀드아웃으로 내린 "이득 없음"
> 결론을, 방법을 21가지로 넓히고 유의성 검정을 붙여 **독립적으로 재확인**한 셈이다.

### 7.2 Pseudo labeling — [`experiments/pseudo_labeling/`](experiments/pseudo_labeling/)

[03](eda/03_feature_engineering.ipynb) 7절이 40개 그룹 · QDA 단독에서 +0.0008 을 관측하고
"유일하게 남은 개선 여지"로 지목한 항목이다. 이를 **512그룹 · 6모델 · 4 config** 로 확장했다.

누수 없는 프로토콜(fold 마다: 학습 fold 로만 적합 → test 예측 → 확신 행에 가짜 라벨 →
학습 fold + 가짜 라벨로 재적합 → 검증 fold 예측)을 그대로 따랐고,
`_assert_no_leak()` 이 재적합 98,858회 전부에서 불변식을 검사한다.

| 구성 | level-1 앙상블 | 최종 8열 평균 | vs baseline |
|---|---|---|---|
| baseline / `no_pseudo` 대조군 | 0.949604 | **0.949605** | — |
| pseudo p > 0.90 | 0.949651 | **0.949614** | +0.000009 |
| pseudo p > 0.99 | 0.949605 | **0.949607** | +0.000002 |

512그룹 대응 비교 (level-1, p > 0.90): **−0.000122 [−0.000292, +0.000048]**, 개선된 그룹 254/512.
**유의한 개선 없음.**

- **대조군이 baseline 을 비트 단위로 재현한다.** `no_pseudo` 변형의 최종 AUC 가
  `0.9496047663578433` 로 baseline 산출물과 마지막 자리까지 같다. 따라서 위 차이는
  pseudo labeling 단계에서만 나온 것이다.
- **파일럿의 +0.0008 은 "QDA 단독 · 40그룹" 이라는 조건에 붙어 있던 값이었다.**
  512그룹에서 QDA 만 보면 +0.00021 로 방향은 남지만 유의하지 않고, 나머지 5개 모델에서는
  무효이거나 손해다. baseline 이 이미 4 config × 6 모델 평균으로 같은 종류의 분산을 제거하고 있어서
  pseudo labeling 이 회수할 여지가 남아 있지 않다.
- **모델마다 가짜 라벨을 만들 수 있는 능력이 극단적으로 다르다.** fold 당 추가 행 수(p > 0.99 기준):
  QDA 491.9 · knn16 177.1 · svc_deg4 28.3 · nusvc_deg4 14.2 · nusvc_deg2 2.5 · **logreg_l1 0.0**.
  `logreg_l1` 은 `C=0.05` 로 강하게 규제돼 p > 0.99 를 한 번도 넘기지 못하고,
  `knn16` 은 확률이 1/16 격자에 갇혀 있다. "임계값에 둔감하다" 던 파일럿의 관찰도 QDA 만의 성질이었다.
- **pooled AUC 와 그룹별 AUC 가 부호까지 갈릴 수 있다.** `nusvc_deg2` 는 p > 0.90 에서
  pooled −0.00139 인데 그룹별 대응은 **+0.00078 (유의)** 다. pseudo labeling 이 확률을 극단으로
  미는 정도가 그룹마다 달라 그룹 *안* 순위는 좋아지고 그룹 *간* 공통 스케일은 망가진다.
  대회 지표는 pooled 쪽이므로, 한 숫자만 보면 정반대 결론이 나온다.
- 축소 실행의 함정: `--magic-limit 40` 에서는 `no_pseudo` 조차 baseline 대비 +0.0008 "유의" 로 나온다.
  pseudo labeling 과 무관하게 메타 모델이 20,475행으로만 학습되기 때문이며,
  512그룹에서는 이 차이가 정확히 0 으로 사라진다. **축소 실행의 최종 AUC 로 판단하면 안 된다.**

실행 시간 75.9분(임계값 2개 + 대조군, stage-1 공유). p > 0.95 는 40그룹에서만 측정했다.

---

## 8. 원본 노트북에서 수정한 오류

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

## 9. 향후 계획

EDA 노트북 02~04 와 7절의 추가 실험 결과에 따라 우선순위를 매기면:

**해볼 가치가 있는 것**
- **모델 다양성 확보** — level-1 모델 간 순위상관이 0.89~0.99 라 앙상블이 작동하지 않는다.
  현재 6개는 모두 "거리/공분산 기반"이다. 성격이 다른 모델(예: 트리 계열, 서로 다른 전처리)을
  넣어야 한다. 7.1 에서 가중치 탐색이 전부 한 열로 퇴화한 것도 같은 원인이다
- **그룹 간 확률 캘리브레이션** — 7.2 의 `nusvc_deg2` 사례(그룹별 +0.00078 인데 pooled −0.00139)와
  [04](eda/04_model_diagnostics.ipynb) 의 `logreg_l1` 사례(그룹별 최하위인데 pooled 2위)가
  같은 현상의 양면이다. 그룹 간 확률 스케일이 pooled AUC 의 실질적 병목인데,
  아직 이를 직접 다룬 실험이 없다. 그룹별 재보정이 pooled AUC 를 올리는지 측정할 가치가 있다
- **파이프라인 단순화** — 기여가 측정되지 않은 구성요소(아래)를 제거하면 성능 손실 없이
  코드와 실행 시간이 줄어든다

**하지 않아도 되는 것 (측정으로 기각됨)**
- ~~pseudo labeling~~ — 512그룹 · 6모델 규모에서 이득 없음. 파일럿의 +0.0008 은
  QDA 단독 조건에 묶인 값이었다 (7.2절)
- ~~가중 앙상블~~ — 21가지 방법 전부에서 유의한 이득 없음. 최적해가 "mlp 단독"으로 퇴화하고
  부트스트랩 CI 가 0 을 포함한다 (7.1절, [04](eda/04_model_diagnostics.ipynb) 5절)
- ~~그룹별 모델/하이퍼파라미터 선택~~ — 오라클 상한이 +0.0034 에 불과하고, 그룹별 AUC 편차의 83%가 측정 오차다
- ~~std 임계값 튜닝~~ — 1.5 ~ 2.5 구간에서 결과가 완전히 동일하다
- ~~순위 평균 앙상블~~ — 그룹 간 스케일 정보를 버려 오히려 손해다 (7.1절에서 재확인)
- ~~GMM 파생 피처 반복 복사 / hist 피처 / GMM stratify~~ — 기여가 측정되지 않는다 (제거하면 코드가 단순해진다)

> 세 갈래(가중 앙상블 · pseudo labeling · 하이퍼파라미터 민감도)가 모두 "차이 없음"으로 끝났다는 것은
> **baseline 이 이 설계 안에서는 평탄한 최적점에 있다**는 뜻이다.
> 남은 개선은 설계를 바꾸는 쪽(모델 다양성, 그룹 간 캘리브레이션)에서 찾아야 한다.
