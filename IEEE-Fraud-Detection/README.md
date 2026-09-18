# IEEE-CIS Fraud Detection

원본 CSV를 분석하는 EDA, [choco9966의 Model2](https://github.com/choco9966/IEEE-CIS-Fraud-Detection/blob/main/Model2.ipynb) baseline,
**1위 fraudsquad의 Chris Deotte / Konstantin Yakovlev 본인이 공개한 XGBoost·CatBoost·LightGBM 코드**를 나눈 실험이다.
Instant-Gratification과 같이 데이터 로딩, EDA, 실험과 산출물을 분리했다. 피처 생성과 학습 코드는 노트북에 직접 두었다.

- 대회: <https://www.kaggle.com/competitions/ieee-fraud-detection>
- 타깃: `isFraud`, 평가지표: ROC AUC, 제출: `TransactionID,isFraud` 확률.
- 출처: 루트 `Model2.ipynb`는 GitHub 원본을 수정 없이 보존한다. `source.json`에 commit과 SHA-256을 기록했다.
- 우승자 출처: `references/winner/`에 공개 원본 12개와 URL/버전/공개 여부/SHA-256을 보존했다. [재현 범위와 변경 사항](references/winner/README.md)을 참고한다.

| 노트북 | 역할 |
|---|---|
| `eda/01_eda_report.ipynb` | 전체 원본 데이터 구조/타깃/결측/시간/ProductCD |
| `eda/02_winner_eda.ipynb` | Transaction reference/V·ID 분석/UID 순도·버전별 연결/시간 일관성 |
| `experiments/baseline/baseline.ipynb` | 요청한 GitHub Model2 baseline |
| `experiments/winner_xgb/winner_xgb.ipynb` | Chris Deotte 공개 XGB95/96; 기본 XGB96 |
| `experiments/winner_catboost/winner_catboost.ipynb` | Konstantin Yakovlev 공개 CatBoost GroupKFold baseline |
| `experiments/winner_lgbm/winner_lgbm.ipynb` | 같은 저자의 공개 LightGBM GroupKFold baseline |
| `experiments/winner_blend/winner_blend.ipynb` | 공개된 best-single CAT/LGB/XGB test 예측 평균 + UID PP; 재학습 아님 |

Cat/LGB는 우승자 본인의 공개 baseline이다. Part 2에 보고된 **최종 우승 모델 전체 FE/스태킹과 동일한 코드로 확인된 것은 아니다.**
17위 등 다른 참가자의 코드를 우승자 구현으로 대체하지 않았다.

## 학습용으로 읽기

작업용 노트북 7개에 코드 셀 앞의 단계별 설명과 출력 뒤의 해석을 넣었다.
작은 거래 예시로 계산한 뒤 실제 변수·함수에 연결하고, 그 방법을 쓸 이유와 정보가 사라지는 지점도 설명한다.
EDA의 관찰에서 피처 가설로, 피처에서 검증과 다음 실험으로 이어서 읽을 수 있다.
기존 학습 코드와 저장된 실행 결과는 유지했고, 미실행 튜닝·제거 실험은 제안으로 표시했다.

처음에는 `01_eda_report` → GitHub `baseline`을 읽는다. 이후 `02_winner_eda` → `winner_xgb` → `winner_lgbm` → `winner_catboost` → `winner_blend`로 이어간다.
계산 예시는 직접 손으로 따라 해보고, 코드 아래의 표·그래프에서 같은 의미의 값을 찾아보면 좋다.
출처 보존용 `Model2.ipynb`와 `references/winner/` 원본은 수정하지 않았다.

## 구조

```text
IEEE-Fraud-Detection/
├── README.md
├── requirements.txt
├── source.json
├── Model2.ipynb                    # 참고 저장소의 원본
├── data/
│   ├── download.py                 # 원본 CSV 다운로드; 공유 캐시 사용
│   ├── loader.py                   # 경로 탐색, id-xx 정규화, left join
│   ├── winner.py                   # 원문 V120 목록, 공개 UID 입력/PP
│   └── winner_fe.py                # 저자의 공개 minification → FE 공통 코드
├── references/winner/              # Part2+직결8개, 추가공개6개 검토/원본/출처
├── eda/
│   ├── 01_eda_report.ipynb          # 구조/타깃/결측/시간/ProductCD/피처군
│   └── 02_winner_eda.ipynb          # 우승자 EDA/UID/시간 일관성
└── experiments/
    ├── winner_xgb/                 # 독립 notebook, README, outputs/<TAG>
    ├── winner_catboost/            # 독립 notebook, README, outputs/<TAG>
    ├── winner_lgbm/                # 독립 notebook, README, outputs/<TAG>
    ├── winner_blend/               # 공개 예측 평균/PP, outputs/released_v2
    └── baseline/
        ├── README.md
        ├── baseline.ipynb          # Model2 피처와 ProductCD별 5-fold 모델
        └── outputs/
            └── <TAG>/              # baseline과 smoke 결과를 구분
                ├── metrics.json
                ├── run.log
                ├── oof_train.npy
                ├── pred_test.npy
                ├── oof.csv
                ├── feature_importance.csv
                └── submission.csv # 전체 실행만; 축소 실행은 sample_predictions.csv
```

CSV와 대용량 예측은 git에서 제외하고 metrics/로그와 실행된 노트북은 남긴다.

## 실행

아래 명령은 이 폴더에서 Windows PowerShell로 실행한다. 이미 만들어진 `.venv`를 그대로 사용할 수 있다.
환경을 다시 만들 때는 다음 두 명령을 실행한다.

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

데이터 접근에는 Kaggle 대회 규칙 동의와 API 인증이 필요하다. `KAGGLE_API_TOKEN` 또는 Kaggle 인증 파일을 사용한다.
토큰을 코드나 저장소에 넣지 않는다.

```powershell
.\.venv\Scripts\python.exe data\download.py
.\.venv\Scripts\python.exe -m jupyter lab
```

Jupyter에서 `eda/01_eda_report.ipynb` → `experiments/baseline/baseline.ipynb` 순서로 읽고 실행한다.
전처리 CSV나 EDA 실행 결과 없이 baseline 자체만 실행할 수 있다.
데이터 경로는 `IEEE_DATA_DIR` → `data/raw/` → `data/data_path.txt` → 공유 kagglehub 캐시 순서로 찾는다.

명령행으로 실행 결과를 노트북에 저장할 수도 있다.

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace eda\01_eda_report.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 experiments\baseline\baseline.ipynb
```

전체 학습 전 빠른 확인은 원본의 앞 30,000행씩, 2-fold, 100 rounds로 실행한다.
피처 생성 단계는 전체 baseline과 같고, 출력은 `outputs/smoke/`에 저장된다.

```powershell
$env:IEEE_NROWS = '30000'
$env:IEEE_FOLDS = '2'
$env:IEEE_ROUNDS = '100'
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace experiments\baseline\baseline.ipynb
Remove-Item Env:IEEE_NROWS, Env:IEEE_FOLDS, Env:IEEE_ROUNDS
```

환경변수를 해제하면 전체 데이터, 5-fold, seed 42, 15,000 rounds가 기본값이다.
CPU 스레드는 8개이며 `IEEE_THREADS`로 바꾼다. 실행 태그는 `IEEE_TAG`로 지정한다.
같은 태그로 다시 실행하면 해당 산출물을 덮어쓴다. 무거운 학습은 한 번에 하나씩 실행한다.

## 우승자 공개 코드 실행

우승자 EDA와 학습 노트북은 전처리 pickle 없이 원본 CSV부터 실행한다. Cat/LGB의 공통 FE만 Python 파일에 분리했다.
UID PP/정밀 UID EDA/공개 예측 평균에 필요한 추가 데이터셋은 `kyakovlev/ieee-submissions-and-uids` version 2를 공유 캐시에 내려받는다.

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace eda\02_winner_eda.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace experiments\winner_blend\winner_blend.ipynb
```

모델 축소 실행은 각 월에서 같은 비율로 표본을 뽑아 6개월을 유지한다. 아래 3개 학습은 순서대로 실행한다.

```powershell
$env:IEEE_WINNER_NROWS = '30000'
$env:IEEE_WINNER_FOLDS = '6'
$env:IEEE_WINNER_ROUNDS = '100'
$env:IEEE_WINNER_LOCAL_ROUNDS = '100'
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace experiments\winner_xgb\winner_xgb.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace experiments\winner_catboost\winner_catboost.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace experiments\winner_lgbm\winner_lgbm.ipynb
Remove-Item Env:IEEE_WINNER_NROWS, Env:IEEE_WINNER_FOLDS, Env:IEEE_WINNER_ROUNDS, Env:IEEE_WINNER_LOCAL_ROUNDS
```

환경변수를 해제하면 전체 데이터/6-fold와 원문 공개 baseline 파라미터가 기본이다.
XGB는 CUDA 최대 5,000 rounds+시간 holdout 2,000 rounds, CAT는 GPU depth 8/lr .07/5,000 trees, LGB는 CPU lr .007/10,000 rounds다.
`IEEE_WINNER_DEVICE=cpu`, `IEEE_WINNER_CAT_DEVICE=CPU`로 장치를 바꾼다. 실행 태그는 `IEEE_WINNER_TAG`로 지정한다.
전체 모델 학습은 아직 실행하지 않았다. 저장된 모델 출력은 각 월에서 뽑은 **30,000 train / 30,002 test, 6-fold, 최대 100 rounds**다.
월별 비율 표본의 반올림 때문에 요청한 행 수와 실제 행 수가 약간 다를 수 있다. smoke에서는 submission/UID PP를 생성하지 않는다.

Cat/LGB의 전체 target mean은 학습 fold 내부로 옮겼고, float16 대신 float32를 유지했다.
두 모델은 원문 min-max OOF와 원래 확률 OOF를 모두 저장한다. 원문 보고 LB, 다른 크기/검증의 Model2 점수와 직접 비교하지 않는다.

## 원본 코드에서 유지한 것

- 이메일 일치/결측/도메인, M 피처, DeviceInfo/OS/브라우저 파생 피처.
- 빈도 인코딩, card/addr UID, 금액/요일/계정 생성일 추정 집계.
- LDA 5개 토픽을 만드는 3개 조합, 범주 조합과 빈도/금액 집계.
- next-click, 상품군별 누적 금액 평균과 min-max 변환, 추가 dist/시간 집계.
- 최빈 ProductCD와 나머지의 두 모델, 그룹별 범주형/제외 열, shuffled StratifiedKFold.
- LightGBM GBDT 파라미터와 early stopping 100, fold별 test 확률 평균.

원본 `add_nmf_feature`는 실제로 LDA를 실행한다. `_NMF_` 열 이름과 알고리즘을 그대로 유지했다.
원본 출력 파일명의 `goss`도 실제 파라미터와 다르므로 이 노트북은 GBDT로 명시한다.

## 수정한 것과 해석 범위

1. 최신 pandas에서 동작하지 않는 dict 중첩 집계를 `groupby.transform`으로 바꾸고, 행별 문자열/날짜 처리를 벡터화했다.
2. test identity의 `id-xx`를 `id_xx`로 통일하고, pandas 3의 문자열 dtype도 인코딩한다.
3. LightGBM의 제거된 `silent`, `early_stopping_rounds`, `verbose_eval` 인자를 현재 callback 방식으로 바꿨다.
4. 원본의 ProductCD/M4 target mean은 전체 정답으로 계산된다. 여기서는 각 상품군의 fold 학습 정답만 사용해 검증 라벨 누수를 없앴다. 원본 점수와 수치가 같다고 보장하지 않는다.
5. 피처를 만든 뒤 실수 열을 float32로 줄였으며, CPU 스레드는 원본 32개에서 8개로 조정했다. LDA는 Windows의 joblib 프로세스 종료 메시지를 피하도록 threading backend로 실행한다. 전체 코드를 64GB에서 실행했다는 원본 설명을 고려한 메모리/실행 조정이다.
6. 동일 학습을 반복하는 마지막 셀과 진단용 플롯을 줄이고, 실행 설정·버전·OOF·로그·ID 검증을 추가했다.

라벨 없는 빈도/집계/LDA는 원본처럼 train+test에서 계산한다. next-click은 다음 거래도 이용한다.
따라서 이 baseline은 전체 배치가 주어지는 대회 방식이며, 실시간 미래 거래 예측을 위한 검증은 아니다.
시간순 검증과 train-only 집계는 baseline을 기준으로 한 별도 실험으로 비교할 수 있다.

## 검증 결과

2026-09-18에 원본 CSV 다운로드와 전체 데이터 EDA를 실행했다. EDA 노트북에 표와 그래프가 저장되어 있다.

| 관찰 | 결과 |
|---|---|
| 결합 후 train/test | 590,540 / 506,691행 |
| 사기 건수/비율 | 20,663건 / 3.499% |
| identity 커버리지 train/test | 24.42% / 28.01% |
| train/test 상대 날짜 | 1–183일 / 213–396일 |
| W 상품군 | 439,670건, 사기 비율 2.04%, identity 커버리지 0% |
| C 상품군 | 68,519건, 사기 비율 11.69% |

baseline의 저장된 출력은 **train/test 앞 30,000행, 2-fold, 최대 100 rounds의 smoke 실행**이다.
모든 원본 피처 생성, 두 상품군 모델 학습과 산출물 저장이 완료되었다. pooled OOF AUC는 **0.894789**, 피처 생성부터 저장까지 **15.5초**였다.
전체 데이터 학습의 검증 점수로 해석하지 않는다. 구체적인 설정은 `outputs/smoke/metrics.json`에 기록했다.

노트북 형식/전체 코드 셀 문법, 원본 SHA-256, fold-local 타깃 인코딩과 미등록 범주 대체,
모든 OOF/test 예측의 완결성·확률 범위·행 수·원본 ID/라벨 대응을 확인했다.
전체 제출 파일의 순서 정렬 분기도 원본 sample_submission 506,691개 ID로 검증했다. 전체 5-fold 학습은 아직 실행하지 않았다.

우승자 EDA는 전체 CSV로 실행해 표/그림/관찰을 저장했다. 우승자 공개 학습 코드의 아래 수치는 월별 **축소 실행 확인**이다.

| 공개 코드 실험 | 피처 | 확률 OOF AUC | 실행 시간 |
|---|---:|---:|---:|
| Chris Deotte XGB96 | 263 | 0.862225 | 17.7초 |
| Konstantin Yakovlev CatBoost baseline | 720 | 0.857846 | 42.8초 |
| Konstantin Yakovlev LGB baseline | 772 | 0.830459 | 40.3초 |

세 모델은 같은 월별 30,000/30,002 표본과 최대 100 rounds로 실행했지만 피처/파라미터는 각 공개 코드의 값이다. 전체 LB/최종 성능 비교 결과로 해석하지 않는다.
전체 1,097,231개 train/test ID에서 옮긴 UID PP와 원문 XGB 셀 45가 같은 결과를 내는지 확인했다.
공개 예측 평균+PP는 전체 test 506,691행에서 실행했으며 공개 최종 blend와의 평균 절대 차이는 0.003513이었다. test AUC를 계산하지 않았다.
원본 source/hash·노트북 문법/실행·월별 fold 분리·OOF 완결성/라벨 대응·확률 범위·출력 ID/순서·미등록 target-mean 대체를 검증했다.
