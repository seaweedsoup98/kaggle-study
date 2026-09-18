# 우승자 공개 자료 검토와 재현 범위

2026-09-18에 Part 2 본문과 본문에서 직접 연결한 8개 자료를 모두 읽었다. 댓글의 모든 링크나 재귀 링크를 조사한 범위는 아니다.
일반 웹 도구에서 내용이 비어 있던 글은 브라우저로 읽었고, 공개 노트북 6개는 공식 Kaggle API로 내려받았다. 접근 실패를 비공개 판정 근거로 사용하지 않았다.
다운로드한 원본은 하위 폴더에 수정 없이 보존했다. `source.json`에 URL, kernel metadata, SHA-256을 기록한다.
노트북은 다운로드 시점의 공개 source이며 2019년 특정 버전이라고 주장하지 않는다.
추가로 우승자의 공개 CatBoost/LightGBM 학습·FE·minification·internal blend 6개를 확인하고 보존했다. 이 6개는 아래 원래 9개와 별도 목록으로 기록한다.

| 자료 | 확인한 내용 | 이 workspace에 적용한 범위 |
|---|---|---|
| [Part 2](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2) | 최종 CAT/LGB/XGB, 평균·LGB 메타모델, PP, V 선택과 시간/고객 검증 | 학습·공개 예측 앙상블·EDA를 분리하고 재현 한계를 명시 |
| [Part 1](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284) | UID로 고객을 식별하는 이유, known/unknown/questionable 고객 | 간단/공개 정밀 UID 순도, seen/new/unresolved 거래 수 분석 |
| [XGB Fraud with Magic](https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600) | 원본 CSV부터 XGB95/96, 월별 CV와 2단계 UID PP까지 실제 코드 | `winner_xgb.ipynb`의 직접 출처 |
| [How to Find UIDs](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111510) | adversarial validation으로 카드/D/V 단서 탐색, UID별 집계, 원시 UID 제외 | 같은 UID 집계 피처 유지. 새 adversarial selection을 수행한 것으로 주장하지 않음 |
| [UID detection v6](https://www.kaggle.com/kyakovlev/ieee-uid-detection-v6) | D1/D3와 V 누적금액 연속성, root 탐색·ambiguity·sanity check로 정밀 UID 복원 | 원본 보존/검토. v6을 실행해 v1/v4를 생성했다고 주장하지 않음 |
| [Basic FE part 1](https://www.kaggle.com/kyakovlev/ieee-basic-fe-part-1) | 6종 D/카드 UID, M/D/금액/거리/V 결측 집계, 4개월→1개월 gap→1개월 테스트 | 원본 보존/비교. 작성자의 공개 FE 예시이며 최종 LGB 전체 학습 코드로 부르지 않음 |
| [FE for local test](https://www.kaggle.com/kyakovlev/ieee-fe-for-local-test) | 시간/은행/카드 UID, D 정규화, 금액/C/디바이스/이메일 인코딩, 단계별 local test | 기존 choco Model2와 겹치는 FE를 확인. 이 코드의 global target mean 누수는 새 XGB에 도입하지 않음 |
| [Transaction Columns Reference](https://www.kaggle.com/alijs1/ieee-transaction-columns-reference) | 앞 150개 열의 숫자/범주별 통계, 빈도, 사기율, 시간/상관 그래프 | 새 EDA의 열별 profile·타깃 그룹 요약·대표 분포/월별 평균 |
| [V and ID EDA](https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id) | V 결측 개수 그룹, 수동 상관 묶음의 대표 선택, 전체/축소 상관, ID/C/D/M 분석 | 새 EDA에 반영; 실제 결측 마스크 확인을 추가 |

## 실행하는 것

- `experiments/winner_xgb/winner_xgb.ipynb`: Chris Deotte의 공개 XGB96 학습. `BUILD95=True`로 UID FE 전 ablation도 가능하다. 전체 데이터 기본값은 75/25 시간 holdout과 월별 GroupKFold 6개다.
- `experiments/winner_catboost/winner_catboost.ipynb`: Konstantin Yakovlev의 공개 CatBoost baseline 학습. 공개 minification→FE와 CatBoost 준비 단계를 원본 CSV부터 연결한다.
- `experiments/winner_lgbm/winner_lgbm.ipynb`: 같은 저자의 공개 LightGBM GroupKFold 모델 학습. 공통 FE는 `data/winner_fe.py`에 분리한다.
- `experiments/winner_blend/winner_blend.ipynb`: 공개 CAT/LGB/XGB best-single **test 예측 파일**을 ID로 정렬하여 1/3씩 평균하고, XGB 공개 PP 코드의 v4→v1 순서를 적용한다.
- `eda/02_winner_eda.ipynb`: 전체 원본 transaction/identity CSV 분석. 공개 v3 정밀 UID를 부가적으로 진단한다.

## 최종 우승 제출 전체와의 차이

Part 2는 최종 단일 CAT/LGB/XGB의 Public/Private AUC를 각각 .9639/.9408, .9617/.9384, .9602/.9324로 보고한다.
두 최종 제출은 CAT+XGB의 LGB 메타모델 방식과 동일 가중치 평균 방식이었고, UID 평균 PP를 사용했다고 설명한다.
이 숫자는 **원문 보고 수치**이며 여기서 실행한 점수가 아니다.

제공된 9개 자료에는 최종 CatBoost 학습 코드·전체 최종 LGB FE/학습 코드·LGB 메타모델의 학습 OOF/정확한 설정이 포함되어 있지 않다.
제공된 9개 밖에서 저자 본인의 공개 CatBoost/LightGBM baseline **학습 코드도 확인했으며 실제로 적용했다.** 공개된 것이 XGB뿐이라는 의미가 아니다.
다만 이 공개 baseline이 Part 2의 최종 CAT/LGB와 정확히 같은 FE/모델이라는 근거는 없다. 공개 baseline에 최종 파라미터를 임의로 합치거나 test 예측으로 스태킹을 학습하지 않았다.
원문은 EDA/고객 진단/PP용 정밀 UID와 모델이 집계를 통해 찾는 UID를 구분한다. 새 XGB도 detection v6의 UID를 학습 피처로 넣지 않는다.

공개 `final_model_blend.csv`는 작성자가 최종 blend/stack+PP 결과라고 설명한 파일이다. 동일 가중치 평균+공개 XGB PP와 동일하다고 가정하지 않고 차이를 측정한다.
정확한 마지막 blend/stack 조합을 복원할 자료가 부족하므로 새 평균 결과를 최종 우승 제출 그 자체라고 부르지 않는다.
test 정답이 없으므로 로컬에서 Public/Private LB를 검증할 수 없다.

## 추가 입력과 의존성

실행 가능한 XGB 학습과 EDA는 대회 원본 CSV부터 시작한다. 기존 Model2나 전처리 pickle의 실행을 선행할 필요가 없다.

[공개 submissions and UIDs 데이터셋](https://www.kaggle.com/datasets/kyakovlev/ieee-submissions-and-uids)의 **version 2**를 고정해 공유 kagglehub 캐시를 사용한다.

| 파일 | 용도 |
|---|---|
| `catboost_best_single.csv`, `lgbm_best_single.csv`, `xgb_best_single.csv` | 새 blend의 입력. 각 506,691개 test 예측, 학습 OOF 아님 |
| `uids_v4_no_multiuid_cleaning..csv`, `uids_v1_no_multiuid_cleaning.csv` | XGB 원문 셀 45의 PP 입력, 순서 v4→v1. 파일명에 점 2개인 v4 이름도 그대로 사용 |
| `train_uids_full_v3.csv`, `test_uids_full_v3.csv` | EDA의 공개 정밀 UID 분석. 원본 train/test보다 ID 수가 적고 UID NaN도 있으므로 left alignment |
| `final_model_blend.csv` | 새 평균+PP와의 차이 비교 |
| `lgbm_meta_model.csv` | 공개 test 예측 존재 확인. 학습 OOF가 없어 새 스태킹 학습에는 사용하지 않음 |

보존한 Basic FE와 UID v6 원본의 입력은 `ieee-data-minification-private`라는 이름의 선행 kernel이 생산한 pickle이다.
Local FE 원본도 `ieee-data-minification` pickle을 읽는다. 두 minification kernel 모두 공식 API에서 **version 7, isPrivate=false**와 실제 CSV 로딩 소스를 확인했다. 이름의 private는 현재 접근 불가를 뜻하지 않는다.
이 선행 pickle과 UID v6의 Windows multiprocessing/전역변수 동작까지 원본 그대로 실행하는 작업은 새 XGB 재현 범위에 포함하지 않는다.
특히 v6 코드를 실행해 XGB가 참조하는 v1/v4 결과를 똑같이 생성할 수 있다고 보장할 근거가 없다. 공개된 정확한 버전의 UID CSV를 추가 입력으로 사용한다.

## 원본에서 바꾼 것

XGB 피처, V120 목록, 공동 인코딩·집계, UID 집계, 학습 파라미터, 월별 CV, PP 순서를 유지했다.
pandas 3/XGBoost 3 API에 맞추고, 반복 플롯/중복 학습을 줄였으며 XGB95는 선택 실행으로 바꿨다.
ID는 정수로 읽고 범주 수가 32,000개를 넘으면 int32를 쓰며 로그/metrics/OOF/출력 ID 검증을 추가했다.
현재 라이브러리로 실행하므로 2019년 버전과 점수/트리가 완전히 같다는 보장은 없다.

EDA는 실제 결측 마스크 검증, 상관행렬의 20,000행 seed 42 표본, 첫/마지막 월 단일 피처 LightGBM 진단을 추가했다.
Reference의 임의 public/private 행 분할 가정을 제외했고, 익명 피처 의미를 공식 정의로 단정하지 않는다.
원문 EDA의 V128과 실제 XGB의 V120을 구분한다. 차이 8개는 V96/98/99/104와 V325/332/335/338이다.

train+test 피처 집계와 전체 train 라벨을 이용한 test UID PP는 대회 배치 방식이다. 검증 OOF에 같은 PP를 적용하면서 검증 정답을 포함하지 않는다.

## 추가로 확인한 우승자 공개 학습 경로

아래 버전/공개 여부는 공식 Kaggle pull API의 metadata에서 확인했다. 원본 파일과 SHA-256은 `source.json`의 `additional_winner_resources`에 저장했다.

| 공개 소스 | 현재 버전 | 사용/확인 내용 |
|---|---|---|
| [CatBoost baseline](https://www.kaggle.com/kyakovlev/ieee-catboost-baseline-with-groupkfold-cv) | 4 | GPU depth 8/lr .07/5,000 trees, 결측 그룹/원래 범주형/지배값/UID 복원, 6개월 CV. 새 CatBoost 노트북의 출처 |
| [LGBM GroupKFold](https://www.kaggle.com/kyakovlev/ieee-lgbm-with-groupkfold-cv) | 2 | lr .007/10,000 rounds/256 leaves/feature .5/bagging .7, 월별 CV, stop 100. 새 LGB 노트북의 출처 |
| [FE with some EDA](https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda) | 1 | minification 입력 → 시간/카드/은행/UID/D/금액/C/디바이스 FE → 학습 pickle. 공통 Python 파일로 직접 옮김 |
| [Data minification](https://www.kaggle.com/kyakovlev/ieee-data-minification) | 7 | CSV → card/ProductCD/M4 빈도 인코딩, M/ID 매핑, 수치 축소. 새 공통 FE의 입력 단계 |
| [Data minification private](https://www.kaggle.com/kyakovlev/ieee-data-minification-private) | 7 | 현재 공개 소스. 시간/추정 시작일 추가와 UID v6/Basic FE의 입력 의존성 확인 |
| [Internal blend](https://www.kaggle.com/kyakovlev/ieee-internal-blend) | 10 | 저자의 공개 코드 경로 안내와 이전 4개 baseline 예측 합. 최종 우승 스태킹 코드가 아님 |

새 공통 FE는 원본 pickle 경로를 원본 CSV 로딩으로 바꾸고 `id-xx` 이름과 pandas 3 문자열/주차 API를 맞췄다.
float16 축소를 float32로 바꾸었으며, ProductCD/M4의 전체 target mean을 학습 fold에서만 계산한다. 범주 미등록 값은 fold 평균으로 대체한다.
CatBoost는 현재 API가 요구하는 기존 category dtype 열까지 명시적으로 범주형에 포함한다. 두 모델 모두 원문의 fold별 min-max OOF와 원래 확률 OOF를 별도 저장한다.
Internal blend 원본은 평균이 아닌 **4개 예측 합**이며 확률 1을 넘을 수 있다. 새 최종 공개 예측 3모델 평균과 같은 코드로 취급하지 않았고, 그대로 실행하지 않는다.
