# relabel — flip 으로 라벨을 정정하고, 정정된 라벨 위에서 ablation 을 다시

**무엇을**: 노이즈로 판정된 학습 행을 버리지(drop) 않고 **뒤집어(flip)** 정정하고, 정정된 라벨로 탐지기를
다시 학습해 반복한 뒤, 그 라벨 위에서 파이프라인 후보 11개(lean 의 ablation 5종 + 단순 모델 6종)를 전부 다시 잰다.

**왜**: 7.4절(denoise)에 대한 지적 세 가지.
1. 뒤집힌 행은 다른 클래스의 완벽한 샘플이다 — 버리면 그룹당 ~25행을 잃는다
2. 정정 후 라벨을 쓰는 단계는 전부 다시 돌아야 한다 — 탐지기 포함(반복)
3. 노이즈 5% 가 있을 때 잰 부품 중요도(7.3절)와 깨끗한 라벨에서의 중요도는 다를 수 있다 —
   **baseline 의 복잡성이 노이즈 보상 장치였을 가능성**을 재야 한다

**주의 — lean 의 피처는 라벨을 쓰지 않는다.** std 선택·KernelPCA·GMM·scaler 는 flip 으로 바뀌지 않는다.
라벨이 들어가는 곳은 (a) 분류기 적합 (b) 탐지기 (c) **라벨을 쓰는 피처 추출**(클래스별 GMM, LDA) — (c) 는 노이즈 때문에
지금까지 불리하게 평가됐을 수 있어 후보에 넣었다.

**평가**: 검증 라벨도 5% 뒤집혀 있어 실제 데이터 AUC 는 0.95 천장에 눌린다(차이가 0.0003 안에 압축).
그래서 같은 코드를 **진짜 라벨을 아는 합성 그룹**에서 돌려 진짜 라벨 기준 AUC 로 ablation 을 읽고, 실제 데이터는 나란히 놓는다.

## 설계

```
그룹·시드마다 피처 집합 6개를 한 번 생성 (라벨 무관):  lean / -kpca / -gmm / -scaler / -transductive / raw
outer fold 마다:
  라벨 정정: lean 피처 + QDA 를 학습 fold 안 내부 5-fold 로 → 확신 있게(τ=0.9) 어긋나는 행 flip → 정정 라벨로 반복 (r1, r2, r3)
  라벨 모드 {noisy, flip_r1, flip_r2, flip_r3, (합성) oracle=진짜 라벨} × 파이프라인 11개 → 검증 fold 예측
```
정정은 fold 마다 **한 번**, lean 탐지기로만 한다 → 모든 파이프라인이 같은 라벨을 쓴다(공정한 ablation).

파이프라인: `lean|qda`, `-kpca|qda`, `-gmm|qda`, `-scaler|qda`, `-transductive|qda`,
`raw|qda`, `raw|qda_reg0.5`, `raw|lda`, `raw|gmm_bayes_k1`, `raw|gmm_bayes_k3`, `lean|lda`

```bash
python experiments/relabel/relabel.py --synthetic --n-groups 60 --seeds 1 2 --tag synth   # 수 분
python experiments/relabel/relabel.py --seeds 1 2 --tag real                             # 512그룹, 약 30분
```

**산출** (`outputs/`): `{tag}_metrics.json`, `{tag}_run.log`(요약표 포함), `{tag}_per_group_auc.csv`,
실제 데이터는 라벨 모드별 lean OOF `.npy` 와 최고 모드의 `real_submission_lean_<mode>.csv`.
합성 실행은 라운드별 정정 품질(뒤집은 수 · 맞게 뒤집은 수 · 잘못 뒤집은 수 · 남은 노이즈)을 함께 기록한다.

## 결과 (상세는 저장소 README 7.6절)

**합성 60그룹 · 진짜 라벨 기준 AUC** (noisy → flip×1 → oracle):

| | noisy | flip ×1 | oracle |
|---|---|---|---|
| lean | 0.9998 | 0.9998 | 0.9999 |
| −GMM proba / −transductive | 0.9878 / 0.9869 | 0.9911 / 0.9898 | 0.9916 / 0.9904 |
| raw QDA / raw 클래스별 GMM k=3 | 0.9881 / 0.9874 | 0.9912 / 0.9926 | 0.9917 / 0.9936 |

정정 품질: ×1 이 행의 5.51% 를 뒤집어 정밀도 88%, 남은 노이즈 5.0% → 0.78%. ×2, ×3 은 잘못 뒤집는 수가 맞게 뒤집는 수를 넘어
노이즈가 다시 늘어난다(0.90%, 1.04%). **반복은 1회.**

**실제 512그룹 · 시드 2개 · pooled(뒤집힌 라벨 채점)**: lean 0.949684 → flip×1 0.949665 (대조군은 simplify 의 lean_2seeds 를 비트 재현).
단순 모델들은 flip 으로 +0.003~0.005 오르지만 0.941~0.944 에 머문다.

- flip 은 노이즈에 상한 단순 모델만 살리고 lean 은 이미 노이즈에 강하다
- **oracle(완벽한) 라벨로 학습해도 단순 모델은 lean 을 못 따라잡는다** — GMM proba 와 transductive 는 노이즈 보상이 아니라 필수 부품
- ablation 순위는 정정 라벨 위에서도 그대로. KernelPCA·scaler 는 그룹 안 성능에 무효(역할은 그룹 간 스케일)
- flip ≈ drop (둘 다 lean 에 영향 없음)
