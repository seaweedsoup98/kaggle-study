# calibrate — 그룹 간 확률 캘리브레이션

**무엇을**: 그룹마다 따로 학습된 QDA 의 확률 스케일을 그룹별 보정기(temperature / Platt / bias)로 맞춰
pooled AUC 가 오르는지 잰다. 재학습 없이 저장된 OOF 예측만 읽는다 (1~2분).

**왜**: 대회 지표는 512개 그룹의 확률을 한 줄로 섞어 순위를 매기는데, 그룹별 AUC 와 pooled AUC 가
여러 번 갈렸다 — `reg_param` 을 키우면 그룹별 +0.0016 인데 pooled 는 평평(7.3절), `minimal` 은 그룹별로
baseline 을 이기고 pooled 로는 진다, `logreg_l1` 은 그룹 내 꼴찌인데 pooled 2위(eda/04). 그룹 간 스케일
불일치가 원인이라면 보정으로 회수될 것이라는 가설.

**입력**: `experiments/lean/outputs/lean_oof_{train,test}.npy` (기본), `--oof-train/--oof-test` 로 다른 OOF 지정 가능.

## 방법

모두 **그룹 안에서의 단조 변환**이라 그룹별 AUC 는 변하지 않고 pooled 만 변한다 — 효과가 순수하게
"그룹 간 비교 가능성"에서 온다.

| 방법 | 변환 | 파라미터/그룹 |
|---|---|---|
| temperature | `sigmoid(logit(p) / T)` | 1 |
| platt | `sigmoid(a·logit(p) + b)` | 2 |
| bias | `sigmoid(logit(p) + b)` | 1 |
| rank (참고) | 그룹 내 순위 | 0, 라벨 불필요 |
| zscore (참고) | 그룹 내 logit 표준화 | 0, 라벨 불필요 |

**누수 방지**: 그룹의 행을 절반으로 나눠 A 로 보정기를 맞추고 B 에 적용, 반대도(교차 적합).
**불확실성**: 그룹을 복원추출하는 부트스트랩 200회로 pooled AUC 차이의 95% CI.

```bash
python experiments/calibrate/calibrate.py                         # lean OOF
python experiments/calibrate/calibrate.py --oof-train ... --name X
```

**산출** (`outputs/`): `{name}_metrics.json`, `{name}_run.log`, 최고 라벨 기반 방법의 `{name}_submission_<method>.csv`.

## 결과 (lean OOF, raw pooled 0.949699)

| 방법 | pooled AUC | Δ | 그룹 부트스트랩 95% CI |
|---|---|---|---|
| temperature (교차 적합) | 0.949062 | **−0.00064** | [−0.00102, −0.00029] |
| platt (교차 적합) | 0.948756 | −0.00094 | [−0.00128, −0.00061] |
| bias (교차 적합) | 0.948791 | −0.00091 | [−0.00130, −0.00044] |
| rank / zscore (라벨 불필요) | 0.948404 / 0.948853 | −0.00130 / −0.00085 | 둘 다 유의하게 손해 |
| platt **in-sample** (같은 행으로 맞춰 같은 행에 적용) | 0.951933 | +0.00223 | 낙관적 — 아래 참고 |
| clip p∈[0.01, 0.99] (라벨·그룹 무관) | 0.949832 | +0.00013 | 잡음 수준 |

- **교차 적합한 모든 보정이 손해다.** 그룹당 512행(절반이면 256행)으로 맞춘 보정기의 잡음이 회수하려는 이득보다 크고,
  두 절반에 다른 보정기가 씌워져 그룹 안 순위마저 깨진다(그룹별 AUC 0.9479 → 0.9470).
- **in-sample 의 +0.0022 는 상한이 아니라 과적합의 크기다.** 라벨의 5% 가 무작위로 뒤집혀 있으므로(denoise/ Step 0),
  그 그룹에 실현된 flip 개수에 보정기가 맞춰지면 pooled 가 오른 것처럼 보인다. 교차 적합에서 부호가 뒤집히는 것이 증거.
- **클리핑의 +0.00013 은 flip 가설과 방향이 맞는다.** 뒤집힌 라벨의 행은 다른 클래스 군집 한가운데 있어 가장 극단적인 확률을 받으므로,
  극단값들 사이의 순서를 동점으로 없애면 손해가 줄어든다 — 그룹별 AUC 가 +0.0019 나 오르는 게 그 흔적이다.
  다만 pooled 로는 잡음 기준(0.0001) 수준이라 채택하지 않는다.

**결론**: 그룹 간 캘리브레이션으로 회수할 것은 없다. eda/04 와 README 7.3 에서 본 pooled/그룹별 분기는
그룹별 스케일 보정으로 고쳐지는 종류가 아니다 — 그룹 AUC 편차 자체가 그룹마다 걸린 flip 수에서 온다
(diagnose: 그룹 확신 오답률 ↔ 그룹 AUC Spearman −0.845).
