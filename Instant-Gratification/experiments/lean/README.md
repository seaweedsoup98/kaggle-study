# lean — 기준(reference) 파이프라인

**무엇을**: baseline 에서 성능을 만드는 부품만 남긴 파이프라인. 이 저장소의 **새 기준**이다.
이후 실험(모델 다양성, 그룹 간 캘리브레이션 등)은 44분짜리 baseline 대신 이 위에서 돈다.

**왜**: [`experiments/simplify/`](../simplify/) 의 512그룹 해부(README 7.3절, [eda/05](../../eda/05_ablation.ipynb))에서
baseline 부품 9개 중 3개만 결정적이고 2개는 해롭다는 것이 확인됐다.
결정적인 부품만 남기고 시드 4개를 평균하면 baseline 최종과 통계적으로 같은 점수가 나온다.

**입력**: `data/loader.py` 가 찾는 train/test. `--check` 옵션은 `experiments/simplify/outputs/followup_metrics.json` 을 읽는다.

## 파이프라인 (그룹마다 · 시드마다)

```
std > 2 인 컬럼 선택                        피처 선택 — 없으면 QDA 성립 불가
-> KernelPCA(cosine, n_components=d)        차원 유지. 그룹 간 확률 스케일을 맞춘다 (그룹별 AUC 엔 무효, pooled 에 필요)
-> GaussianMixture(5, full) 소속확률 x1      메커니즘. train+test 1024행으로 적합 (transductive)
-> StandardScaler                           없으면 -0.0078
-> QDA(reg_param=0.111)                     target 으로 stratify 한 5-fold OOF
시드 {1,2,3,4} 평균                          시드 1개는 ±0.0005 흔들린다
```

baseline 에서 뺀 것: 나머지 5개 모델, LightGBM/MLP 메타, GMM 로그밀도 ×3, hist 밀도 피처,
소속확률 ×5 반복복사, GMM 라벨 stratify. (마지막 둘은 빼면 유의하게 좋아진다.)

## 실행

```bash
python experiments/lean/lean.py --magic-limit 4 --tag smoke    # 스모크 (수 초)
python experiments/lean/lean.py --check                        # 전체, 시드 4개, 약 8.5분 + simplify 결과와 비트 비교
python experiments/lean/lean.py --seeds 1 2                    # 시드 2개 (약 4분, 약간 손해)
```

## 성능 (512그룹 · 5-fold OOF · pooled AUC)

| | pooled AUC | Δ 그룹별 vs baseline 최종 (95% CI) | fit/그룹 | 시간 |
|---|---|---|---|---|
| baseline (6모델 × 4config + LGBM·MLP 메타) | 0.949605 | — | ~136 + 메타 | 2,617s |
| **lean, 시드 4개** | **0.949699** | +0.0005 [−0.0000, +0.0011] | 32 | ~509s |
| lean, 시드 2개 | 0.949684 | −0.0001 [−0.0006, +0.0004] | 16 | ~254s |
| lean, 시드 1개 | 0.9485 ~ 0.9495 (시드에 따라) | | 8 | ~127s |

"더 좋다"는 입증되지 않았다. **같은 성능을 1/5 시간, 모델 1종, 스태킹 없이** 낸다.

**산출물** (`outputs/`): `lean_metrics.json`(시드별·최종 AUC, 재현 검사), `lean_submission.csv`,
`lean_oof_{train,test}.npy`(gitignore — 후속 실험이 재사용), `lean_run.log`.
