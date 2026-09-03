# simplify — baseline 해부 & 단순화

**무엇을**: baseline 파이프라인의 구성요소를 512개 그룹 전체에서 하나씩 제거해 무엇이 결정적인지 재고,
결정적인 부품만 남긴 가벼운 파이프라인이 baseline 최종 점수(0.949605)에 도달하는지 확인한다.

**왜**: baseline 은 6모델 × 4config × 2단 스태킹인데, QDA 하나(config 1개)의 level-1 OOF 가
0.949064 로 최종과 0.0005 차이다. 복잡도 대부분이 기여하지 않는다는 의심에서 출발한다.
eda/03 이 40그룹에서 같은 방향을 봤지만, 그 파일럿의 결론 하나가 전체 규모에서 뒤집힌
전례(pseudo labeling)가 있어 이번엔 처음부터 512그룹으로 잰다.

**입력**: `data/loader.py` 가 찾는 train/test, 그리고 비교 기준으로
`experiments/baseline/outputs/baseline_metrics.json`, `baseline_oof_train_{level1,final}.npy`.

**프로브 모델**: QDA 하나. `baseline_probe` 설정이 baseline 의 (kmeans, seed=1) QDA 를
비트 단위로 재현하는지 실행 초반에 검사한다(`REPRODUCTION CHECK`).

**실행**:
```bash
python experiments/simplify/simplify.py --magic-limit 4 --tag smoke   # 스모크
python experiments/simplify/simplify.py                                # 전체, 약 1시간
```

**산출물** (`outputs/`): `simplify_metrics.json`, `simplify_run.log`, `simplify_per_group_auc.csv`,
설정별 `simplify_oof_{train,test}_<name>.npy`(gitignore), 가장 단순한 합격 후보의 `simplify_submission.csv`.
후속 실행(`--phase followup --tag followup`)은 같은 형식으로 `followup_*` 를 남기며,
조합(combo) 계산 시 1차 실행의 OOF 를 `--reuse-tag` 로 재사용한다.

## 결과 요약

상세는 저장소 README 7.3절. 재현 검사: `baseline_probe` = `0.9490644878826664`, |diff| = 0.

**해부** (baseline 피처 + QDA, 512그룹 대응 Δ):

| 결정적 (빼면 무너짐) | 무효 | 해로움 (빼면 좋아짐) |
|---|---|---|
| transductive −0.0157 / GMM 소속확률 −0.0111 / StandardScaler −0.0078 | KernelPCA(그룹별) / GMM 로그밀도 / hist | GMM 라벨 stratify +0.0005 / 소속확률 ×5 반복 +0.0007 |

GMM 없는 대안(원시 QDA, 클래스별 GMM 생성분류기, diag 공분산)은 전부 0.93대.

**단순화**: `lean` = std>2 선택 → KernelPCA(cosine) → GMM(5) 소속확률 1회 → StandardScaler → QDA,
target stratify, 시드 4개 평균.

| | pooled AUC | Δ vs baseline (95% CI) | fit/그룹 | 시간 |
|---|---|---|---|---|
| baseline | 0.949605 | — | ~136 + 메타 | 2,617s |
| lean × 4 seeds | 0.949699 | +0.0005 [−0.0000, +0.0011] | 32 | 509s |

같은 점수를 1/5 시간에. 시드 1개는 ±0.0005 흔들리므로 4개 권장.
KernelPCA 는 그룹별 AUC 로는 무효지만 그룹 간 확률 스케일을 맞춰 pooled 를 지키므로 남긴다.
