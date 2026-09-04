#!/usr/bin/env python
"""lean — Instant Gratification 의 기준(reference) 파이프라인.

`experiments/simplify/` 의 해부 결과(README 7.3절, eda/05)에서 성능을 만드는 부품만 남긴 것이다.
baseline(6모델 × 4config + 2단 스태킹, 44분)과 통계적으로 같은 점수를 모델 1종·스태킹 없이 약 8.5분에 낸다.

그룹(=`wheezy-copper-turtle-magic` 값)마다, GMM 시드마다:

    std > 2 인 컬럼 선택                       # 그룹별 피처 선택 — 없으면 QDA 가 성립 불가
    -> KernelPCA(cosine, n_components=d)       # 차원 유지. 그룹 간 확률 스케일을 맞추는 역할 (그룹별 AUC 에는 무효)
    -> GaussianMixture(5, full) 소속확률 1회    # 이 문제의 메커니즘. train+test 1024행으로 적합 (transductive)
    -> StandardScaler                          # 스케일 정리 — 없으면 -0.0078
    -> QDA(reg_param=0.111)                    # target 으로 stratify 한 5-fold OOF

시드 4개의 OOF 예측을 평균한 것이 최종 예측이다. 시드 1개는 ±0.0005 흔들리므로 평균이 필수다.

baseline 과 비교해 뺀 것: 나머지 5개 모델, LightGBM/MLP 메타, GMM 로그밀도 ×3, hist 밀도 피처,
GMM 소속확률 ×5 반복복사, GMM 라벨 stratify (마지막 둘은 빼면 유의하게 좋아진다).

실행 예시
---------
    python experiments/lean/lean.py --magic-limit 4 --tag smoke     # 스모크 (수 초)
    python experiments/lean/lean.py                                 # 전체, 시드 4개, 약 8.5분
    python experiments/lean/lean.py --seeds 1 2                     # 시드 2개 (약 4분, 성능 약간 손해)
    python experiments/lean/lean.py --check                         # simplify 후속 실행의 lean_4seeds 와 비트 비교

결과물은 `experiments/lean/outputs/` 에 생성된다:
`{tag}_metrics.json`, `{tag}_submission.csv`, `{tag}_oof_{train,test}.npy`, `{tag}_run.log`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
BASELINE_CODE_DIR = PROJECT_DIR / "experiments" / "baseline"
for _path in (str(PROJECT_DIR), str(BASELINE_CODE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data.loader import load_data, load_sample_submission  # noqa: E402
from baseline import MAGIC_COL, N_MAGIC, STD_THRESHOLD, _make_folds  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "outputs"
SIMPLIFY_OUTPUT_DIR = PROJECT_DIR / "experiments" / "simplify" / "outputs"

DEFAULT_SEEDS = (1, 2, 3, 4)
GMM_K = 5
REG_PARAM = 0.111
N_SPLITS = 5


def run_group(xtr: np.ndarray, xte: np.ndarray, y: np.ndarray, seed: int,
              gmm_k: int, reg_param: float, n_splits: int) -> tuple[np.ndarray, np.ndarray]:
    """그룹 하나, 시드 하나의 (train OOF 예측, test 예측)."""
    n_train = len(xtr)
    allx = np.vstack([xtr, xte])                       # train+test 를 합쳐 변환을 적합 (transductive)

    allx = KernelPCA(n_components=xtr.shape[1], kernel="cosine", random_state=seed).fit_transform(allx)
    gmm = GaussianMixture(n_components=gmm_k, random_state=seed, max_iter=1000, init_params="kmeans").fit(allx)
    allx = np.hstack([allx, gmm.predict_proba(allx)])  # 소속확률 1회만 붙인다
    allx = StandardScaler().fit_transform(allx)

    x_train, x_test = allx[:n_train], allx[n_train:]
    oof, pred_test = np.zeros(n_train), np.zeros(len(xte))
    folds = _make_folds(y, y, n_splits, seed)          # target 으로 stratify
    for trn, val in folds:
        clf = QuadraticDiscriminantAnalysis(reg_param=reg_param).fit(x_train[trn], y[trn])
        oof[val] = clf.predict_proba(x_train[val])[:, 1]
        pred_test += clf.predict_proba(x_test)[:, 1] / len(folds)
    return oof, pred_test


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="lean reference pipeline")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--magic-limit", type=int, default=N_MAGIC)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--gmm-k", type=int, default=GMM_K)
    p.add_argument("--reg-param", type=float, default=REG_PARAM)
    p.add_argument("--n-splits", type=int, default=N_SPLITS)
    p.add_argument("--tag", default="lean")
    p.add_argument("--check", action="store_true",
                   help="simplify 후속 실행(followup_metrics.json)의 lean_4seeds 와 pooled AUC 를 비트 비교")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fh = open(out_dir / f"{args.tag}_run.log", "w", encoding="utf-8")

    def log(msg):
        line = f"[lean] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    t0 = time.time()
    log("loading data ...")
    train_df, test_df = load_data(args.data_dir)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    feature_columns = [c for c in train_df.columns if c not in ("id", "target", MAGIC_COL)]
    magics = list(range(min(args.magic_limit, N_MAGIC)))
    y_all = train_df["target"].to_numpy()
    mask = train_df[MAGIC_COL].isin(magics).to_numpy()
    log(f"groups={len(magics)} seeds={args.seeds} gmm_k={args.gmm_k} reg_param={args.reg_param}")

    # 그룹별 유효 피처 캐시 (baseline 과 같은 pandas 연산 — 재현성)
    groups = {}
    for m in magics:
        x_train = train_df[train_df[MAGIC_COL] == m]
        x_test = test_df[test_df[MAGIC_COL] == m]
        train_std = x_train[feature_columns].std()
        cols = list(train_std.index.values[np.where(train_std > STD_THRESHOLD)])
        groups[m] = (x_train.index.to_numpy(), x_test.index.to_numpy(),
                     x_train[cols].to_numpy(), x_test[cols].to_numpy(), x_train["target"].to_numpy())
    del train_df

    oof_train = np.zeros((len(y_all), len(args.seeds)))
    oof_test = np.zeros((len(test_df), len(args.seeds)))
    per_seed = {}
    for j, seed in enumerate(args.seeds):
        t_seed = time.time()
        for m in tqdm(magics, desc=f"seed {seed}", disable=args.quiet, leave=False):
            tr_idx, te_idx, xtr, xte, y = groups[m]
            oof, pred_test = run_group(xtr, xte, y, seed, args.gmm_k, args.reg_param, args.n_splits)
            oof_train[tr_idx, j] = oof
            oof_test[te_idx, j] = pred_test
        per_seed[seed] = {"pooled_auc": float(roc_auc_score(y_all[mask], oof_train[mask, j])),
                          "elapsed_sec": round(time.time() - t_seed, 1)}
        log(f"seed {seed}: pooled AUC={per_seed[seed]['pooled_auc']:.6f} ({per_seed[seed]['elapsed_sec']}s)")

    final_train = oof_train.mean(axis=1)
    final_test = oof_test.mean(axis=1)
    final_auc = float(roc_auc_score(y_all[mask], final_train[mask]))
    log(f"FINAL (mean of {len(args.seeds)} seeds): pooled AUC = {final_auc:.6f}")

    check = None
    if args.check and len(magics) == N_MAGIC:
        ref_path = SIMPLIFY_OUTPUT_DIR / "followup_metrics.json"
        if ref_path.exists():
            ref = json.loads(ref_path.read_text(encoding="utf-8"))["combos"].get("lean_4seeds", {}).get("pooled_auc")
            if ref is not None and list(args.seeds) == list(DEFAULT_SEEDS):
                check = {"reference_lean_4seeds": ref, "abs_diff": abs(final_auc - ref)}
                log(f"CHECK vs simplify lean_4seeds: ref={ref:.16f} this={final_auc:.16f} "
                    f"|diff|={check['abs_diff']:.2e} -> {'OK (bit-level)' if check['abs_diff'] < 1e-12 else 'MISMATCH'}")

    np.save(out_dir / f"{args.tag}_oof_train.npy", final_train)
    np.save(out_dir / f"{args.tag}_oof_test.npy", final_test)
    submission = load_sample_submission(args.data_dir)
    assert (submission["id"].to_numpy() == test_df["id"].to_numpy()).all(), "sample_submission 과 test 의 id 순서가 다릅니다"
    submission["target"] = final_test
    submission.to_csv(out_dir / f"{args.tag}_submission.csv", index=False)

    metrics = {
        "config": {"magic_limit": len(magics), "seeds": args.seeds, "gmm_k": args.gmm_k,
                   "reg_param": args.reg_param, "n_splits": args.n_splits,
                   "pipeline": "std>2 select -> KernelPCA(cosine, d) -> GMM(k, full) proba x1 -> StandardScaler -> QDA, target-stratified CV, transductive"},
        "per_seed": per_seed,
        "final_auc": final_auc,
        "check": check,
        "elapsed_sec": round(time.time() - t0, 1),
        "outputs": {"submission": f"{args.tag}_submission.csv", "oof_train": f"{args.tag}_oof_train.npy",
                    "oof_test": f"{args.tag}_oof_test.npy"},
    }
    (out_dir / f"{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"done in {metrics['elapsed_sec']}s")
    log_fh.close()
    return metrics


if __name__ == "__main__":
    main()
