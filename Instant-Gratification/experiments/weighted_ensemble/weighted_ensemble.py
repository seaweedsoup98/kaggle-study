#!/usr/bin/env python
"""Instant Gratification 가중 앙상블 실험 스크립트.

`experiments/baseline/` 이 남긴 3단계 스태킹 OOF 예측(.npy)만 재사용해서,
8개 열(level-1 6개 모델 + LightGBM 메타 + MLP 메타)을 **어떻게 결합할지**만 비교한다.
모델 재학습은 전혀 하지 않으므로 수 분 안에 끝난다.

비교 대상 (모두 "열들의 선형 결합" 이라는 하나의 틀로 표현한다)
---------------------------------------------------------------
표현 공간(transform) 3가지
  * ``prob``  : 원본 확률 그대로
  * ``logit`` : log(p / (1-p)) 로 변환 후 결합 (확률 평균의 0/1 근처 압축을 풀어준다)
  * ``rank``  : 각 열을 순위로 바꿔 [0, 1] 로 정규화 (AUC 는 순위만 보므로 스케일 차이에 강함)

가중치 산출(fitter) 6가지
  * ``equal``        : 단순 평균 (가중치 학습 없음, 기준선)
  * ``logreg``       : 로지스틱 회귀 스태킹의 계수
  * ``nnls``         : 비음수 최소제곱 (음수 가중치를 막아 과적합을 줄인다)
  * ``auc_opt``      : ``scipy.optimize.minimize`` (Nelder-Mead) 로 AUC 를 직접 최대화
  * ``subset``       : 8개 열의 모든 부분집합(255가지)을 전수 조사해 단순 평균이 가장 좋은 조합 선택
  * ``simplex_grid`` : 열 2~3개짜리 저용량 결합을 단위 심플렉스 격자에서 전수 탐색

여기에 level-1 6개 열의 평균을 ``level1_mean`` 파생 열로 추가해,
"level-1 대표값 하나 vs 메타 모델" 처럼 자유도가 1~2 뿐인 결합도 함께 비교한다.
가중치를 전혀 학습하지 않는 단일 열 기준선(``mlp_meta_only``, ``lgbm_meta_only``)도
넣어 두었는데, 학습이 없으므로 CV 와 in-sample 이 일치해 "선택 비용" 의 기준이 된다.

과적합 방지
-----------
OOF 예측 위에서 가중치를 학습한 뒤 같은 OOF 로 AUC 를 재면 낙관적으로 부풀려진다.
그래서 **주 지표는 StratifiedKFold 로 학습 폴드에서만 가중치를 적합하고
홀드아웃 폴드에서 잰 AUC 의 평균(cv_auc)** 이며, 전체 적합 후 전체에서 잰
in-sample AUC 는 참고용으로만 함께 기록한다. 두 값의 차이(gap)가 과적합 정도다.
가중치 학습이 없는 ``equal`` 계열의 gap 은 순수한 폴드 분할 잡음이므로,
다른 방법의 gap 을 해석할 때의 기준선 역할을 한다.

실행 예시
---------
    python experiments/weighted_ensemble/weighted_ensemble.py
    python experiments/weighted_ensemble/weighted_ensemble.py --n-folds 10 --seed 7

결과물은 모두 `experiments/weighted_ensemble/outputs/` 에 생성된다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data.loader import get_data_dir, load_sample_submission  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_BASELINE_DIR = PROJECT_DIR / "experiments" / "baseline" / "outputs"

# baseline_oof_*_final.npy 의 열 순서 (baseline.py 의 build_model_list + 메타 2개)
BASE_COLUMN_NAMES = [
    "nusvc_deg4",
    "nusvc_deg2",
    "qda",
    "svc_deg4",
    "knn16",
    "logreg_l1",
    "lgbm_meta",
    "mlp_meta",
]
# level-1 6개 열의 평균을 파생 열(index 8)로 추가한다.
# "level-1 대표값 1개 vs 메타 모델" 처럼 자유도가 아주 낮은 결합을 만들기 위해서다.
COLUMN_NAMES = BASE_COLUMN_NAMES + ["level1_mean"]
LEVEL1_COLUMNS = list(range(6))
ALL_COLUMNS = list(range(8))
LGBM_COL, MLP_COL, L1MEAN_COL = 6, 7, 8

# 비교 기준으로 삼는 baseline 수치 (experiments/baseline/outputs/baseline_metrics.json)
BASELINE_FINAL_AUC = 0.9496047663578433
# 대응 비교(paired comparison)의 기준이 되는 방법
REFERENCE_METHOD = "mean_prob_8"

EPS = 1e-6


# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------
class Logger:
    """stdout 과 run.log 에 동시에 쓰는 아주 단순한 로거."""

    def __init__(self, path: Path) -> None:
        self.handle = path.open("w", encoding="utf-8")

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def fast_auc(y_bin: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney U 로 계산한 AUC. 최적화 루프 안에서만 쓰는 빠른 버전.

    동점(tie)을 평균 순위로 처리하지 않으므로 이론상 sklearn 과 미세하게 다를 수
    있으나, 여기 점수는 연속 실수라 동점이 사실상 없다.
    보고되는 모든 수치는 `sklearn.metrics.roc_auc_score` 로 다시 계산한다.
    """
    order = np.argsort(score)
    y_sorted = y_bin[order]
    n_pos = int(y_sorted.sum())
    n_neg = y_sorted.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pos_ranks = np.flatnonzero(y_sorted) + 1
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# 표현 공간 변환
# ---------------------------------------------------------------------------
def to_logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def to_rank(prob: np.ndarray) -> np.ndarray:
    """열별 순위를 [0, 1] 로 정규화. 동점은 평균 순위로 처리한다."""
    n_rows = prob.shape[0]
    out = np.empty_like(prob, dtype=float)
    for col in range(prob.shape[1]):
        order = np.argsort(prob[:, col], kind="stable")
        ranks = np.empty(n_rows, dtype=float)
        ranks[order] = np.arange(1, n_rows + 1, dtype=float)
        out[:, col] = ranks / n_rows
    return out


def build_spaces(prob: np.ndarray) -> dict[str, np.ndarray]:
    """(n, 8) 확률 행렬에 level1_mean 파생 열을 붙여 3가지 표현 공간을 만든다."""
    level1_mean = prob[:, LEVEL1_COLUMNS].mean(axis=1, keepdims=True)
    full = np.hstack([prob.astype(float), level1_mean])
    return {"prob": full, "logit": to_logit(full), "rank": to_rank(full)}


# ---------------------------------------------------------------------------
# 가중치 산출기 (fitter)
# ---------------------------------------------------------------------------
# 모든 fitter 는 (Z, y) -> 가중치 벡터 w 를 반환한다. 최종 점수는 Z @ w 이며,
# AUC 는 단조 변환에 불변이므로 절편/스케일은 무시해도 된다.
def fit_equal(Z: np.ndarray, y: np.ndarray, rng: np.random.Generator, args) -> np.ndarray:
    return np.full(Z.shape[1], 1.0 / Z.shape[1])


def fit_logreg(Z: np.ndarray, y: np.ndarray, rng: np.random.Generator, args) -> np.ndarray:
    """표준화 후 로지스틱 회귀. 계수를 원래 열 스케일로 되돌려 반환한다."""
    scaler = StandardScaler().fit(Z)
    model = LogisticRegression(C=args.logreg_c, max_iter=2000, solver="lbfgs")
    model.fit(scaler.transform(Z), y)
    return model.coef_.ravel() / scaler.scale_


def fit_nnls(Z: np.ndarray, y: np.ndarray, rng: np.random.Generator, args) -> np.ndarray:
    """비음수 최소제곱: 음수 가중치를 원천 봉쇄해 과적합 여지를 줄인다."""
    scaler = StandardScaler().fit(Z)
    Zs = scaler.transform(Z)
    weights, _ = nnls(Zs, y - y.mean())
    return weights / scaler.scale_


def fit_auc_opt(Z: np.ndarray, y: np.ndarray, rng: np.random.Generator, args) -> np.ndarray:
    """scipy.optimize.minimize (Nelder-Mead) 로 AUC 를 직접 최대화.

    AUC 는 계단 함수라 기울기가 0 이므로 미분 기반 최적화가 통하지 않는다.
    단순 평균에서 출발하는 시도 1회 + 랜덤 시작점 (restarts - 1) 회를 돌려
    학습 데이터 AUC 가 가장 높은 해를 고른다.
    """
    y_bin = y.astype(np.int8)
    n_cols = Z.shape[1]

    def objective(weights: np.ndarray) -> float:
        return -fast_auc(y_bin, Z @ weights)

    starts = [np.full(n_cols, 1.0 / n_cols)]
    for _ in range(max(0, args.opt_restarts - 1)):
        starts.append(rng.dirichlet(np.ones(n_cols)))

    best_w, best_score = starts[0], objective(starts[0])
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={
                "maxfev": args.opt_maxfev,
                "xatol": 1e-4,
                "fatol": 1e-7,
                "adaptive": True,
            },
        )
        if result.fun < best_score:
            best_score, best_w = float(result.fun), np.asarray(result.x, dtype=float)

    total = np.abs(best_w).sum()
    return best_w / total if total > 0 else best_w


def _simplex_points(n_cols: int, steps: int):
    """합이 steps 인 길이 n_cols 의 비음수 정수 조합을 모두 생성."""
    if n_cols == 1:
        yield (steps,)
        return
    for first in range(steps + 1):
        for rest in _simplex_points(n_cols - 1, steps - first):
            yield (first,) + rest


def fit_simplex_grid(Z: np.ndarray, y: np.ndarray, rng: np.random.Generator, args) -> np.ndarray:
    """단위 심플렉스 위 격자 전수 탐색 (비음수 + 합 1).

    자유도가 열 개수 - 1 밖에 없고 가중치 범위도 [0, 1] 로 묶여 있어
    자유로운 선형 결합보다 과적합 여지가 훨씬 작다. 열이 2~3개일 때만 쓴다.
    """
    n_cols = Z.shape[1]
    if n_cols > 4:
        raise ValueError("simplex_grid is only meant for a handful of columns")
    y_bin = y.astype(np.int8)
    # 열이 3개 이상이면 격자점 수가 급격히 늘어나므로 간격을 최소 0.05 로 제한한다.
    step = args.grid_step if n_cols <= 2 else max(args.grid_step, 0.05)
    steps = int(round(1.0 / step))
    best_w, best_auc = None, -1.0
    for counts in _simplex_points(n_cols, steps):
        weights = np.array(counts, dtype=float) / steps
        score = fast_auc(y_bin, Z @ weights)
        if score > best_auc:
            best_auc, best_w = score, weights
    return best_w


def fit_subset(Z: np.ndarray, y: np.ndarray, rng: np.random.Generator, args) -> np.ndarray:
    """열 부분집합 전수 조사 (2^n - 1 가지). 선택된 열에 균등 가중치를 준다."""
    y_bin = y.astype(np.int8)
    n_cols = Z.shape[1]
    best_w, best_auc = None, -1.0
    for size in range(1, n_cols + 1):
        for combo in itertools.combinations(range(n_cols), size):
            weights = np.zeros(n_cols)
            weights[list(combo)] = 1.0 / size
            score = fast_auc(y_bin, Z @ weights)
            if score > best_auc:
                best_auc, best_w = score, weights
    return best_w


FITTERS = {
    "equal": fit_equal,
    "logreg": fit_logreg,
    "nnls": fit_nnls,
    "auc_opt": fit_auc_opt,
    "subset": fit_subset,
    "simplex_grid": fit_simplex_grid,
}

# (method name, transform, columns, fitter)
METHOD_SPECS = [
    # --- 결합 없는 단일 열 기준선 ---
    ("mean_prob_8", "prob", ALL_COLUMNS, "equal"),
    ("mean_prob_6_level1", "prob", LEVEL1_COLUMNS, "equal"),
    ("mlp_meta_only", "prob", [MLP_COL], "equal"),
    ("lgbm_meta_only", "prob", [LGBM_COL], "equal"),
    ("mean_logit_8", "logit", ALL_COLUMNS, "equal"),
    ("mean_rank_8", "rank", ALL_COLUMNS, "equal"),
    ("logreg_stack_logit_8", "logit", ALL_COLUMNS, "logreg"),
    ("logreg_stack_prob_8", "prob", ALL_COLUMNS, "logreg"),
    ("logreg_stack_rank_8", "rank", ALL_COLUMNS, "logreg"),
    ("nnls_logit_8", "logit", ALL_COLUMNS, "nnls"),
    ("nnls_rank_8", "rank", ALL_COLUMNS, "nnls"),
    ("auc_opt_prob_8", "prob", ALL_COLUMNS, "auc_opt"),
    ("auc_opt_logit_8", "logit", ALL_COLUMNS, "auc_opt"),
    ("auc_opt_rank_8", "rank", ALL_COLUMNS, "auc_opt"),
    ("subset_mean_logit_8", "logit", ALL_COLUMNS, "subset"),
    ("subset_mean_rank_8", "rank", ALL_COLUMNS, "subset"),
    # --- 자유도를 1~2 로 줄인 저용량 결합 ---
    ("blend_l1mean_mlp_prob", "prob", [L1MEAN_COL, MLP_COL], "simplex_grid"),
    ("blend_l1mean_mlp_logit", "logit", [L1MEAN_COL, MLP_COL], "simplex_grid"),
    ("blend_l1mean_mlp_rank", "rank", [L1MEAN_COL, MLP_COL], "simplex_grid"),
    ("blend_l1mean_lgbm_mlp_prob", "prob", [L1MEAN_COL, LGBM_COL, MLP_COL], "simplex_grid"),
    ("blend_l1mean_lgbm_mlp_rank", "rank", [L1MEAN_COL, LGBM_COL, MLP_COL], "simplex_grid"),
]


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def evaluate_method(
    spec: tuple,
    spaces: dict[str, np.ndarray],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    args,
    log: Logger,
    reference_fold_aucs: list[float] | None = None,
) -> dict:
    """한 방법에 대해 정직한 CV AUC 와 in-sample AUC 를 모두 계산한다.

    보고하는 세 가지 CV 지표
      * ``cv_auc``        : 폴드별 홀드아웃 AUC 의 평균 (주 지표)
      * ``cv_pooled_auc`` : 폴드별 홀드아웃 점수를 폴드 안에서 순위 정규화한 뒤
                            전체를 이어붙여 한 번에 잰 AUC.
                            폴드 분할로 인한 표본 축소 효과가 없어
                            baseline 의 전체 데이터 AUC(0.949605) 와 눈금이 같다.
      * ``cv_delta_vs_ref``: 같은 폴드에서 잰 단순 평균(mean_prob_8) 대비 차이의 평균.
                            폴드 잡음이 상쇄되는 대응 비교라 방법 간 우열 판단에 가장 예민하다.
    """
    name, transform, columns, fitter_name = spec
    fitter = FITTERS[fitter_name]
    Z = spaces[transform][:, columns]
    started = time.time()

    # --- K-fold: 학습 폴드에서만 가중치 적합 -> 홀드아웃 폴드에서 평가 ---
    fold_aucs: list[float] = []
    pooled_score = np.empty(len(y), dtype=float)
    for fold_index, (trn_idx, val_idx) in enumerate(folds):
        rng = np.random.default_rng(seed + 1000 * fold_index)
        weights = fitter(Z[trn_idx], y[trn_idx], rng, args)
        val_score = Z[val_idx] @ weights
        fold_aucs.append(float(roc_auc_score(y[val_idx], val_score)))
        # 폴드마다 가중치 스케일이 다르므로 순위로 정규화한 뒤에만 이어붙일 수 있다.
        order = np.argsort(val_score, kind="stable")
        normalized = np.empty(len(val_idx), dtype=float)
        normalized[order] = np.arange(1, len(val_idx) + 1, dtype=float) / len(val_idx)
        pooled_score[val_idx] = normalized

    # --- 전체 데이터 적합 -> 전체 데이터 평가 (in-sample, 낙관적) ---
    rng = np.random.default_rng(seed)
    full_weights = fitter(Z, y, rng, args)
    in_sample_auc = float(roc_auc_score(y, Z @ full_weights))

    cv_auc = float(np.mean(fold_aucs))
    cv_pooled_auc = float(roc_auc_score(y, pooled_score))
    result = {
        "transform": transform,
        "fitter": fitter_name,
        "columns": [COLUMN_NAMES[c] for c in columns],
        "cv_auc": cv_auc,
        "cv_auc_std": float(np.std(fold_aucs)),
        "cv_pooled_auc": cv_pooled_auc,
        "fold_aucs": fold_aucs,
        "in_sample_auc": in_sample_auc,
        "gap_in_sample_minus_cv": in_sample_auc - cv_auc,
        "gap_in_sample_minus_pooled": in_sample_auc - cv_pooled_auc,
        "weights": {COLUMN_NAMES[c]: float(w) for c, w in zip(columns, full_weights)},
        "elapsed_sec": round(time.time() - started, 1),
    }
    if reference_fold_aucs is not None:
        deltas = np.array(fold_aucs) - np.array(reference_fold_aucs)
        result["cv_delta_vs_ref"] = float(deltas.mean())
        result["cv_delta_vs_ref_std"] = float(deltas.std())
        result["cv_delta_vs_ref_per_fold"] = [float(d) for d in deltas]

    delta_text = ""
    if "cv_delta_vs_ref" in result:
        delta_text = (
            f" d_vs_mean={result['cv_delta_vs_ref']:+.6f}"
            f"(+-{result['cv_delta_vs_ref_std']:.6f})"
        )
    log(
        f"[wens] {name:<28s} cv={cv_auc:.6f}(+-{result['cv_auc_std']:.6f}) "
        f"pooled={cv_pooled_auc:.6f} in_sample={in_sample_auc:.6f} "
        f"gap={result['gap_in_sample_minus_pooled']:+.6f}{delta_text} "
        f"({result['elapsed_sec']:.1f}s)"
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # NOTE: argparse 의 help 는 --help 시 stdout 으로 나간다. 이 셸의 stdout 인코딩이
    # cp1252 라 한글을 넣으면 UnicodeEncodeError 로 죽으므로 영어로만 적는다.
    parser = argparse.ArgumentParser(description="Instant Gratification weighted ensemble")
    parser.add_argument("--data-dir", default=None, help="directory holding train.csv / sample_submission.csv")
    parser.add_argument(
        "--baseline-dir", default=str(DEFAULT_BASELINE_DIR),
        help="directory holding the baseline OOF .npy files (read only)",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="where metrics.json / submission.csv / run.log are written")
    parser.add_argument("--n-folds", type=int, default=5, help="number of folds used to honestly evaluate the weight fitting")
    parser.add_argument("--seed", type=int, default=42, help="random seed for fold splitting and optimizer restarts")
    parser.add_argument(
        "--opt-restarts", type=int, default=2,
        help="number of auc_opt starting points (first is the simple mean, rest random)",
    )
    parser.add_argument(
        "--opt-maxfev", type=int, default=1000,
        help="max objective evaluations for the auc_opt Nelder-Mead search",
    )
    parser.add_argument("--logreg-c", type=float, default=1.0, help="inverse regularization strength C for the logistic-regression stack")
    parser.add_argument(
        "--grid-step", type=float, default=0.02,
        help="simplex_grid step size (0.02 searches weights in 0.02 increments)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(output_dir / "run.log")
    started = time.time()

    try:
        # ---------------- 입력 ----------------
        baseline_dir = Path(args.baseline_dir)
        oof_train = np.load(baseline_dir / "baseline_oof_train_final.npy")
        oof_test = np.load(baseline_dir / "baseline_oof_test_final.npy")
        log(f"[wens] loaded oof train={oof_train.shape} test={oof_test.shape}")

        # target 컬럼만 읽어 메모리/시간 절약 (train.csv 는 1.3GB)
        data_dir = get_data_dir(args.data_dir)
        y = pd.read_csv(data_dir / "train.csv", usecols=["target"])["target"].to_numpy()
        log(f"[wens] loaded target n={len(y)} positive_rate={y.mean():.6f}")
        if len(y) != oof_train.shape[0]:
            raise ValueError("target length does not match OOF rows")

        train_spaces = build_spaces(oof_train)
        test_spaces = build_spaces(oof_test)

        splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        folds = list(splitter.split(np.zeros(len(y)), y))
        log(f"[wens] folds={args.n_folds} seed={args.seed} methods={len(METHOD_SPECS)}")
        log(f"[wens] reference baseline final AUC = {BASELINE_FINAL_AUC:.6f}")

        # ---------------- 방법별 평가 ----------------
        results: dict[str, dict] = {}
        reference_fold_aucs: list[float] | None = None
        for spec in METHOD_SPECS:
            results[spec[0]] = evaluate_method(
                spec, train_spaces, y, folds, args.seed, args, log, reference_fold_aucs
            )
            if spec[0] == REFERENCE_METHOD:
                reference_fold_aucs = results[spec[0]]["fold_aucs"]

        # ---------------- 최종 선택 & 제출 ----------------
        ranking = sorted(results.items(), key=lambda kv: kv[1]["cv_auc"], reverse=True)
        best_name, best = ranking[0]
        log("[wens] ranking by cv_auc:")
        for rank, (name, res) in enumerate(ranking, 1):
            log(
                f"[wens]   {rank:2d}. {name:<28s} cv={res['cv_auc']:.6f} "
                f"pooled={res['cv_pooled_auc']:.6f} in_sample={res['in_sample_auc']:.6f} "
                f"gap={res['gap_in_sample_minus_pooled']:+.6f}"
            )
        log(
            f"[wens] best={best_name} cv_auc={best['cv_auc']:.6f} "
            f"pooled={best['cv_pooled_auc']:.6f} "
            f"pooled_delta_vs_baseline={best['cv_pooled_auc'] - BASELINE_FINAL_AUC:+.6f}"
        )
        log(f"[wens] best weights: {json.dumps(best['weights'])}")

        columns = [COLUMN_NAMES.index(c) for c in best["columns"]]
        weight_vector = np.array([best["weights"][COLUMN_NAMES[c]] for c in columns])
        test_score = test_spaces[best["transform"]][:, columns] @ weight_vector

        # 제출은 순위만 의미가 있으므로 [0, 1] 로 min-max 정규화해 저장한다.
        span = test_score.max() - test_score.min()
        test_pred = (test_score - test_score.min()) / span if span > 0 else test_score

        submission = load_sample_submission(args.data_dir)
        if len(submission) != len(test_pred):
            raise ValueError("sample_submission length does not match test rows")
        submission["target"] = test_pred
        submission_path = output_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)
        log(f"[wens] wrote {submission_path.name} rows={len(submission)}")

        # 참고: baseline 제출과의 순위 상관
        baseline_test_mean = oof_test.mean(axis=1)
        corr = float(pd.Series(test_pred).corr(pd.Series(baseline_test_mean), method="spearman"))
        log(f"[wens] spearman corr(best submission, baseline mean submission) = {corr:.6f}")

        metrics = {
            "config": {
                "n_folds": args.n_folds,
                "seed": args.seed,
                "opt_restarts": args.opt_restarts,
                "opt_maxfev": args.opt_maxfev,
                "logreg_c": args.logreg_c,
                "columns": COLUMN_NAMES,
                "grid_step": args.grid_step,
                "baseline_dir": str(baseline_dir),
            },
            "reference": {
                "paired_reference_method": REFERENCE_METHOD,
                "baseline_final_auc_8col_mean": BASELINE_FINAL_AUC,
                "baseline_level1_ensemble_auc": 0.949604001101723,
                "baseline_lgbm_meta_auc": 0.9495550173176566,
                "baseline_mlp_meta_auc": 0.9499631329991503,
            },
            "methods": results,
            "best": {
                "name": best_name,
                "cv_auc": best["cv_auc"],
                "cv_pooled_auc": best["cv_pooled_auc"],
                "in_sample_auc": best["in_sample_auc"],
                "cv_delta_vs_ref": best.get("cv_delta_vs_ref"),
                "pooled_delta_vs_baseline_final": best["cv_pooled_auc"] - BASELINE_FINAL_AUC,
                "weights": best["weights"],
                "transform": best["transform"],
                "spearman_vs_baseline_submission": corr,
            },
            "elapsed_sec": round(time.time() - started, 1),
            "outputs": {"metrics": "metrics.json", "submission": "submission.csv", "log": "run.log"},
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log(f"[wens] wrote metrics.json; done in {metrics['elapsed_sec']:.1f}s")
        return metrics
    finally:
        log.close()


if __name__ == "__main__":
    main()
