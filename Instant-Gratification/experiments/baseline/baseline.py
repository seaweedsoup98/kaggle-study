#!/usr/bin/env python
"""Instant Gratification baseline 실험 스크립트.

`1_Instant_Gratification.ipynb` 의 학습 파트를 그대로 옮기되,
최신 라이브러리에서 실행되지 않던 오류들만 수정한 버전이다.

파이프라인 (3-level stacking)
-----------------------------
Level 1 : magic(=`wheezy-copper-turtle-magic`) 512개 그룹별로
          std > 2 인 컬럼만 선택 -> KernelPCA(cosine) -> GMM/히스토그램 파생 피처 추가
          -> StandardScaler -> 6개 분류기의 OOF 예측 생성.
          (gmm_init_params, random_state) 조합 4가지로 반복한 뒤 평균.
Level 2 : level-1 OOF (6열) 를 입력으로 LightGBM / MLP 메타 모델을 seed 4개 x 5-fold 로 학습.
Level 3 : level-1 평균 6열 + LightGBM 메타 + MLP 메타 = 8열의 단순 평균이 최종 예측.

실행 예시
---------
    python experiments/baseline/baseline.py --magic-limit 8      # 빠른 스모크 테스트
    python experiments/baseline/baseline.py                      # 전체 실행

결과물은 모두 `experiments/baseline/outputs/` 에 생성된다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn import linear_model, neighbors, neural_network, svm
from sklearn.base import clone
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data.loader import load_data, load_sample_submission  # noqa: E402

warnings.filterwarnings("ignore")

BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASELINE_DIR / "outputs"
MAGIC_COL = "wheezy-copper-turtle-magic"
N_MAGIC = 512
STD_THRESHOLD = 2.0

# ---------------------------------------------------------------------------
# 모델 하이퍼파라미터 (원본 노트북과 동일)
# ---------------------------------------------------------------------------
SVNU_PARAMS = {
    "probability": True, "kernel": "poly", "degree": 4, "gamma": "auto",
    "nu": 0.4, "coef0": 0.08, "random_state": 4,
}
SVNU2_PARAMS = {
    "probability": True, "kernel": "poly", "degree": 2, "gamma": "auto",
    "nu": 0.4, "coef0": 0.08, "random_state": 4,
}
QDA_PARAMS = {"reg_param": 0.111}
SVC_PARAMS = {
    "probability": True, "kernel": "poly", "degree": 4, "gamma": "auto",
    "random_state": 4,
}
NEIGHBOR_PARAMS = {"n_neighbors": 16}
LR_PARAMS = {"solver": "liblinear", "penalty": "l1", "C": 0.05, "random_state": 42}

LGBM_META_PARAM = {
    "min_child_weight": 6.790,
    "subsample_for_bin": 50000,
    "bagging_seed": 0,
    "boost_from_average": "true",
    "boost": "gbdt",
    "feature_fraction": 0.450,
    "bagging_fraction": 0.343,
    "learning_rate": 0.025,
    "max_depth": 10,
    "metric": "auc",
    "min_data_in_leaf": 78,
    "min_sum_hessian_in_leaf": 8,
    "num_leaves": 18,
    "num_threads": 8,
    "tree_learner": "serial",
    "objective": "binary",
    "verbosity": -1,
    "lambda_l1": 7.961,
    "lambda_l2": 7.781,
}
MLP16_PARAMS = {
    "activation": "relu", "solver": "lbfgs", "tol": 1e-06,
    "hidden_layer_sizes": (16,), "random_state": 42,
}

# (gmm_init_params, random_state) 조합. 원본 노트북의 4회 run 과 동일.
FIRST_LEVEL_CONFIGS = [
    ("kmeans", 1),
    ("kmeans", 2),
    ("random", 1),
    ("random", 2),
]


def build_model_list() -> tuple[list[str], list]:
    """level-1 에서 사용할 분류기 목록."""
    models = [
        ("nusvc_deg4", svm.NuSVC(**SVNU_PARAMS)),
        ("nusvc_deg2", svm.NuSVC(**SVNU2_PARAMS)),
        ("qda", QuadraticDiscriminantAnalysis(**QDA_PARAMS)),
        ("svc_deg4", svm.SVC(**SVC_PARAMS)),
        ("knn16", neighbors.KNeighborsClassifier(**NEIGHBOR_PARAMS)),
        ("logreg_l1", linear_model.LogisticRegression(**LR_PARAMS)),
    ]
    return [name for name, _ in models], [model for _, model in models]


class HistModel:
    """변수별 히스토그램 높이의 평균을 점수로 쓰는 밀도 근사 모델.

    원본 노트북의 `hist_model` 과 동일한 결과를 내지만, obs/feature 이중
    파이썬 루프를 `np.searchsorted` 로 벡터화해 512개 magic 그룹 x 4회 실행에서도
    현실적인 시간 안에 끝나도록 했다.
    """

    def __init__(self, bins: int = 50) -> None:
        self.bins = bins

    def fit(self, X: np.ndarray) -> "HistModel":
        bin_height, bin_edge = [], []
        for var in X.T:
            height, edge = np.histogram(var, bins=self.bins)
            bin_height.append(height)
            bin_edge.append(edge)
        self.bin_height = np.array(bin_height)   # (n_features, bins)
        self.bin_edge = np.array(bin_edge)       # (n_features, bins + 1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros(0)
        scores = np.empty(X.shape, dtype=float)
        for i in range(X.shape[1]):
            # 원본: (var > edge).argmin() - 1  ==  searchsorted(edge, var, "left") - 1
            idx = np.searchsorted(self.bin_edge[i], X[:, i], side="left") - 1
            idx[idx >= self.bins] = -1  # 마지막 edge 를 넘는 값은 원본과 동일하게 마지막 bin
            scores[:, i] = self.bin_height[i, idx]
        return scores.mean(axis=1)


def _make_folds(
    y: np.ndarray,
    stratify_by: np.ndarray,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedKFold 분할을 만든다.

    원본은 target 이 아닌 GMM 군집 라벨로 stratify 한다(= 군집 비율을 유지한 CV).
    다만 군집 크기가 작거나 특정 fold 의 학습 데이터가 한 클래스만 갖게 되면
    분류기가 예외를 던지므로, 그런 경우에만 target stratify 로 되돌린다.
    """
    n_samples = len(y)
    for candidate in (stratify_by, y):
        counts = np.bincount(candidate.astype(int))
        if counts[counts > 0].min() < n_splits:
            continue
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(splitter.split(np.zeros(n_samples), candidate))
        if all(len(np.unique(y[trn])) > 1 for trn, _ in folds):
            return folds
    # 최후의 보루: target 으로 stratify
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.zeros(n_samples), y))


def run_first_level(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    magics: list[int],
    random_state: int,
    gmm_init_params: str = "kmeans",
    n_splits: int = 5,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """magic 그룹별 파이프라인을 돌려 level-1 OOF 예측을 만든다."""
    model_names, model_list = build_model_list()
    n_models = len(model_list)

    oof_train = np.zeros((len(train_df), n_models))
    oof_test = np.zeros((len(test_df), n_models))
    feature_columns = [c for c in train_df.columns if c not in ("id", "target", MAGIC_COL)]

    desc = f"L1 {gmm_init_params}/seed{random_state}"
    for magic in tqdm(magics, desc=desc, disable=not show_progress):
        x_train = train_df[train_df[MAGIC_COL] == magic]
        x_test = test_df[test_df[MAGIC_COL] == magic]

        train_idx_origin = x_train.index.to_numpy()
        test_idx_origin = x_test.index.to_numpy()

        # 이 magic 그룹에서 정보가 있는(std > 2) 컬럼만 사용
        train_std = x_train[feature_columns].std()
        cols = list(train_std.index.values[np.where(train_std > STD_THRESHOLD)])
        if not cols:
            continue

        y_train = x_train["target"].to_numpy()
        x_train = x_train[cols].to_numpy()
        x_test = x_test[cols].to_numpy()
        n_train = x_train.shape[0]

        all_data = np.vstack([x_train, x_test])

        # Kernel PCA (cosine)
        all_data = KernelPCA(
            n_components=len(cols), kernel="cosine", random_state=random_state
        ).fit_transform(all_data)

        # GMM 기반 파생 피처
        gmm = GMM(
            n_components=5, random_state=random_state, max_iter=1000,
            init_params=gmm_init_params,
        ).fit(all_data)
        gmm_pred = gmm.predict_proba(all_data)
        gmm_score = gmm.score_samples(all_data).reshape(-1, 1)
        gmm_label = gmm.predict(all_data)

        # 히스토그램 밀도 피처
        hist_pred = HistModel().fit(all_data).predict(all_data).reshape(-1, 1)

        # 원본과 동일하게 gmm_pred 를 5번, gmm_score 를 3번 반복해 가중치를 준다
        all_data = np.hstack([all_data] + [gmm_pred] * 5)
        all_data = np.hstack([all_data, hist_pred, gmm_score, gmm_score, gmm_score])
        all_data = StandardScaler().fit_transform(all_data)

        x_train = all_data[:n_train]
        x_test = all_data[n_train:]

        folds = _make_folds(y_train, gmm_label[:n_train], n_splits, random_state)
        for trn_idx, val_idx in folds:
            for model_index, base_model in enumerate(model_list):
                clf = clone(base_model)
                clf.fit(x_train[trn_idx], y_train[trn_idx])
                oof_train[train_idx_origin[val_idx], model_index] = clf.predict_proba(
                    x_train[val_idx]
                )[:, 1]

                if x_test.shape[0] == 0:
                    continue
                oof_test[test_idx_origin, model_index] += (
                    clf.predict_proba(x_test)[:, 1] / len(folds)
                )

    return oof_train, oof_test


def run_meta_level(
    train_second: pd.DataFrame,
    test_second: pd.DataFrame,
    y_train: pd.Series,
    eval_mask: np.ndarray,
    seed_number: int = 4,
    n_folds: int = 5,
    boost_rounds: int = 100,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """level-1 OOF 를 입력으로 LightGBM / MLP 메타 모델을 학습한다."""
    n_train, n_test = len(train_second), len(test_second)
    oof_lgbm_train = np.zeros((n_train, seed_number))
    oof_lgbm_test = np.zeros((n_test, seed_number))
    oof_mlp_train = np.zeros((n_train, seed_number))
    oof_mlp_test = np.zeros((n_test, seed_number))

    fit_index = np.where(eval_mask)[0]  # magic 을 제한한 경우 학습에 쓸 행만 사용
    fit_x = train_second.iloc[fit_index]
    fit_y = y_train.iloc[fit_index]

    for seed in range(seed_number):
        mlp_params = {**MLP16_PARAMS, "random_state": seed}
        lgbm_params = {**LGBM_META_PARAM, "seed": seed}
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_index, (trn_index, val_index) in enumerate(folds.split(fit_x, fit_y), 1):
            trn_x, trn_y = fit_x.iloc[trn_index], fit_y.iloc[trn_index]
            val_x, val_y = fit_x.iloc[val_index], fit_y.iloc[val_index]
            val_rows = fit_index[val_index]

            # --- MLP meta ---
            mlp_meta_model = neural_network.MLPClassifier(**mlp_params)
            mlp_meta_model.fit(trn_x, trn_y)
            oof_mlp_train[val_rows, seed] = mlp_meta_model.predict_proba(val_x)[:, 1]
            oof_mlp_test[:, seed] += mlp_meta_model.predict_proba(test_second)[:, 1] / n_folds
            mlp_auc = roc_auc_score(val_y, oof_mlp_train[val_rows, seed])

            # --- LightGBM meta (lightgbm >= 4 API) ---
            dtrain = lgbm.Dataset(trn_x, label=trn_y)
            dcross = lgbm.Dataset(val_x, label=val_y)
            lgbm_meta_model = lgbm.train(
                lgbm_params,
                train_set=dtrain,
                num_boost_round=boost_rounds,
                valid_sets=[dtrain, dcross],
                callbacks=[
                    lgbm.early_stopping(100, verbose=False),
                    lgbm.log_evaluation(0),
                ],
            )
            oof_lgbm_train[val_rows, seed] = lgbm_meta_model.predict(val_x)
            oof_lgbm_test[:, seed] += lgbm_meta_model.predict(test_second) / n_folds
            lgbm_auc = roc_auc_score(val_y, oof_lgbm_train[val_rows, seed])

            if verbose:
                print(
                    f"  [meta] seed={seed} fold={fold_index} "
                    f"MLP AUC={mlp_auc:.6f} LGBM AUC={lgbm_auc:.6f}",
                    flush=True,
                )

    lgbm_train_df = pd.DataFrame(oof_lgbm_train).mean(axis=1).to_frame().rename(columns={0: "lgbm"})
    lgbm_test_df = pd.DataFrame(oof_lgbm_test).mean(axis=1).to_frame().rename(columns={0: "lgbm"})
    mlp_train_df = pd.DataFrame(oof_mlp_train).mean(axis=1).to_frame().rename(columns={0: "mlp"})
    mlp_test_df = pd.DataFrame(oof_mlp_test).mean(axis=1).to_frame().rename(columns={0: "mlp"})
    return lgbm_train_df, lgbm_test_df, mlp_train_df, mlp_test_df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instant Gratification baseline")
    parser.add_argument("--data-dir", default=None, help="train/public_test/sample_submission 위치")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="결과물 저장 경로")
    parser.add_argument(
        "--magic-limit", type=int, default=N_MAGIC,
        help="사용할 magic 그룹 수 (기본 512, 작게 주면 빠른 스모크 테스트)",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="level-1 CV fold 수")
    parser.add_argument("--meta-seeds", type=int, default=4, help="메타 모델 seed 수")
    parser.add_argument("--meta-folds", type=int, default=5, help="메타 모델 fold 수")
    parser.add_argument(
        "--meta-boost-rounds", type=int, default=100,
        help="LightGBM 메타 모델 부스팅 라운드 (원본 노트북의 기본값 100)",
    )
    parser.add_argument("--skip-meta", action="store_true", help="level-2/3 없이 level-1 만 실행")
    parser.add_argument("--tag", default="baseline", help="결과 파일 접두어")
    parser.add_argument("--quiet", action="store_true", help="진행 로그 최소화")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("[baseline] loading data ...", flush=True)
    train_df, test_df = load_data(args.data_dir)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"[baseline] train={train_df.shape} test={test_df.shape}", flush=True)

    magics = list(range(min(args.magic_limit, N_MAGIC)))
    train_mask = train_df[MAGIC_COL].isin(magics).to_numpy()
    test_mask = test_df[MAGIC_COL].isin(magics).to_numpy()
    y_train = train_df["target"]
    y_eval = y_train.to_numpy()[train_mask]
    print(
        f"[baseline] magics={len(magics)} "
        f"train_rows={int(train_mask.sum())} test_rows={int(test_mask.sum())}",
        flush=True,
    )

    model_names, _ = build_model_list()
    metrics: dict = {
        "config": {
            "magic_limit": len(magics),
            "n_splits": args.n_splits,
            "meta_seeds": args.meta_seeds,
            "meta_folds": args.meta_folds,
            "meta_boost_rounds": args.meta_boost_rounds,
            "first_level_configs": [
                {"gmm_init_params": init, "random_state": seed}
                for init, seed in FIRST_LEVEL_CONFIGS
            ],
            "models": model_names,
        },
        "level1": {},
    }

    # ---------------- Level 1 ----------------
    oof_train_sum = np.zeros((len(train_df), len(model_names)))
    oof_test_sum = np.zeros((len(test_df), len(model_names)))
    for init_params, seed in FIRST_LEVEL_CONFIGS:
        run_name = f"{init_params}_seed{seed}"
        run_started = time.time()
        oof_train, oof_test = run_first_level(
            train_df, test_df, magics,
            random_state=seed,
            gmm_init_params=init_params,
            n_splits=args.n_splits,
            show_progress=not args.quiet,
        )
        run_scores = {
            name: float(roc_auc_score(y_eval, oof_train[train_mask, i]))
            for i, name in enumerate(model_names)
        }
        run_scores["mean_of_models"] = float(
            roc_auc_score(y_eval, oof_train[train_mask].mean(axis=1))
        )
        run_scores["elapsed_sec"] = round(time.time() - run_started, 1)
        metrics["level1"][run_name] = run_scores
        print(f"[baseline] level1 {run_name}: {json.dumps(run_scores)}", flush=True)

        oof_train_sum += oof_train
        oof_test_sum += oof_test

    train_second = pd.DataFrame(oof_train_sum / len(FIRST_LEVEL_CONFIGS), columns=model_names)
    test_second = pd.DataFrame(oof_test_sum / len(FIRST_LEVEL_CONFIGS), columns=model_names)
    metrics["level1_ensemble_auc"] = float(
        roc_auc_score(y_eval, train_second.to_numpy()[train_mask].mean(axis=1))
    )
    print(f"[baseline] level1 ensemble AUC = {metrics['level1_ensemble_auc']:.6f}", flush=True)

    np.save(output_dir / f"{args.tag}_oof_train_level1.npy", train_second.to_numpy())
    np.save(output_dir / f"{args.tag}_oof_test_level1.npy", test_second.to_numpy())

    final_train, final_test = train_second, test_second

    # ---------------- Level 2 / 3 ----------------
    if not args.skip_meta:
        lgbm_train_df, lgbm_test_df, mlp_train_df, mlp_test_df = run_meta_level(
            train_second, test_second, y_train, train_mask,
            seed_number=args.meta_seeds,
            n_folds=args.meta_folds,
            boost_rounds=args.meta_boost_rounds,
            verbose=not args.quiet,
        )
        metrics["level2"] = {
            "lgbm_auc": float(roc_auc_score(y_eval, lgbm_train_df["lgbm"].to_numpy()[train_mask])),
            "mlp_auc": float(roc_auc_score(y_eval, mlp_train_df["mlp"].to_numpy()[train_mask])),
        }
        print(f"[baseline] level2: {json.dumps(metrics['level2'])}", flush=True)

        final_train = pd.concat([train_second, lgbm_train_df, mlp_train_df], axis=1)
        final_test = pd.concat([test_second, lgbm_test_df, mlp_test_df], axis=1)

    metrics["final_auc"] = float(
        roc_auc_score(y_eval, final_train.to_numpy()[train_mask].mean(axis=1))
    )
    print(f"[baseline] final ensemble AUC = {metrics['final_auc']:.6f}", flush=True)

    np.save(output_dir / f"{args.tag}_oof_train_final.npy", final_train.to_numpy())
    np.save(output_dir / f"{args.tag}_oof_test_final.npy", final_test.to_numpy())

    # ---------------- submission ----------------
    submission = load_sample_submission(args.data_dir)
    if not (submission["id"].to_numpy() == test_df["id"].to_numpy()).all():
        # 순서가 다르면 id 기준으로 맞춘다
        pred = pd.Series(final_test.mean(axis=1).to_numpy(), index=test_df["id"].to_numpy())
        submission["target"] = submission["id"].map(pred).to_numpy()
    else:
        submission["target"] = final_test.mean(axis=1).to_numpy()
    submission_path = output_dir / f"{args.tag}_submission.csv"
    submission.to_csv(submission_path, index=False)

    metrics["elapsed_sec"] = round(time.time() - started, 1)
    metrics["outputs"] = {
        "submission": submission_path.name,
        "oof_train_level1": f"{args.tag}_oof_train_level1.npy",
        "oof_test_level1": f"{args.tag}_oof_test_level1.npy",
        "oof_train_final": f"{args.tag}_oof_train_final.npy",
        "oof_test_final": f"{args.tag}_oof_test_final.npy",
    }
    metrics_path = output_dir / f"{args.tag}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[baseline] wrote {metrics_path.name} and {submission_path.name}", flush=True)
    print(f"[baseline] done in {metrics['elapsed_sec']:.1f}s", flush=True)
    return metrics


if __name__ == "__main__":
    main()
