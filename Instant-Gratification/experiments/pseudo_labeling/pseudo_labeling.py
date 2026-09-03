#!/usr/bin/env python
"""Instant Gratification pseudo labeling 실험 스크립트.

`experiments/baseline/baseline.py` 의 3단계 스태킹 파이프라인을 그대로 유지한 채,
level-1 의 각 fold 안에 **pseudo labeling 재적합 단계**만 끼워 넣는다.
512개 그룹 / 5-fold / 4개 config / 6개 모델 / 동일한 메타 단계를 쓰므로
최종 숫자를 baseline 의 결과와 그대로 나란히 놓을 수 있다.

누수 없는 프로토콜 (eda/03_feature_engineering.ipynb 7절과 동일)
-----------------------------------------------------------------
fold 마다:
  1. **학습 fold 로만** 적합(stage-1) -> test 전체를 예측
  2. 예측이 확신 구간(p > threshold 또는 p < 1 - threshold)에 드는 **test 행** 에만 가짜 라벨 부여
  3. (학습 fold + 가짜 라벨 test 행) 으로 재적합(stage-2) -> 검증 fold 예측

**불변식**: 검증 fold 의 train 행은 stage-1 에도 stage-2 에도 절대 들어가지 않는다.
가짜 라벨은 오직 test 행에서만 만들어지므로 train 의 정답 라벨이 재적합에 새로 유입되는 경로가 없다.
`_assert_no_leak()` 가 fold 마다 이 불변식을 강제한다.
(피처 변환 자체를 train+test 로 적합하는 transductive 설정은 baseline 과 동일하며,
 라벨을 쓰지 않으므로 여기서 말하는 누수가 아니다.)

stage-1 예측도 함께 저장한다. stage-1 은 pseudo labeling 을 뺀 것 외에는 baseline 과
완전히 같은 fold/seed/피처를 쓰므로, **같은 실행 안에서 얻은 baseline 복제본** 역할을 한다.
실행 간 잡음 없이 대응 비교를 하려면 이쪽이 더 깨끗하다.

코드 재사용
-----------
`HistModel`, `build_model_list`, `_make_folds`, 하이퍼파라미터 상수, `run_meta_level` 은
`baseline` 모듈에서 import 해서 그대로 쓴다. 다만 baseline 의 `run_first_level` 은
피처 생성과 fold 학습이 한 함수에 붙어 있어 pseudo labeling 단계를 끼워 넣을 수 없으므로,
그 중 피처 생성 부분만 `build_group_features()` 로 옮겨 적었다(baseline.py 는 수정하지 않는다).

실행 예시
---------
    python experiments/pseudo_labeling/pseudo_labeling.py --magic-limit 4 --tag smoke
    python experiments/pseudo_labeling/pseudo_labeling.py --threshold 0.90 0.99

결과물은 모두 `experiments/pseudo_labeling/outputs/` 에 생성된다.
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
from sklearn.base import clone
from sklearn.decomposition import KernelPCA
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
BASELINE_CODE_DIR = PROJECT_DIR / "experiments" / "baseline"
for _path in (str(PROJECT_DIR), str(BASELINE_CODE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data.loader import load_data, load_sample_submission  # noqa: E402

# baseline.py 는 `if __name__ == "__main__"` 가드가 있어 import 해도 실행되지 않는다.
from baseline import (  # noqa: E402
    FIRST_LEVEL_CONFIGS,
    MAGIC_COL,
    N_MAGIC,
    STD_THRESHOLD,
    HistModel,
    _make_folds,
    build_model_list,
    run_meta_level,
)

warnings.filterwarnings("ignore")

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_BASELINE_DIR = PROJECT_DIR / "experiments" / "baseline" / "outputs"

# baseline_oof_*_final.npy 의 열 순서
BASE_COLUMN_NAMES = [
    "nusvc_deg4", "nusvc_deg2", "qda", "svc_deg4", "knn16", "logreg_l1",
    "lgbm_meta", "mlp_meta",
]

STAGE1_KEY = "no_pseudo"  # pseudo labeling 을 끄고 돌린 같은 실행 안의 대조군


# ---------------------------------------------------------------------------
# 피처 생성 (baseline.run_first_level 의 앞부분과 동일한 계산)
# ---------------------------------------------------------------------------
def build_group_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    magic: int,
    random_state: int,
    gmm_init_params: str,
) -> tuple | None:
    """magic 그룹 하나의 (x_train, x_test, y, gmm_label, train_idx, test_idx) 를 만든다.

    baseline 과 동일하게 std > 2 컬럼 선택 -> KernelPCA(cosine) -> GMM/히스토그램 파생 피처
    -> StandardScaler 순서이며, 모든 변환은 train+test 를 합쳐 적합한다(transductive).
    선택된 컬럼이 없으면 None 을 돌려준다.
    """
    x_train = train_df[train_df[MAGIC_COL] == magic]
    x_test = test_df[test_df[MAGIC_COL] == magic]

    train_idx_origin = x_train.index.to_numpy()
    test_idx_origin = x_test.index.to_numpy()

    train_std = x_train[feature_columns].std()
    cols = list(train_std.index.values[np.where(train_std > STD_THRESHOLD)])
    if not cols:
        return None

    y_train = x_train["target"].to_numpy()
    x_train = x_train[cols].to_numpy()
    x_test = x_test[cols].to_numpy()
    n_train = x_train.shape[0]

    all_data = np.vstack([x_train, x_test])
    all_data = KernelPCA(
        n_components=len(cols), kernel="cosine", random_state=random_state
    ).fit_transform(all_data)

    gmm = GMM(
        n_components=5, random_state=random_state, max_iter=1000,
        init_params=gmm_init_params,
    ).fit(all_data)
    gmm_pred = gmm.predict_proba(all_data)
    gmm_score = gmm.score_samples(all_data).reshape(-1, 1)
    gmm_label = gmm.predict(all_data)

    hist_pred = HistModel().fit(all_data).predict(all_data).reshape(-1, 1)

    all_data = np.hstack([all_data] + [gmm_pred] * 5)
    all_data = np.hstack([all_data, hist_pred, gmm_score, gmm_score, gmm_score])
    all_data = StandardScaler().fit_transform(all_data)

    return (
        all_data[:n_train],
        all_data[n_train:],
        y_train,
        gmm_label[:n_train],
        train_idx_origin,
        test_idx_origin,
    )


def _assert_no_leak(
    trn_idx: np.ndarray,
    val_idx: np.ndarray,
    n_trn: int,
    n_pseudo: int,
    x_aug: np.ndarray,
    y_aug: np.ndarray,
) -> None:
    """재적합 데이터에 검증 fold 가 섞이지 않았음을 강제한다.

    - 학습 fold 와 검증 fold 는 서로소여야 한다.
    - 증강 데이터의 행 수는 (학습 fold 행) + (가짜 라벨 test 행) 과 정확히 일치해야 한다.
      즉 train 쪽에서 추가로 들어온 행이 하나도 없다는 뜻이다.
    """
    assert np.intersect1d(trn_idx, val_idx).size == 0, "train/val fold overlap"
    assert x_aug.shape[0] == n_trn + n_pseudo, "unexpected rows in augmented set"
    assert y_aug.shape[0] == x_aug.shape[0], "X/y length mismatch"


# ---------------------------------------------------------------------------
# level-1 (pseudo labeling 포함)
# ---------------------------------------------------------------------------
def run_first_level_pseudo(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    magics: list[int],
    thresholds: list[float],
    random_state: int,
    gmm_init_params: str = "kmeans",
    n_splits: int = 5,
    show_progress: bool = True,
) -> dict:
    """magic 그룹별로 stage-1 / stage-2(임계값별) level-1 OOF 예측을 만든다.

    stage-1 은 pseudo labeling 없는 baseline 과 동일한 계산이고, 각 임계값의 stage-2 는
    같은 stage-1 test 예측에서 가짜 라벨을 뽑아 재적합한 결과다.
    임계값들이 stage-1 을 공유하므로 임계값을 따로따로 돌리는 것보다 빠르다.
    """
    model_names, model_list = build_model_list()
    n_models = len(model_list)
    n_train_rows, n_test_rows = len(train_df), len(test_df)
    feature_columns = [c for c in train_df.columns if c not in ("id", "target", MAGIC_COL)]

    keys = [STAGE1_KEY] + [_thr_key(t) for t in thresholds]
    oof_train = {k: np.zeros((n_train_rows, n_models)) for k in keys}
    oof_test = {k: np.zeros((n_test_rows, n_models)) for k in keys}
    # (임계값, 모델) 별로 fold 당 추가된 가짜 라벨 행 수를 누적한다.
    pseudo_rows = {_thr_key(t): np.zeros(n_models) for t in thresholds}
    n_fits = {"stage1": 0, "stage2": 0, "stage2_reused": 0}

    desc = f"PL {gmm_init_params}/seed{random_state}"
    for magic in tqdm(magics, desc=desc, disable=not show_progress):
        built = build_group_features(
            train_df, test_df, feature_columns, magic, random_state, gmm_init_params
        )
        if built is None:
            continue
        x_train, x_test, y_train, gmm_label, train_idx_origin, test_idx_origin = built
        has_test = x_test.shape[0] > 0

        folds = _make_folds(y_train, gmm_label, n_splits, random_state)
        for trn_idx, val_idx in folds:
            x_trn, y_trn = x_train[trn_idx], y_train[trn_idx]
            x_val = x_train[val_idx]

            for model_index, base_model in enumerate(model_list):
                # --- stage 1: 학습 fold 로만 적합 ---
                stage1 = clone(base_model).fit(x_trn, y_trn)
                n_fits["stage1"] += 1
                oof_train[STAGE1_KEY][train_idx_origin[val_idx], model_index] = (
                    stage1.predict_proba(x_val)[:, 1]
                )
                p_test = stage1.predict_proba(x_test)[:, 1] if has_test else None
                if has_test:
                    oof_test[STAGE1_KEY][test_idx_origin, model_index] += p_test / len(folds)

                # --- stage 2: 임계값별로 가짜 라벨을 더해 재적합 ---
                for threshold in thresholds:
                    key = _thr_key(threshold)
                    conf = (
                        (p_test > threshold) | (p_test < 1.0 - threshold)
                        if has_test
                        else np.zeros(0, dtype=bool)
                    )
                    n_pseudo = int(conf.sum())
                    pseudo_rows[key][model_index] += n_pseudo

                    stage2 = stage1
                    if n_pseudo > 0:
                        # 가짜 라벨은 test 행에서만 만든다 (train 행은 학습 fold 그대로).
                        x_aug = np.vstack([x_trn, x_test[conf]])
                        y_aug = np.concatenate(
                            [y_trn, (p_test[conf] > 0.5).astype(y_trn.dtype)]
                        )
                        _assert_no_leak(trn_idx, val_idx, len(trn_idx), n_pseudo, x_aug, y_aug)
                        if len(np.unique(y_aug)) > 1:
                            stage2 = clone(base_model).fit(x_aug, y_aug)
                            n_fits["stage2"] += 1
                        else:
                            n_fits["stage2_reused"] += 1
                    else:
                        n_fits["stage2_reused"] += 1

                    if stage2 is stage1:
                        # 재적합할 것이 없으면 stage-1 예측을 그대로 쓴다 (계산 절약).
                        oof_train[key][train_idx_origin[val_idx], model_index] = (
                            oof_train[STAGE1_KEY][train_idx_origin[val_idx], model_index]
                        )
                        if has_test:
                            oof_test[key][test_idx_origin, model_index] += p_test / len(folds)
                        continue

                    oof_train[key][train_idx_origin[val_idx], model_index] = (
                        stage2.predict_proba(x_val)[:, 1]
                    )
                    if has_test:
                        oof_test[key][test_idx_origin, model_index] += (
                            stage2.predict_proba(x_test)[:, 1] / len(folds)
                        )

    return {
        "model_names": model_names,
        "oof_train": oof_train,
        "oof_test": oof_test,
        "pseudo_rows": pseudo_rows,
        "n_folds": n_splits,
        "n_fits": n_fits,
    }


def _thr_key(threshold: float) -> str:
    return f"p>{threshold:g}"


# ---------------------------------------------------------------------------
# 그룹별 대응 비교
# ---------------------------------------------------------------------------
def per_group_auc(
    y: np.ndarray, pred: np.ndarray, magic_values: np.ndarray, magics: list[int]
) -> np.ndarray:
    """magic 그룹별 ROC AUC. 한 클래스만 있는 그룹은 NaN."""
    out = np.full(len(magics), np.nan)
    for i, magic in enumerate(magics):
        rows = magic_values == magic
        if rows.sum() == 0:
            continue
        y_g = y[rows]
        if len(np.unique(y_g)) < 2:
            continue
        out[i] = roc_auc_score(y_g, pred[rows])
    return out


def paired_comparison(
    treatment: np.ndarray,
    reference: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 0,
) -> dict:
    """그룹별 AUC 의 대응 차이에 대한 평균/95% CI/부트스트랩 CI."""
    diff = treatment - reference
    valid = diff[~np.isnan(diff)]
    n = len(valid)
    mean = float(valid.mean()) if n else float("nan")
    sem = float(valid.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    ci = [mean - 1.96 * sem, mean + 1.96 * sem]

    boot_ci = [float("nan"), float("nan")]
    if n > 1:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        boot_means = valid[idx].mean(axis=1)
        boot_ci = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]

    return {
        "n_groups": int(n),
        "mean_diff": mean,
        "sem": sem,
        "ci95_normal": [float(ci[0]), float(ci[1])],
        "ci95_bootstrap": boot_ci,
        "n_improved": int((valid > 0).sum()),
        "n_worse": int((valid < 0).sum()),
        # 95% CI 가 0 을 포함하지 않을 때만 유의하다고 본다.
        "significant": bool(n > 1 and ci[0] * ci[1] > 0),
    }


def build_comparisons(
    keys: list[str],
    final_arrays: dict[str, np.ndarray],
    model_names: list[str],
    y_all: np.ndarray,
    y_eval: np.ndarray,
    train_mask: np.ndarray,
    magic_values: np.ndarray,
    magics: list[int],
    baseline_dir: Path,
    output_dir: Path,
    prefix: str,
) -> dict:
    """baseline OOF 와의 그룹별 대응 비교 묶음을 만들고 per_group_auc.csv 를 남긴다.

    `final_arrays[key]` 는 (n_train, 8) 배열 = level-1 6개 모델 평균 + lgbm 메타 + mlp 메타이며
    앞 6열은 baseline_oof_train_level1.npy 의 6열과 같은 의미/순서다.

    주의: 8열 평균(final)의 차이에는 **전 그룹에 공통으로 걸리는 메타 모델 재적합** 효과가
    섞여 있어 그룹별 차이가 서로 독립이 아니다. 그룹 단위 CI 는 level-1 비교
    (그룹마다 따로 적합하므로 독립에 가깝다)에서 더 신뢰할 만하다.
    """
    baseline_final_path = baseline_dir / "baseline_oof_train_final.npy"
    baseline_level1_path = baseline_dir / "baseline_oof_train_level1.npy"
    if not baseline_final_path.exists():
        print(f"[pseudo] WARNING: baseline OOF not found at {baseline_final_path}", flush=True)
        return {}

    base_cols = np.load(baseline_level1_path)
    base_final_cols = np.load(baseline_final_path)
    base_group_final = per_group_auc(y_all, base_final_cols.mean(axis=1), magic_values, magics)
    base_group_level1 = per_group_auc(y_all, base_cols.mean(axis=1), magic_values, magics)
    base_group_by_model = {
        name: per_group_auc(y_all, base_cols[:, i], magic_values, magics)
        for i, name in enumerate(model_names)
    }

    out: dict = {
        "baseline_reference": {
            "final_auc": float(roc_auc_score(y_eval, base_final_cols.mean(axis=1)[train_mask])),
            "level1_ensemble_auc": float(
                roc_auc_score(y_eval, base_cols.mean(axis=1)[train_mask])
            ),
            "per_group_mean_final": float(np.nanmean(base_group_final)),
        },
        "paired_comparisons": {},
        "paired_comparisons_per_model": {},
    }

    group_final, group_level1 = {}, {}
    for key in keys:
        cols = final_arrays[key]
        group_final[key] = per_group_auc(y_all, cols.mean(axis=1), magic_values, magics)
        group_level1[key] = per_group_auc(
            y_all, cols[:, : len(model_names)].mean(axis=1), magic_values, magics
        )

    for key in keys:
        entry = {
            "final_vs_baseline_run": paired_comparison(group_final[key], base_group_final),
            "level1_vs_baseline_run": paired_comparison(group_level1[key], base_group_level1),
        }
        if key != STAGE1_KEY:
            # 같은 실행 안의 대조군과의 비교 (실행 간 잡음이 없다)
            entry["final_vs_no_pseudo_same_run"] = paired_comparison(
                group_final[key], group_final[STAGE1_KEY]
            )
            entry["level1_vs_no_pseudo_same_run"] = paired_comparison(
                group_level1[key], group_level1[STAGE1_KEY]
            )
        out["paired_comparisons"][key] = entry

        # 03 파일럿은 QDA 단독이었으므로 모델별 비교도 함께 낸다.
        out["paired_comparisons_per_model"][key] = {
            name: paired_comparison(
                per_group_auc(y_all, final_arrays[key][:, i], magic_values, magics),
                base_group_by_model[name],
            )
            for i, name in enumerate(model_names)
        }

    group_df = pd.DataFrame({"magic": magics, "baseline_final": base_group_final})
    group_df["baseline_level1"] = base_group_level1
    for key in keys:
        group_df[f"{_slug(key)}_final"] = group_final[key]
        group_df[f"{_slug(key)}_level1"] = group_level1[key]
    group_df.to_csv(output_dir / f"{prefix}per_group_auc.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instant Gratification pseudo labeling")
    parser.add_argument("--data-dir", default=None, help="train/public_test/sample_submission location")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="output directory")
    parser.add_argument(
        "--baseline-dir", default=str(DEFAULT_BASELINE_DIR),
        help="directory holding baseline_oof_train_final.npy for the paired comparison",
    )
    parser.add_argument(
        "--threshold", type=float, nargs="+", default=[0.90, 0.99],
        help="pseudo label confidence thresholds (default: 0.90 0.99)",
    )
    parser.add_argument(
        "--magic-limit", type=int, default=N_MAGIC,
        help="number of magic groups to use (default 512; small value = smoke test)",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="level-1 CV folds")
    parser.add_argument("--meta-seeds", type=int, default=4, help="meta model seeds")
    parser.add_argument("--meta-folds", type=int, default=5, help="meta model folds")
    parser.add_argument("--meta-boost-rounds", type=int, default=100, help="LightGBM meta rounds")
    parser.add_argument("--skip-meta", action="store_true", help="run level-1 only")
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="skip training; rebuild the paired comparisons from saved oof_train_final_*.npy",
    )
    parser.add_argument("--tag", default="", help="prefix for output file names")
    parser.add_argument("--quiet", action="store_true", help="disable tqdm progress bars")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.tag}_" if args.tag else ""
    started = time.time()

    thresholds = sorted(args.threshold)
    print("[pseudo] loading data ...", flush=True)
    train_df, test_df = load_data(args.data_dir)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"[pseudo] train={train_df.shape} test={test_df.shape}", flush=True)

    magics = list(range(min(args.magic_limit, N_MAGIC)))
    train_mask = train_df[MAGIC_COL].isin(magics).to_numpy()
    y_train = train_df["target"]
    y_all = y_train.to_numpy()
    y_eval = y_all[train_mask]
    magic_values = train_df[MAGIC_COL].to_numpy()
    print(
        f"[pseudo] magics={len(magics)} thresholds={thresholds} "
        f"train_rows={int(train_mask.sum())}",
        flush=True,
    )

    model_names, _ = build_model_list()
    keys = [STAGE1_KEY] + [_thr_key(t) for t in thresholds]

    # ---------------- 저장된 OOF 로 비교만 다시 계산 ----------------
    if args.analyze_only:
        metrics_path = output_dir / f"{prefix}metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        final_train_arrays = {
            key: np.load(output_dir / f"{prefix}oof_train_final_{_slug(key)}.npy")
            for key in keys
        }
        metrics.update(
            build_comparisons(
                keys, final_train_arrays, model_names, y_all, y_eval, train_mask,
                magic_values, magics, Path(args.baseline_dir), output_dir, prefix,
            )
        )
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[pseudo] rebuilt comparisons in {metrics_path.name}", flush=True)
        return metrics

    metrics: dict = {
        "config": {
            "magic_limit": len(magics),
            "thresholds": thresholds,
            "n_splits": args.n_splits,
            "meta_seeds": args.meta_seeds,
            "meta_folds": args.meta_folds,
            "meta_boost_rounds": args.meta_boost_rounds,
            "first_level_configs": [
                {"gmm_init_params": init, "random_state": seed}
                for init, seed in FIRST_LEVEL_CONFIGS
            ],
            "models": model_names,
            "protocol": (
                "per fold: fit on train fold only -> predict test -> pseudo-label confident "
                "test rows -> refit on (train fold + pseudo rows) -> predict validation fold"
            ),
        },
        "variants": {k: {"level1": {}} for k in keys},
    }

    # ---------------- Level 1 ----------------
    oof_train_sum = {k: np.zeros((len(train_df), len(model_names))) for k in keys}
    oof_test_sum = {k: np.zeros((len(test_df), len(model_names))) for k in keys}
    pseudo_rows_total = {_thr_key(t): np.zeros(len(model_names)) for t in thresholds}
    fit_counts = {"stage1": 0, "stage2": 0, "stage2_reused": 0}

    for init_params, seed in FIRST_LEVEL_CONFIGS:
        run_name = f"{init_params}_seed{seed}"
        run_started = time.time()
        result = run_first_level_pseudo(
            train_df, test_df, magics, thresholds,
            random_state=seed,
            gmm_init_params=init_params,
            n_splits=args.n_splits,
            show_progress=not args.quiet,
        )
        elapsed = round(time.time() - run_started, 1)
        for key in keys:
            oof = result["oof_train"][key]
            scores = {
                name: float(roc_auc_score(y_eval, oof[train_mask, i]))
                for i, name in enumerate(model_names)
            }
            scores["mean_of_models"] = float(
                roc_auc_score(y_eval, oof[train_mask].mean(axis=1))
            )
            scores["elapsed_sec"] = elapsed
            metrics["variants"][key]["level1"][run_name] = scores
            oof_train_sum[key] += oof
            oof_test_sum[key] += result["oof_test"][key]
            print(f"[pseudo] level1 {run_name} [{key}]: {json.dumps(scores)}", flush=True)
        for k, v in result["pseudo_rows"].items():
            pseudo_rows_total[k] += v
        for k in fit_counts:
            fit_counts[k] += result["n_fits"][k]

    n_configs = len(FIRST_LEVEL_CONFIGS)
    n_fold_model_slots = n_configs * len(magics) * args.n_splits * len(model_names)
    metrics["fit_counts"] = fit_counts

    # ---------------- Level 2 / 3 + 그룹별 AUC ----------------
    final_train_arrays: dict[str, np.ndarray] = {}
    final_test_preds: dict[str, np.ndarray] = {}

    for key in keys:
        entry = metrics["variants"][key]
        train_second = pd.DataFrame(oof_train_sum[key] / n_configs, columns=model_names)
        test_second = pd.DataFrame(oof_test_sum[key] / n_configs, columns=model_names)
        entry["level1_ensemble_auc"] = float(
            roc_auc_score(y_eval, train_second.to_numpy()[train_mask].mean(axis=1))
        )
        print(f"[pseudo] [{key}] level1 ensemble AUC = {entry['level1_ensemble_auc']:.6f}", flush=True)

        if key in pseudo_rows_total:
            total = float(pseudo_rows_total[key].sum())
            entry["pseudo_rows_per_fold_per_model"] = round(total / n_fold_model_slots, 1)
            entry["pseudo_rows_per_fold_per_model_by_model"] = {
                name: round(float(pseudo_rows_total[key][i]) / (n_configs * len(magics) * args.n_splits), 1)
                for i, name in enumerate(model_names)
            }
            entry["pseudo_rows_total"] = int(total)

        final_train, final_test = train_second, test_second
        if not args.skip_meta:
            lgbm_tr, lgbm_te, mlp_tr, mlp_te = run_meta_level(
                train_second, test_second, y_train, train_mask,
                seed_number=args.meta_seeds,
                n_folds=args.meta_folds,
                boost_rounds=args.meta_boost_rounds,
                verbose=False,
            )
            entry["level2"] = {
                "lgbm_auc": float(roc_auc_score(y_eval, lgbm_tr["lgbm"].to_numpy()[train_mask])),
                "mlp_auc": float(roc_auc_score(y_eval, mlp_tr["mlp"].to_numpy()[train_mask])),
            }
            print(f"[pseudo] [{key}] level2: {json.dumps(entry['level2'])}", flush=True)
            final_train = pd.concat([train_second, lgbm_tr, mlp_tr], axis=1)
            final_test = pd.concat([test_second, lgbm_te, mlp_te], axis=1)

        final_pred = final_train.to_numpy().mean(axis=1)
        entry["final_auc"] = float(roc_auc_score(y_eval, final_pred[train_mask]))
        print(f"[pseudo] [{key}] final ensemble AUC = {entry['final_auc']:.6f}", flush=True)

        final_train_arrays[key] = final_train.to_numpy()
        final_test_preds[key] = final_test.to_numpy().mean(axis=1)

        np.save(output_dir / f"{prefix}oof_train_final_{_slug(key)}.npy", final_train.to_numpy())
        np.save(output_dir / f"{prefix}oof_test_final_{_slug(key)}.npy", final_test.to_numpy())

    # ---------------- baseline 대비 대응 비교 ----------------
    metrics.update(
        build_comparisons(
            keys, final_train_arrays, model_names, y_all, y_eval, train_mask,
            magic_values, magics, Path(args.baseline_dir), output_dir, prefix,
        )
    )

    # ---------------- submission (가장 좋은 pseudo 설정) ----------------
    pseudo_keys = [k for k in keys if k != STAGE1_KEY]
    best_key = max(pseudo_keys, key=lambda k: metrics["variants"][k]["final_auc"])
    metrics["best_variant"] = best_key
    submission = load_sample_submission(args.data_dir)
    pred_test = final_test_preds[best_key]
    if not (submission["id"].to_numpy() == test_df["id"].to_numpy()).all():
        pred = pd.Series(pred_test, index=test_df["id"].to_numpy())
        submission["target"] = submission["id"].map(pred).to_numpy()
    else:
        submission["target"] = pred_test
    submission_path = output_dir / f"{prefix}submission.csv"
    submission.to_csv(submission_path, index=False)

    metrics["elapsed_sec"] = round(time.time() - started, 1)
    metrics_path = output_dir / f"{prefix}metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[pseudo] best variant = {best_key}", flush=True)
    print(f"[pseudo] wrote {metrics_path.name} and {submission_path.name}", flush=True)
    print(f"[pseudo] done in {metrics['elapsed_sec']:.1f}s", flush=True)
    return metrics


def _slug(key: str) -> str:
    return key.replace(">", "").replace(".", "").replace(" ", "")


if __name__ == "__main__":
    main()
