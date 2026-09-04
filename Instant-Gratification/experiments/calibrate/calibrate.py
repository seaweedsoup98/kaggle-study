#!/usr/bin/env python
"""Step 2 — 그룹 간 확률 캘리브레이션.

가설: 대회 지표(pooled AUC)는 512개 그룹의 확률을 한 줄로 섞어 순위를 매기는데,
그룹마다 QDA 확률의 스케일이 달라 그룹 안 순위가 좋아도 pooled 에서 손해를 본다
(eda/04 3절, README 7.3절의 reg_param·KernelPCA·minimal 관측). 그룹마다 확률을 보정하면 회수될 것이다.

방법 (전부 그룹 안에서 단조 변환 -> 그룹별 AUC 는 변하지 않고 pooled 만 변한다)
  temperature : p' = sigmoid(logit(p) / T),  T 하나를 로그손실 최소화로
  platt       : p' = sigmoid(a·logit(p) + b)
  bias        : p' = sigmoid(logit(p) + b)
  참고용(라벨 불필요): rank (그룹 내 순위), zscore (그룹 내 logit 표준화) — 04 에서 rank 는 손해였다

누수 방지: 그룹의 행을 절반으로 나눠 A 로 보정기를 맞추고 B 에 적용, 반대도 (교차 적합).
불확실성: 그룹 단위 부트스트랩(그룹을 복원추출해 pooled AUC 차이의 95% CI).
test 제출용은 그룹 전체 OOF 로 보정기를 맞춰 test 예측에 적용한다.

실행: python experiments/calibrate/calibrate.py [--oof-train PATH --oof-test PATH --name lean]
산출: experiments/calibrate/outputs/{name}_metrics.json, {name}_run.log, {name}_submission_<best>.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from data.loader import get_data_dir, load_sample_submission  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
LEAN_OUT = PROJECT_DIR / "experiments" / "lean" / "outputs"
MAGIC_COL = "wheezy-copper-turtle-magic"
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def fit_temperature(z, y):
    def nll(log_t):
        q = sigmoid(z / np.exp(log_t))
        return -np.mean(y * np.log(q + EPS) + (1 - y) * np.log(1 - q + EPS))
    return np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x)


def fit_bias(z, y):
    def nll(b):
        q = sigmoid(z + b)
        return -np.mean(y * np.log(q + EPS) + (1 - y) * np.log(1 - q + EPS))
    return minimize_scalar(nll, bounds=(-5, 5), method="bounded").x


def fit_platt(z, y):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply(method, z_fit, y_fit, z_apply):
    if method == "temperature":
        return sigmoid(z_apply / fit_temperature(z_fit, y_fit))
    if method == "platt":
        a, b = fit_platt(z_fit, y_fit); return sigmoid(a * z_apply + b)
    if method == "bias":
        return sigmoid(z_apply + fit_bias(z_fit, y_fit))
    if method == "rank":
        return stats.rankdata(z_apply) / len(z_apply)
    if method == "zscore":
        return sigmoid((z_apply - z_fit.mean()) / (z_fit.std() + EPS))
    raise ValueError(method)


LABEL_METHODS = ("temperature", "platt", "bias")
FREE_METHODS = ("rank", "zscore")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-train", default=str(LEAN_OUT / "lean_oof_train.npy"))
    ap.add_argument("--oof-test", default=str(LEAN_OUT / "lean_oof_test.npy"))
    ap.add_argument("--name", default="lean")
    ap.add_argument("--n-boot", type=int, default=100)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    fh = open(OUT / f"{args.name}_run.log", "w", encoding="utf-8")

    def log(msg):
        line = f"[calib] {msg}"; print(line, flush=True); fh.write(line + "\n"); fh.flush()

    t0 = time.time()
    p = np.load(args.oof_train); p_test = np.load(args.oof_test) if Path(args.oof_test).exists() else None
    lab = pd.read_csv(get_data_dir() / "train.csv", usecols=["target", MAGIC_COL])
    y = lab["target"].to_numpy(); magic = lab[MAGIC_COL].to_numpy()
    te_magic = pd.read_csv(get_data_dir() / "public_test.csv", usecols=[MAGIC_COL])[MAGIC_COL].to_numpy() if p_test is not None else None
    order = np.argsort(magic, kind="stable"); bounds = np.searchsorted(magic[order], np.arange(513))
    slices = [order[bounds[m]:bounds[m + 1]] for m in range(512)]
    z = logit(p)
    raw_auc = roc_auc_score(y, p)
    log(f"input={Path(args.oof_train).name} raw pooled AUC={raw_auc:.6f} group-mean AUC={np.mean([roc_auc_score(y[s], p[s]) for s in slices]):.6f}")

    rng = np.random.default_rng(0)
    calibrated = {}
    for method in LABEL_METHODS + FREE_METHODS:
        out = np.empty(len(p))
        for s in slices:
            perm = rng.permutation(s); half = len(s) // 2
            a, b = perm[:half], perm[half:]
            if method in LABEL_METHODS:       # 교차 적합: A 로 맞춰 B 에, B 로 맞춰 A 에
                out[b] = apply(method, z[a], y[a], z[b]); out[a] = apply(method, z[b], y[b], z[a])
            else:
                out[s] = apply(method, z[s], None, z[s])
        calibrated[method] = out

    # 그룹 단위 부트스트랩으로 pooled AUC 차이의 CI
    results = {}
    for method, q in calibrated.items():
        auc = roc_auc_score(y, q)
        diffs = []
        for _ in range(args.n_boot):
            pick = rng.integers(0, 512, size=512)
            idx = np.concatenate([slices[g] for g in pick])
            diffs.append(roc_auc_score(y[idx], q[idx]) - roc_auc_score(y[idx], p[idx]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        results[method] = {"pooled_auc": float(auc), "delta_vs_raw": float(auc - raw_auc),
                           "boot_ci_low": float(lo), "boot_ci_high": float(hi),
                           "group_auc_mean": float(np.mean([roc_auc_score(y[s], q[s]) for s in slices]))}
        log(f"{method:12s} pooled={auc:.6f} delta={auc - raw_auc:+.6f} boot95%=[{lo:+.6f},{hi:+.6f}] group_mean={results[method]['group_auc_mean']:.6f}")

    # 클리핑: p 를 [eps, 1-eps] 로 자르면 그 바깥의 극단값들이 동점이 된다 (라벨 불필요, 그룹 무관).
    # 극단값들 사이의 순서가 라벨과 반대라면(뒤집힌 라벨은 다른 클래스 군집 한가운데 있어 가장 극단적인 확률을 받는다)
    # 동점으로 묶는 것만으로 pooled AUC 가 오른다. 라벨 노이즈 가설의 또 다른 검증이기도 하다.
    for eps in (1e-6, 1e-4, 1e-3, 1e-2, 0.03, 0.05, 0.1):
        q = np.clip(p, eps, 1 - eps)
        auc = roc_auc_score(y, q)
        results[f"clip_{eps:g}"] = {"pooled_auc": float(auc), "delta_vs_raw": float(auc - raw_auc),
                                    "group_auc_mean": float(np.mean([roc_auc_score(y[s], q[s]) for s in slices])),
                                    "frac_clipped": float(((p < eps) | (p > 1 - eps)).mean())}
        r = results[f"clip_{eps:g}"]
        log(f"{'clip eps=' + format(eps, 'g'):24s} pooled={auc:.6f} delta={auc - raw_auc:+.6f} group_mean={r['group_auc_mean']:.6f} clipped={r['frac_clipped']:.3f}")

    # 낙관적 상한: 그룹 전체로 보정기를 맞춰 같은 그룹에 적용 (in-sample).
    # 교차 적합은 두 절반에 서로 다른 보정기를 씌워 그룹 안 순위까지 흔드는데, 그 잡음 없이
    # "완벽히 맞춘 보정기가 있다면 pooled 가 최대 얼마나 오를 수 있는가" 를 본다.
    for method in LABEL_METHODS:
        q = np.empty(len(p))
        for s in slices:
            q[s] = apply(method, z[s], y[s], z[s])
        auc = roc_auc_score(y, q)
        results[f"{method}_insample_upper_bound"] = {
            "pooled_auc": float(auc), "delta_vs_raw": float(auc - raw_auc),
            "group_auc_mean": float(np.mean([roc_auc_score(y[s], q[s]) for s in slices])),
            "note": "optimistic: fit and applied on the same rows",
        }
        log(f"{method + ' (in-sample UB)':24s} pooled={auc:.6f} delta={auc - raw_auc:+.6f}  <- optimistic upper bound")

    best = max(LABEL_METHODS, key=lambda m: results[m]["pooled_auc"])
    log(f"BEST label-based (cross-fitted): {best} delta={results[best]['delta_vs_raw']:+.6f}")

    if p_test is not None:
        zt = logit(p_test); out_t = np.empty(len(p_test))
        te_order = np.argsort(te_magic, kind="stable"); te_bounds = np.searchsorted(te_magic[te_order], np.arange(513))
        for g, s in enumerate(slices):
            ts = te_order[te_bounds[g]:te_bounds[g + 1]]
            out_t[ts] = apply(best, z[s], y[s], zt[ts])     # 그룹 전체 OOF 로 맞춰 test 에 적용
        sub = load_sample_submission(); sub["target"] = out_t
        sub.to_csv(OUT / f"{args.name}_submission_{best}.csv", index=False)

    metrics = {"input": args.oof_train, "raw_pooled_auc": float(raw_auc), "results": results, "best_label_method": best,
               "elapsed_sec": round(time.time() - t0, 1)}
    (OUT / f"{args.name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"done in {metrics['elapsed_sec']}s"); fh.close()
    return metrics


if __name__ == "__main__":
    main()
