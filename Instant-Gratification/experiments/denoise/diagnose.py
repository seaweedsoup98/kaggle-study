#!/usr/bin/env python
"""Step 0 — 라벨 노이즈 진단: 정말 있고, 얼마나 되고, 탐지기가 얼마나 잡는가.

`make_classification` 의 `flip_y` 는 라벨의 일정 비율을 무작위로 뒤집는다.
원 대회에서는 flip_y ≈ 0.05 로 추정됐다. 이 스크립트는 **재학습 없이** lean 의 OOF 예측만으로
1) 뒤집힌 라벨의 흔적이 있는지, 2) 비율이 얼마인지, 3) "확신 오답" 탐지기가 그중 얼마를 잡는지를 잰다.

측정
----
A. 확신 오답 비율 — p > τ 인데 y=0, 또는 p < 1-τ 인데 y=1 인 행의 비율을 τ 별로.
   뒤집힌 라벨이면 τ 를 완화해도 비율이 어느 수준에서 멈추고, 순수 모델 오류면 계속 늘어난다.
B. 뒤집힘 비율 추정 — 라벨이 무작위로 뒤집혔다면 y=1 행은 (1-f)·진짜양성 + f·진짜음성 의 혼합이다.
   진짜 양성이 p<t 에 거의 없을 만큼 작은 t 에서, a=P(p<t|y=1), b=P(p<t|y=0) 이면 f ≈ a/(a+b).
   (대칭으로 p>1-t 쪽에서도 추정해 둘이 일치하는지 본다.)
C. 그룹별 균일성 — 무작위 flip 이면 그룹당 확신 오답 비율이 이항 잡음(sd≈√(r(1-r)/512)) 안에서 균일해야 한다.
   어려운 그룹에 몰려 있으면 모델 오류다. 관측 sd / 이항 sd 비율과 그룹 AUC 와의 상관을 본다.
D. 합성 대조 — 같은 구조의 가짜 그룹(make_classification, n_informative 33~47, 3 clusters/class)을
   flip 비율 {0, 0.02, 0.05, 0.08} 로 만들어 lean 파이프라인을 돌리고,
   (i) AUC 가 실제(0.9497)와 비슷해지는 flip 값, (ii) 확신 오답 비율, (iii) 추정기 B 의 정확도,
   (iv) 탐지기의 정밀도/재현율(진짜 flip 을 알고 있으므로)을 잰다.

실행: python experiments/denoise/diagnose.py
산출: experiments/denoise/outputs/diagnose_metrics.json, diagnose_run.log
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[2]
for _p in (PROJECT_DIR, PROJECT_DIR / "experiments" / "baseline", PROJECT_DIR / "experiments" / "lean"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data.loader import get_data_dir  # noqa: E402
from lean import run_group  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
LEAN_OUT = PROJECT_DIR / "experiments" / "lean" / "outputs"
MAGIC_COL = "wheezy-copper-turtle-magic"
THRESHOLDS = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6]
SMALL_T = [0.01, 0.02, 0.05, 0.1, 0.2]


def conf_wrong_mask(p: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    return ((p > tau) & (y == 0)) | ((p < 1 - tau) & (y == 1))


def flip_estimates(p: np.ndarray, y: np.ndarray) -> dict:
    """B. f ≈ a/(a+b) 를 작은 t 여러 개에서, 양쪽 꼬리로 추정."""
    out = {}
    for t in SMALL_T:
        a_low = np.mean(p[y == 1] < t)          # y=1 인데 p 가 아주 작음 (뒤집힌 음성일 가능성)
        b_low = np.mean(p[y == 0] < t)          # y=0 이고 p 가 아주 작음 (정상)
        a_high = np.mean(p[y == 0] > 1 - t)     # y=0 인데 p 가 아주 큼
        b_high = np.mean(p[y == 1] > 1 - t)
        f_low = a_low / (a_low + b_low) if (a_low + b_low) > 0 else float("nan")
        f_high = a_high / (a_high + b_high) if (a_high + b_high) > 0 else float("nan")
        out[str(t)] = {"f_from_low_tail": float(f_low), "f_from_high_tail": float(f_high),
                       "P(p<t|y=1)": float(a_low), "P(p<t|y=0)": float(b_low)}
    return out


def synthetic_group(seed: int, flip: float, rng: np.random.Generator):
    """실제 그룹과 같은 구조의 가짜 그룹 하나. 진짜 라벨(뒤집기 전)을 함께 돌려준다."""
    n_inf = int(rng.integers(33, 48))
    X, y_true = make_classification(
        n_samples=1024, n_features=255, n_informative=n_inf, n_redundant=0, n_repeated=0,
        n_classes=2, n_clusters_per_class=3, flip_y=0.0, class_sep=1.0, hypercube=True,
        shuffle=True, random_state=seed,
    )
    y = y_true.copy()
    flipped = np.zeros(1024, dtype=bool)
    if flip > 0:
        idx = rng.choice(1024, size=int(round(flip * 1024)), replace=False)
        y[idx] = 1 - y[idx]
        flipped[idx] = True
    return X, y, y_true, flipped


def run_synthetic(flip: float, n_groups: int, seed0: int = 1000) -> dict:
    rng = np.random.default_rng(seed0 + int(flip * 1000))
    oof_all, y_all, ytrue_all, flip_all = [], [], [], []
    for g in range(n_groups):
        X, y, y_true, flipped = synthetic_group(seed0 + g, flip, rng)
        # 실제 데이터처럼 절반은 train, 절반은 test 로 쓴다 (test 는 transductive 파트너로만)
        tr, te = np.arange(512), np.arange(512, 1024)
        std = X[tr].std(axis=0, ddof=1)
        cols = np.where(std > 2.0)[0]
        oof, _ = run_group(X[np.ix_(tr, cols)], X[np.ix_(te, cols)], y[tr], seed=1,
                           gmm_k=5, reg_param=0.111, n_splits=5)
        oof_all.append(oof); y_all.append(y[tr]); ytrue_all.append(y_true[tr]); flip_all.append(flipped[tr])
    p = np.concatenate(oof_all); y = np.concatenate(y_all); yt = np.concatenate(ytrue_all); fl = np.concatenate(flip_all)
    res = {
        "flip": flip, "n_groups": n_groups,
        "pooled_auc_vs_noisy_labels": float(roc_auc_score(y, p)),
        "pooled_auc_vs_true_labels": float(roc_auc_score(yt, p)),
        "actual_flip_rate": float(fl.mean()),
        "conf_wrong_rate": {str(t): float(conf_wrong_mask(p, y, t).mean()) for t in THRESHOLDS},
        "flip_estimates": flip_estimates(p, y),
        "detector": {},
    }
    for t in THRESHOLDS:
        det = conf_wrong_mask(p, y, t)
        tp = int((det & fl).sum())
        res["detector"][str(t)] = {
            "flagged_rate": float(det.mean()),
            "precision": float(tp / det.sum()) if det.sum() else float("nan"),   # 잡은 것 중 진짜 flip
            "recall": float(tp / fl.sum()) if fl.sum() else float("nan"),        # 진짜 flip 중 잡은 것
        }
    return res


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    fh = open(OUT / "diagnose_run.log", "w", encoding="utf-8")

    def log(msg):
        line = f"[diag] {msg}"
        print(line, flush=True); fh.write(line + "\n"); fh.flush()

    t0 = time.time()
    p = np.load(LEAN_OUT / "lean_oof_train.npy")
    lab = pd.read_csv(get_data_dir() / "train.csv", usecols=["target", MAGIC_COL])
    y = lab["target"].to_numpy(); magic = lab[MAGIC_COL].to_numpy()
    log(f"lean OOF loaded: n={len(p)} pooled AUC={roc_auc_score(y, p):.6f}")

    # A. 확신 오답 비율
    cw = {}
    for t in THRESHOLDS:
        m = conf_wrong_mask(p, y, t)
        cw[str(t)] = {"rate": float(m.mean()), "n": int(m.sum()),
                      "y1_pred_low": float(((p < 1 - t) & (y == 1)).sum() / (y == 1).sum()),
                      "y0_pred_high": float(((p > t) & (y == 0)).sum() / (y == 0).sum())}
        log(f"A. conf-wrong tau={t:<4}: {cw[str(t)]['rate']:.4f} ({cw[str(t)]['n']:,} rows) "
            f"[y=1 & p<{1-t:.2f}: {cw[str(t)]['y1_pred_low']:.4f}, y=0 & p>{t}: {cw[str(t)]['y0_pred_high']:.4f}]")

    # B. 뒤집힘 비율 추정
    fe = flip_estimates(p, y)
    for t, v in fe.items():
        log(f"B. flip estimate t={t:<4}: low-tail {v['f_from_low_tail']:.4f}  high-tail {v['f_from_high_tail']:.4f}")

    # C. 그룹별 균일성 (tau=0.9)
    order = np.argsort(magic, kind="stable"); bounds = np.searchsorted(magic[order], np.arange(513))
    slices = [order[bounds[m]:bounds[m + 1]] for m in range(512)]
    m09 = conf_wrong_mask(p, y, 0.9)
    g_rate = np.array([m09[idx].mean() for idx in slices])
    g_auc = np.array([roc_auc_score(y[idx], p[idx]) for idx in slices])
    r = g_rate.mean()
    binom_sd = float(np.sqrt(r * (1 - r) / 512))
    rho, pval = stats.spearmanr(g_rate, g_auc)
    unif = {"mean_rate": float(r), "observed_sd": float(g_rate.std(ddof=1)), "binomial_sd": binom_sd,
            "sd_ratio": float(g_rate.std(ddof=1) / binom_sd), "spearman_vs_group_auc": float(rho), "p_value": float(pval),
            "min_rate": float(g_rate.min()), "max_rate": float(g_rate.max())}
    log(f"C. per-group conf-wrong(0.9): mean {r:.4f}, observed sd {unif['observed_sd']:.4f}, "
        f"binomial sd {binom_sd:.4f} (ratio {unif['sd_ratio']:.2f}), spearman vs group AUC {rho:+.3f} (p={pval:.1e})")

    # D. 합성 대조
    synth = {}
    for flip in (0.0, 0.02, 0.05, 0.08):
        t1 = time.time()
        s = run_synthetic(flip, n_groups=40)
        synth[str(flip)] = s
        d09 = s["detector"]["0.9"]
        log(f"D. synthetic flip={flip:.2f}: AUC(noisy y)={s['pooled_auc_vs_noisy_labels']:.4f} "
            f"AUC(true y)={s['pooled_auc_vs_true_labels']:.4f} conf-wrong(0.9)={s['conf_wrong_rate']['0.9']:.4f} "
            f"est.f(t=0.05)={s['flip_estimates']['0.05']['f_from_low_tail']:.4f} "
            f"detector(0.9) precision={d09['precision']:.2f} recall={d09['recall']:.2f} ({time.time() - t1:.0f}s)")

    metrics = {"real": {"pooled_auc": float(roc_auc_score(y, p)), "conf_wrong": cw, "flip_estimates": fe, "uniformity": unif},
               "synthetic": synth, "elapsed_sec": round(time.time() - t0, 1)}
    (OUT / "diagnose_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"done in {metrics['elapsed_sec']}s")
    fh.close()
    return metrics


if __name__ == "__main__":
    main()
