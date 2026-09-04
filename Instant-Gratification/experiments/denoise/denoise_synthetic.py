#!/usr/bin/env python
"""Step 1-b — 합성 데이터에서 노이즈 제거의 '진짜' 효과 재기.

실제 데이터에서는 채점 라벨도 뒤집혀 있어서(Step 0: f≈0.05) 노이즈 제거가 모델을 좋게 만들어도
점수에는 거의 보이지 않는다(상한 0.95). 그래서 **진짜 라벨을 아는 합성 그룹**에서
① 노이즈 제거가 모델 자체를 얼마나 개선하는지 (진짜 라벨 기준 AUC)
② 더 단순한 모델(GMM 피처 없는 raw QDA)이 노이즈 제거로 lean 을 따라잡는지
를 잰다. 프로토콜은 denoise.py 와 같다 (내부 OOF → 제거 → 재적합).

실행: python experiments/denoise/denoise_synthetic.py [--n-groups 60] [--flip 0.05]
산출: experiments/denoise/outputs/synthetic_metrics.json, synthetic_run.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parents[1], HERE.parents[1] / "experiments" / "baseline", HERE.parents[1] / "experiments" / "lean"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from baseline import _make_folds  # noqa: E402
from denoise import DCfg, build_lean_features, build_raw_features, clean_train, inner_oof, N_SPLITS, GMM_K  # noqa: E402
from diagnose import synthetic_group  # noqa: E402

OUT = HERE / "outputs"


def make_cfgs(taus):
    cfgs = [DCfg("lean_control"), DCfg("raw_control", features="raw"),
            DCfg("raw_reg0.5", features="raw", reg=0.5)]
    for t in taus:
        cfgs += [DCfg(f"lean_drop_{t}", tau=t), DCfg(f"raw_drop_{t}", tau=t, features="raw"),
                 DCfg(f"lean_flip_{t}", tau=t, action="flip")]
    return cfgs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-groups", type=int, default=60)
    ap.add_argument("--flip", type=float, default=0.05)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.9, 0.99])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    fh = open(OUT / "synthetic_run.log", "w", encoding="utf-8")

    def log(msg):
        line = f"[synth] {msg}"; print(line, flush=True); fh.write(line + "\n"); fh.flush()

    t0 = time.time()
    cfgs = make_cfgs(args.taus)
    rng = np.random.default_rng(4242)
    preds = {c.name: [] for c in cfgs}; removed = {c.name: [] for c in cfgs}; removed_true = {c.name: [] for c in cfgs}
    ys, yts = [], []
    for g in range(args.n_groups):
        X, y, y_true, flipped = synthetic_group(2000 + g, args.flip, rng)
        tr, te = np.arange(512), np.arange(512, 1024)
        cols = np.where(X[tr].std(axis=0, ddof=1) > 2.0)[0]
        xtr_raw, xte_raw, yg, ytg, flg = X[np.ix_(tr, cols)], X[np.ix_(te, cols)], y[tr], y_true[tr], flipped[tr]
        feats = {"lean": build_lean_features(xtr_raw, xte_raw, args.seed, GMM_K), "raw": build_raw_features(xtr_raw, xte_raw)}
        folds = _make_folds(yg, yg, N_SPLITS, args.seed)
        oof = {c.name: np.zeros(512) for c in cfgs}
        for trn, val in folds:
            cache = {}
            for cfg in cfgs:
                x_train, _ = feats[cfg.features]
                xt, yt = x_train[trn], yg[trn]
                keep_mask = np.ones(len(trn), dtype=bool)
                if cfg.tau < 1.0:
                    key = (cfg.features, cfg.reg)
                    if key not in cache:
                        cache[key] = inner_oof(xt, yt, cfg.reg, args.seed)
                    flags = ((cache[key] > cfg.tau) & (yt == 0)) | ((cache[key] < 1 - cfg.tau) & (yt == 1))
                    xt2, yt2, n_rm, canc = clean_train(xt, yt, cache[key], cfg)
                    removed[cfg.name].append(n_rm / len(trn))
                    # 제거/뒤집은 행 중 진짜로 뒤집혀 있던 비율 (정밀도)
                    if flags.sum():
                        removed_true[cfg.name].append(float(flg[trn][flags].mean()))
                    xt, yt = xt2, yt2
                clf = QuadraticDiscriminantAnalysis(reg_param=cfg.reg).fit(xt, yt)
                oof[cfg.name][val] = clf.predict_proba(x_train[val])[:, 1]
        for c in cfgs:
            preds[c.name].append(oof[c.name])
        ys.append(yg); yts.append(ytg)

    y = np.concatenate(ys); yt = np.concatenate(yts)
    results = {}
    for c in cfgs:
        p = np.concatenate(preds[c.name])
        results[c.name] = {
            "cfg": c.__dict__, "auc_noisy_labels": float(roc_auc_score(y, p)), "auc_true_labels": float(roc_auc_score(yt, p)),
            "removed_frac": float(np.mean(removed[c.name])) if removed[c.name] else 0.0,
            "removed_precision": float(np.mean(removed_true[c.name])) if removed_true[c.name] else None,
        }
        r = results[c.name]
        log(f"{c.name:16s} AUC(noisy)={r['auc_noisy_labels']:.4f} AUC(true)={r['auc_true_labels']:.4f} "
            f"removed={r['removed_frac']:.4f} precision={r['removed_precision'] if r['removed_precision'] is None else round(r['removed_precision'], 3)}")
    metrics = {"n_groups": args.n_groups, "flip": args.flip, "results": results, "elapsed_sec": round(time.time() - t0, 1)}
    (OUT / "synthetic_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"done in {metrics['elapsed_sec']}s"); fh.close()
    return metrics


if __name__ == "__main__":
    main()
