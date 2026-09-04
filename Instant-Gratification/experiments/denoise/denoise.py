#!/usr/bin/env python
"""Step 1 — 라벨 노이즈 제거 학습 (lean 파이프라인 위에서).

가설: 학습 라벨의 약 5% 가 `flip_y` 로 뒤집혀 있고, 이 행들이 QDA 의 클래스별 공분산 추정을 오염시킨다.
확신 있게 라벨과 어긋나는 학습 행을 **학습에서만** 빼면(평가는 원래 라벨 그대로) 더 깨끗한 추정으로 성능이 오를 것이다.

누수 없는 프로토콜
------------------
outer fold 마다 (학습 fold T, 검증 fold V):
  1. T 안에서 내부 5-fold 로 T 의 각 행에 대한 내부 OOF 확률 p_inner 를 만든다   <- V 의 라벨은 전혀 보지 않음
  2. (p_inner > tau & y=0) 또는 (p_inner < 1-tau & y=1) 인 T 의 행을 제거(drop) 하거나 라벨을 뒤집는다(flip)
  3. 정리된 T 로 QDA 를 다시 적합해 V 와 test 를 예측한다
V 의 라벨은 3 단계의 채점에만 쓰인다. tau=1.0 (아무것도 안 뺌) 이 대조군이며 lean 과 비트 단위로 같아야 한다.

그룹·시드마다 피처(KernelPCA -> GMM -> scaler)는 한 번만 만들고, 모든 설정이 그 위에서 QDA 만 다시 적합한다.
내부 OOF 도 (피처 종류, reg_param) 별로 한 번만 계산해 tau/action 이 다른 설정들이 공유한다.

설정
----
  control            tau=1.0                             lean 재현 (대조군)
  drop_{tau}         tau in --taus, 제거
  flip_{tau}         tau in --flip-taus, 라벨 뒤집기 (더 공격적)
  drop_{tau}_r2      2회 반복: 1차 제거 후 내부 OOF 를 다시 만들어 한 번 더 제거
  drop_{tau}_reg*    제거 후 QDA 정규화를 바꿔 본다 (오염이 사라지면 정규화가 덜 필요한가)
  raw_control / raw_drop_{tau}   GMM 피처 없이 원시 유효 피처 + QDA  (노이즈를 빼면 단순 모델이 따라잡는가)

실행: python experiments/denoise/denoise.py [--taus 0.8 0.9 0.95 0.99] [--seeds 1 2 3 4] [--magic-limit N]
산출: experiments/denoise/outputs/{tag}_metrics.json, {tag}_run.log, {tag}_per_group_auc.csv,
      {tag}_oof_{train,test}_<config>.npy, {tag}_submission.csv (최고 설정)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
for _p in (PROJECT_DIR, PROJECT_DIR / "experiments" / "baseline", PROJECT_DIR / "experiments" / "lean"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data.loader import load_data, load_sample_submission  # noqa: E402
from baseline import MAGIC_COL, N_MAGIC, STD_THRESHOLD, _make_folds  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "outputs"
LEAN_OUT = PROJECT_DIR / "experiments" / "lean" / "outputs"
N_SPLITS = 5
GMM_K = 5
MIN_CLASS_ROWS = 3   # 제거 후 한 클래스가 이보다 적으면 그 fold 는 제거를 취소한다


@dataclass(frozen=True)
class DCfg:
    name: str
    tau: float = 1.0            # 1.0 = 노이즈 제거 없음
    action: str = "drop"        # "drop" | "flip"
    rounds: int = 1
    reg: float = 0.111
    features: str = "lean"      # "lean" (KPCA+GMM+scaler) | "raw" (scaler 만)


def make_configs(taus, flip_taus, extra_reg, best_tau_guess) -> list[DCfg]:
    cfgs = [DCfg("control")]
    cfgs += [DCfg(f"drop_{t}", tau=t) for t in taus]
    cfgs += [DCfg(f"flip_{t}", tau=t, action="flip") for t in flip_taus]
    cfgs += [DCfg(f"drop_{best_tau_guess}_r2", tau=best_tau_guess, rounds=2)]
    cfgs += [DCfg(f"drop_{best_tau_guess}_reg{r}", tau=best_tau_guess, reg=r) for r in extra_reg]
    cfgs += [DCfg("raw_control", features="raw"), DCfg(f"raw_drop_{best_tau_guess}", tau=best_tau_guess, features="raw")]
    return cfgs


# ---------------------------------------------------------------------------
def build_lean_features(xtr, xte, seed, gmm_k):
    """lean.run_group 과 정확히 같은 순서의 연산 (대조군 비트 재현을 위해)."""
    n_train = len(xtr)
    allx = np.vstack([xtr, xte])
    allx = KernelPCA(n_components=xtr.shape[1], kernel="cosine", random_state=seed).fit_transform(allx)
    gmm = GaussianMixture(n_components=gmm_k, random_state=seed, max_iter=1000, init_params="kmeans").fit(allx)
    allx = np.hstack([allx, gmm.predict_proba(allx)])
    allx = StandardScaler().fit_transform(allx)
    return allx[:n_train], allx[n_train:]


def build_raw_features(xtr, xte):
    n_train = len(xtr)
    allx = StandardScaler().fit_transform(np.vstack([xtr, xte]))
    return allx[:n_train], allx[n_train:]


def inner_oof(x, y, reg, seed):
    """학습 fold 안에서만 만든 OOF 확률 (검증 fold 는 관여하지 않음)."""
    p = np.zeros(len(y))
    if np.bincount(y).min() < N_SPLITS:
        # 너무 작아 내부 분할이 불가능하면 in-sample 로 대체 (실제 데이터에서는 일어나지 않는다)
        return QuadraticDiscriminantAnalysis(reg_param=reg).fit(x, y).predict_proba(x)[:, 1]
    for trn, val in StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(x, y):
        p[val] = QuadraticDiscriminantAnalysis(reg_param=reg).fit(x[trn], y[trn]).predict_proba(x[val])[:, 1]
    return p


def flag_noise(p_inner, y, tau):
    return ((p_inner > tau) & (y == 0)) | ((p_inner < 1 - tau) & (y == 1))


def clean_train(x, y, p_inner, cfg: DCfg):
    """설정에 따라 학습 fold 를 정리한다. (정리된 x, y, 제거/뒤집은 행 수, 취소 여부)"""
    flags = flag_noise(p_inner, y, cfg.tau)
    if cfg.action == "flip":
        y2 = y.copy(); y2[flags] = 1 - y2[flags]
        return x, y2, int(flags.sum()), False
    keep = ~flags
    if np.bincount(y[keep], minlength=2).min() < MIN_CLASS_ROWS:
        return x, y, 0, True
    return x[keep], y[keep], int(flags.sum()), False


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="label-noise removal on top of lean")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--magic-limit", type=int, default=N_MAGIC)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--taus", type=float, nargs="+", default=[0.8, 0.9, 0.95, 0.99])
    p.add_argument("--flip-taus", type=float, nargs="+", default=[0.9, 0.95])
    p.add_argument("--best-tau-guess", type=float, default=0.9, help="반복/정규화/raw 변형에 쓸 tau")
    p.add_argument("--extra-reg", type=float, nargs="+", default=[0.05, 0.3])
    p.add_argument("--tag", default="denoise")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def paired(a, b):
    d = a - b
    sem = d.std(ddof=1) / np.sqrt(len(d))
    return {"mean": float(d.mean()), "ci_low": float(d.mean() - 1.96 * sem), "ci_high": float(d.mean() + 1.96 * sem),
            "wins": int((d > 0).sum()), "n": int(len(d))}


def main(argv=None) -> dict:
    args = parse_args(argv)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / f"{args.tag}_run.log", "w", encoding="utf-8")

    def log(msg):
        line = f"[denoise] {msg}"; print(line, flush=True); fh.write(line + "\n"); fh.flush()

    t0 = time.time()
    cfgs = make_configs(args.taus, args.flip_taus, args.extra_reg, args.best_tau_guess)
    log(f"configs={len(cfgs)}: {[c.name for c in cfgs]}")

    log("loading data ...")
    train_df, test_df = load_data(args.data_dir)
    train_df = train_df.reset_index(drop=True); test_df = test_df.reset_index(drop=True)
    feature_columns = [c for c in train_df.columns if c not in ("id", "target", MAGIC_COL)]
    magics = list(range(min(args.magic_limit, N_MAGIC)))
    y_all = train_df["target"].to_numpy()
    mask = train_df[MAGIC_COL].isin(magics).to_numpy()
    groups = {}
    for m in magics:
        xt = train_df[train_df[MAGIC_COL] == m]; xe = test_df[test_df[MAGIC_COL] == m]
        std = xt[feature_columns].std()
        cols = list(std.index.values[np.where(std > STD_THRESHOLD)])
        groups[m] = (xt.index.to_numpy(), xe.index.to_numpy(), xt[cols].to_numpy(), xe[cols].to_numpy(), xt["target"].to_numpy())
    del train_df
    n_train, n_test = len(y_all), len(test_df)

    oof_tr = {c.name: np.zeros((n_train, len(args.seeds))) for c in cfgs}
    oof_te = {c.name: np.zeros((n_test, len(args.seeds))) for c in cfgs}
    removed = {c.name: [] for c in cfgs}       # fold 당 제거/뒤집은 행 비율
    cancelled = {c.name: 0 for c in cfgs}

    for j, seed in enumerate(args.seeds):
        t_seed = time.time()
        for m in tqdm(magics, desc=f"seed {seed}", disable=args.quiet, leave=False):
            tr_idx, te_idx, xtr_raw, xte_raw, y = groups[m]
            feats = {"lean": build_lean_features(xtr_raw, xte_raw, seed, GMM_K),
                     "raw": build_raw_features(xtr_raw, xte_raw)}
            folds = _make_folds(y, y, N_SPLITS, seed)        # lean 과 동일 (target stratify)
            for trn, val in folds:
                inner_cache = {}
                for cfg in cfgs:
                    x_train, x_test = feats[cfg.features]
                    xt, yt = x_train[trn], y[trn]
                    n_removed_total = 0
                    if cfg.tau < 1.0:
                        for _round in range(cfg.rounds):
                            key = (cfg.features, cfg.reg, _round, cfg.tau, cfg.action) if _round > 0 else (cfg.features, cfg.reg)
                            if key not in inner_cache:
                                inner_cache[key] = inner_oof(xt, yt, cfg.reg, seed)
                            xt2, yt2, n_rm, canc = clean_train(xt, yt, inner_cache[key], cfg)
                            if canc:
                                cancelled[cfg.name] += 1
                            n_removed_total += n_rm
                            if cfg.action == "drop" and n_rm == 0:
                                break
                            xt, yt = xt2, yt2
                    removed[cfg.name].append(n_removed_total / len(trn))
                    clf = QuadraticDiscriminantAnalysis(reg_param=cfg.reg).fit(xt, yt)
                    oof_tr[cfg.name][tr_idx[val], j] = clf.predict_proba(x_train[val])[:, 1]
                    oof_te[cfg.name][te_idx, j] += clf.predict_proba(x_test)[:, 1] / len(folds)
        log(f"seed {seed} done ({time.time() - t_seed:.0f}s): " +
            " ".join(f"{c.name}={roc_auc_score(y_all[mask], oof_tr[c.name][mask, j]):.5f}" for c in cfgs[:4]))

    # ---- 집계 ----
    lean_final = np.load(LEAN_OUT / "lean_oof_train.npy") if (LEAN_OUT / "lean_oof_train.npy").exists() else None
    lean_metrics = json.loads((LEAN_OUT / "lean_metrics.json").read_text(encoding="utf-8")) if (LEAN_OUT / "lean_metrics.json").exists() else None
    slices = {m: groups[m][0] for m in magics}

    def group_aucs(pred):
        return np.array([roc_auc_score(groups[m][4], pred[slices[m]]) for m in magics])

    results, per_group = {}, {}
    ctrl_pred = oof_tr["control"].mean(axis=1)
    ctrl_groups = group_aucs(ctrl_pred)
    for cfg in cfgs:
        pred = oof_tr[cfg.name].mean(axis=1)
        pg = group_aucs(pred); per_group[cfg.name] = pg
        results[cfg.name] = {
            "cfg": asdict(cfg),
            "pooled_auc": float(roc_auc_score(y_all[mask], pred[mask])),
            "per_seed_pooled": [float(roc_auc_score(y_all[mask], oof_tr[cfg.name][mask, j])) for j in range(len(args.seeds))],
            "group_auc_mean": float(pg.mean()),
            "vs_control": paired(pg, ctrl_groups),
            "removed_frac_mean": float(np.mean(removed[cfg.name])) if removed[cfg.name] else 0.0,
            "cancelled_folds": cancelled[cfg.name],
        }
        r = results[cfg.name]; v = r["vs_control"]
        log(f"{cfg.name:20s} pooled={r['pooled_auc']:.6f} group_mean={r['group_auc_mean']:.6f} "
            f"removed={r['removed_frac_mean']:.4f} vs_control={v['mean']:+.5f} [{v['ci_low']:+.5f},{v['ci_high']:+.5f}] wins={v['wins']}/{v['n']}")
        np.save(out_dir / f"{args.tag}_oof_train_{cfg.name}.npy", pred)
        np.save(out_dir / f"{args.tag}_oof_test_{cfg.name}.npy", oof_te[cfg.name].mean(axis=1))

    check = None
    if lean_metrics is not None and len(magics) == N_MAGIC and list(args.seeds) == [1, 2, 3, 4]:
        ref = lean_metrics["final_auc"]; this = results["control"]["pooled_auc"]
        check = {"lean_final_auc": ref, "control_auc": this, "abs_diff": abs(ref - this)}
        log(f"REPRODUCTION CHECK control vs lean: |diff|={check['abs_diff']:.2e} -> {'OK (bit-level)' if check['abs_diff'] < 1e-12 else 'MISMATCH'}")

    best = max((c.name for c in cfgs if c.features == "lean"), key=lambda n: results[n]["pooled_auc"])
    log(f"BEST (lean features): {best} pooled={results[best]['pooled_auc']:.6f} "
        f"({results[best]['pooled_auc'] - results['control']['pooled_auc']:+.6f} vs control)")
    sub = load_sample_submission(args.data_dir)
    sub["target"] = oof_te[best].mean(axis=1)
    sub.to_csv(out_dir / f"{args.tag}_submission.csv", index=False)

    pg_df = pd.DataFrame(per_group, index=magics); pg_df.index.name = "magic"
    pg_df.to_csv(out_dir / f"{args.tag}_per_group_auc.csv")
    metrics = {"config": {"magic_limit": len(magics), "seeds": args.seeds, "taus": args.taus, "flip_taus": args.flip_taus,
                          "best_tau_guess": args.best_tau_guess, "extra_reg": args.extra_reg},
               "results": results, "reproduction_check": check, "best": best, "elapsed_sec": round(time.time() - t0, 1)}
    (out_dir / f"{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"done in {metrics['elapsed_sec']}s"); fh.close()
    return metrics


if __name__ == "__main__":
    main()
