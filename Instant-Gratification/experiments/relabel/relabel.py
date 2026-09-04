#!/usr/bin/env python
"""relabel — flip 으로 라벨을 정정하고, 정정된 라벨 위에서 파이프라인 전체와 ablation 을 다시 돈다.

denoise 실험(7.4절)에 대한 지적에서 출발했다:
  1. 노이즈 행은 버리는(drop) 게 아니라 **뒤집어야(flip)** 한다 — 뒤집힌 행은 다른 클래스의 완벽한 샘플이다.
  2. 정정한 뒤 **라벨을 쓰는 단계를 전부 다시** 돌려야 한다 — 탐지기도 정정된 라벨로 다시 학습해 반복한다.
  3. **ablation 도 정정된 라벨 위에서 다시** 해야 한다 — 노이즈 5% 가 있을 때의 부품 중요도와
     깨끗한 라벨에서의 중요도는 다를 수 있다 (baseline 의 복잡성이 노이즈 보상 장치였을 가능성).

lean 의 피처(std 선택, KernelPCA, GMM, scaler)는 라벨을 쓰지 않으므로 flip 으로 바뀌지 않는다.
라벨이 들어가는 곳은 (a) 분류기 적합 (b) 탐지기 (c) 라벨을 쓰는 피처 추출(클래스별 GMM, LDA) 이다.
그래서 이 스크립트는 피처 집합은 그룹·시드마다 한 번만 만들고, 그 위에서
  라벨 모드 x 파이프라인
격자를 전부 돈다. 라벨 정정은 fold 마다 lean 탐지기 하나로 한 번 수행해 모든 파이프라인이 같은 라벨을 쓴다(공정한 ablation).

평가
----
검증 라벨도 5% 뒤집혀 있어 실제 데이터의 AUC 는 0.95 천장에 눌린다. 그래서 같은 코드를
진짜 라벨을 아는 합성 그룹(--synthetic)에서도 돌려 **진짜 라벨 기준 AUC** 로 ablation 을 읽는다.

라벨 모드
  noisy      원래 라벨
  flip_r{k}  lean 탐지기(학습 fold 안 내부 5-fold, tau) 로 확신 있게 어긋나는 행을 뒤집기, k 회 반복
  oracle     (합성만) 진짜 라벨로 학습 — 정정의 상한

실행
  python experiments/relabel/relabel.py --synthetic --n-groups 60 --seeds 1 2 --tag synth
  python experiments/relabel/relabel.py --seeds 1 2 --tag real          # 512그룹, 약 25분
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
for _p in (PROJECT_DIR, PROJECT_DIR / "experiments" / "baseline", PROJECT_DIR / "experiments" / "denoise"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data.loader import load_data, load_sample_submission  # noqa: E402
from baseline import MAGIC_COL, N_MAGIC, STD_THRESHOLD, _make_folds  # noqa: E402
from diagnose import synthetic_group  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SIMPLIFY_OUT = PROJECT_DIR / "experiments" / "simplify" / "outputs"
N_SPLITS, GMM_K, REG = 5, 5, 0.111

# (피처 집합, 모델) — 피처 집합은 라벨을 쓰지 않으므로 그룹·시드마다 한 번만 만든다
PIPELINES = [
    ("lean", "qda"), ("-kpca", "qda"), ("-gmm", "qda"), ("-scaler", "qda"), ("-transductive", "qda"),
    ("raw", "qda"), ("raw", "qda_reg0.5"), ("raw", "lda"), ("raw", "gmm_bayes_k1"), ("raw", "gmm_bayes_k3"),
    ("lean", "lda"),
]


# ---------------------------------------------------------------------------
def build_feature_sets(xtr, xte, seed):
    """라벨을 쓰지 않는 피처 집합 6개를 한 번에 만든다."""
    n = len(xtr)
    allx = np.vstack([xtr, xte])
    d = xtr.shape[1]
    kp = KernelPCA(n_components=d, kernel="cosine", random_state=seed).fit_transform(allx)
    gmm_kp = GaussianMixture(GMM_K, random_state=seed, max_iter=1000, init_params="kmeans").fit(kp)
    lean_unscaled = np.hstack([kp, gmm_kp.predict_proba(kp)])
    gmm_raw = GaussianMixture(GMM_K, random_state=seed, max_iter=1000, init_params="kmeans").fit(allx)
    nokpca = np.hstack([allx, gmm_raw.predict_proba(allx)])
    # transductive 를 끈 버전: 변환을 train 행으로만 적합
    kp_tr = KernelPCA(n_components=d, kernel="cosine", random_state=seed).fit(xtr)
    kpt = kp_tr.transform(allx)
    gmm_tr = GaussianMixture(GMM_K, random_state=seed, max_iter=1000, init_params="kmeans").fit(kpt[:n])
    notrans = np.hstack([kpt, gmm_tr.predict_proba(kpt)])
    sc = StandardScaler().fit(notrans[:n])
    sets = {
        "lean": StandardScaler().fit_transform(lean_unscaled),
        "-scaler": lean_unscaled,
        "-kpca": StandardScaler().fit_transform(nokpca),
        "-gmm": StandardScaler().fit_transform(kp),
        "-transductive": sc.transform(notrans),
        "raw": StandardScaler().fit_transform(allx),
    }
    return {k: (v[:n], v[n:]) for k, v in sets.items()}


def fit_predict(model, xtr, ytr, xs, seed):
    """모델 이름에 따라 학습하고 xs 각각의 양성 점수를 돌려준다. 학습 불가능하면 None."""
    if len(np.unique(ytr)) < 2:
        return None
    if model == "qda":
        clf = QuadraticDiscriminantAnalysis(reg_param=REG).fit(xtr, ytr)
        return [clf.predict_proba(x)[:, 1] for x in xs]
    if model == "qda_reg0.5":
        clf = QuadraticDiscriminantAnalysis(reg_param=0.5).fit(xtr, ytr)
        return [clf.predict_proba(x)[:, 1] for x in xs]
    if model == "lda":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(xtr, ytr)
        return [clf.predict_proba(x)[:, 1] for x in xs]
    if model.startswith("gmm_bayes_k"):
        k = int(model[-1])
        if np.bincount(ytr).min() < k + 2:
            return None
        ms, pri = [], []
        for c in (0, 1):
            xc = xtr[ytr == c]
            ms.append(GaussianMixture(k, covariance_type="full", reg_covar=0.1, random_state=seed, max_iter=500).fit(xc))
            pri.append(np.log(len(xc) / len(xtr)))
        out = []
        for x in xs:
            llr = (ms[1].score_samples(x) + pri[1]) - (ms[0].score_samples(x) + pri[0])
            out.append(1 / (1 + np.exp(-np.clip(llr, -500, 500))))
        return out
    raise ValueError(model)


def inner_oof(x, y, seed):
    p = np.zeros(len(y))
    if np.bincount(y).min() < N_SPLITS:
        return QuadraticDiscriminantAnalysis(reg_param=REG).fit(x, y).predict_proba(x)[:, 1]
    for trn, val in StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(x, y):
        p[val] = QuadraticDiscriminantAnalysis(reg_param=REG).fit(x[trn], y[trn]).predict_proba(x[val])[:, 1]
    return p


def relabel_rounds(x_lean_trn, y_trn, tau, rounds, seed):
    """lean 탐지기로 라벨을 반복 정정. 각 라운드의 라벨과 뒤집은 행 마스크를 돌려준다."""
    labels, flipped_masks = [], []
    y_cur = y_trn.copy()
    for _ in range(rounds):
        p = inner_oof(x_lean_trn, y_cur, seed)
        flags = ((p > tau) & (y_cur == 0)) | ((p < 1 - tau) & (y_cur == 1))
        y_cur = y_cur.copy(); y_cur[flags] = 1 - y_cur[flags]
        labels.append(y_cur); flipped_masks.append(flags)
        if not flags.any():
            # 더 뒤집을 게 없으면 이후 라운드는 같은 라벨
            while len(labels) < rounds:
                labels.append(y_cur); flipped_masks.append(np.zeros_like(flags))
            break
    return labels, flipped_masks


# ---------------------------------------------------------------------------
def paired(a, b):
    d = a - b; sem = d.std(ddof=1) / np.sqrt(len(d))
    return {"mean": float(d.mean()), "ci_low": float(d.mean() - 1.96 * sem), "ci_high": float(d.mean() + 1.96 * sem),
            "wins": int((d > 0).sum()), "n": int(len(d))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-groups", type=int, default=60)
    ap.add_argument("--flip", type=float, default=0.05, help="합성 데이터의 라벨 뒤집힘 비율")
    ap.add_argument("--magic-limit", type=int, default=N_MAGIC)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    tag = args.tag or ("synth" if args.synthetic else "real")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(OUT_DIR / f"{tag}_run.log", "w", encoding="utf-8")

    def log(msg):
        line = f"[relabel] {msg}"; print(line, flush=True); fh.write(line + "\n"); fh.flush()

    t0 = time.time()
    modes = ["noisy"] + [f"flip_r{r}" for r in range(1, args.rounds + 1)] + (["oracle"] if args.synthetic else [])
    log(f"mode={'synthetic' if args.synthetic else 'real'} seeds={args.seeds} tau={args.tau} rounds={args.rounds} "
        f"pipelines={len(PIPELINES)} label_modes={modes}")

    # ---- 데이터 ----
    groups = {}        # gid -> (tr_idx_global, te_idx_global, xtr, xte, y_noisy, y_true or None)
    if args.synthetic:
        rng = np.random.default_rng(777)
        off_tr = off_te = 0
        for g in range(args.n_groups):
            X, y, y_true, flipped = synthetic_group(3000 + g, args.flip, rng)
            tr, te = np.arange(512), np.arange(512, 1024)
            cols = np.where(X[tr].std(axis=0, ddof=1) > STD_THRESHOLD)[0]
            groups[g] = (np.arange(off_tr, off_tr + 512), np.arange(off_te, off_te + 512),
                         X[np.ix_(tr, cols)], X[np.ix_(te, cols)], y[tr], y_true[tr])
            off_tr += 512; off_te += 512
        n_train = n_test = 512 * args.n_groups
        y_all = np.concatenate([groups[g][4] for g in groups]); ytrue_all = np.concatenate([groups[g][5] for g in groups])
        mask = np.ones(n_train, dtype=bool)
    else:
        train_df, test_df = load_data(None)
        train_df = train_df.reset_index(drop=True); test_df = test_df.reset_index(drop=True)
        fcols = [c for c in train_df.columns if c not in ("id", "target", MAGIC_COL)]
        magics = list(range(min(args.magic_limit, N_MAGIC)))
        for m in magics:
            xt = train_df[train_df[MAGIC_COL] == m]; xe = test_df[test_df[MAGIC_COL] == m]
            std = xt[fcols].std(); cols = list(std.index.values[np.where(std > STD_THRESHOLD)])
            groups[m] = (xt.index.to_numpy(), xe.index.to_numpy(), xt[cols].to_numpy(), xe[cols].to_numpy(),
                         xt["target"].to_numpy(), None)
        y_all = train_df["target"].to_numpy(); ytrue_all = None
        mask = train_df[MAGIC_COL].isin(magics).to_numpy()
        n_train, n_test = len(y_all), len(test_df)
        del train_df
    gids = list(groups)
    log(f"groups={len(gids)} train_rows={int(mask.sum())}")

    keys = [(mode, f"{fs}|{mdl}") for mode in modes for fs, mdl in PIPELINES]
    oof_tr = {k: np.zeros((n_train, len(args.seeds))) for k in keys}
    oof_te = {k: np.zeros((n_test, len(args.seeds))) for k in keys} if not args.synthetic else None
    failed = {k: 0 for k in keys}
    clean_stats = {f"flip_r{r}": {"flipped": 0, "correct": 0, "wrong": 0, "remaining_noise": 0, "rows": 0} for r in range(1, args.rounds + 1)}

    for j, seed in enumerate(args.seeds):
        t_seed = time.time()
        for g in tqdm(gids, desc=f"seed {seed}", disable=args.quiet, leave=False):
            tr_idx, te_idx, xtr_raw, xte_raw, y, y_true = groups[g]
            feats = build_feature_sets(xtr_raw, xte_raw, seed)
            folds = _make_folds(y, y, N_SPLITS, seed)
            for trn, val in folds:
                labels_by_mode = {"noisy": y[trn]}
                lab_rounds, flip_masks = relabel_rounds(feats["lean"][0][trn], y[trn], args.tau, args.rounds, seed)
                for r, (lab, fm) in enumerate(zip(lab_rounds, flip_masks), 1):
                    labels_by_mode[f"flip_r{r}"] = lab
                    if y_true is not None:
                        yt = y_true[trn]
                        cs = clean_stats[f"flip_r{r}"]
                        cs["flipped"] += int(fm.sum()); cs["rows"] += len(trn)
                        # 이번 라운드에 뒤집은 행 중 진짜로 틀려 있었던 것 / 멀쩡했던 것 (뒤집기 직전 라벨 기준)
                        prev = labels_by_mode["noisy"] if r == 1 else lab_rounds[r - 2]
                        cs["correct"] += int((fm & (prev != yt)).sum()); cs["wrong"] += int((fm & (prev == yt)).sum())
                        cs["remaining_noise"] += int((lab != yt).sum())
                if y_true is not None:
                    labels_by_mode["oracle"] = y_true[trn]
                for mode in modes:
                    ytr = labels_by_mode[mode]
                    for fs, mdl in PIPELINES:
                        key = (mode, f"{fs}|{mdl}")
                        x_train, x_test = feats[fs]
                        xs = [x_train[val]] + ([] if args.synthetic else [x_test])
                        out = fit_predict(mdl, x_train[trn], ytr, xs, seed)
                        if out is None:
                            failed[key] += 1; oof_tr[key][tr_idx[val], j] = 0.5
                            continue
                        oof_tr[key][tr_idx[val], j] = out[0]
                        if oof_te is not None:
                            oof_te[key][te_idx, j] += out[1] / len(folds)
        log(f"seed {seed} done ({time.time() - t_seed:.0f}s)")

    # ---- 집계 ----
    def group_aucs(pred, target):
        return np.array([roc_auc_score(target[groups[g][0]], pred[groups[g][0]]) for g in gids])

    results = {}
    ctrl_pred = oof_tr[("noisy", "lean|qda")].mean(axis=1)
    ctrl_groups = group_aucs(ctrl_pred, y_all)
    ctrl_groups_true = group_aucs(ctrl_pred, ytrue_all) if args.synthetic else None
    for key in keys:
        pred = oof_tr[key].mean(axis=1)
        pg = group_aucs(pred, y_all)
        r = {"label_mode": key[0], "pipeline": key[1],
             "pooled_auc_noisy": float(roc_auc_score(y_all[mask], pred[mask])),
             "group_mean_noisy": float(pg.mean()), "vs_control_noisy": paired(pg, ctrl_groups), "failed_fits": failed[key]}
        if args.synthetic:
            pgt = group_aucs(pred, ytrue_all)
            r.update({"pooled_auc_true": float(roc_auc_score(ytrue_all, pred)), "group_mean_true": float(pgt.mean()),
                      "vs_control_true": paired(pgt, ctrl_groups_true)})
        results[f"{key[0]}::{key[1]}"] = r

    # 요약표
    log("SUMMARY pooled AUC" + (" (noisy | TRUE labels)" if args.synthetic else " (noisy labels)"))
    header = f"{'pipeline':22s} " + " ".join(f"{m:>17s}" for m in modes)
    log(header)
    for fs, mdl in PIPELINES:
        cells = []
        for mode in modes:
            r = results[f"{mode}::{fs}|{mdl}"]
            cells.append(f"{r['pooled_auc_noisy']:.4f}|{r['pooled_auc_true']:.4f}" if args.synthetic else f"{r['pooled_auc_noisy']:.6f}")
        log(f"{fs + '|' + mdl:22s} " + " ".join(f"{c:>17s}" for c in cells))
    if args.synthetic:
        for r_, cs in clean_stats.items():
            log(f"CLEAN {r_}: flipped={cs['flipped'] / cs['rows']:.4f} of rows, correct={cs['correct']}, wrong={cs['wrong']}, "
                f"remaining noise={cs['remaining_noise'] / cs['rows']:.4f} (started at {args.flip})")

    check = None
    if not args.synthetic and len(gids) == N_MAGIC and list(args.seeds) == [1, 2]:
        ref_path = SIMPLIFY_OUT / "followup_metrics.json"
        if ref_path.exists():
            ref = json.loads(ref_path.read_text(encoding="utf-8"))["combos"]["lean_2seeds"]["pooled_auc"]
            this = results["noisy::lean|qda"]["pooled_auc_noisy"]
            check = {"ref_lean_2seeds": ref, "this": this, "abs_diff": abs(ref - this)}
            log(f"REPRODUCTION CHECK noisy/lean vs simplify lean_2seeds: |diff|={check['abs_diff']:.2e} -> {'OK' if check['abs_diff'] < 1e-12 else 'MISMATCH'}")

    if oof_te is not None:
        best_mode = max(modes, key=lambda m: results[f"{m}::lean|qda"]["pooled_auc_noisy"])
        sub = load_sample_submission(None); sub["target"] = oof_te[(best_mode, "lean|qda")].mean(axis=1)
        sub.to_csv(OUT_DIR / f"{tag}_submission_lean_{best_mode}.csv", index=False)
        for mode in modes:
            np.save(OUT_DIR / f"{tag}_oof_train_lean_{mode}.npy", oof_tr[(mode, "lean|qda")].mean(axis=1))

    pg_df = pd.DataFrame({f"{k[0]}::{k[1]}": group_aucs(oof_tr[k].mean(axis=1), y_all) for k in keys}, index=gids)
    pg_df.index.name = "group"; pg_df.to_csv(OUT_DIR / f"{tag}_per_group_auc.csv")
    metrics = {"config": vars(args) | {"tag": tag, "label_modes": modes, "pipelines": [f"{a}|{b}" for a, b in PIPELINES]},
               "results": results, "clean_stats": clean_stats if args.synthetic else None,
               "reproduction_check": check, "elapsed_sec": round(time.time() - t0, 1)}
    (OUT_DIR / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"done in {metrics['elapsed_sec']}s"); fh.close()
    return metrics


if __name__ == "__main__":
    main()
