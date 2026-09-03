#!/usr/bin/env python
"""baseline 해부 & 단순화 실험.

두 가지 질문에 답한다.

1. **해부** — baseline 파이프라인의 구성요소 중 무엇이 성능을 결정하는가?
   512개 magic 그룹 **전체** 에서 요소를 하나씩 제거(ablation)해 잰다.
   (eda/03 은 40개 그룹·파일럿이었고, 그 파일럿의 결론 하나가 전체 규모에서 뒤집힌 전례가 있다.)
2. **단순화** — 결정적인 부품만 남긴 가벼운 파이프라인이 baseline 최종 점수(0.949605)와
   같거나 더 나은가?

설계
----
- 프로브(probe) 모델은 **QDA 하나**. baseline 6개 중 최고 성능이면서 학습이 가장 가볍다.
  baseline 의 level-1 OOF 에서 QDA 단독(config 1개)은 0.949064 로, 6모델×4config×스태킹
  전체(0.949605)와 0.0005 차이다. 즉 "무엇이 결정적인가"는 QDA 로 재도 충분하다.
- 모든 설정이 **같은 512그룹·같은 fold** 를 쓰므로 그룹별 대응 비교(paired)가 가능하다.
- `baseline_probe` 설정은 baseline 의 (kmeans, seed=1) QDA 를 **비트 단위로 재현**해야 한다.
  재현되지 않으면 이후의 모든 비교가 의미 없으므로 시작할 때 검사한다.
- 각 설정의 OOF 예측을 `.npy` 로 저장하고, 여러 설정을 평균하는 '단순 앙상블'은
  재실행 없이 저장된 OOF 로 계산한다.

실행 예시
---------
    python experiments/simplify/simplify.py --magic-limit 4 --tag smoke    # 스모크 테스트
    python experiments/simplify/simplify.py                                # 전체 (약 1시간)
    python experiments/simplify/simplify.py --phase simple                 # 단순화 후보만

결과물은 모두 `experiments/simplify/outputs/` 에 생성된다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
BASELINE_CODE_DIR = PROJECT_DIR / "experiments" / "baseline"
for _path in (str(PROJECT_DIR), str(BASELINE_CODE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data.loader import load_data, load_sample_submission  # noqa: E402

# baseline.py 는 __main__ 가드가 있어 import 해도 실행되지 않는다.
from baseline import MAGIC_COL, N_MAGIC, STD_THRESHOLD, HistModel, _make_folds  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "outputs"
BASELINE_OUTPUT_DIR = BASELINE_CODE_DIR / "outputs"
N_SPLITS = 5


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cfg:
    """그룹 하나에 적용할 파이프라인 설정. 기본값 = baseline 과 동일."""

    name: str
    kpca: bool = True             # KernelPCA(cosine, n_components=d)  (차원 유지)
    gmm_k: int = 5                # 0 이면 GMM 파생 피처 없음
    gmm_cov: str = "full"         # GaussianMixture covariance_type
    proba_rep: int = 5            # GMM 소속 확률 블록 반복 횟수
    score_rep: int = 3            # GMM 로그밀도 블록 반복 횟수
    hist: bool = True             # 히스토그램 밀도 피처
    scale: bool = True            # StandardScaler
    transductive: bool = True     # 변환을 train+test 로 적합
    stratify_gmm: bool = True     # CV 를 GMM 라벨로 stratify (False 면 target)
    reg_param: float = 0.111      # QDA 정규화
    seed: int = 1                 # KernelPCA / GMM / fold 시드
    init_params: str = "kmeans"   # GMM 초기화
    model: str = "qda"            # "qda" | "gmm_bayes" (클래스별 GMM 생성 분류기)
    bayes_k: int = 1              # gmm_bayes 의 클래스당 군집 수
    bayes_reg: float = 0.1        # gmm_bayes 의 reg_covar
    group: str = "ablation"       # 보고용 묶음 이름


def make_configs() -> list[Cfg]:
    base = Cfg(name="baseline_probe")
    ablation = [
        base,
        replace(base, name="-kpca", kpca=False),
        replace(base, name="-gmm_score", score_rep=0),
        replace(base, name="-hist", hist=False),
        replace(base, name="-scaler", scale=False),
        replace(base, name="-gmm_stratify", stratify_gmm=False),
        replace(base, name="-repeat", proba_rep=1, score_rep=1),
        replace(base, name="-gmm", gmm_k=0, proba_rep=0, score_rep=0),
        replace(base, name="-transductive", transductive=False),
    ]
    # 결정적인 부품만 남긴 후보: 피처 선택 + GMM 소속확률(1회) + StandardScaler + QDA
    minimal = Cfg(name="minimal", kpca=False, proba_rep=1, score_rep=0, hist=False, group="simple")
    simple = [
        minimal,
        replace(minimal, name="minimal_diag", gmm_cov="diag"),
        replace(minimal, name="minimal_-transductive", transductive=False),
        replace(minimal, name="minimal_-gmm_stratify", stratify_gmm=False),
        # QDA 정규화 스윕
        replace(minimal, name="minimal_reg0.05", reg_param=0.05),
        replace(minimal, name="minimal_reg0.3", reg_param=0.3),
        replace(minimal, name="minimal_reg0.5", reg_param=0.5),
        # GMM 없이 QDA 만 (가장 단순한 형태)
        replace(minimal, name="raw_qda", gmm_k=0, proba_rep=0, score_rep=0, stratify_gmm=False),
        replace(minimal, name="raw_qda_reg0.5", gmm_k=0, proba_rep=0, score_rep=0,
                stratify_gmm=False, reg_param=0.5),
        # 원리적으로 가장 단순한 모델: 클래스별 GMM 생성 분류기 (피처 엔지니어링 없음)
        replace(minimal, name="gmm_bayes_k1", gmm_k=0, proba_rep=0, score_rep=0,
                stratify_gmm=False, model="gmm_bayes", bayes_k=1),
        replace(minimal, name="gmm_bayes_k2", gmm_k=0, proba_rep=0, score_rep=0,
                stratify_gmm=False, model="gmm_bayes", bayes_k=2),
        replace(minimal, name="gmm_bayes_k3", gmm_k=0, proba_rep=0, score_rep=0,
                stratify_gmm=False, model="gmm_bayes", bayes_k=3),
    ]
    # 단순 앙상블 재료: minimal 을 GMM 시드/군집 수만 바꿔 여러 번 (QDA 는 그대로)
    family = []
    for k in (3, 4, 5, 6, 7):
        for seed in (1, 2, 3, 4):
            if k != 5 and seed > 2:
                continue  # k=5 는 seed 4개, 나머지 k 는 seed 2개
            if k == 5 and seed == 1:
                continue  # == minimal
            family.append(replace(minimal, name=f"minimal_k{k}_s{seed}", gmm_k=k, seed=seed,
                                  group="family"))
    # 1차 결과에서 나온 후속 후보.
    #  - full_ts : baseline 피처 그대로, CV 만 target 으로 stratify (-gmm_stratify 가 단일 설정 최고였음)
    #  - lean    : full_ts 에서 기여가 없던 반복복사·hist·score 를 뺀 것 (kpca 는 pooled 에 기여하므로 유지)
    full_ts = replace(base, stratify_gmm=False, group="followup")
    lean = replace(full_ts, proba_rep=1, score_rep=0, hist=False)
    followup = [
        replace(full_ts, name="full_ts_s2", seed=2),
        replace(full_ts, name="full_ts_s3", seed=3),
        replace(full_ts, name="full_ts_s4", seed=4),
        replace(full_ts, name="full_ts_reg0.3_s1", reg_param=0.3),
        replace(lean, name="lean_s1"),
        replace(lean, name="lean_s2", seed=2),
        replace(lean, name="lean_s3", seed=3),
        replace(lean, name="lean_s4", seed=4),
    ]
    return ablation + simple + family + followup


# ---------------------------------------------------------------------------
# 데이터 캐시
# ---------------------------------------------------------------------------
class GroupCache:
    """그룹별 (유효 피처 행렬, 정답, 원본 행 위치) 를 한 번만 계산해 둔다.

    피처 선택(std > 2)은 baseline 과 정확히 같은 pandas 연산으로 한다 — 비트 단위 재현을 위해.
    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, magics: list[int]):
        feature_columns = [c for c in train_df.columns if c not in ("id", "target", MAGIC_COL)]
        self.magics = magics
        self.n_train, self.n_test = len(train_df), len(test_df)
        self.y_all = train_df["target"].to_numpy()
        self.magic_all = train_df[MAGIC_COL].to_numpy()
        self.groups: dict[int, dict] = {}
        for m in tqdm(magics, desc="cache", leave=False):
            x_train = train_df[train_df[MAGIC_COL] == m]
            x_test = test_df[test_df[MAGIC_COL] == m]
            train_std = x_train[feature_columns].std()
            cols = list(train_std.index.values[np.where(train_std > STD_THRESHOLD)])
            self.groups[m] = {
                "train_idx": x_train.index.to_numpy(),
                "test_idx": x_test.index.to_numpy(),
                "xtr": x_train[cols].to_numpy(),
                "xte": x_test[cols].to_numpy(),
                "y": x_train["target"].to_numpy(),
                "n_cols": len(cols),
            }
        self.train_mask = np.isin(self.magic_all, magics)


# ---------------------------------------------------------------------------
# 그룹 하나 실행
# ---------------------------------------------------------------------------
def build_features(g: dict, cfg: Cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(train 피처, test 피처, gmm 라벨(train), fit 호출 수)."""
    xtr, xte = g["xtr"], g["xte"]
    n_train = len(xtr)
    allx = np.vstack([xtr, xte])
    fit_slice = slice(None) if cfg.transductive else slice(0, n_train)
    n_fits = 0

    if cfg.kpca:
        kp = KernelPCA(n_components=g["n_cols"], kernel="cosine", random_state=cfg.seed)
        # transductive 일 때 baseline 과 동일하게 fit_transform 을 쓴다 (수치가 미세하게 다를 수 있어서)
        allx = kp.fit_transform(allx) if cfg.transductive else kp.fit(allx[fit_slice]).transform(allx)
        n_fits += 1

    blocks = [allx]
    gmm_label = np.zeros(len(allx), dtype=int)
    if cfg.gmm_k > 0:
        gmm = GMM(n_components=cfg.gmm_k, random_state=cfg.seed, max_iter=1000,
                  init_params=cfg.init_params, covariance_type=cfg.gmm_cov).fit(allx[fit_slice])
        n_fits += 1
        gmm_label = gmm.predict(allx)
        proba = gmm.predict_proba(allx)
        score = gmm.score_samples(allx).reshape(-1, 1)
        blocks += [proba] * cfg.proba_rep
        if cfg.hist:
            blocks.append(HistModel().fit(allx[fit_slice]).predict(allx).reshape(-1, 1))
            n_fits += 1
        blocks += [score] * cfg.score_rep
    elif cfg.hist:
        blocks.append(HistModel().fit(allx[fit_slice]).predict(allx).reshape(-1, 1))
        n_fits += 1

    allx = np.hstack(blocks)
    if cfg.scale:
        sc = StandardScaler()
        allx = sc.fit_transform(allx) if cfg.transductive else sc.fit(allx[fit_slice]).transform(allx)
        n_fits += 1
    return allx[:n_train], allx[n_train:], gmm_label[:n_train], n_fits


def _fit_predict(cfg: Cfg, xtr: np.ndarray, ytr: np.ndarray, x_list: list[np.ndarray]) -> list[np.ndarray]:
    """설정에 맞는 분류기를 학습해 x_list 각각의 양성 점수를 돌려준다."""
    if cfg.model == "qda":
        clf = QuadraticDiscriminantAnalysis(reg_param=cfg.reg_param).fit(xtr, ytr)
        return [clf.predict_proba(x)[:, 1] for x in x_list]
    if cfg.model == "gmm_bayes":
        # 클래스별 GMM 을 적합해 로그 우도비를 점수로 쓴다 (생성 모델 분류기).
        # bayes_k=1 이면 정규화 방식만 다른 QDA 와 같다.
        models, priors = [], []
        for c in (0, 1):
            xc = xtr[ytr == c]
            models.append(GMM(n_components=cfg.bayes_k, covariance_type="full", reg_covar=cfg.bayes_reg,
                              random_state=cfg.seed, max_iter=500).fit(xc))
            priors.append(np.log(len(xc) / len(xtr)))
        out = []
        for x in x_list:
            llr = (models[1].score_samples(x) + priors[1]) - (models[0].score_samples(x) + priors[0])
            out.append(1.0 / (1.0 + np.exp(-np.clip(llr, -500, 500))))
        return out
    raise ValueError(cfg.model)


def run_config(cfg: Cfg, cache: GroupCache, show_progress: bool) -> dict:
    oof_train = np.zeros(cache.n_train)
    oof_test = np.zeros(cache.n_test)
    per_group, dims, fits = [], [], []
    started = time.time()
    for m in tqdm(cache.magics, desc=cfg.name, disable=not show_progress, leave=False):
        g = cache.groups[m]
        x_train, x_test, gmm_label, n_fits = build_features(g, cfg)
        y = g["y"]
        strat = gmm_label if cfg.stratify_gmm else y
        folds = _make_folds(y, strat, N_SPLITS, cfg.seed)
        oof = np.zeros(len(y))
        for trn, val in folds:
            p_val, p_test = _fit_predict(cfg, x_train[trn], y[trn], [x_train[val], x_test])
            oof[val] = p_val
            oof_test[g["test_idx"]] += p_test / len(folds)
            n_fits += 1
        oof_train[g["train_idx"]] = oof
        per_group.append(roc_auc_score(y, oof))
        dims.append(x_train.shape[1])
        fits.append(n_fits)
    mask = cache.train_mask
    return {
        "cfg": cfg,
        "oof_train": oof_train,
        "oof_test": oof_test,
        "per_group": np.array(per_group),
        "pooled_auc": float(roc_auc_score(cache.y_all[mask], oof_train[mask])),
        "elapsed_sec": round(time.time() - started, 1),
        "feature_dim_mean": float(np.mean(dims)),
        "fits_per_group": float(np.mean(fits)),
    }


# ---------------------------------------------------------------------------
# 비교 도구
# ---------------------------------------------------------------------------
def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """그룹별 대응 차이 a - b 의 평균과 95% CI."""
    d = a - b
    sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    return {
        "mean": float(d.mean()),
        "ci_low": float(d.mean() - 1.96 * sem),
        "ci_high": float(d.mean() + 1.96 * sem),
        "wins": int((d > 0).sum()),
        "n": int(len(d)),
        "significant": bool(abs(d.mean()) > 1.96 * sem > 0),
    }


def group_aucs(pred: np.ndarray, cache: GroupCache) -> np.ndarray:
    return np.array([roc_auc_score(cache.groups[m]["y"], pred[cache.groups[m]["train_idx"]])
                     for m in cache.magics])


def stack_oof(columns: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """저장된 OOF 열들을 로지스틱 회귀로 5-fold 스태킹한다 (참고용, 단순화의 반대 방향)."""
    out = np.zeros(len(y))
    for trn, val in StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(columns, y):
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(columns[trn], y[trn])
        out[val] = clf.predict_proba(columns[val])[:, 1]
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="baseline ablation & simplification")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--magic-limit", type=int, default=N_MAGIC)
    p.add_argument("--phase", choices=["all", "ablation", "simple", "family", "followup"], default="all")
    p.add_argument("--only", nargs="*", default=None, help="이름을 지정해 일부 설정만 실행")
    p.add_argument("--tag", default="simplify")
    p.add_argument("--reuse-tag", default="simplify",
                   help="조합(combo) 계산 시 이 태그로 저장된 이전 실행의 OOF 를 재사용")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def log(msg: str, fh) -> None:
    line = f"[simp] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def main(argv=None) -> dict:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / f"{args.tag}_run.log", "w", encoding="utf-8")
    t0 = time.time()

    log("loading data ...", fh)
    train_df, test_df = load_data(args.data_dir)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    magics = list(range(min(args.magic_limit, N_MAGIC)))
    cache = GroupCache(train_df, test_df, magics)
    del train_df
    log(f"groups={len(magics)} train_rows={int(cache.train_mask.sum())} "
        f"useful_cols mean={np.mean([g['n_cols'] for g in cache.groups.values()]):.1f}", fh)

    # baseline 참조값
    baseline_metrics = json.loads((BASELINE_OUTPUT_DIR / "baseline_metrics.json").read_text(encoding="utf-8"))
    ref_qda_single = baseline_metrics["level1"]["kmeans_seed1"]["qda"]
    ref_final = baseline_metrics["final_auc"]
    ref_final_pred = np.load(BASELINE_OUTPUT_DIR / "baseline_oof_train_final.npy").mean(axis=1)
    ref_l1 = np.load(BASELINE_OUTPUT_DIR / "baseline_oof_train_level1.npy")
    ref_qda_avg4 = ref_l1[:, baseline_metrics["config"]["models"].index("qda")]
    full_scale = len(magics) == N_MAGIC
    base_final_groups = group_aucs(ref_final_pred, cache)
    base_qda4_groups = group_aucs(ref_qda_avg4, cache)
    log(f"reference: baseline final={ref_final:.6f}  qda single(kmeans,s1)={ref_qda_single:.6f}  "
        f"qda 4-config avg={roc_auc_score(cache.y_all[cache.train_mask], ref_qda_avg4[cache.train_mask]):.6f}", fh)

    configs = make_configs()
    if args.phase != "all":
        keep = {"ablation": {"ablation"}, "simple": {"simple"}, "family": {"family"},
                "followup": {"followup"}}[args.phase]
        anchors = () if args.phase == "followup" else ("baseline_probe", "minimal")
        configs = [c for c in configs if c.group in keep or c.name in anchors]
    if args.only:
        configs = [c for c in configs if c.name in set(args.only)]
    log(f"configs={len(configs)} phase={args.phase}", fh)

    results: dict[str, dict] = {}
    probe_groups = None
    for cfg in configs:
        r = run_config(cfg, cache, show_progress=not args.quiet)
        results[cfg.name] = r
        np.save(out_dir / f"{args.tag}_oof_train_{cfg.name}.npy", r["oof_train"])
        np.save(out_dir / f"{args.tag}_oof_test_{cfg.name}.npy", r["oof_test"])
        if cfg.name == "baseline_probe":
            probe_groups = r["per_group"]
            if full_scale:
                diff = abs(r["pooled_auc"] - ref_qda_single)
                log(f"REPRODUCTION CHECK: probe={r['pooled_auc']:.16f} baseline={ref_qda_single:.16f} "
                    f"|diff|={diff:.2e} -> {'OK (bit-level)' if diff < 1e-12 else 'MISMATCH'}", fh)
        vs_probe = paired(r["per_group"], probe_groups) if probe_groups is not None else None
        vs_final = paired(r["per_group"], base_final_groups)
        r["vs_probe"] = vs_probe
        r["vs_baseline_final"] = vs_final
        r["vs_qda_avg4"] = paired(r["per_group"], base_qda4_groups)
        log(f"{cfg.name:24s} pooled={r['pooled_auc']:.6f} group_mean={r['per_group'].mean():.6f} "
            f"dim={r['feature_dim_mean']:.0f} fits/grp={r['fits_per_group']:.0f} "
            f"vs_probe={'%+.5f' % vs_probe['mean'] if vs_probe else '  ---  '} "
            f"vs_final={vs_final['mean']:+.5f} [{vs_final['ci_low']:+.5f},{vs_final['ci_high']:+.5f}] "
            f"({r['elapsed_sec']}s)", fh)

    # ---- 저장된 OOF 로 단순 앙상블 조합 ----
    combos: dict[str, dict] = {}
    y = cache.y_all
    mask = cache.train_mask

    # 이전 실행(--reuse-tag)의 저장 결과: 조합 계산에서 이번에 안 돌린 설정을 재사용한다
    reuse_metrics = {}
    reuse_path = out_dir / f"{args.reuse_tag}_metrics.json"
    if args.reuse_tag != args.tag and reuse_path.exists():
        reuse_metrics = json.loads(reuse_path.read_text(encoding="utf-8")).get("results", {})

    def get_member(n):
        """(oof_train, oof_test, elapsed, fits) — 이번 결과에 없으면 디스크에서 읽는다."""
        if n in results:
            r = results[n]
            return r["oof_train"], r["oof_test"], r["elapsed_sec"], r["fits_per_group"]
        p_tr = out_dir / f"{args.reuse_tag}_oof_train_{n}.npy"
        p_te = out_dir / f"{args.reuse_tag}_oof_test_{n}.npy"
        if p_tr.exists() and p_te.exists():
            rm = reuse_metrics.get(n, {})
            return np.load(p_tr), np.load(p_te), rm.get("elapsed_sec", 0.0), rm.get("fits_per_group", 0.0)
        return None

    def add_combo(name, members, note):
        found = [(n, get_member(n)) for n in members]
        found = [(n, m) for n, m in found if m is not None]
        if len(found) < 2:
            return
        members = [n for n, _ in found]
        pred_tr = np.mean([m[0] for _, m in found], axis=0)
        pred_te = np.mean([m[1] for _, m in found], axis=0)
        pg = group_aucs(pred_tr, cache)
        combos[name] = {
            "members": members, "note": note,
            "pooled_auc": float(roc_auc_score(y[mask], pred_tr[mask])),
            "group_auc_mean": float(pg.mean()),
            "vs_baseline_final": paired(pg, base_final_groups),
            "elapsed_sec": round(sum(m[2] for _, m in found), 1),
            "fits_per_group": float(sum(m[3] for _, m in found)),
            "_pred_tr": pred_tr, "_pred_te": pred_te,
        }
        v = combos[name]["vs_baseline_final"]
        log(f"COMBO {name:22s} pooled={combos[name]['pooled_auc']:.6f} members={len(members)} "
            f"vs_final={v['mean']:+.5f} [{v['ci_low']:+.5f},{v['ci_high']:+.5f}] wins={v['wins']}/{v['n']}", fh)

    add_combo("minimal_4seeds", ["minimal", "minimal_k5_s2", "minimal_k5_s3", "minimal_k5_s4"],
              "minimal, GMM seed 4개 평균 (k=5)")
    add_combo("minimal_5k", ["minimal_k3_s1", "minimal_k4_s1", "minimal", "minimal_k6_s1", "minimal_k7_s1"],
              "minimal, k=3..7 평균 (seed 1)")
    add_combo("minimal_5k_x2seeds",
              ["minimal_k3_s1", "minimal_k4_s1", "minimal", "minimal_k6_s1", "minimal_k7_s1",
               "minimal_k3_s2", "minimal_k4_s2", "minimal_k5_s2", "minimal_k6_s2", "minimal_k7_s2"],
              "minimal, k=3..7 x seed 2개 = 10개 평균")
    add_combo("minimal_all", [n for n in results if n.startswith("minimal_k") or n == "minimal"],
              "minimal 계열 전부 평균")
    # 후속: baseline 피처 + target stratify 를 시드 4개 평균 / 그중 기여 없는 부품을 뺀 lean 버전
    add_combo("full_ts_4seeds", ["-gmm_stratify", "full_ts_s2", "full_ts_s3", "full_ts_s4"],
              "baseline 피처, target stratify, QDA, GMM 시드 4개 평균")
    add_combo("lean_4seeds", ["lean_s1", "lean_s2", "lean_s3", "lean_s4"],
              "kpca + GMM proba x1 + scaler, target stratify, QDA, 시드 4개 평균")
    add_combo("full_ts_2seeds", ["-gmm_stratify", "full_ts_s2"], "full_ts 시드 2개 평균")
    add_combo("lean_2seeds", ["lean_s1", "lean_s2"], "lean 시드 2개 평균")

    # 참고: 스태킹을 붙이면? (단순화의 반대 방향이지만 상한을 알기 위해)
    fam = [n for n in results if n.startswith("minimal_k") or n == "minimal"]
    stack_info = None
    if len(fam) >= 3 and full_scale:
        cols = np.column_stack([results[n]["oof_train"] for n in fam])
        st = stack_oof(cols, y)
        pg = group_aucs(st, cache)
        stack_info = {"members": fam, "pooled_auc": float(roc_auc_score(y, st)),
                      "vs_baseline_final": paired(pg, base_final_groups)}
        log(f"STACK  logreg over {len(fam)} minimal cols: pooled={stack_info['pooled_auc']:.6f} "
            f"vs_final={stack_info['vs_baseline_final']['mean']:+.5f}", fh)

    # ---- 가장 단순한 '합격' 후보 고르기: baseline final 이상인 것 중 fits 가 가장 적은 것 ----
    candidates = []
    for name, r in results.items():
        candidates.append((name, r["pooled_auc"], r["fits_per_group"], r["oof_test"], r["vs_baseline_final"]))
    for name, c in combos.items():
        candidates.append((name, c["pooled_auc"], c["fits_per_group"], c["_pred_te"], c["vs_baseline_final"]))
    passing = [c for c in candidates if c[1] >= ref_final] if full_scale else []
    best_simple = min(passing, key=lambda c: c[2]) if passing else max(candidates, key=lambda c: c[1])
    log(f"BEST SIMPLE: {best_simple[0]} pooled={best_simple[1]:.6f} fits/grp={best_simple[2]:.0f} "
        f"({'>= baseline final' if best_simple[1] >= ref_final else 'below baseline final'})", fh)

    submission = load_sample_submission(args.data_dir)
    submission["target"] = best_simple[3]
    submission.to_csv(out_dir / f"{args.tag}_submission.csv", index=False)

    # ---- 저장 ----
    per_group_df = pd.DataFrame({n: r["per_group"] for n, r in results.items()}, index=magics)
    per_group_df["baseline_final"] = base_final_groups
    per_group_df["baseline_qda_avg4"] = base_qda4_groups
    per_group_df.index.name = "magic"
    per_group_df.to_csv(out_dir / f"{args.tag}_per_group_auc.csv")

    metrics = {
        "config": {"magic_limit": len(magics), "n_splits": N_SPLITS, "phase": args.phase,
                   "probe_model": "qda"},
        "reference": {"baseline_final_auc": ref_final, "baseline_qda_single_kmeans_s1": ref_qda_single,
                      "baseline_qda_avg4": float(roc_auc_score(y[mask], ref_qda_avg4[mask]))},
        "results": {
            n: {k: v for k, v in r.items() if k not in ("oof_train", "oof_test", "per_group", "cfg")}
            | {"cfg": asdict(r["cfg"]), "group_auc_mean": float(r["per_group"].mean())}
            for n, r in results.items()
        },
        "combos": {n: {k: v for k, v in c.items() if not k.startswith("_")} for n, c in combos.items()},
        "stack": stack_info,
        "best_simple": {"name": best_simple[0], "pooled_auc": best_simple[1],
                        "fits_per_group": best_simple[2], "vs_baseline_final": best_simple[4]},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / f"{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    log(f"done in {metrics['elapsed_sec']}s", fh)
    fh.close()
    return metrics


if __name__ == "__main__":
    main()
