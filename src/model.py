from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .config import HALF_BLOCKS, KICKOFF, LABELS, graded_indices
from .features import WEAK_LABEL_COLS
from .postprocess import PostParams, count_possessions, decode

CLASS_ID = {c: i for i, c in enumerate(LABELS)}
PUBLIC_WINDOW = (11_185, 18_684)


def make_model(seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        n_estimators=220,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=40,
        reg_lambda=10.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
    )


def labelled(gt: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    m = gt["label"].isin(CLASS_ID.keys())
    frames = gt.loc[m, "frame"].to_numpy()
    y = gt.loc[m, "label"].map(CLASS_ID).to_numpy()
    order = np.argsort(frames)
    return frames[order], y[order]


def fit(X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> XGBClassifier:
    m = make_model(seed)
    m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return m


def blocked_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    frames: np.ndarray,
    seed: int,
    post: PostParams,
    n_blocks: int = 3,
) -> None:
    edges = np.linspace(0, len(frames), n_blocks + 1).astype(int)
    all_i = np.arange(len(frames))
    print(f"{'blok':>5} {'macroF1 argmax':>15} {'macroF1 viterbi':>16} {'#poss pred':>11} {'#poss true':>11}")
    for b in range(n_blocks):
        te = all_i[edges[b]:edges[b + 1]]
        tr = np.setdiff1d(all_i, te)
        m = fit(X.iloc[frames[tr]], y[tr], seed)
        pr = m.predict_proba(X.iloc[frames[te]])
        f_arg = f1_score(y[te], pr.argmax(1), average="macro", labels=[0, 1, 2])
        path = decode(pr, post)
        f_vit = f1_score(y[te], path, average="macro", labels=[0, 1, 2])
        print(f"{b:>5} {f_arg:>15.4f} {f_vit:>16.4f} {count_possessions(path):>11} {count_possessions(y[te]):>11}")


def _purified_weak_indices(
    X: pd.DataFrame, exclude_public: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior = X["poss_prior"].to_numpy()
    interior = (X["prior_interior"].to_numpy() == 1.0) if "prior_interior" in X.columns else np.ones(len(X), bool)
    sp = X["set_piece"].to_numpy()
    oob = X["ball_oob"].to_numpy()
    miss = X["ball_missing"].to_numpy()
    dh = np.nan_to_num(X["d_h"].to_numpy(), nan=99.0)
    da = np.nan_to_num(X["d_a"].to_numpy(), nan=99.0)
    bspeed = np.nan_to_num(X["bspeed"].to_numpy(), nan=0.0)
    ctrl_last = X["ctrl_last"].to_numpy()

    eligible = np.zeros(len(X), bool)
    eligible[graded_indices()] = True
    if exclude_public:
        eligible[PUBLIC_WINDOW[0]:PUBLIC_WINDOW[1] + 1] = False

    live_ok = eligible & interior & (sp == 0) & (oob == 0)

    agree_h = ((miss == 0) & (dh <= 3.5) & (dh + 0.5 < da)) | ((miss == 1) & (ctrl_last == 1.0))
    agree_a = ((miss == 0) & (da <= 3.5) & (da + 0.5 < dh)) | ((miss == 1) & (ctrl_last == -1.0))

    fr_h = np.flatnonzero(live_ok & (prior > 0) & agree_h)
    fr_a = np.flatnonzero(live_ok & (prior < 0) & agree_a)

    pre_ko_2 = np.zeros(len(X), bool)
    pre_ko_2[HALF_BLOCKS[2][0]:KICKOFF[2]] = True
    dead_sp = eligible & ((oob == 1) | ((sp == 1) & (bspeed < 0.2) & (np.minimum(dh, da) > 3.5)))
    fr_d = np.flatnonzero(pre_ko_2 | dead_sp)

    if len(fr_h) < 500 or len(fr_a) < 500 or len(fr_d) < 500:
        raise RuntimeError(
            f"Label lemah terlalu sedikit (H={len(fr_h)}, A={len(fr_a)}, D={len(fr_d)}); periksa fitur."
        )
    return fr_h, fr_a, fr_d


def fit_weak(X: pd.DataFrame, seed: int = 42):
    fr_h, fr_a, _ = _purified_weak_indices(X, exclude_public=True)
    fr = np.concatenate([fr_h, fr_a])
    y = np.concatenate([np.zeros(len(fr_h), int), np.ones(len(fr_a), int)])
    cols = [c for c in X.columns if c not in WEAK_LABEL_COLS]

    m = make_model(seed)
    m.set_params(objective="binary:logistic")
    m.fit(X.iloc[fr][cols], y, sample_weight=compute_sample_weight("balanced", y))
    return m, cols


def fit_dead(
    X: pd.DataFrame,
    pub_frames: np.ndarray,
    pub_y: np.ndarray,
    seed: int = 42,
    pub_boost: float = 5.0,
):
    fr_h, fr_a, fr_d = _purified_weak_indices(X, exclude_public=True)
    w_fr = np.concatenate([fr_h, fr_a, fr_d])
    w_y = np.concatenate([np.zeros(len(fr_h) + len(fr_a), int), np.ones(len(fr_d), int)])

    p_y = (pub_y == 2).astype(int)
    all_fr = np.concatenate([w_fr, pub_frames])
    all_y = np.concatenate([w_y, p_y])
    sw = compute_sample_weight("balanced", all_y)
    sw[:len(w_fr)] *= 0.25
    sw[len(w_fr):] *= pub_boost

    cols = [c for c in X.columns if c not in WEAK_LABEL_COLS]
    m = make_model(seed)
    m.set_params(objective="binary:logistic")
    m.fit(X.iloc[all_fr][cols], all_y, sample_weight=sw)
    return m, cols


def fit_live_gt(
    X: pd.DataFrame,
    pub_frames: np.ndarray,
    pub_y: np.ndarray,
    seed: int = 42,
):
    live = pub_y != 2
    fr = pub_frames[live]
    y = pub_y[live]
    cols = [c for c in X.columns if c not in WEAK_LABEL_COLS]

    m = make_model(seed)
    m.set_params(objective="binary:logistic")
    m.fit(X.iloc[fr][cols], y, sample_weight=compute_sample_weight("balanced", y))
    return m, cols


def combine_decoupled(
    p_d: np.ndarray,
    q_h: np.ndarray,
    sp_active_oob: np.ndarray | None = None,
) -> np.ndarray:
    pd_eff = np.clip(p_d, 0.0, 1.0)
    if sp_active_oob is not None:
        pd_eff = np.where(sp_active_oob > 0, np.maximum(pd_eff, 0.85), pd_eff)
    qh_eff = np.clip(q_h, 0.0, 1.0)
    out = np.empty((len(pd_eff), 3), dtype=np.float64)
    out[:, 0] = (1.0 - pd_eff) * qh_eff
    out[:, 1] = (1.0 - pd_eff) * (1.0 - qh_eff)
    out[:, 2] = pd_eff
    return out


def decoupled_oof_proba(
    X: pd.DataFrame,
    pub_frames: np.ndarray,
    pub_y: np.ndarray,
    q_h_weak_pub: np.ndarray,
    weak_weight: float = 0.85,
    seed: int = 42,
    pub_boost: float = 5.0,
    n_blocks: int = 5,
) -> np.ndarray:
    edges = np.linspace(0, len(pub_frames), n_blocks + 1).astype(int)
    all_i = np.arange(len(pub_frames))
    oof = np.zeros((len(pub_frames), 3), dtype=np.float64)
    for b in range(n_blocks):
        te = all_i[edges[b]:edges[b + 1]]
        tr = np.setdiff1d(all_i, te)
        dm, dcols = fit_dead(X, pub_frames[tr], pub_y[tr], seed=seed, pub_boost=pub_boost)
        p_d_te = dm.predict_proba(X.iloc[pub_frames[te]][dcols])[:, 1]
        lm, lcols = fit_live_gt(X, pub_frames[tr], pub_y[tr], seed=seed)
        q_h_gt_te = lm.predict_proba(X.iloc[pub_frames[te]][lcols])[:, 0]
        q_h_te = weak_weight * q_h_weak_pub[te] + (1.0 - weak_weight) * q_h_gt_te
        sp_oob_te = (
            X.iloc[pub_frames[te]]["sp_active_oob"].to_numpy()
            if "sp_active_oob" in X.columns
            else None
        )
        oof[te] = combine_decoupled(p_d_te, q_h_te, sp_oob_te)
    return oof


def weak_oof_proba(
    X: pd.DataFrame,
    pub_frames: np.ndarray,
    pub_y: np.ndarray,
    q_h_weak_pub: np.ndarray,
    weak_weight: float = 0.85,
    seed: int = 42,
    n_blocks: int = 5,
) -> np.ndarray:
    edges = np.linspace(0, len(pub_frames), n_blocks + 1).astype(int)
    all_i = np.arange(len(pub_frames))
    oof = np.zeros((len(pub_frames), 3), dtype=np.float64)
    for b in range(n_blocks):
        te = all_i[edges[b]:edges[b + 1]]
        tr = np.setdiff1d(all_i, te)
        m = fit(X.iloc[pub_frames[tr]], pub_y[tr], seed=seed)
        pr_te = m.predict_proba(X.iloc[pub_frames[te]])
        oof[te] = blend_weak(pr_te, q_h_weak_pub[te], weak_weight)
    return oof


def fit_hybrid(
    X: pd.DataFrame,
    pub_frames: np.ndarray,
    pub_y: np.ndarray,
    seed: int = 42,
    pub_boost: float = 2.5,
):
    fr_h, fr_a, fr_d = _purified_weak_indices(X, exclude_public=True)
    w_fr = np.concatenate([fr_h, fr_a, fr_d])
    w_y = np.concatenate([
        np.zeros(len(fr_h), int),
        np.ones(len(fr_a), int),
        np.full(len(fr_d), 2, int),
    ])

    all_fr = np.concatenate([w_fr, pub_frames])
    all_y = np.concatenate([w_y, pub_y])
    sw = compute_sample_weight("balanced", all_y)
    sw[len(w_fr):] *= pub_boost

    cols = [c for c in X.columns if c not in WEAK_LABEL_COLS]
    m = make_model(seed)
    m.fit(X.iloc[all_fr][cols], all_y, sample_weight=sw)
    return m, cols


def hybrid_oof_proba(
    X: pd.DataFrame,
    pub_frames: np.ndarray,
    pub_y: np.ndarray,
    seed: int = 42,
    pub_boost: float = 2.5,
    n_blocks: int = 5,
) -> np.ndarray:
    edges = np.linspace(0, len(pub_frames), n_blocks + 1).astype(int)
    all_i = np.arange(len(pub_frames))
    oof = np.zeros((len(pub_frames), 3), dtype=np.float64)
    for b in range(n_blocks):
        te = all_i[edges[b]:edges[b + 1]]
        tr = np.setdiff1d(all_i, te)
        m, cols = fit_hybrid(X, pub_frames[tr], pub_y[tr], seed=seed, pub_boost=pub_boost)
        oof[te] = m.predict_proba(X.iloc[pub_frames[te]][cols])
    return oof


def blend_weak(proba: np.ndarray, q_h: np.ndarray, w: float = 1.0) -> np.ndarray:
    p_d = proba[:, 2]
    main_h = proba[:, 0] / np.maximum(proba[:, 0] + proba[:, 1], 1e-9)
    qh = w * q_h + (1 - w) * main_h
    out = proba.copy()
    out[:, 0] = (1 - p_d) * qh
    out[:, 1] = (1 - p_d) * (1 - qh)
    return out