from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import FPS, HALF_SPLIT, KICKOFF, N_FRAMES, PITCH_L, PITCH_W
from .ingestion import Tracking

CAP = 750

ATTACK = {
    "BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
    "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX",
}
DEFEND = {
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION",
}

WEAK_LABEL_COLS = {"tag_h", "tag_a", "poss_prior", "sp_team", "prior_interior"}
NEUTRAL_TAG_COLS = [
    "tag_na", "set_piece", "sp_team_inferred", "sp_active_oob",
    "goal", "n_tags", "since_sp", "since_sp_end", "to_sp",
]
TAG_COLS = list(WEAK_LABEL_COLS) + NEUTRAL_TAG_COLS
CTRL_DIST = 2.5


def _nan(fn, a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=axis)


def interp_short_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    isn = s.isna()
    grp = (isn != isn.shift()).cumsum()
    gap_len = isn.groupby(grp).transform("sum")
    filled = s.interpolate(limit_area="inside")
    return s.where(~(isn & (gap_len <= max_gap)), filled)


def attack_direction(tr: Tracking) -> dict[int, float]:
    d: dict[int, float] = {}
    for h, k in KICKOFF.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mh, ma = np.nanmean(tr.hx[k:k + FPS]), np.nanmean(tr.ax[k:k + FPS])
        d[h] = (1.0 if mh < ma else -1.0) if np.isfinite(mh) and np.isfinite(ma) else np.nan
    if np.isnan(d[1]):
        d[1] = -d[2] if np.isfinite(d[2]) else 1.0
    if np.isnan(d[2]):
        d[2] = -d[1]
    return d


def _team(prefix, x, y, s, bxm, bym, dirvec):
    X, Y = x * PITCH_L, y * PITCH_W
    with np.errstate(invalid="ignore"):
        d = np.hypot(X - bxm[:, None], Y - bym[:, None])
        no_ball = np.isnan(bxm)
        out = {f"d_{prefix}": _nan(np.nanmin, d)}
        for r in (5, 12):
            c = (d < r).sum(1).astype(np.float32)
            c[no_ball] = np.nan
            out[f"n{r}_{prefix}"] = c
    out[f"n_{prefix}"] = (~np.isnan(x)).sum(1).astype(np.float32)
    xn = np.where(dirvec[:, None] == 1, x, 1 - x)
    out[f"cx_{prefix}"] = _nan(np.nanmean, xn)
    out[f"cy_{prefix}"] = _nan(np.nanmean, y)
    out[f"sx_{prefix}"] = _nan(np.nanstd, xn)
    out[f"sy_{prefix}"] = _nan(np.nanstd, y)
    out[f"spd_{prefix}"] = _nan(np.nanmean, s)
    return out


def control_features(d_h: np.ndarray, d_a: np.ndarray) -> dict[str, np.ndarray]:
    n = len(d_h)
    idx = np.arange(n)
    dh = np.where(np.isnan(d_h), 1e9, d_h)
    da = np.where(np.isnan(d_a), 1e9, d_a)
    dmin = np.minimum(dh, da)
    team = np.where(dh <= da, 1.0, -1.0)
    raw_ctrl = np.where(dmin <= CTRL_DIST, team, 0.0)

    s_raw = pd.Series(raw_ctrl)
    r_min = s_raw.rolling(6, min_periods=6).min()
    r_max = s_raw.rolling(6, min_periods=6).max()
    sustained = np.where((r_min == r_max) & (r_min != 0.0), r_min, 0.0)

    has = sustained != 0
    last_i = pd.Series(np.where(has, idx, np.nan))
    prev_i = last_i.ffill().to_numpy()
    prev_val = pd.Series(np.where(has, sustained, np.nan)).ffill().to_numpy()
    f = {
        "ctrl_last": np.nan_to_num(prev_val, nan=0.0),
        "ctrl_age": np.minimum(np.where(np.isnan(prev_i), CAP, idx - prev_i), CAP),
    }
    ch = pd.Series((sustained == 1).astype(float))
    ca = pd.Series((sustained == -1).astype(float))
    for w in (50, 125):
        sh = ch.rolling(w, center=True, min_periods=1).sum().to_numpy()
        sa = ca.rolling(w, center=True, min_periods=1).sum().to_numpy()
        tot = sh + sa
        with np.errstate(invalid="ignore", divide="ignore"):
            f[f"ctrl_share_h_{w}"] = np.where(tot > 0, sh / tot, np.nan)
        f[f"ctrl_frac_{w}"] = tot / w
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


def tagging_features(
    tag: pd.DataFrame,
    d_h: np.ndarray,
    d_a: np.ndarray,
    ball_oob: np.ndarray,
    n: int = N_FRAMES,
) -> dict[str, np.ndarray]:
    z = lambda: np.zeros(n, np.float32)
    f = {
        k: z()
        for k in [
            "tag_h", "tag_a", "tag_na", "set_piece", "sp_team",
            "sp_team_inferred", "sp_active_oob", "goal", "poss_prior", "n_tags",
        ]
    }
    sp_starts, sp_ends = [], []

    for r in tag.itertuples():
        s, e = max(int(r.start), 0), min(int(r.end), n - 1) + 1
        if e <= s:
            continue
        f["n_tags"][s:e] += 1
        f[{"H": "tag_h", "A": "tag_a"}.get(r.team, "tag_na")][s:e] += 1
        sign = {"H": 1.0, "A": -1.0}.get(r.team, 0.0)

        if r.code in ATTACK:
            f["poss_prior"][s:e] += sign
        elif r.code in DEFEND:
            f["poss_prior"][s:e] -= sign
        elif r.code == "SET PIECES":
            f["set_piece"][s:e] = 1.0
            f["sp_team"][s:e] = sign
            if sign == 0.0:
                dh_med = float(_nan(np.nanmedian, d_h[s:e], axis=0))
                da_med = float(_nan(np.nanmedian, d_a[s:e], axis=0))
                dh_v = dh_med if np.isfinite(dh_med) else 1e9
                da_v = da_med if np.isfinite(da_med) else 1e9
                if min(dh_v, da_v) < 4.0:
                    sign = 1.0 if dh_v <= da_v else -1.0
            f["sp_team_inferred"][s:e] = sign
            sp_starts.append(s)
            sp_ends.append(e - 1)
        elif r.code == "GOALS":
            f["goal"][s:e] = 1.0

    f["sp_active_oob"] = (f["set_piece"] * ball_oob).astype(np.float32)

    idx = np.arange(n)
    if sp_starts:
        st = np.sort(np.array(sp_starts))
        en = np.sort(np.array(sp_ends))
        ks = np.searchsorted(st, idx, side="right")
        ke = np.searchsorted(en, idx, side="right")
        since_s = np.where(ks > 0, idx - st[np.maximum(ks - 1, 0)], CAP)
        since_e = np.where(ke > 0, idx - en[np.maximum(ke - 1, 0)], CAP)
        to_s = np.where(ks < len(st), st[np.minimum(ks, len(st) - 1)] - idx, CAP)
    else:
        since_s = since_e = to_s = np.full(n, CAP)

    f["since_sp"] = np.minimum(since_s, CAP).astype(np.float32)
    f["since_sp_end"] = np.minimum(since_e, CAP).astype(np.float32)
    f["to_sp"] = np.minimum(to_s, CAP).astype(np.float32)
    return f


def build_features(
    tr: Tracking,
    tag: pd.DataFrame,
    homo: pd.DataFrame | None = None,
    max_gap: int = 10,
) -> pd.DataFrame:
    n = N_FRAMES
    idx = np.arange(n)
    dirs = attack_direction(tr)
    dirvec = np.where(idx < HALF_SPLIT, dirs[1], dirs[2])
    f: dict[str, np.ndarray] = {}

    bx_raw, by_raw = pd.Series(tr.bx), pd.Series(tr.by)
    seen = bx_raw.notna().to_numpy()
    f["ball_missing"] = (~seen).astype(np.float32)
    last = pd.Series(np.where(seen, idx, np.nan)).ffill().to_numpy()
    f["since_seen"] = np.minimum(np.where(np.isnan(last), CAP, idx - last), CAP).astype(np.float32)

    bx = interp_short_gaps(bx_raw, max_gap).to_numpy()
    by = interp_short_gaps(by_raw, max_gap).to_numpy()
    f["ball_interp"] = (~seen & ~np.isnan(bx)).astype(np.float32)

    ball_oob = ((bx < 0.0) | (bx > 1.0) | (by < 0.0) | (by > 1.0)).astype(np.float32)
    f["ball_oob"] = ball_oob

    bxm, bym = bx * PITCH_L, by * PITCH_W
    bvx = np.diff(bxm, prepend=np.nan) * FPS
    bvy = np.diff(bym, prepend=np.nan) * FPS
    bspeed = np.hypot(bvx, bvy)
    f["bspeed"] = np.clip(bspeed, 0, 40).astype(np.float32)
    f["bvx_n"] = np.clip(np.where(dirvec == 1, bvx, -bvx), -40, 40).astype(np.float32)
    f["bvy"] = np.clip(bvy, -40, 40).astype(np.float32)
    f["baccel"] = np.clip(np.diff(f["bspeed"], prepend=np.nan) * FPS, -150, 150).astype(np.float32)

    bxn = np.where(dirvec == 1, bx, 1 - bx)
    f["bx_n"], f["by"] = bxn, by

    H = _team("h", tr.hx, tr.hy, tr.hs, bxm, bym, dirvec)
    A = _team("a", tr.ax, tr.ay, tr.as_, bxm, bym, dirvec)
    f.update(H), f.update(A)

    cx_h_s = pd.Series(H["cx_h"]).ffill().bfill()
    cx_a_s = pd.Series(A["cx_a"]).ffill().bfill()
    f["vcx_h"] = (
        (cx_h_s.diff().fillna(0.0) * PITCH_L * FPS)
        .rolling(25, center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )
    f["vcx_a"] = (
        (cx_a_s.diff().fillna(0.0) * PITCH_L * FPS)
        .rolling(25, center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )

    f["d_min_any"] = np.fmin(H["d_h"], A["d_a"])
    f["d_diff"] = A["d_a"] - H["d_h"]
    f["d_logratio"] = np.log1p(A["d_a"]) - np.log1p(H["d_h"])
    f["d_diff_ffill"] = pd.Series(f["d_diff"]).ffill(limit=75).to_numpy()
    f["cdiff_x"] = H["cx_h"] - A["cx_a"]
    f["b_minus_cx_h"] = bxn - H["cx_h"]
    f["b_minus_cx_a"] = bxn - A["cx_a"]

    S = np.concatenate([tr.hs, tr.as_], axis=1)
    f["spd_all"] = _nan(np.nanmean, S)
    den = (~np.isnan(S)).sum(1).astype(float)
    den[den == 0] = np.nan
    with np.errstate(invalid="ignore"):
        f["still_frac"] = ((S < 0.5).sum(1) / den).astype(np.float32)

    f.update(control_features(H["d_h"], A["d_a"]))

    for col in ("d_diff", "d_logratio", "bspeed", "spd_all", "ball_missing"):
        s = pd.Series(f[col])
        for w in (25, 50, 100):
            f[f"{col}_r{w}"] = s.rolling(w, center=True, min_periods=1).mean().to_numpy()

    f.update(tagging_features(tag, H["d_h"], A["d_a"], ball_oob, n))

    active_prior = pd.Series((f["poss_prior"] != 0).astype(np.float32))
    f["prior_interior"] = active_prior.rolling(51, center=True, min_periods=51).min().fillna(0.0).to_numpy()

    if homo is not None:
        reproj = homo["reprojection_error"].to_numpy(np.float32)
        scale = np.hypot(homo["h00"].to_numpy(np.float32), homo["h11"].to_numpy(np.float32))
        f["homo_reproj"] = reproj
        f["homo_scale"] = scale
        scale_s = pd.Series(scale).ffill().bfill()
        f["homo_scale_spd"] = (
            scale_s.diff()
            .abs()
            .fillna(0.0)
            .rolling(25, center=True, min_periods=1)
            .mean()
            .to_numpy(np.float32)
        )

    return pd.DataFrame(f, index=idx).astype(np.float32)