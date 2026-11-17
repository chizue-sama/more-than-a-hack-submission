from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import HALF_BLOCKS, KICKOFF, LABELS, N_FRAMES


@dataclass
class PostParams:
    stay_poss: float = 0.9980
    stay_dead: float = 0.994
    min_poss: int = 58
    min_dead: int = 12
    temp: float = 1.0
    abstain_conf: float = 0.0
    prob_shift: int = 3


def viterbi(logp: np.ndarray, p: PostParams) -> np.ndarray:
    T, K = logp.shape
    sp, sd = p.stay_poss, p.stay_dead
    P = np.array([
        [sp, (1 - sp) / 2, (1 - sp) / 2],
        [(1 - sp) / 2, sp, (1 - sp) / 2],
        [(1 - sd) / 2, (1 - sd) / 2, sd],
    ])
    lt = np.log(P)
    dp = np.empty((T, K))
    bp = np.zeros((T, K), dtype=np.int8)
    dp[0] = logp[0]
    for t in range(1, T):
        s = dp[t - 1][:, None] + lt
        bp[t] = s.argmax(0)
        dp[t] = s.max(0) + logp[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = dp[-1].argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def runs(a: np.ndarray):
    if len(a) == 0:
        return []
    ch = np.flatnonzero(np.diff(a) != 0) + 1
    st = np.r_[0, ch]
    en = np.r_[ch, len(a)]
    return list(zip(st.tolist(), en.tolist(), a[st].tolist()))


def merge_short(labels: np.ndarray, proba: np.ndarray, p: PostParams) -> np.ndarray:
    labels = labels.copy()
    min_len = {0: p.min_poss, 1: p.min_poss, 2: p.min_dead}
    while True:
        rs = runs(labels)
        if len(rs) <= 1:
            break
        short = [(e - s, i) for i, (s, e, l) in enumerate(rs) if (e - s) < min_len[l]]
        if not short:
            break
        _, i = min(short)
        s, e, _l = rs[i]
        cands = []
        if i > 0:
            cands.append(rs[i - 1][2])
        if i < len(rs) - 1:
            cands.append(rs[i + 1][2])
        labels[s:e] = max(cands, key=lambda c: proba[s:e, c].mean())
    return labels


def decode(proba: np.ndarray, p: PostParams) -> np.ndarray:
    logp = np.log(np.clip(proba, 1e-4, 1.0)) / p.temp
    return merge_short(viterbi(logp, p), proba, p)


def count_possessions(path: np.ndarray) -> int:
    return sum(1 for *_, l in runs(path) if l in (0, 1))


def build_labels(
    proba: np.ndarray,
    p: PostParams,
    occluded_mask: np.ndarray | None = None,
) -> np.ndarray:
    lab = np.full(N_FRAMES, "X", dtype="<U1")
    chars = np.array(LABELS)
    for half, (s, e) in HALF_BLOCKS.items():
        k = KICKOFF[half]
        if k > s:
            lab[s:k] = "D"
        seg_proba = proba[k:e + 1]
        if p.prob_shift > 0 and len(seg_proba) > p.prob_shift:
            seg_proba = np.vstack([np.repeat(seg_proba[:1], p.prob_shift, axis=0), seg_proba[:-p.prob_shift]])
        decoded = decode(seg_proba, p)
        seg_chars = chars[decoded]

        if occluded_mask is not None and p.abstain_conf > 0:
            occ = occluded_mask[k:e + 1]
            max_ha = np.maximum(seg_proba[:, 0], seg_proba[:, 1])
            low_conf = occ & (max_ha < p.abstain_conf) & (decoded != 2)
            is_boundary = np.r_[True, np.diff(decoded) != 0] | np.r_[np.diff(decoded) != 0, True]
            seg_chars[low_conf & ~is_boundary] = "X"

        lab[k:e + 1] = seg_chars
    return lab