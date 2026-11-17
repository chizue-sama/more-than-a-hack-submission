from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HALF_BLOCKS, LABELS, N_FRAMES, ROOT, graded_indices
from .features import build_features
from .ingestion import load_homography, load_public_gt, load_tagging, load_tracking
from .model import (
    blend_weak,
    blocked_cv,
    combine_decoupled,
    decoupled_oof_proba,
    fit,
    fit_dead,
    fit_hybrid,
    fit_live_gt,
    fit_weak,
    hybrid_oof_proba,
    labelled,
    weak_oof_proba,
)
from .postprocess import PostParams, build_labels, count_possessions

REQUIRED_CACHE_COLS = {
    "ball_oob", "ctrl_last", "sp_team_inferred", "since_sp_end",
    "prior_interior", "homo_reproj", "baccel", "vcx_h", "sp_active_oob",
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--weak", action="store_true")
    ap.add_argument("--weak-weight", type=float, default=None)
    ap.add_argument("--decoupled", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--oof-public", action="store_true")
    ap.add_argument("--pub-boost", type=float, default=5.0)
    ap.add_argument("--stay-poss", type=float, default=PostParams.stay_poss)
    ap.add_argument("--stay-dead", type=float, default=PostParams.stay_dead)
    ap.add_argument("--min-poss", type=int, default=PostParams.min_poss)
    ap.add_argument("--min-dead", type=int, default=PostParams.min_dead)
    ap.add_argument("--temp", type=float, default=PostParams.temp)
    ap.add_argument("--abstain-conf", type=float, default=PostParams.abstain_conf)
    ap.add_argument("--prob-shift", type=int, default=PostParams.prob_shift)
    a = ap.parse_args(argv)

    random.seed(a.seed)
    np.random.seed(a.seed)
    weak_weight = a.weak_weight if a.weak_weight is not None else (0.85 if a.decoupled else 1.0)
    post = PostParams(a.stay_poss, a.stay_dead, a.min_poss, a.min_dead, a.temp, a.abstain_conf, a.prob_shift)

    cache = ROOT / "cache" / "features.pkl"
    X = None
    if a.use_cache and cache.exists():
        cached = pd.read_pickle(cache)
        if REQUIRED_CACHE_COLS.issubset(cached.columns):
            print("[1/5] fitur dari cache")
            X = cached
        else:
            print("[1/5] cache lama terdeteksi; membangun ulang fitur ...")

    if X is None:
        print("[1/5] memuat tracking + Smart Tagging + homography dan membangun fitur ...")
        X = build_features(load_tracking(), load_tagging(), load_homography())
        if a.use_cache:
            cache.parent.mkdir(exist_ok=True)
            X.to_pickle(cache)
    print(f"      {X.shape[1]} fitur x {len(X)} frame; bola hilang di {X['ball_missing'].mean():.1%} frame")

    print("[2/5] menyiapkan data latih & melatih XGBoost ...")
    frames, y = labelled(load_public_gt())
    print(f"      {len(frames)} frame GT publik; kelas H/A/D = {np.bincount(y, minlength=3).tolist()}")
    if a.cv:
        blocked_cv(X, y, frames, a.seed, post)

    g = graded_indices()
    proba = np.full((N_FRAMES, 3), np.nan)

    if a.decoupled:
        dm, dcols = fit_dead(X, frames, y, seed=a.seed, pub_boost=a.pub_boost)
        wm, wcols = fit_weak(X, seed=a.seed)
        lm, lcols = fit_live_gt(X, frames, y, seed=a.seed)
        p_d = dm.predict_proba(X.iloc[g][dcols])[:, 1]
        q_h_weak = wm.predict_proba(X.iloc[g][wcols])[:, 0]
        q_h_gt = lm.predict_proba(X.iloc[g][lcols])[:, 0]
        q_h = weak_weight * q_h_weak + (1.0 - weak_weight) * q_h_gt
        sp_oob = X.iloc[g]["sp_active_oob"].to_numpy() if "sp_active_oob" in X.columns else None
        proba[g] = combine_decoupled(p_d, q_h, sp_oob)
        print(f"[3/5] model decoupled 2-tahap ({len(wcols)} fitur, weak_weight={weak_weight}) ...")
        if a.oof_public:
            q_h_weak_pub = wm.predict_proba(X.iloc[frames][wcols])[:, 0]
            proba[frames] = decoupled_oof_proba(
                X, frames, y, q_h_weak_pub,
                weak_weight=weak_weight, seed=a.seed, pub_boost=a.pub_boost,
            )
            print("      jendela publik diisi probabilitas out-of-fold 5-blok (Stage 1 + Stage 2)")
    elif a.hybrid:
        model, hcols = fit_hybrid(X, frames, y, seed=a.seed, pub_boost=a.pub_boost)
        print(f"[3/5] prediksi probabilitas model hybrid ({len(hcols)} fitur, pub_boost={a.pub_boost}) ...")
        proba[g] = model.predict_proba(X.iloc[g][hcols])
        if a.oof_public:
            proba[frames] = hybrid_oof_proba(X, frames, y, seed=a.seed, pub_boost=a.pub_boost)
            print("      jendela publik diisi probabilitas out-of-fold 5-blok")
    else:
        model = fit(X.iloc[frames], y, a.seed)
        print("[3/5] prediksi probabilitas pada frame yang dinilai ...")
        proba[g] = model.predict_proba(X.iloc[g])
        if a.weak:
            wm, wcols = fit_weak(X, a.seed)
            q_h = wm.predict_proba(X.iloc[g][wcols])[:, 0]
            proba[g] = blend_weak(proba[g], q_h, weak_weight)
            print(f"      model lemah terpurifikasi dicampur (bobot {weak_weight}, {len(wcols)} fitur)")
            if a.oof_public:
                q_h_pub = wm.predict_proba(X.iloc[frames][wcols])[:, 0]
                proba[frames] = weak_oof_proba(
                    X, frames, y, q_h_pub, weak_weight=weak_weight, seed=a.seed,
                )
                print("      jendela publik diisi probabilitas out-of-fold 5-blok")

    print("[4/5] Viterbi + penggabungan segmen pendek ...")
    occluded_mask = ((X["ball_missing"] == 1) & (X["since_seen"] > 12)).to_numpy()
    labels = build_labels(proba, post, occluded_mask=occluded_mask)

    print("[5/5] menulis CSV ...")
    out = ROOT / "submissions" / f"{a.team}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels}).to_csv(out, index=False, lineterminator="\n")
    print(f"      {out}  sha256={hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")

    idx = {c: i for i, c in enumerate(LABELS + ["X"])}
    num = np.array([idx[c] for c in labels])
    for h, (s, e) in HALF_BLOCKS.items():
        seg = num[s:e + 1]
        print(
            f"      babak {h}: possession={count_possessions(seg)}  "
            f"H={np.mean(seg == 0):.1%} A={np.mean(seg == 1):.1%} "
            f"D={np.mean(seg == 2):.1%} X={np.mean(seg == 3):.1%}"
        )

    if a.results_dir:
        d = Path(a.results_dir)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(out, d / out.name)
    if a.validate:
        v = ROOT / "evaluation" / "validate_submission.py"
        if v.exists():
            sys.exit(subprocess.call([sys.executable, str(v), str(out)]))
        print("      validate_submission.py tidak ditemukan -- jalankan dari clone repo starter")


if __name__ == "__main__":
    main()