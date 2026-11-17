from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FPS, N_FRAMES, PUBLIC_GT, get_paths


@dataclass
class Tracking:
    bx: np.ndarray
    by: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    hs: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    as_: np.ndarray


def _prefix_candidates(p) -> list[str]:
    c = [str(p.player_id)]
    jn = getattr(p, "jersey_no", None)
    if jn is not None:
        c.append(str(jn))
    return c


def _team_slots(ds, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    from kloppy.domain import Ground

    out = {Ground.HOME: [], Ground.AWAY: []}
    for team in ds.metadata.teams:
        for p in team.players:
            for c in _prefix_candidates(p):
                if f"{c}_x" in df.columns:
                    out[team.ground].append(c)
                    break
    home, away = out[Ground.HOME], out[Ground.AWAY]
    if not home or not away or set(home) & set(away):
        cols = [c for c in df.columns if c.endswith("_x")][:8]
        raise RuntimeError(
            "Pemetaan slot->tim gagal. Jalankan `python scripts/inspect_data.py` dan lihat "
            f"nama kolom (contoh: {cols}) vs ds.metadata.teams[*].players[*].player_id/jersey_no."
        )
    return home, away


def _block(df, prefixes, suffix):
    cols = [f"{c}_{suffix}" for c in prefixes]
    if any(c not in df.columns for c in cols):
        return np.full((len(df), len(prefixes)), np.nan)
    return df[cols].to_numpy(dtype=np.float64)


def load_tracking() -> Tracking:
    from kloppy import metrica

    paths = get_paths()
    ds = metrica.load_tracking_epts(meta_data=str(paths["atd_meta"]), raw_data=str(paths["atd_raw"]))
    df = ds.to_df().copy()
    df["frame"] = df["frame_id"].astype(int) - 1
    df = df.drop_duplicates("frame").set_index("frame").reindex(range(N_FRAMES))
    home, away = _team_slots(ds, df)
    return Tracking(
        bx=df["ball_x"].to_numpy(np.float64), by=df["ball_y"].to_numpy(np.float64),
        hx=_block(df, home, "x"), hy=_block(df, home, "y"), hs=_block(df, home, "s"),
        ax=_block(df, away, "x"), ay=_block(df, away, "y"), as_=_block(df, away, "s"),
    )


def _team_code(v) -> str:
    s = str(v).lower()
    if "barc" in s:
        return "H"
    if "atl" in s or "ath" in s:
        return "A"
    return "N"


def load_tagging() -> pd.DataFrame:
    from kloppy import sportscode

    ev = sportscode.load(str(get_paths()["smart_tagging"])).to_df()
    ev.columns = [str(c).lower() for c in ev.columns]
    start = (pd.to_timedelta(ev["timestamp"]).dt.total_seconds() * FPS).round().astype(int)
    end = (pd.to_timedelta(ev["end_timestamp"]).dt.total_seconds() * FPS).round().astype(int)
    team = ev["team"].map(_team_code) if "team" in ev.columns else pd.Series("N", index=ev.index)
    typ = ev["type"].astype(str).str.lower() if "type" in ev.columns else pd.Series("", index=ev.index)
    return pd.DataFrame({
        "start": start,
        "end": end,
        "code": ev["code"].astype(str).str.upper(),
        "team": team,
        "type": typ,
    }).reset_index(drop=True)


def load_homography() -> pd.DataFrame:
    df = pd.read_csv(get_paths()["homography"])
    df["frame"] = df["frame"].astype(int)
    return df.drop_duplicates("frame").set_index("frame").reindex(range(N_FRAMES))


def load_public_gt() -> pd.DataFrame:
    gt = pd.read_csv(PUBLIC_GT)
    gt["label"] = gt["label"].astype(str).str.strip().str.upper()
    return gt