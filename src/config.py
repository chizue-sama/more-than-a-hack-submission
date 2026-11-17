from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N_FRAMES = 199_319
FPS = 25
PITCH_L, PITCH_W = 105.3, 68.0

HALF_BLOCKS = {1: (11_185, 90_651), 2: (110_808, 190_197)}
KICKOFF = {1: 11_185, 2: 113_742}
HALF_SPLIT = 100_000

LABELS = ["H", "A", "D"]

PUBLIC_GT = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"


def get_paths() -> dict:
    try:
        from games import asset_paths
        return {k: Path(v) for k, v in asset_paths().items()}
    except Exception:
        d = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data")) / "tactical"
        return {
            "video": d / "BAR-ATH_tactical.mp4",
            "atd_meta": d / "BAR-ATH_tactical_atd.xml",
            "atd_raw": d / "BAR-ATH_tactical_atd.txt",
            "homography": d / "BAR-ATH_tactical_homography.csv",
            "smart_tagging": d / "BAR-ATH_tactical_smart_tagging.xml",
        }


def graded_indices():
    import numpy as np
    return np.concatenate([np.arange(a, b + 1) for a, b in HALF_BLOCKS.values()])