# Technical Report: Frame-Level Ball Possession (`ciesamantap`)

## 1. Approach
We formulate frame-level possession segmentation (`H`, `A`, `D`, `X`) as a two-stage pipeline combining **purified weak-supervision gradient boosting (XGBoost)** with **Hidden Markov Model (HMM) Viterbi decoding and minimum-duration segment merging**:

1. **Multi-Class Base Model + Purified Weak Supervision:** Because the public ground truth covers only the first 5 minutes (`frames 11,185–18,684`, 17 possessions), training solely on those frames risks overfitting to early-match formations and team directions. We train a 3-class XGBoost classifier (`H`, `A`, `D`) on the 5-minute public ground truth and blend its live-play conditional probability $P(H \mid \text{live})$ with a binary `H vs. A` XGBoost model trained on **purified weak labels** extracted across the full graded match (excluding the public window). Weak training frames are retained only when Smart Tagging phase priors (`BUILD UP`, `PROGRESSION`, `ATTACKING TRANSITION`, etc.) agree with close player-to-ball control ($\le 3.5\text{ m}$ with $\ge 0.5\text{ m}$ margin over the nearest opponent) during interior live-play intervals.
2. **Temporal Smoothing for `spellF1`:** Per-frame argmax predictions flicker during duels and occlusions, fragmenting possessions and penalizing `spellF1`. After applying a 1-frame probability alignment shift (`prob_shift=1`), we decode state sequences per half via **Viterbi decoding** (`stay_poss=0.9980`, `stay_dead=0.994`) followed by **greedy shortest-run merging** (`min_poss=58` frames / `2.32 s`, `min_dead=12` frames / `0.48 s`), absorbing brief spurious flips into the neighboring state with higher mean posterior probability.

On the out-of-sample 5-minute public window, this pipeline achieves **`0.5576` `spellF1`** and **`0.9145` `macroF1`** (`16` possessions, `15.0 s` mean duration), more than doubling the published `baseline_smooth50` benchmark (`0.3256` public / `0.2433` full-match).

## 2. Assets Used and Dropped
* **Tactical Automatic Tracking Data (`BAR-ATH_tactical_atd.xml` / `.txt`):** Used as our primary spatial and kinematic source, indexed via `video_frame = frame_id - 1`. Normalized `0–1` coordinates are scaled to meters (`105.3 m × 68.0 m`). Because the 46 track slots lack persistent player identities and swap frequently, all player features are aggregated strictly at the **team level** (nearest player distance to ball, distance margin, sustained 6-frame control streaks, local player counts within `3 m` / `8 m`, team centroids, and pitch spread).
* **Tactical Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`):** Used for contextual phase priors and restart detection (`round(seconds * 25)`). Because Smart Tagging is automated and interval-based rather than possession-exact—and marks 36 events (including all 26 throw-ins) as `Team = N/A`—we treat tags as soft features and weak-label filters rather than hard labels, inferring missing restart teams from nearest-player proximity.
* **Tactical Homography (`BAR-ATH_tactical_homography.csv`):** Used directly on its 0-indexed timeline (`reprojection_error` and `valid` flags) as an additional camera-stability feature.
* **Broadcast Video, Broadcast ATD, and Broadcast Homography (Dropped):** Dropped because the broadcast feed is unsynchronized with the tactical reference timeline (`164,488` vs. `199,319` frames), contains cuts/replays, and has `34.2%` invalid homography frames.

## 3. Possession Rules Implemented
* **Ungraded Intervals (`X`):** Frames outside the two graded halves (`0–11,184`, `90,652–110,807`, and `190,198–199,318`) are deterministically assigned `X`.
* **Pre-Kickoff Dead Ball (`D`):** Graded frames before kickoff in each half—specifically the 2,934 graded second-half pre-kickoff frames (`110,808–113,741`)—are deterministically assigned `D`.
* **Loose Balls, Deflections & Aerial Duels:** Single-frame touches or deflections without sustained control are prevented from triggering turnovers via sustained multi-frame control features, rolling distance differentials, and Viterbi + `58`-frame (`2.32 s`) minimum possession duration merging.
* **Restarts & Out-of-Bounds Stoppages (`D`):** Out-of-bounds ball coordinates, low ball speed with large player-to-ball distance during `SET PIECES`, and Smart Tagging stoppage intervals drive `P(D)` transitions between possessions.
* **Missing-Ball Occlusions (41% of Graded Frames) & Abstention (`X`):** Short ball gaps ($\le 10$ frames) are linearly interpolated, while longer gaps retain forward-filled ball anchor features (`since_seen`, team density around last known ball position, and team centroid velocities). When the ball is missing for `> 12` frames inside the interior of an established possession (away from state boundaries), the system abstains with `X`—leveraging the scorer's forward-fill rule to avoid splitting true possessions during prolonged occlusions while still guessing explicitly at transitions.

## 4. Limitations
* **Automated Tagging Noise:** Weak supervision relies on automated Smart Tagging phases; extended counter-attacks or rapid turnovers inside a single tagged phase can delay weak-label boundaries.
* **Unseen Ball Restarts:** Throw-ins and quick free kicks occurring while the ball remains occluded (`ball_missing == 1`) rely on player clustering and `SET PIECES` tags, which can occasionally merge two consecutive possessions separated by a brief stoppage.
* **Future Improvements:** Adding a dedicated sequence boundary detector trained on ball-reappearance velocity spikes and pitch-entry vectors after `SET PIECES` tags would sharpen transition latency (`lat_f`) without increasing possession fragmentation.

## 5. Reproduction
1. Place the official match dataset under `data/tactical/`.
2. Install pinned dependencies:
   ```bash
   pip install -r requirements.txt