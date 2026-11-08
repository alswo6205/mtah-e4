"""새 모델용 데이터셋 (BALL_STYLE=v3 기본: alive 짧은 결측 + dead 긴 결측 / BALL_STYLE=v1: alive 41% 가림, 기존 방식): 5 fps, 프레임마다 '그 순간 잡힌 선수 집합' (신원 없음).

선수 토큰 특징 (10):
  side(왼쪽 팀 -1 / 오른쪽 팀 +1), x, y (정규화), vx, vy, speed (0.2초 차분), vel_known,
  공까지 dx, dy, 거리 (공이 안 보이면 0)
공 정보 (3): x, y, visible
라벨: 0 = 왼쪽 팀 소유, 1 = 오른쪽 팀 소유, 2 = dead
잡음 버전은 25 fps 위치에 결측/오차/중복을 넣은 뒤 5 fps 로 줄인다. 공 가림은 공중볼(z>2m), 혼전(3 m 안 3명+)에서 더 자주.
"""
import os, sys, re
EXP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, EXP)
import numpy as np, pandas as pd
from noise import corrupt_positions, duplicate_detections, gap_mask

PATHCRF = os.path.expanduser(os.environ.get("PATHCRF_DIR", "~/pathcrf"))
ROOT = os.path.join(PATHCRF, "data/sportec/tracking_processed")
OUT = os.path.join(EXP, os.environ.get("DS_DIR", "ds")); os.makedirs(OUT, exist_ok=True)
NMAX = 26; F = 10; STEP = 5


# 공 결측 흉내 설정. 대회 ATD 로 잰 값이 있으면 여기만 바꾸면 됨 (atd_ball_stats.py 참고).
BALL = dict(
    alive_missing=0.32,   # alive 구간에서 공이 안 보이는 비율.
                          # 추정: 대회 채점 프레임 결측 41% = (1-dead비율)*alive_missing + dead비율*dead_missing,
                          #       dead비율 0.20(공개 5분), dead_missing 0.75 -> alive_missing ~= 0.32
    dead_missing=0.75,    # dead 구간을 통째로 가릴 확률 (추정값)
    alive_gap_s=0.6,      # alive 구간 짧은 결측의 평균 길이(초)
    air_hide=0.9,         # 공 높이 2 m 초과 프레임을 가릴 확률
    crowd_start=0.08,     # 공 3 m 안 3명 이상일 때 프레임마다 가림이 시작될 확률
    crowd_gap_s=1.0,      # 혼전 가림 평균 길이(초)
)


def runs_of(mask):
    """True 가 이어지는 구간들의 (시작, 끝+1)."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def ball_mask(bx, by, bz, PX, PY, alive, rng, cfg=BALL, fps=25):
    """True = 공 안 보임.
    dead: dead 구간을 통째로 dead_missing 확률로 가림 (공을 주우러 가는 동안 길게 사라지는 것 흉내)
    alive: 공중볼 + 혼전 가림 + 짧은 무작위 결측. 짧은 결측의 양은 alive 결측이 alive_missing 이 되도록 자동 조절."""
    T = len(bx)
    m = np.zeros(T, bool)
    dead = ~alive
    for s, e in runs_of(dead):
        if rng.random() < cfg["dead_missing"]:
            m[s:e] = True
    crowd = np.nansum(np.sqrt((PX - bx[:, None]) ** 2 + (PY - by[:, None]) ** 2) < 3.0, 1) >= 3
    air = np.nan_to_num(bz) > 2.0
    m |= alive & air & (rng.random(T) < cfg["air_hide"])
    for s in np.flatnonzero(alive & crowd & (rng.random(T) < cfg["crowd_start"])):
        m[s:s + int(rng.exponential(cfg["crowd_gap_s"] * fps)) + 1] = True
    # alive 짧은 결측: 목표 비율이 될 때까지 결측 비율을 이분 탐색 (프레임 단위로 덜어 내지 않음)
    target_alive = cfg["alive_missing"]
    base = m.copy(); lo, hi = 0.0, 0.95; best = base
    seed = int(rng.integers(1 << 31))
    for _ in range(12):
        mid = (lo + hi) / 2
        cand = base | (alive & gap_mask(T, mid, cfg["alive_gap_s"] * fps, np.random.default_rng(seed)))
        if cand[alive].mean() < target_alive: lo = mid
        else: hi = mid
        best = cand
    return best


def ball_mask_v1(bx, by, bz, PX, PY, alive, rng, target=0.41):
    T = len(bx)
    crowd = np.nansum(np.sqrt((PX - bx[:, None]) ** 2 + (PY - by[:, None]) ** 2) < 3.0, 1) >= 3
    air = np.nan_to_num(bz) > 2.0
    m = gap_mask(T, 0.25, 30, rng)                     # 무작위 구간 가림
    m |= air & (rng.random(T) < 0.9)
    starts = crowd & (rng.random(T) < 0.08)            # 혼전에서 시작하는 가림
    for s in np.flatnonzero(starts):
        m[s:s + int(rng.exponential(25)) + 1] = True
    # 목표 비율(alive 기준)에 맞춰 무작위 구간을 더하거나 덜어냄
    for _ in range(20):
        r = m[alive].mean()
        if abs(r - target) < 0.01:
            break
        if r < target:
            m |= gap_mask(T, (target - r) * 0.8, 30, rng)
        else:
            keep = rng.random(T) < (r - target) / r
            m &= ~keep
    return m


def build(mid, level, seed, use_noise):
    rng = np.random.default_rng(seed)
    tr = pd.read_parquet(f"{ROOT}/{mid}.parquet")
    outs = []
    for per, g in tr.groupby("period_id"):
        g = g.sort_values("frame_id")
        teams = {}
        for team in ["home", "away"]:
            pl = sorted({c[:-2] for c in tr.columns if re.match(rf"{team}_\d+_x$", c)})
            X = g[[p + "_x" for p in pl]].to_numpy(float); Y = g[[p + "_y" for p in pl]].to_numpy(float)
            teams[team] = (X, Y)
        mean_x = {k: np.nanmean(v[0]) for k, v in teams.items()}
        left = min(mean_x, key=mean_x.get)
        bx = g.ball_x.to_numpy(float); by = g.ball_y.to_numpy(float); bz = g.ball_z.to_numpy(float)
        alive = (g.ball_state == "alive").to_numpy()
        allX = np.concatenate([teams["home"][0], teams["away"][0]], 1); allY = np.concatenate([teams["home"][1], teams["away"][1]], 1)
        if use_noise:
            mask_fn = ball_mask_v1 if os.environ.get("BALL_STYLE", "v3") == "v1" else ball_mask
            bvis = ~mask_fn(bx, by, bz, allX, allY, alive, rng)
            from noise import ar1
            bx = bx + ar1(len(bx), 0.3, rng); by = by + ar1(len(by), 0.3, rng)
        else:
            bvis = np.ones(len(bx), bool)
        toks = []
        for team, (X, Y) in teams.items():
            if use_noise:
                NX, NY = corrupt_positions(X, Y, level, 25, rng)
                DX, DY = duplicate_detections(X, Y, level, 25, rng)
                NX = np.concatenate([NX, DX], 1); NY = np.concatenate([NY, DY], 1)
            else:
                NX, NY = X, Y
            side = -1.0 if team == left else 1.0
            toks.append((NX, NY, side))
        idx = np.arange(0, len(g), STEP)
        T5 = len(idx)
        feat = np.zeros((T5, NMAX, F), np.float32); mask = np.zeros((T5, NMAX), bool)
        cnt = np.zeros(T5, int)
        b5x, b5y, b5v = bx[idx], by[idx], bvis[idx]
        for NX, NY, side in toks:
            x = NX[idx]; y = NY[idx]
            prev = np.maximum(idx - STEP, 0)
            vx = (NX[idx] - NX[prev]) / 0.2; vy = (NY[idx] - NY[prev]) / 0.2
            vk = ~np.isnan(vx)
            vx = np.nan_to_num(vx); vy = np.nan_to_num(vy)
            for j in range(x.shape[1]):
                ok = ~np.isnan(x[:, j])
                rows = np.flatnonzero(ok)
                rows = rows[cnt[rows] < NMAX]
                c = cnt[rows]
                dx = np.where(b5v[rows], x[rows, j] - b5x[rows], 0); dy = np.where(b5v[rows], y[rows, j] - b5y[rows], 0)
                feat[rows, c] = np.stack([np.full(len(rows), side), x[rows, j] / 52.5 - 1, y[rows, j] / 34 - 1,
                                          vx[rows, j] / 8, vy[rows, j] / 8, np.hypot(vx[rows, j], vy[rows, j]) / 8,
                                          vk[rows, j], dx / 20, dy / 20, np.hypot(dx, dy) / 20], 1)
                mask[rows, c] = True; cnt[rows] += 1
        ball = np.stack([np.where(b5v, b5x / 52.5 - 1, 0), np.where(b5v, b5y / 34 - 1, 0), b5v.astype(float)], 1).astype(np.float32)
        own = g.ball_owning_team_id.to_numpy()[idx]
        home_id = None
        pid = g.player_id.astype("object")
        known = pid.notna() & pid.str.startswith(("home", "away"), na=False)
        ct = pd.crosstab(g.ball_owning_team_id[known], pid[known].str.split("_").str[0])
        home_id = ct["home"].idxmax()
        own_home = own == home_id
        own_left = own_home if left == "home" else ~own_home
        lab = np.where(alive[idx], np.where(own_left, 0, 1), 2).astype(np.int64)
        outs.append(dict(feat=feat, mask=mask, ball=ball, label=lab, frame=g.frame_id.to_numpy()[idx],
                         period=np.full(T5, per), home_left=np.full(T5, left == "home")))
    d = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    name = f"{mid}_{'clean' if not use_noise else level}_{seed}"
    np.savez_compressed(os.path.join(OUT, name + ".npz"), **d)
    alive5 = d["label"] != 2
    print(name, d["feat"].shape, "mean players", d["mask"].sum(1).mean().round(2),
          "ball missing all/alive/dead", round(1 - d["ball"][:, 2].mean(), 3), round(1 - d["ball"][alive5, 2].mean(), 3),
          round(1 - d["ball"][~alive5, 2].mean(), 3), flush=True)


if __name__ == "__main__":
    mid = sys.argv[1]
    build(mid, "none", 0, False)
    jobs = [("mid", 1)] if mid in ("J03WR9", "J03WQQ") else [("low", 1), ("mid", 2), ("high", 3), ("mid", 4)]
    if mid == "J03WR9":
        jobs += [("high", 5)]
    for lv, s in jobs:
        build(mid, lv, s, True)
