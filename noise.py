"""우리 대회 데이터(자동 추적) 수준의 잡음을 Sportec 위치 데이터에 흉내 낸다.

수준별 설정 (mid = DATA.md 의 '채점 프레임 평균 20.5명' -> 선수-프레임 약 7% 결측):
  miss     : 선수-프레임 결측 비율
  gap_s    : 한 번 사라질 때 평균 길이(초)
  jitter   : 좌표 오차 표준편차(m), 천천히 변하는 AR(1) 잡음
  dup_rate : 프레임당 '한 사람을 두 번 잡음' 시작 확률 (팀별)
"""
import numpy as np
from scipy.signal import lfilter

LEVELS = {
    "none": dict(miss=0.0, gap_s=1.0, jitter=0.0, dup_rate=0.0),
    "low": dict(miss=0.03, gap_s=1.5, jitter=0.2, dup_rate=0.002),
    "mid": dict(miss=0.07, gap_s=2.0, jitter=0.4, dup_rate=0.005),
    "high": dict(miss=0.15, gap_s=3.0, jitter=0.8, dup_rate=0.01),
}


def gap_mask(n, miss, gap_frames, rng):
    """보임/안 보임이 번갈아 나오는 구간 마스크. True = 안 보임."""
    mask = np.zeros(n, bool)
    if miss <= 0:
        return mask
    mean_off = gap_frames
    mean_on = mean_off * (1 - miss) / miss
    i = int(rng.exponential(mean_on))
    while i < n:
        L = max(1, int(rng.exponential(mean_off)))
        mask[i:i + L] = True
        i += L + max(1, int(rng.exponential(mean_on)))
    return mask


def ar1(n, sigma, rng, rho=0.98):
    if sigma <= 0:
        return np.zeros(n)
    e = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n)
    return lfilter([1.0], [1.0, -rho], e)


def corrupt_positions(X, Y, level, fps, rng):
    """X, Y: (T, P) 위치 (NaN = 경기장에 없음). 결측, 좌표 오차를 넣은 복사본을 돌려준다."""
    cfg = LEVELS[level]
    X = X.copy(); Y = Y.copy()
    T, P = X.shape
    for p in range(P):
        on = ~np.isnan(X[:, p])
        if not on.any():
            continue
        X[:, p] += ar1(T, cfg["jitter"], rng)
        Y[:, p] += ar1(T, cfg["jitter"], rng)
        m = gap_mask(T, cfg["miss"], cfg["gap_s"] * fps, rng)
        X[m, p] = np.nan; Y[m, p] = np.nan
    return X, Y


def duplicate_detections(X, Y, level, fps, rng):
    """'한 사람을 두 번 잡음'을 흉내 낸 추가 검출 (T, D) 배열. 원래 선수 근처 0.5 m."""
    cfg = LEVELS[level]
    T, P = X.shape
    DX = np.full((T, 2), np.nan); DY = np.full((T, 2), np.nan)
    if cfg["dup_rate"] <= 0:
        return DX, DY
    t = 0
    slot = 0
    while t < T:
        t += max(1, int(rng.exponential(1 / cfg["dup_rate"])))
        if t >= T:
            break
        act = np.flatnonzero(~np.isnan(X[t]))
        if len(act) == 0:
            continue
        p = rng.choice(act)
        L = max(1, int(rng.exponential(fps)))
        e = min(T, t + L)
        ang = rng.uniform(0, 2 * np.pi)
        DX[t:e, slot] = X[t:e, p] + 0.5 * np.cos(ang)
        DY[t:e, slot] = Y[t:e, p] + 0.5 * np.sin(ang)
        slot = 1 - slot
    return DX, DY
