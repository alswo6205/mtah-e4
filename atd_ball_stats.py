"""대회 ATD 에서 잡음 흉내에 필요한 통계를 잰다. 정답은 쓰지 않고 ATD 만 읽음. 요약 숫자만 출력.

  python atd_ball_stats.py --repo <대회 저장소> --data-dir <데이터 폴더>

출력
  - 하프 안 공 결측 비율, 공 결측 구간 길이 분포 (짧은 결측 vs 긴 결측)
  - 프레임당 잡힌 선수 수
  - 슬롯이 한 번 보이면 평균 몇 초 이어지는지 (선수 사라짐 흉내의 근거)
  - 0.2초 사이 비정상 속도(12 m/s 초과) 비율 (슬롯 신원 바뀜 흔적)
이 숫자를 알려 주면 build_ds.py 의 BALL 과 noise.py 의 LEVELS 를 맞출 수 있음.
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run import slot_teams, read_raw, HALVES_VIDEO


def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--data-dir")
    ap.add_argument("--atd-meta"); ap.add_argument("--atd-raw"); ap.add_argument("--n-frames", type=int, default=199319)
    ap.add_argument("--halves", default=None, help="테스트용 'a,b;c,d'")
    a = ap.parse_args(); sys.path.insert(0, a.repo)
    global HALVES_VIDEO
    if a.halves: HALVES_VIDEO = {i + 1: tuple(int(v) for v in h.split(",")) for i, h in enumerate(a.halves.split(";"))}
    if a.atd_meta and a.atd_raw: p = {"atd_meta": a.atd_meta, "atd_raw": a.atd_raw}
    else:
        from games import asset_paths
        p = asset_paths(data_dir=a.data_dir)
    team = slot_teams(p["atd_meta"]); X, Y, BX, BY = read_raw(p["atd_raw"], len(team), a.n_frames)
    sl = np.concatenate([np.arange(s, e + 1) for s, e in HALVES_VIDEO.values()])
    miss = np.isnan(BX[sl])
    print(f"하프 안 프레임 {len(sl)}개, 공 결측 {miss.mean():.3f}")
    L = np.concatenate([runs(np.isnan(BX[s:e + 1])) for s, e in HALVES_VIDEO.values()]) / 25
    tot = L.sum()
    print(f"공 결측 구간 {len(L)}개, 길이 중앙값 {np.median(L):.2f}초, 평균 {L.mean():.2f}초")
    for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 30), (30, 1e9)]:
        m = (L >= lo) & (L < hi)
        print(f"  {lo:>4}~{hi if hi < 1e9 else '':<4}초: 구간 {m.sum():5d}개, 결측 프레임의 {L[m].sum()/tot:.3f}")
    cnt = (~np.isnan(X[sl])).sum(1)
    print(f"프레임당 잡힌 선수: 평균 {cnt.mean():.2f}, 22명 초과 프레임 {np.mean(cnt > 22):.3f}")
    seg = np.concatenate([runs(~np.isnan(X[s:e + 1, j])) for s, e in HALVES_VIDEO.values() for j in range(X.shape[1])]) / 25
    seg = seg[seg > 0]
    print(f"슬롯이 한 번 보이면 이어지는 시간: 중앙값 {np.median(seg):.1f}초, 평균 {seg.mean():.1f}초, 구간 {len(seg)}개")
    gaps = np.concatenate([runs(np.isnan(X[s:e + 1, j]) & (np.cumsum(~np.isnan(X[s:e + 1, j])) > 0)
                                & (np.cumsum(~np.isnan(X[s:e + 1, j])[::-1])[::-1] > 0))
                           for s, e in HALVES_VIDEO.values() for j in range(X.shape[1])]) / 25
    gaps = gaps[gaps > 0]
    print(f"슬롯 안의 끊김(다시 나타나기까지): 중앙값 {np.median(gaps):.2f}초, 평균 {gaps.mean():.2f}초, {len(gaps)}개")
    v = np.hypot(X[sl][5:] - X[sl][:-5], Y[sl][5:] - Y[sl][:-5]) / 0.2
    ok = ~np.isnan(v)
    print(f"0.2초 속도 > 12 m/s 비율: {np.mean(v[ok] > 12):.4f} (선수-시점 {ok.sum()}개 중)")


if __name__ == "__main__":
    main()
