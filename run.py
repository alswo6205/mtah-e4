"""학습한 소유 모델을 대회 ATD 에 적용해 제출 CSV 를 만든다 (CPU 로 충분, 몇 분).

필요: models_new.py, model_final.pt, params.json (train.py final 이 만듦), numpy, pandas, torch
대회 데이터 값은 화면에 출력하지 않는다 (진행 상황과 최종 점수만).

사용 예 (대회 저장소 폴더에서):
  python run.py --repo ~/more-than-a-hack-2027 --data-dir ~/mtah_data --out e2_submission.csv --score

순서
  1. 메타 XML 에서 46개 트랙 슬롯의 팀(홈/원정) 읽기
  2. raw ATD 를 한 줄씩 읽어 선수·공 위치(0~1)를 배열로, 미터로 변환 (105.3 x 68.0)
  3. 하프마다: 왼쪽 팀 정하기 -> 5 fps 로 줄이기 -> 선수 토큰 10개 특징 + 공 3개 특징 (학습 때와 똑같이)
  4. 모델 예측 (10초 창, 5초 간격으로 겹쳐서 평균)
  5. Viterbi (전환 비용 4, dead 비용 10, 전환 확률 반영 1 : 검증 경기로 고른 값)
  6. 25 fps 로 펼쳐 H/A/D, 하프 밖은 X -> CSV
  7. (--score) 공개 5분 정답으로 채점
"""
import argparse, os, sys, time
import xml.etree.ElementTree as ET
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from models_new import FrameEncoder

PITCH = (105.3, 68.0)
HALVES_VIDEO = {1: (11185, 90651), 2: (110808, 190197)}   # games.py 의 halves(IdFrame) - offsets
SW, DD, BETA = 4.0, 10.0, 1.0   # params.json 이 있으면 그 값으로 덮어씀
T, STEP, NMAX = 50, 5, 26


class Net(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.enc = FrameEncoder(10, d, True)
        self.gru = nn.GRU(2 * d, d, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(2 * d, 3); self.bhead = nn.Linear(2 * d, 1)

    def forward(self, feat, mask, ball):
        B, T_, N, F = feat.shape
        s, _, _ = self.enc(feat.reshape(B * T_, N, F), mask.reshape(B * T_, N), ball.reshape(B * T_, 3))
        h, _ = self.gru(s.reshape(B, T_, -1))
        return self.head(h), self.bhead(h).squeeze(-1)


def slot_teams(meta_path):
    """대회 baseline.py 의 team_of_each_track 과 같은 방식: 첫 번째 Team = 홈."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")], np.int8)


def read_raw(raw_path, n_slots, n_frames):
    """'<frame_id>:<p1>;<p2>;...:<ball>' 각 그룹 'x,y,...' (NaN 은 없음). frame_id 는 1부터."""
    X = np.full((n_frames, n_slots), np.nan, np.float32); Y = X.copy()
    BX = np.full(n_frames, np.nan, np.float32); BY = BX.copy()
    with open(raw_path) as f:
        for line in f:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n_frames:
                continue
            for j, g in enumerate(parts[1].split(";")[:n_slots]):
                if g and g[0] != "N":
                    v = g.split(","); X[i, j] = float(v[0]); Y[i, j] = float(v[1])
            b = parts[2]
            if b and b[0] != "N":
                v = b.split(","); BX[i] = float(v[0]); BY[i] = float(v[1])
    return X * PITCH[0], Y * PITCH[1], BX * PITCH[0], BY * PITCH[1]


def half_features(X, Y, BX, BY, team):
    """build_ds.py 와 같은 특징. X,Y: (n, slots) 미터. 반환 feat (n5, 26, 10), mask, ball (n5, 3), home_left."""
    mean_x = [np.nanmean(X[:, team == k]) for k in (0, 1)]
    home_left = mean_x[0] < mean_x[1]
    side_of = np.where(team == 0, -1.0 if home_left else 1.0, 1.0 if home_left else -1.0)
    idx = np.arange(0, len(X), STEP); prev = np.maximum(idx - STEP, 0); n5 = len(idx)
    bx, by = BX[idx], BY[idx]; bv = ~np.isnan(bx)
    feat = np.zeros((n5, NMAX, 10), np.float32); mask = np.zeros((n5, NMAX), bool); cnt = np.zeros(n5, int)
    for j in range(X.shape[1]):
        x, y = X[idx, j], Y[idx, j]
        vx = (X[idx, j] - X[prev, j]) / 0.2; vy = (Y[idx, j] - Y[prev, j]) / 0.2
        vk = ~np.isnan(vx); vx = np.nan_to_num(vx); vy = np.nan_to_num(vy)
        rows = np.flatnonzero(~np.isnan(x)); rows = rows[cnt[rows] < NMAX]; c = cnt[rows]
        dx = np.where(bv[rows], x[rows] - bx[rows], 0); dy = np.where(bv[rows], y[rows] - by[rows], 0)
        feat[rows, c] = np.stack([np.full(len(rows), side_of[j]), x[rows] / 52.5 - 1, y[rows] / 34 - 1,
                                  vx[rows] / 8, vy[rows] / 8, np.hypot(vx[rows], vy[rows]) / 8, vk[rows],
                                  dx / 20, dy / 20, np.hypot(dx, dy) / 20], 1)
        mask[rows, c] = True; cnt[rows] += 1
    ball = np.stack([np.where(bv, bx / 52.5 - 1, 0), np.where(bv, by / 34 - 1, 0), bv.astype(float)], 1).astype(np.float32)
    return feat, mask, ball, home_left


@torch.no_grad()
def predict(model, feat, mask, ball):
    n = len(feat); lp = np.zeros((n, 3)); bl = np.zeros(n); w = np.zeros(n)
    starts = sorted(set(list(range(0, max(1, n - T + 1), 25)) + [max(0, n - T)]))
    tri = np.minimum(np.arange(1, T + 1), np.arange(T, 0, -1)).astype(float)
    for k in range(0, len(starts), 16):
        ss = starts[k:k + 16]
        o, b = model(*[torch.from_numpy(np.stack([a[s:s + T] for s in ss])) for a in (feat, mask, ball)])
        o = Fn.log_softmax(o, -1).numpy(); b = b.numpy()
        for j, s in enumerate(ss):
            L = min(T, n - s); lp[s:s + L] += o[j, :L] * tri[:L, None]; bl[s:s + L] += b[j, :L] * tri[:L]; w[s:s + L] += tri[:L]
    w = np.maximum(w, 1e-9); return lp / w[:, None], bl / w


def viterbi(lp, bl):
    n = len(lp); swt = np.maximum(0.0, SW - BETA * bl); back = np.zeros((n, 3), int); score = lp[0].copy()
    for t in range(1, n):
        C = np.array([[0, swt[t], DD], [swt[t], 0, DD], [DD, DD, 0]])
        cand = score[:, None] - C; back[t] = cand.argmax(0); score = cand.max(0) + lp[t]
    out = np.zeros(n, int); s = int(score.argmax())
    for t in range(n - 1, -1, -1): out[t] = s; s = back[t, s]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="대회 저장소 폴더")
    ap.add_argument("--data-dir", default=None, help="tactical/ 폴더가 들어 있는 데이터 폴더")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--model", default=os.path.join(HERE, "model_final.pt"))
    ap.add_argument("--params", default=os.path.join(HERE, "params.json"))
    ap.add_argument("--score", action="store_true", help="공개 5분 정답으로 채점")
    ap.add_argument("--atd-meta"); ap.add_argument("--atd-raw"); ap.add_argument("--n-frames", type=int, default=199319)
    ap.add_argument("--halves", default=None, help="테스트용: 'a,b;c,d' (video frame, 끝 포함)")
    ap.add_argument("--gt", default=None, help="테스트용 정답 CSV (기본: 공개 5분 정답)")
    a = ap.parse_args()
    global SW, DD, BETA
    if os.path.exists(a.params):
        import json
        p = json.load(open(a.params)); SW, DD, BETA = float(p["sw"]), float(p["dd"]), float(p["beta"])
    print(f"모델 {os.path.basename(a.model)}, Viterbi 전환 비용 {SW}, dead 비용 {DD}, 전환 확률 반영 {BETA}", flush=True)
    sys.path.insert(0, a.repo); sys.path.insert(0, os.path.join(a.repo, "evaluation"))
    import possession_metrics as pm
    if a.atd_meta and a.atd_raw:
        meta, raw = a.atd_meta, a.atd_raw
    else:
        from games import asset_paths
        p = asset_paths(data_dir=a.data_dir); meta, raw = p["atd_meta"], p["atd_raw"]
    halves = HALVES_VIDEO if a.halves is None else {
        i + 1: tuple(int(v) for v in h.split(",")) for i, h in enumerate(a.halves.split(";"))}
    t0 = time.time()
    team = slot_teams(meta); print(f"[1] 트랙 슬롯 {len(team)}개 (홈 {int((team == 0).sum())}, 원정 {int((team == 1).sum())})", flush=True)
    X, Y, BX, BY = read_raw(raw, len(team), a.n_frames); print(f"[2] ATD 읽기 완료 {time.time()-t0:.0f}s", flush=True)
    model = Net(); model.load_state_dict(torch.load(a.model, map_location="cpu")); model.eval()
    labels = np.full(a.n_frames, "X", dtype=object)
    for h, (s, e) in halves.items():
        sl = slice(s, e + 1)
        feat, mask, ball, home_left = half_features(X[sl], Y[sl], BX[sl], BY[sl], team)
        lp, bl = predict(model, feat, mask, ball); lab5 = viterbi(lp, bl)
        pos = np.minimum(np.arange(e + 1 - s) // STEP, len(lab5) - 1); lab = lab5[pos]
        left, right = ("H", "A") if home_left else ("A", "H")
        labels[sl] = np.where(lab == 0, left, np.where(lab == 1, right, "D"))
        print(f"[3-6] {h}반 완료: 5fps {len(lab5)}스텝, 홈팀 {'왼쪽' if home_left else '오른쪽'} 시작, {time.time()-t0:.0f}s", flush=True)
    import pandas as pd
    pd.DataFrame({"frame": np.arange(a.n_frames), "label": labels}).to_csv(a.out, index=False)
    print(f"저장: {a.out}", flush=True)
    if a.score:
        gt_path = a.gt or os.path.join(a.repo, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
        r = pm.evaluate(pm.label_codes(pm.read_label_csv(gt_path)), pm.label_codes(pm.read_label_csv(a.out)))
        print(f"[7] 채점 ({os.path.basename(gt_path)}): spellF1 {r['spell_f1_mean']:.4f}  macroF1 {r['macro_f1']:.4f}  "
              f"소유 구간 예측 {r['n_pred_spells']}개 / 정답 {r['n_gt_spells']}개")


if __name__ == "__main__":
    main()
