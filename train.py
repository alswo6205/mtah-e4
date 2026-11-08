"""소유 모델 학습. 세 가지 모드:
  python train.py val   <이름> [epochs=20] [max_minutes=60]   학습 5경기, 검증 J03WQQ 로 에폭·Viterbi 값 선택
  python train.py test  <이름>                                테스트 J03WR9 채점 (마지막에 한 번만)
  python train.py final <이름>                                7경기 전부로 다시 학습 -> model_final.pt, params.json
  (대조군: 이름 끝에 _ctrl 을 붙이면 전환 장치 없이 학습)
aux=1: 전환 예측 머리 + 전환 근처(±2스텝) 손실 3배 + 짧은 구간(5초 미만) 포함 창 2배 샘플링 + 전환 확률로 Viterbi 전환 비용 조절
aux=0: 대조군 (같은 에폭, 같은 데이터, 위 장치 없음)
설정 선택은 검증 J03WQQ(mid) 로만. 테스트는 --test 로 따로 한 번."""
import os, sys, glob, time, json, itertools
EXP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, EXP)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, pandas as pd
from models_new import FrameEncoder
from teamscore import score_frame_labels
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
MODE, name = sys.argv[1], sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 20; MAXMIN = float(sys.argv[4]) if len(sys.argv) > 4 else 60
AUX = not name.endswith("_ctrl"); TEST_ONLY = MODE == "test"
DSS = [os.path.join(EXP, d) if not os.path.isabs(d) else d for d in os.environ.get("DS_DIR", "ds").split(",")]
DS = DSS[0]; T = 50   # DS_DIR 에 쉼표로 여러 폴더를 주면 섞어서 학습, 검증은 폴더별 점수의 평균
TRAIN = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]; VAL = "J03WQQ"; TEST = "J03WR9"
STOP_AFTER = None
if MODE == "final":
    vj = json.load(open(os.path.join(EXP, f"v2_{name}_val.json")))
    EPOCHS = vj["epochs"]; STOP_AFTER = vj["best_epoch"] + 1; MAXMIN = 1e9
    TRAIN = TRAIN + [VAL, TEST]


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


def load(p):
    z = np.load(p); return {k: z[k] for k in z.files}


def change_targets(lab):
    """L/R 사이 전환이 있는 경계 앞뒤 1스텝 = 1. dead 를 사이에 두면 전환으로 보지 않음."""
    b = np.zeros(len(lab), np.float32); near = np.zeros(len(lab), bool)
    ch = np.flatnonzero((lab[1:] != lab[:-1]) & (lab[1:] != 2) & (lab[:-1] != 2)) + 1
    for c in ch:
        b[max(0, c - 1):c + 1] = 1; near[max(0, c - 2):c + 2] = True
    return b, near


def short_steps(lab, maxlen=25):
    s = np.zeros(len(lab), bool); i = 0
    while i < len(lab):
        j = i
        while j < len(lab) and lab[j] == lab[i]: j += 1
        if lab[i] != 2 and j - i < maxlen: s[i:j] = True
        i = j
    return s


def windows(d, stride):
    out = []
    for p in np.unique(d["period"]):
        ix = np.flatnonzero(d["period"] == p)
        out += list(range(ix[0], ix[-1] - T + 2, stride))
    return out


model = Net()


@torch.no_grad()
def predict(d):
    model.eval(); n = len(d["label"]); lp = np.zeros((n, 3)); bl = np.zeros(n); w = np.zeros(n)
    st = windows(d, 25)
    for p in np.unique(d["period"]):
        st.append(np.flatnonzero(d["period"] == p)[-1] - T + 1)
    st = sorted(set(st)); tri = np.minimum(np.arange(1, T + 1), np.arange(T, 0, -1)).astype(float)
    for k in range(0, len(st), 16):
        ss = st[k:k + 16]
        o, b = model(*[torch.from_numpy(np.stack([d[key][s:s + T] for s in ss])) for key in ("feat", "mask", "ball")])
        o = Fn.log_softmax(o, -1).numpy(); b = b.numpy()
        for j, s in enumerate(ss):
            lp[s:s + T] += o[j] * tri[:, None]; bl[s:s + T] += b[j] * tri; w[s:s + T] += tri
    model.train(); w = np.maximum(w, 1e-9)
    return lp / w[:, None], bl / w


def viterbi(lp, per, sw, dd, bl=None, beta=0.0):
    """L<->R 전환 비용 = max(0, sw - beta * 전환 로짓). D 들어가고 나오는 비용 dd."""
    out = np.zeros(len(lp), int)
    for p in np.unique(per):
        ix = np.flatnonzero(per == p); l = lp[ix]; n = len(ix)
        swt = np.full(n, sw, float)
        if bl is not None and beta > 0: swt = np.maximum(0.0, sw - beta * bl[ix])
        back = np.zeros((n, 3), int); score = l[0].copy()
        for t in range(1, n):
            C = np.array([[0, swt[t], dd], [swt[t], 0, dd], [dd, dd, 0]])
            cand = score[:, None] - C; back[t] = cand.argmax(0); score = cand.max(0) + l[t]
        s = int(score.argmax())
        for t in range(n - 1, -1, -1): out[ix[t]] = s; s = back[t, s]
    return out


TRCACHE = {}
def to_frames(d, lab5, mid):
    if mid not in TRCACHE:
        TRCACHE[mid] = pd.read_parquet(os.path.join(os.path.expanduser(os.environ.get("PATHCRF_DIR", "~/pathcrf")), f"data/sportec/tracking_processed/{mid}.parquet"),
                                       columns=["frame_id", "period_id", "ball_state", "ball_owning_team_id", "player_id", "episode_id"])
    tr = TRCACHE[mid]
    pos = np.clip(np.searchsorted(d["frame"], tr.frame_id.to_numpy(), side="right") - 1, 0, len(d["frame"]) - 1)
    hl = d["home_left"][pos]; lab = lab5[pos]
    team = np.where(lab == 0, np.where(hl, "H", "A"), np.where(lab == 1, np.where(hl, "A", "H"), None)).astype(object)
    return tr, team, lab == 2


def evaluate(d, mid, sw, dd, beta, pred=None):
    lp, bl = predict(d) if pred is None else pred
    lab = viterbi(lp, d["period"], sw, dd, bl, beta)
    labt = viterbi(np.where(np.arange(3) == 2, -1e9, lp), d["period"], sw, 1e9, bl, beta)
    tr, tv, _ = to_frames(d, labt, mid); tr, tf, dead = to_frames(d, lab, mid)
    tf = np.where(tf == None, tv, tf)
    r = score_frame_labels(tr, tv, pred_dead=dead)
    r["pred_dead"] = score_frame_labels(tr, tf, pred_dead=dead)["pred_dead"]
    return r


vals = [load(f"{d}/{VAL}_mid_1.npz") for d in DSS] if MODE != "final" else []
val = vals[0] if vals else None

def val_score(sw, dd, be, preds):
    return float(np.mean([evaluate(v, VAL, sw, dd, be, p)["pred_dead"]["spellF1"] for v, p in zip(vals, preds)]))
if not TEST_ONLY:
    sets = [load(f) for d in DSS for m in TRAIN for f in sorted(glob.glob(f"{d}/{m}_*.npz"))]
    aux = [change_targets(s["label"]) for s in sets]; shorts = [short_steps(s["label"]) for s in sets]
    idx, wts = [], []
    for i, s in enumerate(sets):
        cs = np.concatenate([[0], np.cumsum(shorts[i])])
        for st in windows(s, 10):
            idx.append((i, st)); wts.append(2.0 if (AUX and cs[st + T] - cs[st] > 0) else 1.0)
    wts = np.array(wts) / np.sum(wts)
    print(name, "windows", len(idx), "params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best = (-1, None); t0 = time.time()
    for ep in range(EPOCHS):
        sel_all = np.random.choice(len(idx), 6000, p=wts); tot = 0; nb = 0
        for k in range(0, 6000, 32):
            sel = [idx[j] for j in sel_all[k:k + 32]]
            F_ = np.stack([sets[i]["feat"][s:s + T] for i, s in sel]).copy(); M_ = np.stack([sets[i]["mask"][s:s + T] for i, s in sel])
            B_ = np.stack([sets[i]["ball"][s:s + T] for i, s in sel]).copy(); L_ = np.stack([sets[i]["label"][s:s + T] for i, s in sel]).copy()
            Bt = np.stack([aux[i][0][s:s + T] for i, s in sel]); Nr = np.stack([aux[i][1][s:s + T] for i, s in sel])
            fl = np.random.rand(len(sel), 2) < 0.5
            for b in range(len(sel)):
                if fl[b, 0]: F_[b, ..., [0, 1, 3, 7]] *= -1; B_[b, :, 0] *= -1; L_[b] = np.where(L_[b] == 2, 2, 1 - L_[b])
                if fl[b, 1]: F_[b, ..., [2, 4, 8]] *= -1; B_[b, :, 1] *= -1
            o, bo = model(torch.from_numpy(F_), torch.from_numpy(M_), torch.from_numpy(B_))
            ce = Fn.cross_entropy(o.reshape(-1, 3), torch.from_numpy(L_).reshape(-1), reduction="none")
            if AUX:
                w = torch.from_numpy(np.where(Nr, 3.0, 1.0).astype(np.float32)).reshape(-1)
                loss = (ce * w).sum() / w.sum() + 0.5 * Fn.binary_cross_entropy_with_logits(
                    bo, torch.from_numpy(Bt), pos_weight=torch.tensor(5.0))
            else:
                loss = ce.mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if MODE == "final":
            print(f"ep {ep} loss {tot/nb:.4f} {(time.time()-t0)/60:.1f}min", flush=True)
            if ep + 1 >= STOP_AFTER: break
            continue
        pvs = [predict(v) for v in vals]
        vs = val_score(2, 10, 0.0, pvs)   # 고정 설정으로 에폭 선택 (spellF1 기준, 검증 폴더 평균)
        print(f"ep {ep} loss {tot/nb:.4f} val_spellF1 {vs:.4f} {(time.time()-t0)/60:.1f}min", flush=True)
        if vs > best[0]: best = (vs, {k: v.clone() for k, v in model.state_dict().items()}, ep)
        if (time.time() - t0) / 60 > MAXMIN: print("time budget reached", flush=True); break
    if MODE == "final":
        from build_ds import BALL
        torch.save(model.state_dict(), os.path.join(EXP, "model_final.pt"))
        p = {"sw": vj["val_top"][0][1], "dd": vj["val_top"][0][2], "beta": vj["val_top"][0][3],
             "epochs_trained": STOP_AFTER, "train_matches": TRAIN, "ball_noise": BALL}
        json.dump(p, open(os.path.join(EXP, "params.json"), "w"), ensure_ascii=False, indent=1)
        print("saved model_final.pt, params.json", p, flush=True); sys.exit(0)
    torch.save(best[1], os.path.join(EXP, f"model_{name}.pt")); print("best epoch", best[2], flush=True)
    model.load_state_dict(best[1])
    pvs = [predict(v) for v in vals]
    betas = [0, 0.5, 1, 2, 3] if AUX else [0]
    rows = sorted(((val_score(sw, dd, be, pvs), sw, dd, be)
                   for sw, dd, be in itertools.product([1, 2, 4, 6, 8], [5, 10, 20], betas)), reverse=True)
    print("val top", rows[:5], flush=True)
    json.dump({"best_epoch": best[2], "epochs": EPOCHS, "val_top": rows[:10]}, open(os.path.join(EXP, f"v2_{name}_val.json"), "w"))
else:
    model.load_state_dict(torch.load(os.path.join(EXP, f"model_{name}.pt")))
    v = json.load(open(os.path.join(EXP, f"v2_{name}_val.json"))); _, sw, dd, be = v["val_top"][0]
    res = {"sw": sw, "dd": dd, "beta": be, "val_spellF1": v["val_top"][0][0]}
    for d in DSS:
        for tag in ["clean_0", "mid_1", "high_5"]:
            k = f"{os.path.basename(os.path.normpath(d))}/{tag}"
            res[k] = evaluate(load(f"{d}/{TEST}_{tag}.npz"), TEST, sw, dd, be)
            print(k, json.dumps(res[k], ensure_ascii=False), flush=True)
    json.dump(res, open(os.path.join(EXP, f"v2_{name}_test.json"), "w"), ensure_ascii=False, indent=1)
