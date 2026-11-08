"""팀 소유 채점 (공통). 정답 = 제공사 ball_owning_team_id / ball_state, 대회 채점 코드 사용."""
import os, sys
sys.path.insert(0, os.path.join(os.path.expanduser(os.environ.get("MTAH_REPO", "~/more-than-a-hack-2027")), "evaluation"))
import numpy as np, pandas as pd
import possession_metrics as pm


def gt_labels(tr):
    t = tr.set_index("frame_id")
    pid = t.player_id.astype("object")
    known = pid.notna() & pid.str.startswith(("home", "away"), na=False)
    m = pd.crosstab(t.ball_owning_team_id[known], pid[known].str.split("_").str[0])
    home_id = m["home"].idxmax()
    alive = (t.ball_state == "alive").to_numpy()
    team = np.where(t.ball_owning_team_id == home_id, "H", "A").astype(object)
    return t, alive, team


def to_codes(lab, period):
    out = []
    for i, p in enumerate(np.unique(period)):
        if i:
            out.append(np.array(["X"], dtype=object))
        out.append(lab[period == p])
    return pm.label_codes(np.concatenate(out))


def competition_scores(gt_lab, pred_lab, period):
    r = pm.evaluate(to_codes(gt_lab, period), to_codes(pred_lab, period))
    return dict(macroF1=round(r["macro_f1"], 4), spellF1=round(r["spell_f1_mean"], 4),
                n_pred=int(r["n_pred_spells"]), n_gt=int(r["n_gt_spells"]),
                dur_s=round(r["mean_spell_duration_s"], 1))


def score_frame_labels(tr, pred_team, pred_dead=None):
    """pred_team: 프레임별 'H'/'A' (alive 구간), pred_dead: 프레임별 bool (None 이면 정답 dead 사용)."""
    t, alive, team = gt_labels(tr)
    pred_team = np.asarray(pred_team, dtype=object)
    acc = float((pred_team[alive] == team[alive]).mean())
    gt_lab = np.where(alive, team, "D").astype(object)
    per = t.period_id.to_numpy()
    res = dict(alive_team_acc=round(acc, 4))
    res["given_dead"] = competition_scores(gt_lab, np.where(alive, pred_team, "D").astype(object), per)
    if pred_dead is not None:
        pl = np.where(pred_dead, "D", pred_team).astype(object)
        res["pred_dead"] = competition_scores(gt_lab, pl, per)
        tp = (pred_dead & ~alive).sum(); fp = (pred_dead & alive).sum(); fn = (~pred_dead & ~alive).sum()
        res["dead_f1"] = round(float(2 * tp / max(1, 2 * tp + fp + fn)), 4)
    return res


def score_team(tr, micro):
    """PathCRF 엣지 -> 팀 ('보내는 쪽' 규칙)."""
    t = tr.set_index("frame_id")
    f = lambda x: "H" if isinstance(x, str) and x.startswith("home") else ("A" if isinstance(x, str) and x.startswith("away") else None)
    src = micro.edge_src.reindex(t.index).map(f); dst = micro.edge_dst.reindex(t.index).map(f)
    ours = src.where(src.notna(), dst).groupby(t.episode_id).ffill().groupby(t.episode_id).bfill()
    ours = ours.fillna("H")
    return score_frame_labels(tr, ours.to_numpy())
