"""신원이 필요 없는 소유 모델 두 가지.

FrameSummary (①②): 프레임마다 선수 집합 -> Set Attention 2층 -> 시드 4개로 요약(PMA) -> 시간 방향 bi-GRU -> L/R/D
SpaceTime   (③)  : 위와 같은 선수 임베딩을 창 안 '모든 (선수, 시점)' 토큰으로 두고,
                   각 시점의 요약값이 앞뒤 ±1초의 모든 선수 토큰에 cross-attention (신원 없이 시간 연결) -> bi-GRU
"""
import torch, torch.nn as nn, torch.nn.functional as Fn


class SAB(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.att = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)

    def forward(self, x, pad):
        x = self.n1(x + self.att(x, x, x, key_padding_mask=pad, need_weights=False)[0])
        return self.n2(x + self.ff(x))


class PMA(nn.Module):
    def __init__(self, d, k=4, h=4):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, k, d) * 0.1)
        self.att = nn.MultiheadAttention(d, h, batch_first=True)

    def forward(self, x, pad):
        return self.att(self.seed.expand(x.size(0), -1, -1), x, x, key_padding_mask=pad, need_weights=False)[0]


class FrameEncoder(nn.Module):
    def __init__(self, fin, d, use_ball):
        super().__init__()
        self.use_ball = use_ball
        self.inp = nn.Linear(fin, d)
        self.ball = nn.Linear(3, d)
        self.sab1 = SAB(d); self.sab2 = SAB(d)
        self.pma = PMA(d, 4)
        self.out = nn.Linear(4 * d, 2 * d)

    def forward(self, feat, mask, ball):
        # feat (B*T, N, F), mask (B*T, N) True=있음, ball (B*T, 3)
        if not self.use_ball:
            feat = feat.clone(); feat[..., 7:] = 0; ball = torch.zeros_like(ball)
        x = self.inp(feat)
        bt = self.ball(ball).unsqueeze(1)
        x = torch.cat([bt, x], 1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), mask], 1) == False  # 공 토큰은 항상 있음
        x = self.sab2(self.sab1(x, pad), pad)
        s = self.pma(x, pad).flatten(1)
        return self.out(s), x, pad


class FrameSummary(nn.Module):
    def __init__(self, fin=10, d=64, use_ball=True):
        super().__init__()
        self.enc = FrameEncoder(fin, d, use_ball)
        self.gru = nn.GRU(2 * d, d, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(2 * d, 3)

    def forward(self, feat, mask, ball):
        B, T, N, F = feat.shape
        s, _, _ = self.enc(feat.reshape(B * T, N, F), mask.reshape(B * T, N), ball.reshape(B * T, 3))
        h, _ = self.gru(s.reshape(B, T, -1))
        return self.head(h)


class SpaceTime(nn.Module):
    def __init__(self, fin=10, d=64, use_ball=True, radius=5):
        super().__init__()
        self.enc = FrameEncoder(fin, d, use_ball)
        self.radius = radius
        self.q = nn.Linear(2 * d, d)
        self.tpos = nn.Embedding(2 * radius + 1, d)
        self.cross = nn.MultiheadAttention(d, 4, batch_first=True)
        self.n = nn.LayerNorm(d)
        self.gru = nn.GRU(2 * d + d, d, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(2 * d, 3)

    def forward(self, feat, mask, ball):
        B, T, N, F = feat.shape
        s, tok, pad = self.enc(feat.reshape(B * T, N, F), mask.reshape(B * T, N), ball.reshape(B * T, 3))
        N1 = tok.size(1); d = tok.size(2)
        tok = tok.reshape(B, T, N1, d); pad = pad.reshape(B, T, N1)
        q = self.q(s).reshape(B, T, d)
        # 시점 t 의 질의가 시점 t+k (|k|<=radius) 의 모든 선수 토큰을 봄. 상대 시간 임베딩을 토큰에 더함.
        R = self.radius
        tokp = Fn.pad(tok, (0, 0, 0, 0, R, R)); padp = Fn.pad(pad, (0, 0, R, R), value=True)
        keys = torch.stack([tokp[:, k:k + T] + self.tpos.weight[k] for k in range(2 * R + 1)], 2)  # B,T,2R+1,N1,d
        kpad = torch.stack([padp[:, k:k + T] for k in range(2 * R + 1)], 2)
        keys = keys.reshape(B * T, (2 * R + 1) * N1, d); kpad = kpad.reshape(B * T, -1)
        c = self.cross(q.reshape(B * T, 1, d), keys, keys, key_padding_mask=kpad, need_weights=False)[0]
        c = self.n(c.squeeze(1) + q.reshape(B * T, d)).reshape(B, T, d)
        h, _ = self.gru(torch.cat([s.reshape(B, T, -1), c], -1))
        return self.head(h)
