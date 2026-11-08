"""The Deep Growth Network and the differentiable investing simulator it learns in.

Policy
------
At each month-end the network reads, for every industry, 13 point-in-time
features plus where its current (drifted) holding sits relative to the
market, and a market-level context token. A transformer lets every industry
attend to every other, so a decision about Oil is made knowing what Utilities
and Business Equipment look like. It outputs:

* a target portfolio: market weights tilted by up to ``max_tilt`` either way,
  ``softmax(log w_mkt + log(max_tilt) * tanh(score))``. The tilt head starts at
  zero, so an untrained network holds exactly the market: it must earn every
  deviation from the index on validation data.
* a trade fraction kappa in (0, 1) from the context token: how far to move
  from current holdings toward the target this month. Holding still is a
  decision the network can learn, which is how "buy and sit" enters the model.

New holdings are ``w + kappa * (target - w)``: long-only and fully invested by
construction, never short, never levered.

Training signal
---------------
The same arithmetic as engine.py, in PyTorch: holdings drift with prices,
every trade pays ``cost`` per unit, and the loss is minus the average monthly
log growth of wealth (the Kelly criterion; ``risk_aversion`` > 1 gives the
more conservative constant-relative-risk-aversion utility). Gradients flow
back through every month of a multi-year episode, so the network learns that
a trade today changes the holdings, costs and risks of the years after it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class AttentionBlock(nn.Module):
    """Pre-norm transformer block over the industry tokens (lean: one fused projection, SDPA)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, n, d = h.shape
        q, k, v = self.qkv(self.norm1(h)).view(b, n, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0)
        h = h + self.drop(self.proj(a.transpose(1, 2).reshape(b, n, d)))
        return h + self.drop(self.ff(self.norm2(h)))


class DeepGrowthNet(nn.Module):
    def __init__(self, n_features: int, n_context: int, d_model: int, n_heads: int, n_layers: int,
                 dropout: float, max_tilt: float, init_trade: float, attention: bool = True, anchored: bool = True) -> None:
        super().__init__()
        self.attention = attention
        self.anchored = anchored
        self.log_max_tilt = math.log(max_tilt)
        self.asset_in = nn.Sequential(nn.Linear(n_features + 1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.context_in = nn.Sequential(nn.Linear(n_context, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        if attention:
            self.encoder = nn.Sequential(*[AttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        else:  # ablation: each industry decided on its own features and the shared context only
            self.encoder = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.tilt_head = nn.Linear(d_model, 1)
        self.trade_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.tilt_head.weight)
        nn.init.zeros_(self.tilt_head.bias)
        nn.init.zeros_(self.trade_head.weight)
        nn.init.constant_(self.trade_head.bias, math.log(init_trade / (1 - init_trade)))

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, w_now: torch.Tensor, w_mkt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, N, F], ctx [B, G], w_now and w_mkt [B, N] -> target [B, N], kappa [B]."""
        hold = torch.log((w_now + 1e-6) / w_mkt).clamp(-3.0, 3.0).unsqueeze(-1)
        a = self.asset_in(torch.cat([x, hold], dim=-1))
        c = self.context_in(ctx).unsqueeze(1)
        if self.attention:
            h = self.encoder(torch.cat([c, a], dim=1))
            h_ctx, h_assets = h[:, 0], h[:, 1:]
        else:
            h_assets = a + self.encoder(a + c)
            h_ctx = c.squeeze(1) + self.encoder(c).squeeze(1)
        h_assets, h_ctx = self.norm(h_assets), self.norm(h_ctx)
        score = self.tilt_head(h_assets).squeeze(-1)
        if self.anchored:
            target = torch.softmax(torch.log(w_mkt) + self.log_max_tilt * torch.tanh(score), dim=-1)
        else:  # ablation: no market anchor, the network allocates from scratch
            target = torch.softmax(score, dim=-1)
        kappa = torch.sigmoid(self.trade_head(h_ctx).squeeze(-1))
        invested = (w_now.sum(-1) > 0.5).to(kappa.dtype)          # from cash, buy the whole target
        kappa = invested * kappa + (1 - invested)
        return target, kappa


def crra(log_growth: torch.Tensor, risk_aversion: float) -> torch.Tensor:
    """Per-month utility of the wealth multiple exp(log_growth): log for risk_aversion = 1 (Kelly)."""
    if abs(risk_aversion - 1.0) < 1e-12:
        return log_growth
    return torch.expm1((1.0 - risk_aversion) * log_growth) / (1.0 - risk_aversion)


def simulate(model: nn.Module, X: torch.Tensor, C: torch.Tensor, R: torch.Tensor, W: torch.Tensor,
             starts: torch.Tensor, length: int, cost: float, w0: torch.Tensor | None = None,
             trade_all: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Play ``length`` months from each start decision; returns (log growth [B, L], weights [B, L, N], turnover [B, L]).

    X [T, N, F], C [T, G], R [T, N] (month-t returns), W [T, N] market weights at decision t.
    Decisions at t = s .. s+L-1 earn R[t+1]. Holdings start at the market (or ``w0``).
    ``trade_all`` forces kappa = 1 (ablation: no learned holding behaviour)."""
    w = W[starts].clone() if w0 is None else w0
    logs, held, traded_all = [], [], []
    for k in range(length):
        t = starts + k
        target, kappa = model(X[t], C[t], w, W[t])
        if trade_all:
            kappa = torch.ones_like(kappa)
        w_new = w + kappa.unsqueeze(-1) * (target - w)
        traded = (w_new - w).abs().sum(-1)
        r = R[t + 1]
        g = (w_new * r).sum(-1)
        logs.append(torch.log1p(-cost * traded) + torch.log1p(g))
        held.append(w_new)
        traded_all.append(traded)
        w = w_new * (1.0 + r) / (1.0 + g).unsqueeze(-1)
    return torch.stack(logs, 1), torch.stack(held, 1), torch.stack(traded_all, 1)
