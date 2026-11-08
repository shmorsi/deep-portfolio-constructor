"""AlphaFormer: a cross-sectional transformer that forecasts next month's relative returns.

Why a cross-sectional model
---------------------------
A market-neutral book only needs to know which stocks will do better *than
each other* next month, not where the market goes. So the target is the
next-month return standardised across stocks, and the loss is the negative
information coefficient (IC): the correlation between predicted and realised
relative returns in each month. Correlation is scale-free, which is exactly
what a ranking problem needs, and it is differentiable, so it trains with
plain backpropagation.

Architecture (per month)
------------------------
1. Temporal encoder, shared by all stocks: a stock's last 12 monthly returns
   are 12 tokens -> transformer (attention over months) -> one vector.
2. Static encoder: volatilities, factor betas, the three classical signals and
   the sector -> MLP -> added to that vector.
3. Cross-sectional encoder: self-attention ACROSS stocks with a padding mask
   and no positional encoding, so the model is permutation-equivariant
   (relabelling the stocks relabels the forecasts) and can learn relative,
   peer-aware patterns (e.g. "cheap versus its sector").
4. Linear head -> one score per stock.
"""

from __future__ import annotations

import torch
from torch import nn

from dpc.config import AlphaModelConfig


def _encoder(ac: AlphaModelConfig, n_layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        ac.d_model, ac.n_heads, 2 * ac.d_model, ac.dropout, batch_first=True, norm_first=True
    )
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


class AlphaFormer(nn.Module):
    def __init__(self, n_seq_channels: int, n_static: int, ac: AlphaModelConfig):
        super().__init__()
        d = ac.d_model
        self.seq_proj = nn.Linear(n_seq_channels, d)
        self.pos = nn.Parameter(torch.zeros(1, ac.seq_len_months, d))
        nn.init.normal_(self.pos, std=0.02)
        self.temporal = _encoder(ac, ac.temporal_layers)
        self.static = nn.Sequential(nn.Linear(n_static, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.cross = _encoder(ac, ac.cross_sectional_layers)
        self.head = nn.Linear(d, 1)

    def forward(self, seq: torch.Tensor, static: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """seq (b,N,L,C), static (b,N,F), mask (b,N) -> scores (b,N) [0 where masked], embeddings (b,N,d)."""
        b, N, L, C = seq.shape
        h = self.seq_proj(seq.reshape(b * N, L, C)) + self.pos[:, :L]
        h = self.temporal(h)
        h = torch.cat([h[:, -1:], h.mean(dim=1, keepdim=True)], dim=1).mean(dim=1).reshape(b, N, -1)
        h = self.norm(h + self.static(static))
        h = self.cross(h, src_key_padding_mask=~mask)
        h = h * mask.unsqueeze(-1)
        return self.head(h).squeeze(-1) * mask, h


def masked_zscore(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardise each row over its masked entries; 0 elsewhere."""
    m = mask.to(x.dtype)
    n = m.sum(-1, keepdim=True).clamp_min(1.0)
    mean = (x * m).sum(-1, keepdim=True) / n
    var = (((x - mean) * m) ** 2).sum(-1, keepdim=True) / (n - 1).clamp_min(1.0)
    return (x - mean) / torch.sqrt(var + eps) * m


def information_coefficient(score: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-row Pearson correlation between score and target over masked entries -> (b,)."""
    m = mask.to(score.dtype)
    n = m.sum(-1, keepdim=True).clamp_min(1.0)
    s = (score - (score * m).sum(-1, keepdim=True) / n) * m
    t = (target - (target * m).sum(-1, keepdim=True) / n) * m
    return (s * t).sum(-1) / torch.sqrt((s**2).sum(-1) * (t**2).sum(-1) + eps)


class AlphaEnsemble(nn.Module):
    """Average of several AlphaFormers' standardised scores (variance reduction)."""

    def __init__(self, members: list[AlphaFormer]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, seq: torch.Tensor, static: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores, embeds = [], []
        for m in self.members:
            s, h = m(seq, static, mask)
            scores.append(masked_zscore(s, mask))
            embeds.append(h)
        return masked_zscore(torch.stack(scores).mean(0), mask), torch.stack(embeds).mean(0)
