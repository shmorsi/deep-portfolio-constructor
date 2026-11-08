"""Deep covariance model: temporal encoder + cross-sectional attention -> factor covariance.

For an origin date t the network outputs, for every stock i,
    b_i  in R^K   (loadings on K latent factors)
    psi_i > 0     (idiosyncratic variance)
and the forecast of the next-month *daily* covariance is

    Sigma = D (B B' + diag(psi)) D,    D = diag(trailing vol)

Why this parameterisation
-------------------------
* **Always valid**: B B' is positive semi-definite and diag(psi) > 0, so
  Sigma is symmetric positive definite by construction. No eigenvalue
  clipping, no Cholesky failures.
* **Cheap**: with K << N, log-det and inverse use the Woodbury identity in
  O(N K^2) rather than O(N^3).
* **Interpretable**: B are learned risk factors; psi is stock-specific risk.
* **Sensible prior**: at initialisation B ~ 0 and psi = 1, i.e. Sigma is the
  trailing-volatility diagonal. The network must *earn* any correlation it adds.

Architecture
------------
1. Temporal encoder (shared across stocks): each stock's L-day sequence of
   [own standardised return, 6 Fama-French factor returns] -> Transformer
   (or LSTM) -> embedding h_i.
2. Static features (log vols, sector) are added to h_i.
3. Cross-sectional encoder: self-attention *across stocks* (a set
   transformer with padding mask, no positional encoding -> permutation
   equivariant). This is where stocks "see" each other, so co-movement
   structure can be learned.
4. Heads: linear -> b_i, softplus -> psi_i.

Loss: Gaussian negative log-likelihood of the next H daily returns, which is
the proper scoring rule for a covariance forecast.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from dpc.config import RiskModelConfig

LOG_2PI = math.log(2 * math.pi)


class TemporalEncoder(nn.Module):
    """Per-stock sequence encoder. Days are grouped into non-overlapping
    patches (e.g. weeks), each patch is one token (PatchTST, Nie et al. 2023):
    fewer tokens, cheaper attention, and each token carries local context."""

    def __init__(self, n_in: int, rc: RiskModelConfig):
        super().__init__()
        self.arch = rc.architecture
        self.patch = rc.patch_len_days
        self.proj = nn.Linear(n_in * self.patch, rc.d_model)
        if self.arch == "transformer":
            self.pos = nn.Parameter(torch.zeros(1, rc.seq_len_days // self.patch, rc.d_model))
            layer = nn.TransformerEncoderLayer(
                rc.d_model, rc.n_heads, 2 * rc.d_model, rc.dropout, batch_first=True, norm_first=True
            )
            self.enc = nn.TransformerEncoder(layer, rc.temporal_layers, enable_nested_tensor=False)
        else:
            self.enc = nn.LSTM(rc.d_model, rc.d_model, rc.temporal_layers, batch_first=True)
        self.norm = nn.LayerNorm(rc.d_model)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:  # (M, L, n_in) -> (M, d)
        M, L, n_in = seq.shape
        h = self.proj(seq.reshape(M, L // self.patch, self.patch * n_in))
        if self.arch == "transformer":
            h = self.enc(h + self.pos[:, : h.shape[1]])
            out = torch.cat([h[:, -1:], h.mean(dim=1, keepdim=True)], dim=1).mean(dim=1)
        else:
            out, _ = self.enc(h)
            out = out[:, -1]
        return self.norm(out)


class DeepRiskModel(nn.Module):
    def __init__(self, n_factor_inputs: int, n_static: int, rc: RiskModelConfig):
        super().__init__()
        self.rc = rc
        self.temporal = TemporalEncoder(1 + n_factor_inputs, rc)
        self.static = nn.Sequential(nn.Linear(n_static, rc.d_model), nn.GELU(), nn.Linear(rc.d_model, rc.d_model))
        layer = nn.TransformerEncoderLayer(
            rc.d_model, rc.n_heads, 2 * rc.d_model, rc.dropout, batch_first=True, norm_first=True
        )
        self.cross = nn.TransformerEncoder(layer, rc.cross_sectional_layers, enable_nested_tensor=False)
        self.head_b = nn.Linear(rc.d_model, rc.n_factors)
        self.head_psi = nn.Linear(rc.d_model, 1)
        # Start at the trailing-vol diagonal prior: B ~ 0, psi = softplus(c) = 1.
        nn.init.normal_(self.head_b.weight, std=1e-3)
        nn.init.zeros_(self.head_b.bias)
        nn.init.zeros_(self.head_psi.weight)
        nn.init.constant_(self.head_psi.bias, math.log(math.e - 1))

    def embed(self, x: torch.Tensor, f: torch.Tensor, static: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x (b,L,N), f (b,L,nf), static (b,N,ns), mask (b,N) -> stock embeddings (b,N,d)."""
        b, L, N = x.shape
        seq = torch.cat([x.permute(0, 2, 1).unsqueeze(-1), f.unsqueeze(1).expand(b, N, L, f.shape[-1])], dim=-1)
        h = self.temporal(seq.reshape(b * N, L, -1)).reshape(b, N, -1)
        h = h + self.static(static)
        h = self.cross(h, src_key_padding_mask=~mask)
        return h * mask.unsqueeze(-1)

    def forward(self, x, f, static, mask) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns loadings B (b,N,K), idiosyncratic variance psi (b,N) in
        *standardised* units, and the embeddings (b,N,d) for downstream use."""
        h = self.embed(x, f, static, mask)
        B = self.head_b(h) * mask.unsqueeze(-1)
        psi = nn.functional.softplus(self.head_psi(h).squeeze(-1)) + self.rc.min_idio_var
        psi = torch.where(mask, psi, torch.ones_like(psi))
        return B, psi, h


def factor_gaussian_nll(
    B: torch.Tensor, psi: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean per-stock-per-day Gaussian NLL of ``target`` under Sigma = D(BB'+diag(psi))D.

    Masked stocks get B=0, psi=1, sigma=1, r=0 so they contribute exactly 0:
    the covariance is block-diagonal and their block is the identity.
    Uses Woodbury:  (BB'+Psi)^-1 = Psi^-1 - Psi^-1 B (I + B'Psi^-1 B)^-1 B'Psi^-1.
    Shapes: B (b,N,K), psi/sigma/mask (b,N), target (b,H,N).
    """
    m = mask.to(B.dtype)
    sig = torch.where(mask, sigma, torch.ones_like(sigma))
    z = target / sig.unsqueeze(1) * m.unsqueeze(1)  # (b,H,N)
    Bm = B * m.unsqueeze(-1)
    pinv = 1.0 / psi
    K = B.shape[-1]
    cap = torch.eye(K, dtype=B.dtype, device=B.device) + Bm.transpose(1, 2) @ (Bm * pinv.unsqueeze(-1))  # (b,K,K)
    Lc = torch.linalg.cholesky(cap)
    logdet = 2 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1) + torch.log(psi).sum(-1) \
        + 2 * (torch.log(sig) * m).sum(-1)  # (b,)
    zp = z * pinv.unsqueeze(1)  # Psi^-1 z
    q1 = (z * zp).sum(-1)  # (b,H)
    u = zp @ Bm  # (b,H,K)
    sol = torch.cholesky_solve(u.transpose(1, 2), Lc)  # (b,K,H)
    q2 = (u * sol.transpose(1, 2)).sum(-1)  # (b,H)
    quad = q1 - q2  # z'C^-1 z  (= r'Sigma^-1 r; the D scaling sits in logdet)
    H = target.shape[1]
    n_valid = m.sum(-1)  # (b,)
    nll = 0.5 * (H * n_valid * LOG_2PI + H * logdet + quad.sum(-1))  # (b,)
    return nll.sum() / (H * n_valid).sum().clamp_min(1.0)


def to_return_space(B: torch.Tensor, psi: torch.Tensor, sigma: torch.Tensor, horizon_days: int):
    """Convert standardised outputs to return-space factor form for the optimiser:
    Sigma_h = Bh Bh' + diag(sh^2) over ``horizon_days`` (e.g. 21 for monthly)."""
    scale = math.sqrt(horizon_days)
    Bh = B * sigma.unsqueeze(-1) * scale
    sh = torch.sqrt(psi) * sigma * scale
    return Bh, sh
