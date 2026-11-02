"""Command-conditioned visual residual on predicted tokens (spec 6.6, eq. 12).

    z_hat = LN0[ z0 + B_o tanh( A_o z0 + M(cond) ) ]

cond is [h_t, x_{t+1}, e(U_t), e_r, e_v, pos(j)]. In baseline B1 no belief exists,
so belief_dim is 0 and h_t, x_{t+1} are absent. Setting belief_dim=320 later adds
them with no other change. This is the deliberate B1 deviation from eq. 12.

The residual sees commands directly, which covers effects that displacement alone
does not determine, such as contact with little visible arm motion.

B_o starts at zero. LN0 is then applied to an already normalised token, so the
output equals the base prediction up to the epsilon of LN0, not bit for bit.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_norm_0(z: torch.Tensor) -> torch.Tensor:
    """Parameter-free LN0 on the channel axis, eps 1e-5 as in upstream F.layer_norm."""
    return F.layer_norm(z, (z.size(-1),))


class _RegimeResidual(nn.Module):
    def __init__(self, token_dim: int, bottleneck: int, cond_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.a_o = nn.Linear(token_dim, bottleneck)
        layers: list[nn.Module] = []
        width = cond_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.SiLU()]
            width = size
        self.modulation = nn.Sequential(*layers)
        self.modulation_out = nn.Linear(width, bottleneck)
        self.b_o = nn.Linear(bottleneck, token_dim)
        nn.init.zeros_(self.b_o.weight)
        nn.init.zeros_(self.b_o.bias)

    def forward(self, z0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.b_o(torch.tanh(self.a_o(z0) + self.modulation_out(self.modulation(cond))))


class TokenResidual(nn.Module):
    def __init__(
        self,
        token_dim: int = 1664,
        bottleneck: int = 32,
        hidden: tuple[int, ...] = (256, 256),
        command_dim: int = 64,
        regime_dim: int = 16,
        view_dim: int = 8,
        num_view_ids: int = 7,
        belief_dim: int = 0,
        regimes: int = 1,
    ):
        super().__init__()
        self.belief_dim = belief_dim
        self.regime_embedding = nn.Embedding(regimes, regime_dim)
        self.view_embedding = nn.Embedding(num_view_ids, view_dim)
        cond_dim = belief_dim + command_dim + regime_dim + view_dim + 2
        self.regimes = nn.ModuleList(_RegimeResidual(token_dim, bottleneck, cond_dim, hidden) for _ in range(regimes))

    def forward(
        self,
        z0: torch.Tensor,
        command: torch.Tensor,
        view_ids: torch.Tensor,
        regime: int = 0,
        belief: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """z0: [B, T, V, H, W, C], command: [B, T, Dc], view_ids: [B, V]."""
        if self.belief_dim and belief is None:
            raise ValueError("this residual was built with belief inputs; pass belief")
        if not self.belief_dim and belief is not None:
            raise ValueError("this residual was built without belief inputs")

        B, T, V, H, W, _ = z0.shape
        rows = torch.linspace(0.0, 1.0, H, device=z0.device) if H > 1 else torch.zeros(1, device=z0.device)
        cols = torch.linspace(0.0, 1.0, W, device=z0.device) if W > 1 else torch.zeros(1, device=z0.device)
        position = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)

        pieces = [
            command[:, :, None, None, None, :].expand(B, T, V, H, W, -1),
            self.regime_embedding.weight[regime].expand(B, T, V, H, W, -1),
            self.view_embedding(view_ids)[:, None, :, None, None, :].expand(B, T, V, H, W, -1),
            position.expand(B, T, V, H, W, 2).to(command.dtype),
        ]
        if belief is not None:
            pieces.insert(0, belief[:, :, None, None, None, :].expand(B, T, V, H, W, -1))
        cond = torch.cat(pieces, dim=-1)
        return layer_norm_0(z0 + self.regimes[regime](z0, cond).to(z0.dtype))

    def online_parameters(self, regime: int) -> list[nn.Parameter]:
        """Spec 9.8: online updates reach only output and modulation parameters."""
        module = self.regimes[regime]
        return [*module.modulation_out.parameters(), *module.b_o.parameters()]
