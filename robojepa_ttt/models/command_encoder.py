"""Command encoder e(U): a GRU of width 64 over ordered issued targets (spec 6.3).

Input per raw tick: six compact targets, tick duration, validity. The input is an
explicit logged command history, never hidden state carried between calls, so any
replay window reproduces its encoding exactly. Only offline training updates it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

COMMAND_FEATURES = 8


def command_features(compact_targets: torch.Tensor, dt_s: torch.Tensor, valid: torch.Tensor, tick_s: float):
    """Stack [..., L, 6] targets with durations and validity into [..., L, 8].

    Durations are expressed in ticks so that the nominal value is 1. Invalid ticks
    are zeroed, so padding cannot carry stale values into the GRU.
    """
    valid = valid.to(compact_targets.dtype)
    return torch.cat(
        [compact_targets * valid.unsqueeze(-1), (dt_s / tick_s).unsqueeze(-1) * valid.unsqueeze(-1), valid.unsqueeze(-1)],
        dim=-1,
    )


class CommandEncoder(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(COMMAND_FEATURES, hidden, batch_first=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [..., L, 8]. Returns [..., hidden]."""
        leading = features.shape[:-2]
        flat = features.reshape(-1, features.shape[-2], COMMAND_FEATURES)
        # Recurrent kernels have no bf16 path; keep this small model in fp32.
        with torch.autocast(device_type=features.device.type, enabled=False):
            _, last = self.gru(flat.float())
        return last[0].reshape(*leading, self.hidden)
