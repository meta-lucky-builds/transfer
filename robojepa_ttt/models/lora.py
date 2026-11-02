"""Regime-specific LoRA on Q, V and attention-output projections (spec 6.6, eq. 11).

    W^(r) = W_0 + (alpha / k) B^(r) A^(r),  k = 8, alpha = 16

Upstream RoPEAttention has two layouts, both handled here:

* num_kv_heads == num_heads (300M): one fused Linear `qkv` with output
  [q | k | v], each of width dim, read by unflatten(-1, (3, heads, -1)).
* num_kv_heads != num_heads (1B, 2B and larger): `q` of width dim and `kv` of
  width 2 * head_dim * num_kv_heads with output [k | v].

K is left untouched, as the spec lists only Q, V and output. Base parameters are
never replaced; LoRALinear holds the original Linear as `.base`, so the frozen
tensors stay shared and the checkpoint must be loaded before wrapping.

With B zero-initialised the wrapper is exactly the base layer, and A receives no
gradient on the first update because dL/dA is proportional to B. That is
expected from the product form, and it is why functional anchoring, not a norm on
A or B, measures drift (spec 9.6).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRADelta(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


class LoRALinear(nn.Module):
    """A frozen Linear plus per-regime additive deltas on selected output slices."""

    def __init__(self, base: nn.Linear, slices: list[tuple[int, int]], rank: int, alpha: float, regimes: int):
        super().__init__()
        ordered = sorted(slices)
        for (_, end), (start, _) in zip(ordered, ordered[1:]):
            if end > start:
                raise ValueError("LoRA slices must not overlap")
        if ordered[0][0] < 0 or ordered[-1][1] > base.out_features:
            raise ValueError(f"slices {ordered} exceed output width {base.out_features}")
        self.base = base
        self.slices = ordered
        self.active_regime = 0
        self.deltas = nn.ModuleList(
            nn.ModuleList(LoRADelta(base.in_features, end - start, rank, alpha) for start, end in ordered)
            for _ in range(regimes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        pieces, cursor = [], 0
        for (start, end), delta in zip(self.slices, self.deltas[self.active_regime]):
            if start > cursor:
                pieces.append(out[..., cursor:start])
            pieces.append(out[..., start:end] + delta(x).to(out.dtype))
            cursor = end
        if cursor < out.shape[-1]:
            pieces.append(out[..., cursor:])
        return torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]


def attach_lora(predictor: nn.Module, last_blocks: int, rank: int, alpha: float, regimes: int = 1) -> list[str]:
    """Freeze the predictor, then wrap Q, V and output in the final blocks.

    Returns the wrapped module names and raises on any count other than exactly
    three adapted projections per block, so a renamed upstream layer cannot leave
    a block silently unadapted.
    """
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)

    blocks = predictor.predictor_blocks
    if not 0 < last_blocks <= len(blocks):
        raise ValueError(f"last_blocks={last_blocks} invalid for depth {len(blocks)}")

    wrapped, delta_count = [], 0
    first = len(blocks) - last_blocks
    for index in range(first, len(blocks)):
        attn = blocks[index].attn
        if getattr(attn, "_tp_head_major", False):
            raise ValueError("tensor-parallel head-major layout is not supported by LoRA")
        if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
            dim = attn.qkv.in_features
            if attn.qkv.out_features != 3 * dim:
                raise ValueError(f"unexpected fused qkv width {attn.qkv.out_features} for dim {dim}")
            attn.qkv = LoRALinear(attn.qkv, [(0, dim), (2 * dim, 3 * dim)], rank, alpha, regimes)
            wrapped.append(f"predictor_blocks.{index}.attn.qkv")
            delta_count += 2
        elif isinstance(getattr(attn, "q", None), nn.Linear) and isinstance(getattr(attn, "kv", None), nn.Linear):
            width = attn.kv.out_features
            if width % 2:
                raise ValueError(f"kv width {width} is not k + v")
            attn.q = LoRALinear(attn.q, [(0, attn.q.out_features)], rank, alpha, regimes)
            attn.kv = LoRALinear(attn.kv, [(width // 2, width)], rank, alpha, regimes)
            wrapped += [f"predictor_blocks.{index}.attn.q", f"predictor_blocks.{index}.attn.kv"]
            delta_count += 2
        else:
            raise ValueError(f"block {index} attention has no recognised Q/V projections")
        if not isinstance(attn.proj, nn.Linear):
            raise ValueError(f"block {index} output projection is not a Linear")
        attn.proj = LoRALinear(attn.proj, [(0, attn.proj.out_features)], rank, alpha, regimes)
        wrapped.append(f"predictor_blocks.{index}.attn.proj")
        delta_count += 1

    if delta_count != 3 * last_blocks:
        raise ValueError(f"adapted {delta_count} projections, expected {3 * last_blocks}")
    return wrapped


def set_lora_regime(predictor: nn.Module, regime: int) -> None:
    for module in predictor.modules():
        if isinstance(module, LoRALinear):
            module.active_regime = regime


def lora_parameters(predictor: nn.Module, regime: int) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in predictor.modules():
        if isinstance(module, LoRALinear):
            params.extend(module.deltas[regime].parameters())
    return params
