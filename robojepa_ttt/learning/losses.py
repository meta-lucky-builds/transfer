"""Loss terms of spec section 9. All reductions run in fp32."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def token_metric(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Masked per-token L1 in LN0 space, eq. 22.

    predicted, target: [B, T, V, H, W, C]. mask, weights: broadcastable to
    [B, T, V, H, W]. Frames are averaged within a sequence and then sequences are
    averaged, so differing view counts cannot reweight the objective. Returns None
    when nothing is valid; the caller must skip the term, not count it as zero.
    """
    predicted = F.layer_norm(predicted.float(), (predicted.size(-1),))
    target = F.layer_norm(target.float(), (target.size(-1),))
    per_token = (predicted - target).abs().mean(-1)

    weight = torch.ones_like(per_token)
    if weights is not None:
        weight = weight * weights.to(per_token.dtype)
    if mask is not None:
        weight = weight * mask.to(per_token.dtype)

    denominator = weight.sum(dim=(2, 3, 4))
    numerator = (per_token * weight).sum(dim=(2, 3, 4))
    frame_valid = denominator > 0
    frame_loss = numerator / denominator.clamp_min(1e-12)
    sequence_frames = frame_valid.sum(1)
    sequence_valid = sequence_frames > 0
    if not sequence_valid.any():
        return None
    sequence_loss = (frame_loss * frame_valid).sum(1)[sequence_valid] / sequence_frames[sequence_valid]
    return sequence_loss.mean()


def discounted_rollout_loss(step_losses: Sequence[Optional[torch.Tensor]], gamma: float) -> Optional[torch.Tensor]:
    """Eq. 25, normalised by the weights of the steps that were actually valid."""
    total, normaliser = None, 0.0
    for k, loss in enumerate(step_losses):
        if loss is None:
            continue
        weight = gamma**k
        total = weight * loss if total is None else total + weight * loss
        normaliser += weight
    return None if total is None else total / normaliser


def proximal_penalty(
    parameters: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    floor: float = 1e-3,
) -> Optional[torch.Tensor]:
    """Eq. 30. The floor keeps zero-initialised tensors from dividing by zero."""
    total, count = None, 0
    for name, value in parameters.items():
        ref = reference[name].to(value.device, value.dtype)
        rms = ref.pow(2).mean().sqrt().clamp_min(floor)
        term = ((value - ref) / rms).pow(2).sum()
        total = term if total is None else total + term
        count += value.numel()
    return None if total is None else total / count


def joint_huber(predicted: torch.Tensor, target: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Eq. 27: Huber with threshold 1 on scale-normalised joint error."""
    error = (predicted - target) / scale
    return F.huber_loss(error, torch.zeros_like(error), delta=1.0)
