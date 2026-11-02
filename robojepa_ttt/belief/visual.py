"""The belief inside the visual predictor: baseline B4 (spec 6.6 eq. 12, 9.2, 9.4, 9.5).

    z_hat_{t+1} = LN0[ z0 + B_o tanh(A_o z0 + M([h_t, x_{t+1}, e(U_t), e_r, e_v, pos])) ]

* Motion. In a command rollout, the joints of future keyframes are the belief's
  prior joint means instead of the servo alone. The compact transition already
  contains the servo, plus its learned residual.
* Conditioning. Position t of the visual sequence predicts z_{t+1}; its belief input
  is [h_t, x_{t+1}^-]: the history after assimilating frame t, and the prior state
  mean for frame t+1. For context positions both come from filtering real frames.
  From the last context frame on, frames are not observed: the belief steps forward
  with every coordinate masked, so the history sees prior means flagged as missing,
  never self-generated measurements (spec 7.6, 14.4 step 6).
* Gradient boundary. Everything the belief produces is detached before the visual
  model sees it. Visual losses can therefore never make the servo bridge or filter
  invent motion that merely explains an image (spec 9.4). The compact model trains
  only on its own likelihood objectives.
* Consistency (eq. 28). The sketch of each predicted token frame is pulled toward
  the stopped compact prediction of the visual sketch for that frame.

The deterministic baseline uses the prior mean, not samples (spec 9.2 allows either
and requires naming it). Sampling belongs to planning (planning/belief_cem.py).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from ..learning.losses import discounted_rollout_loss, token_metric
from ..models.adapted_predictor import AdaptedPredictor
from ..models.token_residual import TokenResidual
from .compact import JOINTS, CompactBelief, StepInput
from .sketch import ObservationSketch


def belief_dim(compact: CompactBelief) -> int:
    return compact.config.recurrent_hidden + compact.config.state_dim


def expand_residual_for_belief(residual: TokenResidual, extra: int) -> TokenResidual:
    """A belief-conditioned copy of a B1 residual that starts as the same function.

    The belief is the first block of the residual's condition vector, so the first
    modulation layer gets `extra` new input columns, initialised to zero. Every other
    weight is copied. A stage-3 B1 checkpoint therefore continues exactly where it
    was, and training then learns how much to use the belief.
    """
    if residual.belief_dim:
        raise ValueError("this residual already has belief inputs")
    expanded = copy.deepcopy(residual)
    expanded.belief_dim = extra
    for regime in expanded.regimes:
        first = regime.modulation[0] if len(regime.modulation) else regime.modulation_out
        new = torch.nn.Linear(first.in_features + extra, first.out_features, device=first.weight.device, dtype=first.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, extra:] = first.weight
            new.bias.copy_(first.bias)
        if len(regime.modulation):
            regime.modulation[0] = new
        else:
            regime.modulation_out = new
    return expanded


def check_contract(compact: CompactBelief, sketch: ObservationSketch, model: Optional[AdaptedPredictor] = None) -> None:
    """Refuse mismatched observation or timing contracts instead of silently misaligning.

    The compact model's visual slots must be the sketch's contract views with the same
    number of coordinates, and its command blocks must use the visual model's stride,
    history and tick, because both read the same tick arrays of one batch.
    """
    c = compact.config
    if tuple(c.view_ids) != tuple(sketch.view_ids) or c.visual_per_view != sketch.components:
        raise ValueError(
            f"compact contract views {c.view_ids} x {c.visual_per_view} differ from the sketch's "
            f"{sketch.view_ids} x {sketch.components}"
        )
    if model is not None:
        if (compact.stride, compact.history_ticks) != (model.stride, model.history_ticks) or abs(compact.tick_s - model.tick_s) > 1e-12:
            raise ValueError("compact and visual models use different stride, history or tick")
        if compact.units != model.units:
            raise ValueError("compact and visual models use different unit conventions")


def compact_steps(compact: CompactBelief, sketch: ObservationSketch, batch: dict, frames: int) -> list[StepInput]:
    """StepInputs for keyframes 0..frames-1 of a visual batch, in the compact model's dtype.

    The visual batch holds tick commands with `history` ticks before keyframe 0, so
    the block U_{k-1} that leads into keyframe k ends at tick index history + k*stride
    and its own history starts `history` ticks before the block. Keyframe 0's block
    ends at tick `history`, so its history reaches `stride` ticks before the data:
    those ticks hold the first measured position with validity 0 and nominal
    duration, as data.command_block and windows.extract_window do before a segment.

    Visual windows are admitted only with every keyframe's joints and views valid
    (windows.window_is_valid), so joints are all valid here; a contract view that
    the episode lacks is masked by the sketch.
    """
    check_contract(compact, sketch)
    dtype = next(compact.parameters()).dtype
    stride, history = compact.stride, compact.history_ticks
    length = history + stride
    needed = history + (frames - 1) * stride
    if batch["issued"].shape[1] < needed:
        raise ValueError(f"batch has {batch['issued'].shape[1]} ticks, {frames} keyframes need {needed}")
    issued = batch["issued"].to(dtype)
    tick_dt = batch["dt"].to(dtype)
    valid = batch["command_valid"].bool()
    pad = stride
    hold = batch["position"][:, :1].to(dtype).expand(-1, pad, -1)
    issued = torch.cat([hold, issued], 1)
    tick_dt = torch.cat([torch.full_like(tick_dt[:, :1], compact.tick_s).expand(-1, pad), tick_dt], 1)
    valid = torch.cat([torch.zeros_like(valid[:, :1]).expand(-1, pad), valid], 1)

    joints = compact.units.physical_to_compact(batch["position"][:, :frames].to(dtype))
    B = joints.shape[0]
    visual, view_valid = [], []
    for b in range(B):
        v, ok = sketch.observe(batch["features"][b, :frames], batch["view_ids"][b].tolist())
        visual.append(v)
        view_valid.append(ok)
    visual = torch.stack(visual).to(dtype)
    view_valid = torch.stack(view_valid)
    joint_valid = torch.ones(B, frames, JOINTS, dtype=torch.bool, device=joints.device)
    mask = torch.cat([joint_valid, view_valid.repeat_interleave(sketch.components, -1)], -1)
    steps = []
    for k in range(frames):
        end = pad + history + k * stride
        block_dt = tick_dt[:, end - stride : end]
        steps.append(
            StepInput(
                observation=torch.cat([joints[:, k], visual[:, k]], -1),
                obs_mask=mask[:, k],
                issued=issued[:, end - length : end],
                tick_dt=tick_dt[:, end - length : end],
                command_valid=valid[:, end - length : end],
                # Frame 0 has no predecessor; its interval is nominal, as in data.py.
                dt=block_dt.sum(-1) if k else torch.full((B,), stride * compact.tick_s, dtype=dtype, device=joints.device),
            )
        )
    return steps


@dataclass
class BeliefConditioning:
    motion: Optional[torch.Tensor]  # [B, H, 6] physical joints of keyframes context..context+H-1
    belief: torch.Tensor  # [B, positions, belief_dim]: [h_t, x_{t+1}^-]
    predicted_visual: torch.Tensor  # [B, positions, visual]: compact prior sketch of frames 1..positions
    view_valid: torch.Tensor  # [B, positions, V] whether that frame's view was actually observed


@torch.no_grad()
def belief_conditioning(
    compact: CompactBelief,
    steps: list[StepInput],
    context: int,
    horizon: int,
    regimes=(0,),
) -> BeliefConditioning:
    """Filter frames 0..context-1, then predict blind through context..context+horizon-1.

    Returns detached inputs for positions 0..context+horizon-2. With horizon=0 the
    filter assimilates every frame in `steps` (the realized-motion path), and motion
    is None.
    """
    if horizon < 0 or context < 1:
        raise ValueError("need context >= 1 and horizon >= 0")
    frames = context + horizon
    if horizon == 0:
        context = frames = len(steps)
    first = steps[0]
    state = compact.initial_state(first.observation[:, :JOINTS], first.obs_mask[:, :JOINTS].all(-1), regimes)
    state, _ = compact.step(state, first, predict=False)
    histories, priors, visuals, valids, joints = [state.h], [], [], [], []
    per_view = compact.config.visual_per_view
    for k in range(1, frames):
        step = steps[k]
        if k >= context:
            state, out = compact.predict_only(state, step)
            joints.append(out.prior_joint_mean)
        else:
            state, out = compact.step(state, step)
        priors.append(out.prior_state_mean)
        visuals.append(out.predicted_observation[:, JOINTS:])
        valids.append(step.obs_mask[:, JOINTS:].reshape(step.obs_mask.shape[0], -1, per_view).all(-1) & (k < context))
        histories.append(state.h)
    positions = frames - 1
    belief = torch.cat([torch.stack(histories[:positions], 1), torch.stack(priors, 1)], -1)
    motion = None
    if horizon > 0:
        motion = compact.units.compact_to_physical(torch.stack(joints, 1))
    return BeliefConditioning(motion, belief, torch.stack(visuals, 1), torch.stack(valids, 1))


def consistency_loss(
    sketch: ObservationSketch, predicted_tokens: torch.Tensor, compact_visual: torch.Tensor, view_ids: torch.Tensor
) -> Optional[torch.Tensor]:
    """Eq. 28: mean L1 between the sketch of predicted tokens and the stopped compact
    prediction of the sketch, over contract views present in the batch.

    predicted_tokens [B, T, V_ep, H, W, C]; compact_visual [B, T, visual].
    """
    total, count = None, 0
    for b in range(predicted_tokens.shape[0]):
        ids = view_ids[b].tolist()
        for slot, view_id in enumerate(sketch.view_ids):
            if view_id not in ids:
                continue
            coords = sketch.transform_slot(predicted_tokens[b, :, ids.index(view_id)], slot)
            start = slot * sketch.components
            target = compact_visual[b, :, start : start + sketch.components].detach().to(coords.dtype)
            term = (coords - target).abs().sum()
            total = term if total is None else total + term
            count += coords.numel()
    return None if total is None else total / count


def belief_visual_loss(
    model: AdaptedPredictor,
    batch: dict,
    context: int,
    horizon: int,
    config,
    compact: CompactBelief,
    sketch: ObservationSketch,
    weight_consistency: float = 0.05,
    regimes=(0,),
    use_realized: bool = True,
):
    """L_V of spec 9.7 for B4: 0.25 realized + 0.5 cmdTF + 0.5 AR + 0.05 consistency.

    Same shape as offline_train.visual_loss, so OfflineTrainer can take it as loss_fn.
    The compact model must be frozen while this trains (spec 10.6); everything it
    produces is computed under no_grad, so this loss cannot reach it in any case.
    The residual must have been built or expanded with belief_dim(compact) inputs.
    """
    check_contract(compact, sketch, model)
    if horizon < 1:
        raise ValueError("the command losses need a horizon of at least one step")
    frames = batch["features"].shape[1]
    features_dtype = batch["features"].dtype
    steps = compact_steps(compact, sketch, batch, frames)
    rollout = belief_conditioning(compact, steps[: context + horizon], context, horizon, regimes)
    target = batch["features"][:, context : context + horizon]
    predicted = model.rollout(
        batch,
        context,
        horizon,
        motion=rollout.motion.to(batch["position"].dtype),
        belief=rollout.belief.to(features_dtype),
    )
    step_losses = [token_metric(predicted[:, k : k + 1], target[:, k : k + 1]) for k in range(horizon)]
    cmd_tf, rollout_loss = step_losses[0], discounted_rollout_loss(step_losses, config.gamma)
    loss = config.weight_cmd_tf * cmd_tf + config.weight_rollout * rollout_loss
    parts = {"cmd_tf": float(cmd_tf.detach()), "rollout": float(rollout_loss.detach())}
    cons = consistency_loss(sketch, predicted, rollout.predicted_visual[:, context - 1 :], batch["view_ids"])
    if cons is not None and weight_consistency:
        loss = loss + weight_consistency * cons
        parts["consistency"] = float(cons.detach())
    if use_realized:
        filtered = belief_conditioning(compact, steps, frames, 0, regimes)
        realized = model.realized_forward(batch, frames, belief=filtered.belief.to(features_dtype))
        realized_loss = token_metric(realized, batch["features"][:, 1:])
        loss = loss + config.weight_realized * realized_loss
        parts["realized"] = float(realized_loss.detach())
    return loss, parts
