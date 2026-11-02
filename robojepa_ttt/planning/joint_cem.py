"""Six-joint CEM over bounded incremental references (spec 14.3, 14.5, 14.6).

The released CEM samples translation, rotation and gripper, so changing action_dim
alone is not enough. This planner searches normalised increments v in [-1, 1]^6:

    p_ref[k+1] = clip(p_ref[k] + v[k] * delta_max)            eq. 35

and scores the commands the robot would actually receive, because candidates pass
through the same limiter as the local supervisor. This is the deterministic mean
rollout of spec 14.6, which is a debugging ablation of the belief planner, not
uncertainty-aware planning. Belief particles need the compact model.

The KV cache holds only the frames whose actions are already known. The current
frame shares attention with its action, so it is processed per candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from ..models.adapted_predictor import AdaptedPredictor, repeat_cache
from ..robot_io.command_limiter import LimiterConfig, limit_reference


@dataclass(frozen=True)
class CemConfig:
    horizon: int = 5
    candidates: int = 128
    iterations: int = 4
    elites: int = 16
    initial_std: float = 0.35
    std_floor: float = 0.05
    mean_momentum: float = 0.2
    std_momentum: float = 0.5
    movement_weight: float = 0.01
    smoothness_weight: float = 0.05
    view_weights: tuple[float, ...] = (1.0, 0.0)
    microbatch: int = 32


@dataclass
class PlanContext:
    """Everything observed up to now. Nothing here is from the future."""

    features: torch.Tensor  # [1, c, V, H, W, C]
    position: torch.Tensor  # [1, c, 6] physical
    issued_past: torch.Tensor  # [1, history + (c-1) * stride, 6]
    dt_past: torch.Tensor  # same ticks
    valid_past: torch.Tensor  # same ticks
    view_ids: torch.Tensor  # [1, V]
    goal: torch.Tensor  # [1, 1, V, H, W, C] encoded goal image
    reference: torch.Tensor  # [1, 6] current local reference
    reference_velocity: torch.Tensor  # [1, 6]


@dataclass
class Plan:
    commands: torch.Tensor  # [stride, 6] physical targets for the next block
    increments: torch.Tensor  # [H, 6]
    cost: float
    is_hold: bool
    version: str


class JointCem:
    def __init__(
        self,
        model: AdaptedPredictor,
        limiter: LimiterConfig,
        delta_max: tuple[float, ...],
        config: CemConfig,
        goal_scale: float = 1.0,
    ):
        if goal_scale <= 0:
            raise ValueError("goal scale must be positive")
        self.model = model
        self.limiter = limiter
        self.delta_max = delta_max
        self.config = config
        self.goal_scale = goal_scale

    def commands_for(self, increments: torch.Tensor, context: PlanContext) -> torch.Tensor:
        """Translate [N, H, 6] increments to the raw ticks the robot would receive."""
        N, H, _ = increments.shape
        stride = self.model.stride
        delta = increments.new_tensor(self.delta_max)
        lo = increments.new_tensor(self.limiter.position_min)
        hi = increments.new_tensor(self.limiter.position_max)
        previous = context.reference.expand(N, -1)
        ramps = []
        fractions = increments.new_tensor([(i + 1) / stride for i in range(stride)]).view(1, stride, 1)
        for k in range(H):
            following = (previous + increments[:, k] * delta).clamp(lo, hi)
            ramps.append(previous[:, None] + fractions * (following - previous)[:, None])
            previous = following
        commands, _, _ = limit_reference(
            context.reference.expand(N, -1),
            context.reference_velocity.expand(N, -1),
            torch.cat(ramps, dim=1),
            self.model.tick_s,
            self.limiter,
        )
        return commands

    def _batch(self, context: PlanContext, commands: torch.Tensor) -> dict:
        N, ticks, _ = commands.shape
        return {
            "position": context.position.expand(N, -1, -1),
            "issued": torch.cat([context.issued_past.expand(N, -1, -1), commands], dim=1),
            "dt": torch.cat(
                [context.dt_past.expand(N, -1), commands.new_full((N, ticks), self.model.tick_s)], dim=1
            ),
            "command_valid": torch.cat(
                [context.valid_past.expand(N, -1), torch.ones(N, ticks, dtype=torch.bool, device=commands.device)], dim=1
            ),
        }

    @torch.no_grad()
    def _terminal_tokens(self, context: PlanContext, commands: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregress every candidate. Returns terminal tokens and predicted joints."""
        model = self.model
        N = commands.shape[0]
        H = self.config.horizon
        c = context.features.shape[1]
        batch = self._batch(context, commands)
        actions, states = model.sequence_inputs(batch, c, H)
        command = model.encode_commands(batch, c + H - 1)
        predicted_joints = model.predicted_positions(batch, c, H)
        view_ids = context.view_ids.expand(N, -1)

        prefix = None
        if c > 1:
            prefix = model.prime_context(
                context.features[:, : c - 1], actions[:1, : c - 1], states[:1, : c - 1], context.view_ids
            )
            prefix.cache = repeat_cache(prefix.cache, N)
        frame = context.features[:, c - 1 : c].expand(N, -1, -1, -1, -1, -1)
        for k in range(H):
            t = c - 1 + k
            frame, prefix = model.step_cached(
                frame, actions[:, t : t + 1], states[:, t : t + 1], view_ids, command[:, t : t + 1], prefix, t
            )
        return frame, predicted_joints

    def score(self, increments: torch.Tensor, context: PlanContext, previous_first: Optional[torch.Tensor] = None):
        """Return (cost [N], feasible [N]), eq. 34 and 36 for a deterministic rollout."""
        cfg = self.config
        commands = self.commands_for(increments, context)
        costs, feasible = [], []
        lo = increments.new_tensor(self.limiter.position_min)
        hi = increments.new_tensor(self.limiter.position_max)
        for start in range(0, increments.shape[0], cfg.microbatch):
            part = commands[start : start + cfg.microbatch]
            tokens, joints = self._terminal_tokens(context, part)
            costs.append(self._goal_distance(tokens, context.goal))
            feasible.append(((joints >= lo) & (joints <= hi)).all(-1).all(-1))
        goal_cost = torch.cat(costs) / self.goal_scale

        movement = increments.pow(2).sum(-1).mean(-1) / 6
        last = previous_first if previous_first is not None else torch.zeros(6, device=increments.device)
        steps = torch.cat([last.expand(increments.shape[0], 1, 6), increments], dim=1)
        smoothness = (steps[:, 1:] - steps[:, :-1]).pow(2).sum(-1).mean(-1) / 6
        return goal_cost + cfg.movement_weight * movement + cfg.smoothness_weight * smoothness, torch.cat(feasible)

    def _goal_distance(self, tokens: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Weighted per-token L1 in LN0 space, eq. 34. Zero-weight views are ignored."""
        tokens = F.layer_norm(tokens.float(), (tokens.size(-1),))
        goal = F.layer_norm(goal.float(), (goal.size(-1),))
        per_view = (tokens - goal).abs().mean(dim=(-1, -2, -3))[:, 0]
        weights = per_view.new_tensor(self.config.view_weights[: per_view.shape[1]])
        return (per_view * weights).sum(1) / weights.sum()

    @torch.no_grad()
    def plan(self, context: PlanContext, warm_start: Optional[torch.Tensor] = None, seed: int = 0) -> Plan:
        cfg = self.config
        device = context.features.device
        lo = context.position.new_tensor(self.limiter.position_min)
        hi = context.position.new_tensor(self.limiter.position_max)
        for name, value in (("arm", context.position[:, -1]), ("reference", context.reference)):
            if (value < lo - 1e-6).any() or (value > hi + 1e-6).any():
                raise RuntimeError(f"hold is infeasible: {name} is outside its validated limits")
        generator = torch.Generator(device=device).manual_seed(seed)
        H = cfg.horizon
        mean = torch.zeros(H, 6, device=device)
        if warm_start is not None:
            mean[: H - 1] = warm_start[1:H]
        std = torch.full((H, 6), cfg.initial_std, device=device)
        hold = torch.zeros(1, H, 6, device=device)
        previous = warm_start[None, :H].to(device) if warm_start is not None else None
        executed = warm_start[0].to(device) if warm_start is not None else None  # v_{k-1} of eq. 36

        best_cost, best = float("inf"), None
        for _ in range(cfg.iterations):
            noise = torch.randn(cfg.candidates, H, 6, generator=generator, device=device)
            samples = (mean + std * noise).clamp(-1.0, 1.0)
            samples[0] = 0.0  # candidate 0 is always hold
            if previous is not None:
                samples[1] = previous[0]
            costs, feasible = self.score(samples, context, executed)
            order = costs.masked_fill(~feasible, float("inf")).argsort()
            order = order[feasible[order]]
            if len(order) == 0:
                break
            if costs[order[0]] < best_cost:
                best_cost, best = float(costs[order[0]]), samples[order[0]].clone()
            elites = samples[order[: cfg.elites]]
            mean = cfg.mean_momentum * mean + (1 - cfg.mean_momentum) * elites.mean(0)
            std = (cfg.std_momentum * std + (1 - cfg.std_momentum) * elites.std(0, unbiased=False)).clamp_min(cfg.std_floor)

        # The final mean need not be feasible, so it is re-evaluated (spec 14.6).
        final = torch.stack([mean.clamp(-1.0, 1.0), hold[0]])
        final_costs, final_feasible = self.score(final, context, executed)
        if final_feasible[0] and final_costs[0] < best_cost:
            best_cost, best = float(final_costs[0]), final[0]
        if best is None:
            if not final_feasible[1]:
                raise RuntimeError("hold is infeasible; the arm is outside its validated limits")
            best_cost, best = float(final_costs[1]), hold[0]
        is_hold = bool((best == 0).all())
        commands = self.commands_for(best[None], context)[0, : self.model.stride]
        return Plan(commands, best, best_cost, is_hold, self.model.version)
