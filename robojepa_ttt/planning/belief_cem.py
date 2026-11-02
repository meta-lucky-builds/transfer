"""Belief-sampled six-joint CEM with a bounded risk preference (spec 14.4 to 14.6, eq. 36, 37).

For each candidate command sequence U (built and limited exactly as in JointCem):

    1. particles: regimes stratified over pi_t, states x ~ N(mu_r, P_r)        14.4.1
    2. roll the compact F with sampled process noise and sticky regime switches  14.4.2
    3. the particle's joints are the predicted realized motion                   14.4.3
    4. roll the visual predictor with that motion, the same issued commands and
       the belief residual input [h_t, x_{t+1}] of that particle                 14.4.4
    5. the imagined history h sees the particle's predicted observation flagged
       missing; nothing generated is ever assimilated                            14.4.5, 14.4.6, 7.6
    6. J^(m) of eq. 36 per particle, then
       score = (1 - beta) E_w[J] + beta CVaR_alpha,w(J)                          eq. 37

Stratification. Every regime whose weight exceeds the planning cutoff (spec 8.2,
config change_detection.planning_mode_probability_cutoff) gets at least one particle,
the remaining particles go to the largest shortfalls n_r < pi_r M, and each particle
of regime r carries weight pi_r / n_r (pi renormalised over the kept regimes). A rare
but plausible regime therefore always appears, and the weighted estimator stays
unbiased for the kept mixture. Regimes below the cutoff are dropped, as the spec's
planning set prescribes; that is a deliberate truncation of the mixture.

Common random numbers. Regimes, state draws, process noise and switching uniforms are
drawn once per CEM iteration and shared by every candidate, so identical candidates
score identically and score differences come from the commands. The final
re-evaluation of the mean, the best sample and hold uses one fresh shared draw, so
the three are compared on the same particles (JointCem compares a stale best cost,
which is exact only because it is deterministic).

CVaR is the exact weighted empirical form: min over tau of tau + E_w[(J - tau)_+] /
(1 - alpha) is convex and piecewise linear with breakpoints at the particle costs,
and it is minimised at the weighted alpha-quantile. With four particles this tail
estimate is crude and is not calibrated tail safety (spec 14.5).

Feasibility. A candidate is rejected when any particle's predicted joints leave the
validated limits at any keyframe (hard constraint of eq. 36). The soft term
C_limit is not defined numerically by the spec; here it is the mean over keyframes
and joints of ((margin - distance to the nearer limit)_+ / margin)^2, and it is off
(weight 0) unless a margin is configured. Hold is candidate 0 and the previous
feasible plan candidate 1. If every candidate including hold is rejected, planning
raises, as JointCem does: particles near a limit can make even hold infeasible, and
the local supervisor's commissioned hold policy then applies.

Deviations, stated: the epistemic term of eq. 37 is not built (it needs
independently trained heads; config epistemic_weight 0) and the planner refuses a
nonzero weight. Particle motion is absolute (the particle's joints), matching the
B4 training path where motion is the prior joint mean, rather than the measured
position plus the particle's displacement. Gaussian draws inside a regime are
plain, not antithetic or quasi-random.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..belief.compact import JOINTS, BeliefState, CompactBelief, StepInput
from ..belief.imm import robust_cholesky, sticky_log_transition
from ..belief.sketch import ObservationSketch
from ..belief.visual import compact_steps
from ..models.adapted_predictor import AdaptedPredictor, repeat_cache
from ..robot_io.command_limiter import LimiterConfig
from .joint_cem import CemConfig, JointCem, Plan, PlanContext


@dataclass(frozen=True)
class BeliefCemConfig(CemConfig):
    """CemConfig plus the belief planner's values from config_so101.yaml `planning`."""

    particles: int = 4  # particles_after_deterministic_debug
    cvar_alpha: float = 0.8
    risk_weight: float = 0.25  # cvar_mixture_weight, beta_risk
    regime_cutoff: float = 0.05  # change_detection.planning_mode_probability_cutoff
    limit_weight: float = 0.0  # lambda_lim; the spec gives no value
    limit_margin: Optional[tuple[float, ...]] = None  # physical, per joint
    deterministic: bool = False  # one noiseless mean particle: the debugging ablation of 14.6
    epistemic_weight: float = 0.0

    def __post_init__(self):
        if self.particles < 1:
            raise ValueError("need at least one particle")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must lie in (0, 1)")
        if not 0.0 <= self.risk_weight <= 1.0:
            raise ValueError("risk_weight must lie in [0, 1]")
        if self.deterministic and self.particles != 1:
            raise ValueError("the deterministic ablation uses exactly one particle")
        if self.limit_weight < 0 or (self.limit_weight > 0 and (self.limit_margin is None or len(self.limit_margin) != 6 or min(self.limit_margin) <= 0)):
            raise ValueError("a soft limit weight needs six positive margins")
        if self.epistemic_weight != 0.0:
            raise ValueError("the epistemic term needs independently trained heads, which do not exist (spec 14.5)")


@dataclass
class BeliefPlanContext(PlanContext):
    """PlanContext plus the belief filtered through the last context frame c-1.

    belief is the posterior after assimilating frame c-1 (batch of one), including
    h_{c-1}. The visual prefix cache needs no belief: the residual acts on outputs,
    and outputs at prefix positions are not used.
    """

    belief: BeliefState = None

    def __post_init__(self):
        if self.belief is None or self.belief.mean.shape[0] != 1:
            raise ValueError("a belief plan context needs the filtered belief of one robot")


@dataclass
class ParticleDraws:
    """One iteration's common random numbers, shared by every candidate."""

    slot: torch.Tensor  # [M] long, index into belief.regimes
    weight: torch.Tensor  # [M], sums to 1
    state: torch.Tensor  # [M, n] the sampled x_{c-1}
    process: Optional[torch.Tensor]  # [M, H, n] standard normal, None when deterministic
    switch: Optional[torch.Tensor]  # [M, H] uniforms for regime switches, None when deterministic


def weighted_cvar(values: torch.Tensor, weights: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exact CVaR_alpha of a discrete distribution, eq. 37. values [..., M], weights [M] or [..., M].

    CVaR = min_tau tau + sum_i w_i (J_i - tau)_+ / (1 - alpha). The objective's slope
    is 1 - P(J > tau) / (1 - alpha), which changes sign at the weighted alpha-quantile
    (the smallest J whose cumulative weight reaches alpha), so that is a minimiser.
    Where cumulative weight equals alpha exactly the objective is flat between two
    breakpoints, so the tolerance below cannot change the value.
    """
    weights = weights.to(values.dtype).expand_as(values)
    if (weights < 0).any() or not torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]), atol=1e-6):
        raise ValueError("CVaR weights must be nonnegative and sum to one")
    ordered, index = values.sort(-1)
    cumulative = weights.gather(-1, index).cumsum(-1)
    first = (cumulative < alpha - 1e-9).sum(-1, keepdim=True).clamp(max=values.shape[-1] - 1)
    tau = ordered.gather(-1, first)
    return tau.squeeze(-1) + (weights * (values - tau).clamp_min(0)).sum(-1) / (1.0 - alpha)


def stratified_counts(probabilities: torch.Tensor, particles: int, cutoff: float) -> tuple[list[int], torch.Tensor]:
    """Particles per regime and the renormalised kept probabilities (module docstring).

    Kept regimes: probability above cutoff, always including the most probable one.
    Each gets one particle; the rest go one at a time to the largest pi_r M - n_r,
    ties to the lower index. Refuses when there are more kept regimes than particles.
    """
    p = probabilities.detach().double().cpu()
    keep = p > cutoff
    keep[int(p.argmax())] = True
    kept = torch.where(keep, p, torch.zeros_like(p))
    kept = kept / kept.sum()
    regimes = int(keep.sum())
    if regimes > particles:
        raise ValueError(f"{regimes} plausible regimes need at least as many particles, got {particles}")
    counts = [int(k) for k in keep]
    for _ in range(particles - regimes):
        shortfall = torch.where(keep, kept * particles - torch.tensor(counts, dtype=kept.dtype), torch.full_like(kept, -torch.inf))
        counts[int(shortfall.argmax())] += 1
    return counts, kept


def belief_from_context(compact: CompactBelief, sketch: ObservationSketch, context: PlanContext, regimes=(0,)) -> BeliefState:
    """Filter the plan context's own frames 0..c-1 (spec 7.4 initialisation at frame 0).

    For debugging and tests. A live system plans from its running belief
    (FixedLagBelief), which has seen far more than the c context frames.
    """
    c = context.features.shape[1]
    batch = {
        "features": context.features,
        "position": context.position,
        "issued": context.issued_past,
        "dt": context.dt_past,
        "command_valid": context.valid_past,
        "view_ids": context.view_ids,
    }
    steps = compact_steps(compact, sketch, batch, c)
    with torch.no_grad():
        state = compact.initial_state(steps[0].observation[:, :JOINTS], steps[0].obs_mask[:, :JOINTS].all(-1), regimes)
        state, _ = compact.step(state, steps[0], predict=False)
        for step in steps[1:]:
            state, _ = compact.step(state, step)
    return state


class BeliefCem(JointCem):
    def __init__(
        self,
        model: AdaptedPredictor,
        compact: CompactBelief,
        limiter: LimiterConfig,
        delta_max: tuple[float, ...],
        config: BeliefCemConfig,
        goal_scale: float = 1.0,
    ):
        super().__init__(model, limiter, delta_max, config, goal_scale)
        if (compact.stride, compact.history_ticks) != (model.stride, model.history_ticks) or abs(compact.tick_s - model.tick_s) > 1e-12:
            raise ValueError("compact and visual models use different stride, history or tick")
        if compact.units != model.units:
            raise ValueError("compact and visual models use different unit conventions")
        expected = compact.config.recurrent_hidden + compact.config.state_dim
        if model.residual.belief_dim != expected:
            raise ValueError(f"the visual residual takes {model.residual.belief_dim} belief inputs, the compact model gives {expected}")
        self.compact = compact

    # -------------------------------------------------------------- particles

    @torch.no_grad()
    def draw_particles(self, belief: BeliefState, generator: torch.Generator) -> ParticleDraws:
        cfg, compact = self.config, self.compact
        dtype = belief.mean.dtype
        device = belief.mean.device
        n, H = belief.mean.shape[-1], cfg.horizon
        probabilities = belief.log_pi[0].exp()
        if cfg.deterministic:
            slot = probabilities.argmax().reshape(1)
            return ParticleDraws(slot, torch.ones(1, dtype=dtype, device=device), belief.mean[0, slot], None, None)
        counts, kept = stratified_counts(probabilities, cfg.particles, cfg.regime_cutoff)
        slot = torch.repeat_interleave(torch.arange(len(counts), device=device), torch.tensor(counts, device=device))
        weight = torch.cat([torch.full((c,), float(kept[r]) / c, dtype=dtype) for r, c in enumerate(counts) if c]).to(device)
        factor = robust_cholesky(belief.cov[0], compact.config.jitter, compact.repairs, "planning particle covariance")
        epsilon = torch.randn(cfg.particles, n, generator=generator, device=device, dtype=dtype)
        state = belief.mean[0, slot] + (factor[slot] @ epsilon[..., None]).squeeze(-1)
        process = torch.randn(cfg.particles, H, n, generator=generator, device=device, dtype=dtype)
        switch = torch.rand(cfg.particles, H, generator=generator, device=device, dtype=dtype)
        return ParticleDraws(slot, weight, state, process, switch)

    def _switch(self, slot: torch.Tensor, uniforms: torch.Tensor, dt: torch.Tensor, regimes: int) -> torch.Tensor:
        """Sample the next regime from the sticky prior of eq. 14 by inverse CDF."""
        if regimes == 1:
            return slot
        transition = sticky_log_transition(regimes, dt, self.compact.config.mode_residence_s).exp()  # [R, K, K]
        cdf = transition[torch.arange(slot.shape[0], device=slot.device), slot].cumsum(-1)
        return (cdf < uniforms[:, None]).sum(-1).clamp(max=regimes - 1)

    # ---------------------------------------------------------------- rollout

    @torch.no_grad()
    def _particle_rollout(self, context: BeliefPlanContext, commands: torch.Tensor, draws: ParticleDraws):
        """Roll every (candidate, particle) pair. Rows are candidate-major.

        Returns terminal tokens [N*M, 1, V, H, W, C] and particle joints [N, M, H, 6]
        in physical units at keyframes c..c+H-1.
        """
        model, compact, cfg = self.model, self.compact, self.config
        N, M, H = commands.shape[0], draws.slot.shape[0], cfg.horizon
        rows = N * M
        c = context.features.shape[1]
        stride, history = model.stride, model.history_ticks
        belief = context.belief
        cdtype = belief.mean.dtype

        batch = {k: v.repeat_interleave(M, dim=0) for k, v in self._batch(context, commands).items()}
        issued = batch["issued"].to(cdtype)
        tick_dt = batch["dt"].to(cdtype)
        valid = batch["command_valid"]
        x = draws.state.repeat(N, 1)
        slot = draws.slot.repeat(N)
        h = belief.h.expand(rows, -1)
        age = belief.age_s.expand(rows, -1)
        obs_dim = compact.config.obs_dim
        motion, conditioning = [], []
        for k in range(H):
            end = history + (c + k) * stride
            start = end - history - stride
            step = StepInput(
                observation=torch.zeros(rows, obs_dim, dtype=cdtype, device=x.device),
                obs_mask=torch.zeros(rows, obs_dim, dtype=torch.bool, device=x.device),
                issued=issued[:, start:end],
                tick_dt=tick_dt[:, start:end],
                command_valid=valid[:, start:end],
                dt=tick_dt[:, end - stride : end].sum(-1),
            )
            if draws.switch is not None:
                slot = self._switch(slot, draws.switch.repeat(N, 1)[:, k], step.dt, len(belief.regimes))
            noise = None if draws.process is None else draws.process.repeat(N, 1, 1)[:, k]
            x_next, h_next, age, _ = compact.imagine(x, h, age, slot, belief.regimes, step, noise)
            conditioning.append(torch.cat([h, x_next], -1))  # position c-1+k: [h_t, x_{t+1}]
            motion.append(compact.units.compact_to_physical(x_next[:, :JOINTS]))
            x, h = x_next, h_next
        motion = torch.stack(motion, 1).to(context.position.dtype)
        conditioning = torch.stack(conditioning, 1).to(context.features.dtype)

        actions, states = model.sequence_inputs(batch, c, H, motion)
        command = model.encode_commands(batch, c + H - 1)
        view_ids = context.view_ids.expand(rows, -1)
        prefix = None
        if c > 1:
            prefix = model.prime_context(context.features[:, : c - 1], actions[:1, : c - 1], states[:1, : c - 1], context.view_ids)
            prefix.cache = repeat_cache(prefix.cache, rows)
        frame = context.features[:, c - 1 : c].expand(rows, -1, -1, -1, -1, -1)
        for k in range(H):
            t = c - 1 + k
            frame, prefix = model.step_cached(
                frame, actions[:, t : t + 1], states[:, t : t + 1], view_ids, command[:, t : t + 1], prefix, t, conditioning[:, k : k + 1]
            )
        return frame, motion.reshape(N, M, H, JOINTS)

    # ----------------------------------------------------------------- score

    def _soft_limit(self, joints: torch.Tensor) -> torch.Tensor:
        """C_limit per particle [N, M] (definition in the module docstring)."""
        cfg = self.config
        if cfg.limit_weight == 0:
            return joints.new_zeros(joints.shape[:2])
        lo = joints.new_tensor(self.limiter.position_min)
        hi = joints.new_tensor(self.limiter.position_max)
        margin = joints.new_tensor(cfg.limit_margin)
        distance = torch.minimum(joints - lo, hi - joints)
        return ((margin - distance).clamp_min(0) / margin).pow(2).mean(dim=(-1, -2))

    def score_particles(
        self,
        increments: torch.Tensor,
        context: BeliefPlanContext,
        draws: ParticleDraws,
        previous_first: Optional[torch.Tensor] = None,
    ):
        """Return (score [N], feasible [N], particle costs J [N, M]), eq. 36 and 37."""
        cfg = self.config
        commands = self.commands_for(increments, context)
        lo = increments.new_tensor(self.limiter.position_min)
        hi = increments.new_tensor(self.limiter.position_max)
        M = draws.slot.shape[0]
        goal, feasible, soft = [], [], []
        for start in range(0, increments.shape[0], cfg.microbatch):
            part = commands[start : start + cfg.microbatch]
            tokens, joints = self._particle_rollout(context, part, draws)
            goal.append(self._goal_distance(tokens, context.goal).reshape(-1, M))
            feasible.append(((joints >= lo) & (joints <= hi)).all(-1).all(-1).all(-1))
            soft.append(self._soft_limit(joints))
        goal_cost = torch.cat(goal) / self.goal_scale

        movement = increments.pow(2).sum(-1).mean(-1) / 6
        last = previous_first if previous_first is not None else torch.zeros(6, device=increments.device)
        steps = torch.cat([last.expand(increments.shape[0], 1, 6), increments], dim=1)
        smoothness = (steps[:, 1:] - steps[:, :-1]).pow(2).sum(-1).mean(-1) / 6
        particle = (
            goal_cost
            + (cfg.movement_weight * movement + cfg.smoothness_weight * smoothness)[:, None]
            + cfg.limit_weight * torch.cat(soft)
        )
        weight = draws.weight.to(particle.dtype)
        expected = (particle * weight).sum(-1)
        score = (1 - cfg.risk_weight) * expected + cfg.risk_weight * weighted_cvar(particle, weight, cfg.cvar_alpha)
        return score, torch.cat(feasible), particle

    def score(self, increments, context, previous_first=None):
        raise TypeError("BeliefCem scores with shared particle draws: use score_particles")

    # ------------------------------------------------------------------ plan

    @torch.no_grad()
    def plan(self, context: BeliefPlanContext, warm_start: Optional[torch.Tensor] = None, seed: int = 0) -> Plan:
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
        hold = torch.zeros(H, 6, device=device)
        previous = warm_start[None, :H].to(device) if warm_start is not None else None
        executed = warm_start[0].to(device) if warm_start is not None else None  # v_{k-1} of eq. 36

        best, best_cost = None, float("inf")
        for _ in range(cfg.iterations):
            draws = self.draw_particles(context.belief, generator)
            noise = torch.randn(cfg.candidates, H, 6, generator=generator, device=device)
            samples = (mean + std * noise).clamp(-1.0, 1.0)
            samples[0] = 0.0  # candidate 0 is always hold
            if previous is not None:
                samples[1] = previous[0]
            costs, feasible, _ = self.score_particles(samples, context, draws, executed)
            order = costs.masked_fill(~feasible, float("inf")).argsort()
            order = order[feasible[order]]
            if len(order) == 0:
                break
            if costs[order[0]] < best_cost:
                best_cost, best = float(costs[order[0]]), samples[order[0]].clone()
            elites = samples[order[: cfg.elites]]
            mean = cfg.mean_momentum * mean + (1 - cfg.mean_momentum) * elites.mean(0)
            std = (cfg.std_momentum * std + (1 - cfg.std_momentum) * elites.std(0, unbiased=False)).clamp_min(cfg.std_floor)

        # Re-evaluate the final mean, the best sample and hold on one shared fresh draw.
        finalists = [mean.clamp(-1.0, 1.0), hold] + ([best] if best is not None else [])
        draws = self.draw_particles(context.belief, generator)
        final_costs, final_feasible, _ = self.score_particles(torch.stack(finalists), context, draws, executed)
        if not bool(final_feasible.any()):
            raise RuntimeError("no candidate is feasible for every particle, not even hold")
        choice = int(final_costs.masked_fill(~final_feasible, float("inf")).argmin())
        chosen = finalists[choice]
        is_hold = bool((chosen == 0).all())
        commands = self.commands_for(chosen[None], context)[0, : self.model.stride]
        return Plan(commands, chosen, float(final_costs[choice]), is_hold, self.model.version)
