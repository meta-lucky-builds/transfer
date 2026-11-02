"""Compact recurrent belief model and its filtering step (spec 6.3 to 6.5, 7, eq. 5 to 10).

State and observation, all in compact (normalised) units:

    x = [p_bar (6) ; xi (48) ; d (10)]       eq. 5, n = 64
    y = [p_bar measured (6) ; visual sketch (V * 64)]   eq. 4, m = 134 for two views
    h = GRU history (256)

One macro step t, in the order the spec requires:

    1. e(U_{t-1})             command encoding of the block just executed, with its
                              one second of logged history
    2. predict                IMM mixing, then x_t^- = F(x_{t-1}, h_{t-1}, U_{t-1}, dt)
    3. observation model      H(h_{t-1}, e(U_{t-1})), b(...): past only
    4. predictive likelihood  of y_t BEFORE assimilation (the training NLL, eq. 26)
    5. assimilate             masked IMM-EKF update
    6. history                h_t = GRU(h_{t-1}, [y~_t, p_bar_t, e(U_{t-1}), mask,
                              age, dt]), only AFTER assimilation, so the likelihood
                              of y_t never sees y_t through h (spec 6.3)

Missing coordinates are masked out of the likelihood. In the GRU input they are
replaced by the prior predicted mean and flagged by the mask and age, so they never
become measurements. The GRU input carries no joint mask, because the spec fixes it
at 203 values; a missing joint reading is replaced by its prediction, and the step
reports motion_allowed=False, since missing joint feedback blocks new motion in the
first deployment profile.

Initialisation (spec 7.4). The first frame uses measured joints as the joint mean
with the measured sensor variance, which already is the posterior after that joint
reading. The first step therefore assimilates only the visual block: assimilating
the same joint reading again would count it twice.

Precision. The compact model and filter run in float32 with autocast disabled
(spec 7.3: BF16 is not for covariance factorisations). Tests run float64.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.command_encoder import CommandEncoder, command_features
from ..models.servo import ServoParams, rollout_servo
from ..robot_io.so101_units import So101Units
from .imm import (
    JitterPolicy,
    RepairLog,
    block_nis,
    ekf_predict,
    imm_mix,
    masked_update,
    mixture_predictive,
    posterior_log_weights,
    sticky_log_transition,
)

JOINTS = 6


@dataclass(frozen=True)
class CompactConfig:
    """Values from config_so101.yaml `belief`. Hardware values have no defaults."""

    joint_sensor_std: tuple[float, ...]  # measured repeatability, compact units, six values
    residual_bound: tuple[float, ...]  # a_max of the joint residual, compact units per macro step
    view_ids: tuple[int, ...] = (6, 1)  # observation contract, must match the sketch
    visual_per_view: int = 64
    fast_dims: int = 48
    slow_dims: int = 10
    recurrent_hidden: int = 256
    command_hidden: int = 64
    regime_dim: int = 16
    max_regimes: int = 4
    transition_hidden: tuple[int, ...] = (256, 256, 256)
    observation_hidden: tuple[int, ...] = (256, 256)
    nominal_macro_dt_s: float = 0.2
    fast_decay_s: float = 10.0
    fast_coefficient: float = 0.1
    observation_matrix_bound: float = 0.1
    process_variance_init: tuple[float, float, float] = (1e-4, 1e-3, 1e-5)  # joint, fast, slow per 0.2 s
    joint_fast_variance_bounds: tuple[float, float] = (1e-6, 1e-1)
    slow_variance_bounds: tuple[float, float] = (1e-8, 1e-2)
    visual_std_init: float = 0.2
    # Not in the spec, which gives no bound for the visual noise: a sketch coordinate
    # is standardised, so 0.01 to 3 standard deviations covers every sane value.
    visual_std_bounds: tuple[float, float] = (0.01, 3.0)
    initial_latent_variance: float = 1.0
    mode_residence_s: float = 600.0
    # Not in the spec: age inputs are clipped at this many seconds and divided by it.
    age_cap_s: float = 10.0
    correction_rank: int = 8
    jitter: JitterPolicy = JitterPolicy(1e-7, 1e-3)

    def __post_init__(self):
        if len(self.joint_sensor_std) != JOINTS or min(self.joint_sensor_std) <= 0:
            raise ValueError("joint_sensor_std needs six measured positive values (spec 6.5)")
        if len(self.residual_bound) != JOINTS or min(self.residual_bound) <= 0:
            raise ValueError("residual_bound needs six positive values from validated motion (spec 6.4)")

    @property
    def state_dim(self) -> int:
        return JOINTS + self.fast_dims + self.slow_dims

    @property
    def views(self) -> int:
        return len(self.view_ids)

    @property
    def obs_dim(self) -> int:
        return JOINTS + self.views * self.visual_per_view

    @property
    def gru_input_dim(self) -> int:
        return self.views * self.visual_per_view + JOINTS + self.command_hidden + 2 * self.views + 1

    def blocks(self) -> dict[str, tuple[int, int]]:
        out = {"joints": (0, JOINTS)}
        for slot, view_id in enumerate(self.view_ids):
            start = JOINTS + slot * self.visual_per_view
            out[f"view{view_id}"] = (start, start + self.visual_per_view)
        return out


def _bounded(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Map an unconstrained value into [low, high] uniformly in log space."""
    return low * (high / low) ** torch.sigmoid(raw)


def _unbounded(value: float, low: float, high: float) -> float:
    fraction = math.log(value / low) / math.log(high / low)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"initial value {value} outside ({low}, {high})")
    return math.log(fraction / (1.0 - fraction))


def _mlp(width_in: int, hidden: Sequence[int]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    width = width_in
    for size in hidden:
        layers += [nn.Linear(width, size), nn.SiLU()]
        width = size
    return nn.Sequential(*layers), width


@dataclass
class BeliefState:
    log_pi: torch.Tensor  # [B, K]
    mean: torch.Tensor  # [B, K, n]
    cov: torch.Tensor  # [B, K, n, n]
    h: torch.Tensor  # [B, H]
    age_s: torch.Tensor  # [B, V], seconds since each view was last actually observed
    regimes: tuple[int, ...]  # embedding index of each active slot

    def detach(self) -> "BeliefState":
        return BeliefState(
            self.log_pi.detach(), self.mean.detach(), self.cov.detach(), self.h.detach(), self.age_s.detach(), self.regimes
        )

    def select(self, keep: torch.Tensor, other: "BeliefState") -> "BeliefState":
        """Per batch element: self where keep, else other."""
        def pick(a, b):
            k = keep.reshape(-1, *([1] * (a.dim() - 1)))
            return torch.where(k, a, b)

        return BeliefState(
            pick(self.log_pi, other.log_pi),
            pick(self.mean, other.mean),
            pick(self.cov, other.cov),
            pick(self.h, other.h),
            pick(self.age_s, other.age_s),
            self.regimes,
        )


@dataclass
class StepInput:
    """Everything known at macro step t. Nothing after t.

    observation [B, m] in compact units (joints then visual slots), obs_mask [B, m].
    issued [B, L, 6] physical targets for ticks of the block U_{t-1} preceded by its
    history, so the block occupies the last `stride` ticks; tick_dt [B, L]; command_valid
    [B, L]. dt [B] seconds from frame t-1 to frame t. step_valid [B] marks padding.
    quality_scale [B, V], optional R multipliers >= 1 per view for measured blur or
    partial visibility (spec 7.5).
    """

    observation: torch.Tensor
    obs_mask: torch.Tensor
    issued: torch.Tensor
    tick_dt: torch.Tensor
    command_valid: torch.Tensor
    dt: torch.Tensor
    step_valid: Optional[torch.Tensor] = None
    quality_scale: Optional[torch.Tensor] = None


@dataclass
class StepOutput:
    log_predictive: torch.Tensor  # [B], log p(y_t | past) over valid coordinates
    valid_count: torch.Tensor  # [B]
    prior_joint_mean: torch.Tensor  # [B, 6], compact, before assimilation
    prior_joint_var: torch.Tensor  # [B, 6], marginal predictive variance of joint coordinates
    log_prior: torch.Tensor  # [B, K] regime weights before assimilation
    log_posterior: torch.Tensor  # [B, K]
    motion_allowed: torch.Tensor  # [B], False when joint feedback is missing
    predicted_observation: torch.Tensor  # [B, m] mixture prior mean
    nis: Optional[dict] = None


class CompactBelief(nn.Module):
    def __init__(
        self,
        config: CompactConfig,
        units: So101Units,
        servo: ServoParams,
        stride: int = 6,
        tick_s: float = 1.0 / 30.0,
        history_ticks: int = 30,
    ):
        super().__init__()
        if history_ticks < servo.max_delay:
            raise ValueError("command history must cover the servo delay")
        self.config = config
        self.units = units
        self.servo = servo
        self.stride = stride
        self.tick_s = tick_s
        self.history_ticks = history_ticks
        n, c = config.state_dim, config
        self.command_encoder = CommandEncoder(c.command_hidden)
        self.regime_embedding = nn.Embedding(c.max_regimes, c.regime_dim)

        # Transition residual MLP, eq. 7: input [x, h, e(U), dt, e_r].
        self.transition_body, width = _mlp(n + c.recurrent_hidden + c.command_hidden + 1 + c.regime_dim, c.transition_hidden)
        self.transition_out = nn.Linear(width, JOINTS + c.fast_dims)
        # Zero output: the untrained transition is exactly the servo plus decay.
        nn.init.zeros_(self.transition_out.weight)
        nn.init.zeros_(self.transition_out.bias)
        # Online mean correction, rank-limited, B zero-initialised, added before tanh
        # (config online_transition_correction). Frozen offline.
        self.correction_a = nn.Parameter(torch.empty(c.correction_rank, width))
        self.correction_b = nn.Parameter(torch.zeros(JOINTS + c.fast_dims, c.correction_rank))
        nn.init.kaiming_uniform_(self.correction_a, a=math.sqrt(5))

        # Observation MLP, eq. 10: input [h_{t-1}, e(U_{t-1}), e_r].
        visual = c.views * c.visual_per_view
        self.observation_body, width = _mlp(c.recurrent_hidden + c.command_hidden + c.regime_dim, c.observation_hidden)
        self.observation_matrix = nn.Linear(width, visual * n)
        self.observation_bias = nn.Linear(width, visual)
        for layer in (self.observation_matrix, self.observation_bias):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.history = nn.GRUCell(c.gru_input_dim, c.recurrent_hidden)

        # Bounded diagonal process noise per regime, eq. 7 and spec 6.4.
        joint, fast, slow = c.process_variance_init
        lo, hi = c.joint_fast_variance_bounds
        slo, shi = c.slow_variance_bounds
        init = [_unbounded(joint, lo, hi)] * JOINTS + [_unbounded(fast, lo, hi)] * c.fast_dims + [_unbounded(slow, slo, shi)] * c.slow_dims
        self.process_raw = nn.Parameter(torch.tensor(init).repeat(c.max_regimes, 1))
        low = torch.tensor([lo] * (JOINTS + c.fast_dims) + [slo] * c.slow_dims)
        high = torch.tensor([hi] * (JOINTS + c.fast_dims) + [shi] * c.slow_dims)
        self.register_buffer("process_low", low)
        self.register_buffer("process_high", high)

        # Observation noise: joints measured and fixed, visual learned within bounds.
        vlo, vhi = c.visual_std_bounds
        self.visual_std_raw = nn.Parameter(torch.full((visual,), _unbounded(c.visual_std_init, vlo, vhi)))
        self.register_buffer("joint_variance", torch.tensor(c.joint_sensor_std).pow(2))
        # Calibration multipliers on R per block (joints, then each view), spec 7.5.
        # Fitted on calibration episodes after training, never by gradient.
        self.register_buffer("calibration", torch.ones(1 + c.views))
        self.register_buffer("residual_bound", torch.tensor(c.residual_bound))
        self.repairs = RepairLog()

    # ------------------------------------------------------------ parameters

    def offline_parameters(self) -> list[nn.Parameter]:
        """Everything except the online correction, which stays at zero offline."""
        skip = {id(self.correction_a), id(self.correction_b)}
        return [p for p in self.parameters() if id(p) not in skip]

    def online_parameters(self) -> list[nn.Parameter]:
        """Spec 9.8: online, only the small final mean correction of F adapts."""
        return [self.correction_a, self.correction_b]

    def fast_output_weights(self) -> torch.Tensor:
        """Rows of the transition output layer that produce the fast residual (L_regularity)."""
        return self.transition_out.weight[JOINTS:]

    # -------------------------------------------------------------- noise

    def process_variance(self, regimes: Sequence[int], dt_s: torch.Tensor) -> torch.Tensor:
        """Q for this interval, [B, K, n]: per-0.2 s variance scaled by dt / 0.2."""
        q = self.process_low * (self.process_high / self.process_low) ** torch.sigmoid(self.process_raw[list(regimes)])
        return q[None] * (dt_s / self.config.nominal_macro_dt_s)[:, None, None]

    def observation_variance(self, quality_scale: Optional[torch.Tensor], batch: int) -> torch.Tensor:
        """Diagonal R [B, m] with calibration and optional per-view quality multipliers."""
        c = self.config
        visual = _bounded(self.visual_std_raw, *c.visual_std_bounds).pow(2)
        per_view = visual.reshape(c.views, c.visual_per_view) * self.calibration[1:, None]
        joints = self.joint_variance * self.calibration[0]
        r = torch.cat([joints, per_view.reshape(-1)])[None].expand(batch, -1)
        if quality_scale is not None:
            if (quality_scale < 1).any():
                raise ValueError("quality scales inflate R and must be >= 1")
            scale = torch.cat([torch.ones_like(quality_scale[:, :1]).expand(-1, JOINTS), quality_scale.repeat_interleave(c.visual_per_view, 1)], 1)
            r = r * scale
        return r

    # ---------------------------------------------------------- components

    def encode_commands(self, step: StepInput) -> torch.Tensor:
        compact = self.units.physical_to_compact(step.issued)
        return self.command_encoder(command_features(compact, step.tick_dt, step.command_valid, self.tick_s))

    def nominal_joints(self, joints: torch.Tensor, issued: torch.Tensor, tick_dt: torch.Tensor) -> torch.Tensor:
        """Eq. 8 in compact units. joints [N, 6] at the block start; returns [N, 6] at its end.

        The servo runs in physical units; the affine unit maps cancel in its Jacobian,
        which is diag(prod(1 - gain)) with respect to the start position.
        """
        start = self.units.compact_to_physical(joints)
        end = rollout_servo(start, issued, tick_dt, self.servo, self.history_ticks)[..., -1, :]
        return self.units.physical_to_compact(end)

    def transition(self, x: torch.Tensor, h, command, dt_s, issued, tick_dt, regimes) -> torch.Tensor:
        """F for every regime. x [B, K, n] -> [B, K, n]; rows are independent."""
        c = self.config
        B, K, n = x.shape
        flat = x.reshape(B * K, n)
        def per_k(t):
            return t[:, None].expand(B, K, *t.shape[1:]).reshape(B * K, *t.shape[1:])

        dt = per_k(dt_s[:, None])[:, 0]
        emb = self.regime_embedding.weight[list(regimes)][None].expand(B, K, -1).reshape(B * K, -1)
        inputs = torch.cat([flat, per_k(h), per_k(command), (dt / c.nominal_macro_dt_s)[:, None], emb], -1)
        hidden = self.transition_body(inputs)
        raw = self.transition_out(hidden) + F.linear(F.linear(hidden, self.correction_a), self.correction_b)
        joints = self.nominal_joints(flat[:, :JOINTS], per_k(issued), per_k(tick_dt))
        joints = joints + self.residual_bound * torch.tanh(raw[:, :JOINTS])
        fast = flat[:, JOINTS : JOINTS + c.fast_dims]
        decay = torch.exp(-dt / c.fast_decay_s)[:, None]
        fast = decay * fast + (dt / c.nominal_macro_dt_s)[:, None] * c.fast_coefficient * torch.tanh(raw[:, JOINTS:])
        slow = flat[:, JOINTS + c.fast_dims :]
        return torch.cat([joints, fast, slow], -1).reshape(B, K, n)

    def observation_model(self, h: torch.Tensor, command: torch.Tensor, regimes) -> tuple[torch.Tensor, torch.Tensor]:
        """H [B, K, m, n] and b [B, K, m]. Joint rows are [I_6, 0] with zero bias (eq. 10)."""
        c = self.config
        B, K, n = h.shape[0], len(regimes), c.state_dim
        visual = c.views * c.visual_per_view
        emb = self.regime_embedding.weight[list(regimes)]
        inputs = torch.cat([h[:, None].expand(B, K, -1), command[:, None].expand(B, K, -1), emb[None].expand(B, K, -1)], -1)
        hidden = self.observation_body(inputs)
        matrix = c.observation_matrix_bound * torch.tanh(self.observation_matrix(hidden)).reshape(B, K, visual, n)
        joint_rows = torch.eye(JOINTS, n, dtype=matrix.dtype, device=matrix.device).expand(B, K, JOINTS, n)
        bias = torch.cat([matrix.new_zeros(B, K, JOINTS), self.observation_bias(hidden)], -1)
        return torch.cat([joint_rows, matrix], -2), bias

    # --------------------------------------------------------------- state

    def initial_state(
        self,
        joints: torch.Tensor,
        joint_valid: torch.Tensor,
        regimes: Sequence[int] = (0,),
        slow_mean: Optional[torch.Tensor] = None,
        slow_var: Optional[torch.Tensor] = None,
    ) -> BeliefState:
        """Spec 7.4. joints [B, 6] compact measured at the first frame.

        Raises when a joint reading is missing: the belief cannot start without one.
        slow_mean/slow_var [B, slow] may carry a compatible saved slow prior.
        """
        if not bool(joint_valid.all()):
            raise ValueError("cannot initialise a belief without valid joint readings")
        c = self.config
        B, K, n = joints.shape[0], len(regimes), c.state_dim
        if len(set(regimes)) != K or max(regimes) >= c.max_regimes:
            raise ValueError("regimes must be distinct embedding indices below max_regimes")
        dtype, device = joints.dtype, joints.device
        mean = torch.zeros(B, n, dtype=dtype, device=device)
        mean[:, :JOINTS] = joints
        var = torch.full((B, n), c.initial_latent_variance, dtype=dtype, device=device)
        var[:, :JOINTS] = (self.joint_variance * self.calibration[0]).to(dtype)
        if slow_mean is not None:
            mean[:, JOINTS + c.fast_dims :] = slow_mean
        if slow_var is not None:
            var[:, JOINTS + c.fast_dims :] = slow_var
        return BeliefState(
            log_pi=torch.full((B, K), -math.log(K), dtype=dtype, device=device),
            mean=mean[:, None].expand(B, K, n).clone(),
            cov=torch.diag_embed(var)[:, None].expand(B, K, n, n).clone(),
            h=torch.zeros(B, c.recurrent_hidden, dtype=dtype, device=device),
            age_s=torch.full((B, c.views), c.age_cap_s, dtype=dtype, device=device),
            regimes=tuple(int(r) for r in regimes),
        )

    # ---------------------------------------------------------------- step

    def step(
        self,
        state: BeliefState,
        step: StepInput,
        predict: bool = True,
        differentiable_jacobian: bool = True,
        compute_nis: bool = False,
    ) -> tuple[BeliefState, StepOutput]:
        """One macro step in the order of the module docstring.

        predict=False is the first frame after initial_state: no transition, and the
        joint rows are masked because that reading already initialised the belief.
        """
        with torch.autocast(device_type=state.mean.device.type, enabled=False):
            return self._step(state, step, predict, differentiable_jacobian, compute_nis)

    def _step(self, state, step, predict, differentiable_jacobian, compute_nis):
        c = self.config
        B = state.mean.shape[0]
        regimes = state.regimes
        command = self.encode_commands(step)

        if predict:
            log_t = sticky_log_transition(len(regimes), step.dt, c.mode_residence_s)
            log_prior, mixed_mean, mixed_cov = imm_mix(state.log_pi, state.mean, state.cov, log_t)

            def f(x):
                return self.transition(x, state.h, command, step.dt, step.issued, step.tick_dt, regimes)

            prior_mean, prior_cov, _ = ekf_predict(
                mixed_mean, mixed_cov, f, self.process_variance(regimes, step.dt), differentiable_jacobian
            )
        else:
            log_prior, prior_mean, prior_cov = state.log_pi, state.mean, state.cov

        mask = step.obs_mask.bool()
        if not predict:
            mask = mask.clone()
            mask[:, :JOINTS] = False

        matrix, bias = self.observation_model(state.h, command, regimes)
        variance = self.observation_variance(step.quality_scale, B).to(prior_mean.dtype)
        update = masked_update(prior_mean, prior_cov, step.observation, mask, matrix, bias, variance, c.jitter, self.repairs)
        log_post = posterior_log_weights(log_prior, update.log_likelihood)
        log_predictive = torch.logsumexp(log_prior + update.log_likelihood, -1)

        predicted, predictive_cov = mixture_predictive(log_prior, update)
        # Joint predictive variance from the unmasked prior, valid whether or not the
        # joint rows were assimilated: sum_k pi_k (P_k + R) + spread of regime means.
        weights = torch.exp(log_prior)
        joint_means = prior_mean[..., :JOINTS]
        joint_mean = torch.einsum("bk,bkj->bj", weights, joint_means)
        joint_diag = torch.diagonal(prior_cov, dim1=-2, dim2=-1)[..., :JOINTS] + variance[:, None, :JOINTS]
        prior_joint_var = torch.einsum("bk,bkj->bj", weights, joint_diag + (joint_means - joint_mean[:, None]).pow(2))

        # History update after assimilation, eq. 6.
        visual_mask = mask[:, JOINTS:]
        joints_valid = step.obs_mask.bool()[:, :JOINTS]
        stand_in = torch.where(step.obs_mask.bool(), torch.nan_to_num(step.observation), predicted)
        view_valid = visual_mask.reshape(B, c.views, c.visual_per_view).all(-1)
        age = torch.where(view_valid, torch.zeros_like(state.age_s), (state.age_s + step.dt[:, None]).clamp(max=c.age_cap_s))
        gru_input = torch.cat(
            [
                stand_in[:, JOINTS:],
                torch.where(joints_valid, stand_in[:, :JOINTS], predicted[:, :JOINTS]),
                command,
                view_valid.to(command.dtype),
                age / c.age_cap_s,
                (step.dt / c.nominal_macro_dt_s)[:, None],
            ],
            -1,
        ).to(state.h.dtype)
        h = self.history(gru_input, state.h)

        new_state = BeliefState(log_post, update.mean, update.cov, h, age, regimes)
        valid_step = step.step_valid if step.step_valid is not None else torch.ones(B, dtype=torch.bool, device=h.device)
        new_state = new_state.select(valid_step, state)

        nis = None
        if compute_nis:
            nis = block_nis(step.observation, mask, predicted, predictive_cov, c.blocks())
        output = StepOutput(
            log_predictive=log_predictive,
            valid_count=update.valid_count * valid_step,
            prior_joint_mean=joint_mean,
            prior_joint_var=prior_joint_var,
            log_prior=log_prior,
            log_posterior=log_post,
            motion_allowed=joints_valid.all(-1) & valid_step,
            predicted_observation=predicted,
            nis=nis,
        )
        return new_state, output

    def predict_only(self, state: BeliefState, step: StepInput) -> tuple[BeliefState, StepOutput]:
        """Imagination / prediction through a gap: no coordinate is assimilated (spec 7.6).

        Equivalent to a step with every observation missing. The GRU sees the prior
        predicted means flagged as missing, never a self-generated measurement.
        """
        blind = replace(step, obs_mask=torch.zeros_like(step.obs_mask, dtype=torch.bool))
        return self.step(state, blind)
