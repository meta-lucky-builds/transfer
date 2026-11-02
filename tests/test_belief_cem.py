"""Belief-sampled CEM (spec 14.4 to 14.6): CVaR, stratification, common random numbers,
imagination without assimilation, hard limits, and reduction to the deterministic planner."""

import itertools

import pytest
import torch

from conftest import make_batch, make_model, set_adapters_nonzero
from test_belief_visual import b4_model, make_compact, make_sketch
from robojepa_ttt.belief.compact import StepInput
from robojepa_ttt.belief.visual import belief_dim, expand_residual_for_belief
from robojepa_ttt.planning.belief_cem import (
    BeliefCem,
    BeliefCemConfig,
    BeliefPlanContext,
    belief_from_context,
    stratified_counts,
    weighted_cvar,
)
from robojepa_ttt.planning.joint_cem import CemConfig, JointCem
from robojepa_ttt.robot_io.command_limiter import LimiterConfig

LIMITS = LimiterConfig((-2.0,) * 6, (2.0,) * 6, (3.0,) * 6, (100.0,) * 6)
SMALL = dict(horizon=3, candidates=6, iterations=2, elites=3, microbatch=4, view_weights=(1.0, 0.0))


def context_for(compact, sketch, c=3, belief=None, position_offset=0.0):
    batch = make_batch(batch=1, frames=c + 3)
    ticks = 30 + (c - 1) * 6
    fields = dict(
        features=batch["features"][:, :c],
        position=batch["position"][:, :c] * 0 + position_offset,
        issued_past=torch.zeros(1, ticks, 6) + position_offset,
        dt_past=torch.full((1, ticks), 1 / 30),
        valid_past=torch.ones(1, ticks, dtype=torch.bool),
        view_ids=batch["view_ids"],
        goal=batch["features"][:, -1:],
        reference=torch.zeros(1, 6) + position_offset,
        reference_velocity=torch.zeros(1, 6),
    )
    if belief is None:
        from robojepa_ttt.planning.joint_cem import PlanContext

        belief = belief_from_context(compact, sketch, PlanContext(**fields))
    return BeliefPlanContext(**fields, belief=belief)


def planner(compact=None, **changes):
    compact = compact or make_compact()
    model = b4_model(compact=compact, nonzero_belief=True)
    return BeliefCem(model, compact, LIMITS, (0.2,) * 6, BeliefCemConfig(**{**SMALL, **changes})), compact


# ------------------------------------------------------------------ CVaR


def cvar_brute_force(values, weights, alpha):
    """min over tau of tau + E_w (J - tau)_+ / (1 - alpha), on every breakpoint and a fine grid."""
    lo, hi = float(values.min()) - 1, float(values.max()) + 1
    taus = torch.cat([values.double(), torch.linspace(lo, hi, 20001, dtype=torch.float64)])
    objective = taus + (weights.double()[None] * (values.double()[None] - taus[:, None]).clamp_min(0)).sum(-1) / (1 - alpha)
    return float(objective.min())


@pytest.mark.parametrize("alpha", [0.5, 0.8, 0.95])
def test_weighted_cvar_equals_brute_force_minimisation(alpha):
    generator = torch.Generator().manual_seed(0)
    for M in (1, 2, 4, 7):
        values = torch.randn(10, M, generator=generator, dtype=torch.float64)
        weights = torch.rand(M, generator=generator, dtype=torch.float64) + 0.05
        weights = weights / weights.sum()
        exact = weighted_cvar(values, weights, alpha)
        for i in range(10):
            assert exact[i].item() == pytest.approx(cvar_brute_force(values[i], weights, alpha), abs=1e-9)


def test_weighted_cvar_closed_forms_and_refusals():
    values = torch.tensor([[1.0, 4.0, 2.0, 3.0]], dtype=torch.float64)
    uniform = torch.full((4,), 0.25, dtype=torch.float64)
    # alpha 0.8 with four equal particles: the worst 20% lies inside the worst particle.
    assert weighted_cvar(values, uniform, 0.8).item() == pytest.approx(4.0)
    # alpha 0.5: the mean of the worst half; a tie of cumulative weight at alpha is exact.
    assert weighted_cvar(values, uniform, 0.5).item() == pytest.approx(3.5)
    # Ties in the values.
    assert weighted_cvar(torch.tensor([[2.0, 2.0, 2.0, 2.0]]), uniform.float(), 0.8).item() == pytest.approx(2.0)
    with pytest.raises(ValueError):
        weighted_cvar(values, torch.full((4,), 0.3, dtype=torch.float64), 0.8)


# --------------------------------------------------------- stratification


def test_stratified_counts_keep_rare_plausible_regimes():
    counts, kept = stratified_counts(torch.tensor([0.88, 0.12]), 4, 0.05)
    assert counts == [3, 1] and torch.allclose(kept, torch.tensor([0.88, 0.12], dtype=torch.float64))
    counts, kept = stratified_counts(torch.tensor([0.97, 0.03]), 4, 0.05)
    assert counts == [4, 0] and kept.tolist() == [1.0, 0.0], "below the planning cutoff a regime is dropped"
    counts, _ = stratified_counts(torch.tensor([0.5, 0.3, 0.2]), 8, 0.05)
    assert counts == [4, 2, 2] and sum(counts) == 8
    with pytest.raises(ValueError, match="particles"):
        stratified_counts(torch.tensor([0.4, 0.3, 0.3]), 2, 0.05)


def test_particle_draws_weight_regimes_by_pi_over_count():
    cem, compact = planner()
    sketch = make_sketch()
    context = context_for(compact, sketch)
    two = context.belief
    two_regimes = type(two)(
        torch.log(torch.tensor([[0.88, 0.12]])),
        two.mean.repeat(1, 2, 1),
        two.cov.repeat(1, 2, 1, 1),
        two.h,
        two.age_s,
        (0, 1),
    )
    draws = cem.draw_particles(two_regimes, torch.Generator().manual_seed(0))
    assert draws.slot.tolist() == [0, 0, 0, 1]
    assert torch.allclose(draws.weight, torch.tensor([0.88 / 3] * 3 + [0.12]))


# ------------------------------------------------------------ imagination


def test_imagined_step_equals_a_fully_missing_filter_step():
    """For one regime and no noise, imagine() reproduces predict_only: same state mean,
    same history, same predicted observation. Nothing is assimilated."""
    compact = make_compact(dtype=torch.float64)
    sketch = make_sketch()
    context = context_for(compact, sketch)
    state = context.belief
    L = compact.history_ticks + compact.stride
    g = torch.Generator().manual_seed(3)
    step = StepInput(
        observation=torch.randn(1, compact.config.obs_dim, generator=g, dtype=torch.float64) * 100,  # must be ignored
        obs_mask=torch.ones(1, compact.config.obs_dim, dtype=torch.bool),
        issued=torch.randn(1, L, 6, generator=g, dtype=torch.float64).cumsum(1) * 0.02,
        tick_dt=torch.full((1, L), 1 / 30, dtype=torch.float64),
        command_valid=torch.ones(1, L, dtype=torch.bool),
        dt=torch.full((1,), 0.2, dtype=torch.float64),
    )
    after, out = compact.predict_only(state, step)
    x, h, age, predicted = compact.imagine(state.mean[:, 0], state.h, state.age_s, torch.zeros(1, dtype=torch.long), (0,), step)
    assert torch.allclose(x, after.mean[:, 0], atol=1e-12)
    assert torch.allclose(h, after.h, atol=1e-12)
    assert torch.allclose(age, after.age_s) and torch.allclose(predicted, out.predicted_observation, atol=1e-12)
    # With noise the state moves by exactly sqrt(Q) * noise.
    noise = torch.randn(1, compact.config.state_dim, generator=g, dtype=torch.float64)
    x_noisy, *_ = compact.imagine(state.mean[:, 0], state.h, state.age_s, torch.zeros(1, dtype=torch.long), (0,), step, noise)
    q = compact.process_variance((0,), step.dt)[:, 0]
    assert torch.allclose(x_noisy - x, q.sqrt() * noise, atol=1e-12)


# ---------------------------------------------------------------- scoring


def test_common_random_numbers_make_identical_candidates_score_identically():
    cem, compact = planner()
    context = context_for(compact, make_sketch())
    g = torch.Generator().manual_seed(1)
    increments = torch.randn(4, 3, 6, generator=g).clamp(-1, 1)
    increments[2] = increments[0]
    draws = cem.draw_particles(context.belief, torch.Generator().manual_seed(5))
    score, feasible, particle = cem.score_particles(increments, context, draws)
    assert torch.equal(score[0], score[2]) and torch.equal(particle[0], particle[2])
    assert particle.shape == (4, 4)
    assert not torch.allclose(particle[0, 0], particle[0, 1]), "particles differ from each other"
    # Microbatching splits rows, not the draws.
    again, _, again_particle = cem.score_particles(increments, context, draws)
    assert torch.equal(score, again)
    # Aggregation is exactly eq. 37 on the particle costs.
    beta, alpha = 0.25, 0.8
    expected = (1 - beta) * (particle * draws.weight).sum(-1) + beta * weighted_cvar(particle, draws.weight, alpha)
    assert torch.allclose(score, expected)


def test_plans_are_deterministic_for_a_seed_and_respect_the_limiter():
    cem, compact = planner()
    context = context_for(compact, make_sketch())
    plan = cem.plan(context, seed=3)
    again = cem.plan(context, seed=3)
    assert torch.equal(plan.commands, again.commands) and plan.cost == again.cost
    assert plan.commands.shape == (6, 6) and plan.commands.abs().max() <= 2.0
    assert plan.version == cem.model.version


def test_one_noiseless_particle_reduces_to_the_deterministic_joint_cem():
    """Untrained compact transition = servo, belief started at the measured joints,
    zero belief columns = the B1 residual: scores equal JointCem's, so rankings agree."""
    compact = make_compact(perturb=0.0)
    model = make_model()
    set_adapters_nonzero(model)
    joint = JointCem(model, LIMITS, (0.2,) * 6, CemConfig(**SMALL))
    b4 = make_model()
    set_adapters_nonzero(b4)
    b4.residual = expand_residual_for_belief(b4.residual, belief_dim(compact))
    cem = BeliefCem(b4, compact, LIMITS, (0.2,) * 6, BeliefCemConfig(**SMALL, particles=1, deterministic=True))
    sketch = make_sketch()
    plain = context_for(compact, sketch)
    start = compact.initial_state(compact.units.physical_to_compact(plain.position[:, -1]), torch.ones(1, dtype=torch.bool))
    context = context_for(compact, sketch, belief=start)
    increments = torch.randn(6, 3, 6, generator=torch.Generator().manual_seed(2)).clamp(-1, 1)
    expected, expected_feasible = joint.score(increments, context)
    draws = cem.draw_particles(context.belief, torch.Generator().manual_seed(0))
    score, feasible, particle = cem.score_particles(increments, context, draws)
    assert torch.allclose(score, expected, atol=1e-5), (score - expected).abs().max()
    assert torch.equal(feasible, expected_feasible)
    assert torch.equal(score.argsort(), expected.argsort())


def test_a_limit_violation_in_any_particle_rejects_the_candidate():
    cem, compact = planner()
    sketch = make_sketch()
    plain = context_for(compact, sketch)
    wide = plain.belief
    cov = wide.cov.clone()
    cov[..., :6, :6] = torch.eye(6) * 100.0  # joint uncertainty far beyond the +-2 limits
    uncertain = type(wide)(wide.log_pi, wide.mean, cov, wide.h, wide.age_s, wide.regimes)
    context = context_for(compact, sketch, belief=uncertain)
    increments = torch.zeros(2, 3, 6)
    draws = cem.draw_particles(uncertain, torch.Generator().manual_seed(0))
    _, feasible, _ = cem.score_particles(increments, context, draws)
    assert not feasible.any()
    with pytest.raises(RuntimeError, match="not even hold"):
        cem.plan(context, seed=0)
    # The mean alone is feasible: the deterministic ablation accepts hold.
    det = BeliefCem(cem.model, compact, LIMITS, (0.2,) * 6, BeliefCemConfig(**SMALL, particles=1, deterministic=True))
    _, feasible, _ = det.score_particles(increments, context, det.draw_particles(uncertain, torch.Generator()))
    assert feasible.all()


def test_soft_limit_term_and_config_refusals():
    cem, compact = planner(limit_weight=1.0, limit_margin=(0.5,) * 6)
    joints = torch.tensor([[[[1.75] * 6]]])  # 0.25 from the limit with margin 0.5
    assert cem._soft_limit(joints).item() == pytest.approx(0.25)
    with pytest.raises(ValueError):
        BeliefCemConfig(limit_weight=1.0)
    with pytest.raises(ValueError):
        BeliefCemConfig(epistemic_weight=0.1)
    with pytest.raises(ValueError):
        BeliefCemConfig(deterministic=True, particles=4)
    with pytest.raises(ValueError, match="belief inputs"):
        BeliefCem(make_model(), compact, LIMITS, (0.2,) * 6, BeliefCemConfig(**SMALL))


def test_two_regime_belief_plans_with_switching_draws():
    cem, compact = planner()
    sketch = make_sketch()
    one = context_for(compact, sketch).belief
    two = type(one)(
        torch.log(torch.tensor([[0.6, 0.4]])), one.mean.repeat(1, 2, 1), one.cov.repeat(1, 2, 1, 1), one.h, one.age_s, (0, 1)
    )
    context = context_for(compact, sketch, belief=two)
    plan = cem.plan(context, seed=1)
    assert torch.isfinite(torch.tensor(plan.cost))
    # Sticky prior of eq. 14 with a 600 s residence: over 0.2 s a particle stays unless
    # its uniform falls in the top epsilon = 1 - exp(-0.2 / 600) of the row's CDF.
    epsilon = 1 - torch.exp(torch.tensor(-0.2 / 600)).item()
    slot = torch.tensor([0, 0, 1, 1])
    uniforms = torch.tensor([0.5, 1 - epsilon / 2, 0.5, epsilon / 2])
    assert cem._switch(slot, uniforms, torch.full((4,), 0.2), 2).tolist() == [0, 1, 1, 0]
