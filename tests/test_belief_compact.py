"""Compact belief model: structure, ordering, causality, masking, gradients; the sketch."""

import math

import numpy as np
import pytest
import torch

from robojepa_ttt.belief.compact import BeliefState, CompactBelief, CompactConfig, StepInput
from robojepa_ttt.belief.sketch import ObservationSketch, pooled_tokens
from robojepa_ttt.models.servo import ServoParams
from robojepa_ttt.robot_io.so101_units import So101Units

UNITS = So101Units(gripper_direction_verified=True)
SERVO = ServoParams((0.1,) * 6, (2,) * 6)


@pytest.fixture(autouse=True)
def attention_layout():
    yield None


def small_config(**changes):
    values = dict(
        joint_sensor_std=(0.01,) * 6,
        residual_bound=(0.2,) * 6,
        view_ids=(6, 1),
        visual_per_view=4,
        fast_dims=6,
        slow_dims=2,
        recurrent_hidden=16,
        command_hidden=8,
        regime_dim=4,
        transition_hidden=(16, 16),
        observation_hidden=(16,),
    )
    values.update(changes)
    return CompactConfig(**values)


def make_belief(seed=0, dtype=torch.float64, **changes):
    torch.manual_seed(seed)
    return CompactBelief(small_config(**changes), UNITS, SERVO, stride=6, tick_s=1 / 30, history_ticks=30).to(dtype)


def make_step(model, batch=2, seed=1, dtype=torch.float64, mask=None):
    g = torch.Generator().manual_seed(seed)
    c = model.config
    L = model.history_ticks + model.stride
    obs = torch.randn(batch, c.obs_dim, generator=g, dtype=dtype) * 0.5
    return StepInput(
        observation=obs,
        obs_mask=torch.ones(batch, c.obs_dim, dtype=torch.bool) if mask is None else mask,
        issued=(torch.randn(batch, L, 6, generator=g, dtype=dtype).cumsum(1) * 0.01),
        tick_dt=torch.full((batch, L), 1 / 30, dtype=dtype),
        command_valid=torch.ones(batch, L, dtype=torch.bool),
        dt=torch.full((batch,), 0.2, dtype=dtype),
    )


def run(model, steps, state=None, regimes=(0,)):
    if state is None:
        state = model.initial_state(steps[0].observation[:, :6], torch.ones(steps[0].observation.shape[0], dtype=torch.bool), regimes)
    outputs = []
    for t, step in enumerate(steps):
        state, out = model.step(state, step, predict=t > 0)
        outputs.append(out)
    return state, outputs


def test_dimensions_follow_the_spec_at_full_size():
    c = CompactConfig(joint_sensor_std=(0.01,) * 6, residual_bound=(0.1,) * 6)
    assert c.state_dim == 64 and c.obs_dim == 134 and c.gru_input_dim == 203
    with pytest.raises(ValueError):
        CompactConfig(joint_sensor_std=(0.01,) * 5, residual_bound=(0.1,) * 6)


def test_untrained_transition_is_servo_plus_decay_and_its_jacobian_is_exact():
    model = make_belief()
    step = make_step(model)
    state = model.initial_state(step.observation[:, :6], torch.ones(2, dtype=torch.bool))
    x = state.mean.clone()
    x[..., 6:] = torch.randn_like(x[..., 6:])
    out = model.transition(x, state.h, model.encode_commands(step), step.dt, step.issued, step.tick_dt, (0,))
    nominal = model.nominal_joints(x[:, 0, :6], step.issued, step.tick_dt)
    assert torch.allclose(out[:, 0, :6], nominal)
    assert torch.allclose(out[:, 0, 6:12], math.exp(-0.2 / 10) * x[:, 0, 6:12])
    assert torch.equal(out[:, 0, 12:], x[:, 0, 12:])
    from robojepa_ttt.belief.imm import batched_jacobian

    jac = batched_jacobian(
        lambda z: model.transition(z, state.h, model.encode_commands(step), step.dt, step.issued, step.tick_dt, (0,)), x, False
    )
    gain = 1 - math.exp(-(1 / 30) / 0.1)
    assert torch.allclose(jac[0, 0, :6, :6], torch.eye(6, dtype=torch.float64) * (1 - gain) ** 6, atol=1e-12)
    assert torch.allclose(jac[0, 0, :6, 6:], torch.zeros(6, 8, dtype=torch.float64))


def test_observation_rows_noise_bounds_and_process_scaling():
    model = make_belief()
    with torch.no_grad():
        model.observation_matrix.bias.fill_(100.0)  # tanh saturates at the bound
    H, b = model.observation_model(torch.zeros(1, 16, dtype=torch.float64), torch.zeros(1, 8, dtype=torch.float64), (0,))
    assert torch.equal(H[0, 0, :6], torch.eye(6, 14, dtype=torch.float64)) and torch.equal(b[0, 0, :6], torch.zeros(6, dtype=torch.float64))
    assert H[0, 0, 6:].abs().max() <= 0.1 + 1e-12
    q = model.process_variance((0,), torch.tensor([0.2, 0.4], dtype=torch.float64))
    assert q[0, 0, 0].item() == pytest.approx(1e-4) and q[0, 0, 6].item() == pytest.approx(1e-3) and q[0, 0, -1].item() == pytest.approx(1e-5)
    assert torch.allclose(q[1], 2 * q[0])
    r = model.observation_variance(None, 1)
    assert r[0, 0].item() == pytest.approx(1e-4) and r[0, 6].item() == pytest.approx(0.04)


def test_first_frame_does_not_count_the_joint_reading_twice():
    model = make_belief()
    step = make_step(model)
    state = model.initial_state(step.observation[:, :6], torch.ones(2, dtype=torch.bool))
    after, out = model.step(state, step, predict=False)
    joint_var = torch.diagonal(after.cov, dim1=-2, dim2=-1)[:, 0, :6]
    # Untrained H has zero visual rows, so only a second joint update could shrink this.
    assert torch.allclose(joint_var, torch.full_like(joint_var, 1e-4))
    assert torch.equal(out.valid_count, torch.full((2,), 8))


def test_prediction_of_y_t_is_causal():
    """The predicted observation and every earlier output ignore y_t and later data."""
    model = make_belief()
    torch.manual_seed(5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    steps = [make_step(model, seed=10 + t) for t in range(5)]
    _, ref = run(model, steps)
    tampered = [s for s in steps]
    for t in (3, 4):
        s = steps[t]
        tampered[t] = StepInput(s.observation + 7.0, s.obs_mask, s.issued, s.tick_dt, s.command_valid, s.dt)
    _, again = run(model, tampered)
    for t in range(3):
        assert torch.equal(ref[t].log_predictive, again[t].log_predictive)
    assert torch.equal(ref[3].predicted_observation, again[3].predicted_observation), "y_3 leaked into its own prediction"
    assert not torch.equal(ref[3].log_predictive, again[3].log_predictive)


def test_missing_observations_predict_only_and_never_tighten_spuriously():
    model = make_belief()
    step = make_step(model)
    state, _ = run(model, [step, make_step(model, seed=2)])
    blind, out = model.predict_only(state, make_step(model, seed=3))
    assert torch.equal(out.valid_count, torch.zeros(2, dtype=torch.long))
    assert torch.allclose(blind.log_pi, out.log_prior)
    # Joint variance grows by at least Q when nothing is observed (servo factor < 1 shrinks the carried part).
    before = torch.diagonal(state.cov, dim1=-2, dim2=-1)[:, 0, 12:]
    after = torch.diagonal(blind.cov, dim1=-2, dim2=-1)[:, 0, 12:]
    assert (after >= before).all(), "slow coordinates cannot become more certain without data"
    assert not out.motion_allowed.any()
    assert torch.equal(blind.age_s, (state.age_s + 0.2).clamp(max=10.0))


def test_padding_steps_leave_the_state_unchanged():
    model = make_belief()
    state, _ = run(model, [make_step(model)])
    step = make_step(model, seed=4)
    step.step_valid = torch.tensor([True, False])
    after, out = model.step(state, step)
    assert torch.equal(after.mean[1], state.mean[1]) and torch.equal(after.h[1], state.h[1])
    assert not torch.equal(after.mean[0], state.mean[0])
    assert out.valid_count[1] == 0


def test_masked_views_and_joints_flow_through_the_step():
    model = make_belief()
    state, _ = run(model, [make_step(model)])
    mask = torch.ones(2, model.config.obs_dim, dtype=torch.bool)
    mask[0, 6:10] = False  # view 6 missing
    mask[1, :6] = False  # joints missing
    step = make_step(model, seed=6, mask=mask)
    step.observation[0, 6:10] = float("nan")
    after, out = model.step(state, step, compute_nis=True)
    assert torch.isfinite(after.mean).all() and torch.isfinite(out.log_predictive).all()
    assert out.valid_count.tolist() == [10, 8]
    assert out.motion_allowed.tolist() == [True, False]
    assert after.age_s[0, 0] > 0 and after.age_s[0, 1] == 0
    assert out.nis["view6"][1][0] == 0 and out.nis["joints"][1][1] == 0


def test_regime_bank_runs_and_weights_stay_normalised():
    model = make_belief()
    steps = [make_step(model, seed=20 + t) for t in range(4)]
    state, outs = run(model, steps, regimes=(0, 2))
    assert state.mean.shape[1] == 2
    assert torch.allclose(state.log_pi.exp().sum(-1), torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError):
        model.initial_state(steps[0].observation[:, :6], torch.ones(2, dtype=torch.bool), (0, 0))


def test_gradients_reach_every_offline_parameter_but_not_the_online_correction():
    model = make_belief()
    torch.manual_seed(7)
    with torch.no_grad():  # move off the zero-initialised saddle so all paths carry gradient
        for layer in (model.transition_out, model.observation_matrix, model.observation_bias):
            layer.weight.add_(0.05 * torch.randn_like(layer.weight))
    _, outs = run(model, [make_step(model, seed=30 + t) for t in range(4)])
    loss = -sum(o.log_predictive.sum() for o in outs[1:])
    loss.backward()
    names = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    for required in ("transition_body.0.weight", "transition_out.weight", "observation_matrix.weight", "history.weight_ih",
                     "process_raw", "visual_std_raw", "command_encoder.gru.weight_ih_l0", "regime_embedding.weight"):
        assert required in names, required
    assert {id(p) for p in model.online_parameters()}.isdisjoint({id(p) for p in model.offline_parameters()})


def test_init_refuses_missing_joints_and_quality_must_inflate():
    model = make_belief()
    with pytest.raises(ValueError, match="joint"):
        model.initial_state(torch.zeros(1, 6, dtype=torch.float64), torch.tensor([False]))
    with pytest.raises(ValueError, match="inflate"):
        model.observation_variance(torch.full((1, 2), 0.5), 1)


def test_float32_matches_float64():
    m64 = make_belief(dtype=torch.float64)
    m32 = make_belief(dtype=torch.float32)
    steps64 = [make_step(m64, seed=40 + t) for t in range(4)]
    steps32 = [StepInput(s.observation.float(), s.obs_mask, s.issued.float(), s.tick_dt.float(), s.command_valid, s.dt.float()) for s in steps64]
    _, o64 = run(m64, steps64)
    _, o32 = run(m32, steps32)
    for a, b in zip(o64, o32):
        assert torch.allclose(a.log_predictive, b.log_predictive.double(), rtol=1e-4, atol=1e-3)


# ---------------------------------------------------------------- sketch


def test_sketch_recovers_a_low_rank_subspace_and_standardises():
    g = torch.Generator().manual_seed(0)
    C, k, N = 16, 3, 400
    basis = torch.linalg.qr(torch.randn(12 * C, k, generator=g))[0]
    latent = torch.randn(N, k, generator=g) * torch.tensor([5.0, 3.0, 2.0])
    pooled_target = latent @ basis.T
    # Build tokens whose pooled-LN0 features are close to a k-dim subspace: repeat each cell over a 2x2 patch.
    cells = pooled_target.reshape(N, 3, 4, C) + 50.0  # offset keeps LN0 well conditioned
    tokens = cells.repeat_interleave(2, 1).repeat_interleave(2, 2)
    sketch = ObservationSketch((6,), token_dim=C, components=k)
    report = sketch.fit({6: tokens})
    assert report[6]["explained_total"] > 0.9
    coords = sketch.transform_slot(tokens, 0)
    assert torch.allclose(coords.mean(0), torch.zeros(k), atol=1e-4)
    assert torch.allclose(coords.std(0), torch.ones(k), atol=1e-3)
    pooled = pooled_tokens(tokens)
    projector = sketch.basis[0] @ sketch.basis[0].T
    residual = (pooled - sketch.token_mean[0]) - (pooled - sketch.token_mean[0]) @ projector
    assert residual.norm() / (pooled - sketch.token_mean[0]).norm() < 0.3


def test_sketch_contract_masks_missing_views_and_ignores_others():
    g = torch.Generator().manual_seed(1)
    C = 8
    frames = {v: torch.randn(50, 6, 8, C, generator=g) for v in (6, 1)}
    sketch = ObservationSketch((6, 1), token_dim=C, components=4)
    sketch.fit(frames)
    tokens = torch.randn(2, 2, 6, 8, C, generator=g)
    visual, valid = sketch.observe(tokens, [6, 2])  # view 1 absent, wrist 2 not in the contract
    assert valid.tolist() == [[True, False], [True, False]]
    assert torch.equal(visual[:, 4:], torch.zeros(2, 4))
    frame_valid = torch.tensor([[True, True], [False, True]])
    visual, valid = sketch.observe(tokens, [6, 1], frame_valid)
    assert valid.tolist() == [[True, True], [False, True]] and torch.equal(visual[1, :4], torch.zeros(4))
    assert sketch.digest() == sketch.digest()
    with pytest.raises(RuntimeError):
        ObservationSketch((6,), token_dim=C, components=4).transform_slot(tokens[:, 0], 0)
    with pytest.raises(ValueError):
        ObservationSketch((6, 6))


def test_sketch_buffers_are_not_parameters():
    sketch = ObservationSketch((6,), token_dim=8, components=2)
    assert list(sketch.parameters()) == []


def test_filter_refuses_tf32_on_cuda(monkeypatch):
    """NGC containers enable TF32 by default; it cost 0.4% relative error on GPU."""
    from robojepa_ttt.belief.compact import check_full_precision

    cuda = torch.device("cuda")
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
        check_full_precision(cuda)
        check_full_precision(torch.device("cpu"))
        monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
        with pytest.raises(RuntimeError, match="TF32"):
            check_full_precision(cuda)
        monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
        torch.set_float32_matmul_precision("high")
        with pytest.raises(RuntimeError, match="precision"):
            check_full_precision(cuda)
        check_full_precision(torch.device("cpu"))  # CPU has no TF32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous[0]
        torch.set_float32_matmul_precision(previous[1])
