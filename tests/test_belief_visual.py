"""The belief inside the visual predictor (B4): residual expansion, causality, gradient
boundary, cached parity and the stage-3 trainer with the B4 loss.

Runs on the real tiny predictor (C=32, grid 3x4) over both attention layouts.
"""

from functools import partial

import pytest
import torch

from conftest import SERVO, UNITS, make_batch, make_model, set_adapters_nonzero
from robojepa_ttt.belief.compact import CompactBelief, CompactConfig
from robojepa_ttt.belief.sketch import ObservationSketch
from robojepa_ttt.belief.visual import (
    belief_conditioning,
    belief_dim,
    belief_visual_loss,
    check_contract,
    compact_steps,
    consistency_loss,
    expand_residual_for_belief,
)
from robojepa_ttt.learning.offline_train import OfflineConfig, OfflineTrainer

VIEWS = (0, 2)  # make_batch's view ids


def make_compact(seed=0, dtype=torch.float32, perturb=0.05, view_ids=VIEWS):
    torch.manual_seed(seed)
    config = CompactConfig(
        joint_sensor_std=(0.01,) * 6,
        residual_bound=(0.2,) * 6,
        view_ids=view_ids,
        visual_per_view=4,
        fast_dims=6,
        slow_dims=2,
        recurrent_hidden=16,
        command_hidden=8,
        regime_dim=4,
        transition_hidden=(16, 16),
        observation_hidden=(16,),
    )
    compact = CompactBelief(config, UNITS, SERVO, stride=6, tick_s=1 / 30, history_ticks=30).to(dtype)
    if perturb:
        # A trained-looking model: nonzero residual dynamics and observation rows.
        generator = torch.Generator().manual_seed(seed + 100)
        with torch.no_grad():
            for name, p in compact.named_parameters():
                if not name.startswith("correction"):
                    p.add_(perturb * torch.randn(p.shape, generator=generator, dtype=p.dtype))
    return compact


def make_sketch(seed=0, view_ids=VIEWS):
    generator = torch.Generator().manual_seed(seed)
    sketch = ObservationSketch(view_ids, token_dim=32, components=4)
    frames = {v: torch.randn(40, 3, 4, 32, generator=generator) for v in view_ids}
    sketch.fit(frames)
    return sketch


def b4_model(seed=0, compact=None, nonzero_belief=False):
    """A B1 model with nonzero adapters, its residual expanded with belief inputs."""
    model = make_model(seed)
    set_adapters_nonzero(model)
    compact = compact or make_compact()
    model.residual = expand_residual_for_belief(model.residual, belief_dim(compact))
    if nonzero_belief:
        generator = torch.Generator().manual_seed(seed + 7)
        with torch.no_grad():
            for regime in model.residual.regimes:
                first = regime.modulation[0]
                first.weight[:, : belief_dim(compact)] = 0.3 * torch.randn(
                    first.weight.shape[0], belief_dim(compact), generator=generator
                )
    return model


def conditioning(compact, sketch, batch, context, horizon):
    steps = compact_steps(compact, sketch, batch, context + horizon)
    return belief_conditioning(compact, steps, context, horizon)


# ---------------------------------------------------------------- expansion


def test_zero_belief_columns_reproduce_the_b1_residual_exactly():
    model = make_model()
    set_adapters_nonzero(model)
    batch = make_batch()
    with torch.no_grad():
        reference = model.rollout(batch, 3, 3)
        realized_reference = model.realized_forward(batch, 6)
    compact = make_compact()
    model.residual = expand_residual_for_belief(model.residual, belief_dim(compact))
    belief = torch.randn(2, 5, belief_dim(compact), generator=torch.Generator().manual_seed(4)) * 3
    with torch.no_grad():
        again = model.rollout(batch, 3, 3, belief=belief)
        realized = model.realized_forward(batch, 6, belief=torch.randn(2, 5, belief_dim(compact)))
    # Zero columns add exact zeros to every dot product; the remaining sums are the
    # same numbers in the same order, so the result is bitwise identical on CPU.
    assert torch.equal(reference, again)
    assert torch.equal(realized_reference, realized)
    with pytest.raises(ValueError, match="pass belief"):
        model.rollout(batch, 3, 3)
    with pytest.raises(ValueError, match="already has belief"):
        expand_residual_for_belief(model.residual, 4)


def test_belief_columns_are_first_in_the_condition_vector():
    model = make_model()
    compact = make_compact()
    D = belief_dim(compact)
    old_first = model.residual.regimes[0].modulation[0].weight.detach().clone()
    model.residual = expand_residual_for_belief(model.residual, D)
    new_first = model.residual.regimes[0].modulation[0].weight
    assert new_first.shape[1] == old_first.shape[1] + D
    assert torch.equal(new_first[:, :D], torch.zeros_like(new_first[:, :D]))
    assert torch.equal(new_first[:, D:], old_first)


# ---------------------------------------------------------------- conditioning


def test_conditioning_shapes_and_meaning():
    """Position t carries [h_t, x_{t+1}^-]; motion is the prior joint mean of the
    future keyframes, which for an untrained transition is the servo prediction."""
    compact = make_compact(perturb=0.0)
    sketch = make_sketch()
    model = make_model()
    batch = make_batch()
    context, horizon = 3, 3
    cond = conditioning(compact, sketch, batch, context, horizon)
    D = belief_dim(compact)
    assert cond.belief.shape == (2, context + horizon - 1, D)
    assert cond.motion.shape == (2, horizon, 6)
    assert cond.predicted_visual.shape == (2, context + horizon - 1, 8)
    assert not cond.view_valid[:, context - 1 :].any(), "future frames are never observed"
    # Untrained compact transition = servo. The rollout starts from the filtered joint
    # mean of frame c-1 (not the raw reading), so compare with the servo from there.
    steps = compact_steps(compact, sketch, batch, context + horizon)
    state = compact.initial_state(steps[0].observation[:, :6], torch.ones(2, dtype=torch.bool))
    state, _ = compact.step(state, steps[0], predict=False)
    for k in range(1, context):
        state, _ = compact.step(state, steps[k])
    from_filter = dict(batch)
    from_filter["position"] = batch["position"].clone()
    from_filter["position"][:, context - 1] = compact.units.compact_to_physical(state.mean[:, 0, :6])
    servo = model.predicted_positions(from_filter, context, horizon)
    assert torch.allclose(cond.motion, servo, atol=1e-5), (cond.motion - servo).abs().max()
    assert not torch.allclose(state.mean[:, 0, :6], steps[context - 1].observation[:, :6]), "the reading was filtered"
    # The prior state block of the last position is the motion's last keyframe.
    assert torch.allclose(
        compact.units.compact_to_physical(cond.belief[:, -1, compact.config.recurrent_hidden :][:, :6]), cond.motion[:, -1], atol=1e-5
    )
    assert len(steps) == context + horizon


def test_compact_steps_align_command_blocks_with_the_visual_model():
    """Block U_{k-1} of compact step k must be the same ticks encode_commands uses."""
    compact = make_compact()
    sketch = make_sketch()
    batch = make_batch(frames=5)
    steps = compact_steps(compact, sketch, batch, 5)
    for k in range(1, 5):
        block = batch["issued"][:, 30 + (k - 1) * 6 : 30 + k * 6]
        assert torch.equal(steps[k].issued[:, -6:], block)
        assert torch.equal(steps[k].issued, batch["issued"][:, (k - 1) * 6 : 30 + k * 6])
        assert torch.allclose(steps[k].dt, torch.full((2,), 0.2))
    # Keyframe 0: the last 30 ticks are the logged history, the first 6 a held, invalid pad.
    assert torch.equal(steps[0].issued[:, 6:], batch["issued"][:, :30])
    assert torch.equal(steps[0].issued[:, :6], batch["position"][:, :1].expand(-1, 6, -1))
    assert not steps[0].command_valid[:, :6].any() and steps[0].command_valid[:, 6:].all()


def test_contract_mismatches_are_refused():
    compact = make_compact(view_ids=(0, 1))
    with pytest.raises(ValueError, match="contract views"):
        check_contract(compact, make_sketch())
    model = make_model(stride=5)
    with pytest.raises(ValueError, match="stride"):
        check_contract(make_compact(), make_sketch(), model)


# ---------------------------------------------------------------- causality


def test_belief_rollout_is_causal():
    """Frames and joints after the context change neither the belief inputs nor the
    predictions; changing the belief inputs does change the predictions."""
    compact = make_compact()
    sketch = make_sketch()
    model = b4_model(compact=compact, nonzero_belief=True)
    batch = make_batch()
    context, horizon = 3, 4
    cond = conditioning(compact, sketch, batch, context, horizon)
    with torch.no_grad():
        reference = model.rollout(batch, context, horizon, motion=cond.motion, belief=cond.belief)

    tampered = {k: v.clone() for k, v in batch.items()}
    tampered["features"][:, context:] = 99.0
    tampered["position"][:, context:] = 99.0
    cond2 = conditioning(compact, sketch, tampered, context, horizon)
    assert torch.equal(cond.belief, cond2.belief) and torch.equal(cond.motion, cond2.motion)
    with torch.no_grad():
        again = model.rollout(tampered, context, horizon, motion=cond2.motion, belief=cond2.belief)
    assert torch.equal(reference, again)

    # Commands issued after the block that drives frame `context` cannot reach its prediction.
    later = {k: v.clone() for k, v in batch.items()}
    cut = 30 + context * 6
    later["issued"][:, cut:] += 0.5
    cond3 = conditioning(compact, sketch, later, context, horizon)
    assert torch.equal(cond3.belief[:, : context], cond.belief[:, : context])
    with torch.no_grad():
        moved = model.rollout(later, context, horizon, motion=cond3.motion, belief=cond3.belief)
    assert torch.allclose(moved[:, 0], reference[:, 0], atol=1e-6)
    assert not torch.allclose(moved[:, 1:], reference[:, 1:], atol=1e-4)

    # The belief itself matters, with motion held fixed.
    shifted = cond.belief.clone()
    shifted[:, context - 1 :] += 1.0
    with torch.no_grad():
        changed = model.rollout(batch, context, horizon, motion=cond.motion, belief=shifted)
    assert not torch.allclose(changed, reference, atol=1e-4)
    # Changing only a future position's belief leaves earlier predictions untouched.
    last_only = cond.belief.clone()
    last_only[:, -1] += 1.0
    with torch.no_grad():
        partial_change = model.rollout(batch, context, horizon, motion=cond.motion, belief=last_only)
    assert torch.allclose(partial_change[:, :-1], reference[:, :-1], atol=1e-6)
    assert not torch.allclose(partial_change[:, -1], reference[:, -1], atol=1e-4)


def test_belief_motion_drives_the_rollout():
    compact = make_compact()
    sketch = make_sketch()
    model = b4_model(compact=compact)
    batch = make_batch()
    cond = conditioning(compact, sketch, batch, 3, 3)
    with torch.no_grad():
        a = model.rollout(batch, 3, 3, motion=cond.motion, belief=cond.belief)
        b = model.rollout(batch, 3, 3, motion=cond.motion + 0.2, belief=cond.belief)
    assert not torch.allclose(a, b, atol=1e-4)
    with pytest.raises(ValueError, match="motion covers"):
        model.rollout(batch, 3, 3, motion=cond.motion[:, :2], belief=cond.belief)
    with pytest.raises(ValueError, match="belief covers"):
        model.rollout(batch, 3, 3, motion=cond.motion, belief=cond.belief[:, :3])


# ---------------------------------------------------------------- gradients


def test_visual_loss_never_reaches_the_compact_model():
    """Spec 9.4: visual losses must not train the servo bridge or the filter."""
    compact = make_compact()
    for p in compact.parameters():
        p.requires_grad_(True)  # even if someone forgot to freeze it
    sketch = make_sketch()
    model = b4_model(compact=compact)
    model.set_trainable("offline")
    loss, parts = belief_visual_loss(model, make_batch(), 3, 2, OfflineConfig(), compact=compact, sketch=sketch)
    assert set(parts) == {"cmd_tf", "rollout", "consistency", "realized"}
    loss.backward()
    assert all(p.grad is None for p in compact.parameters())
    # The belief columns of the residual do receive gradient: the belief is used.
    first = model.residual.regimes[0].modulation[0]
    assert first.weight.grad[:, : belief_dim(compact)].abs().sum() > 0


def test_consistency_loss_matches_eq_28_and_skips_absent_views():
    sketch = make_sketch()
    generator = torch.Generator().manual_seed(2)
    tokens = torch.randn(2, 3, 2, 3, 4, 32, generator=generator, requires_grad=True)
    compact_visual = torch.randn(2, 3, 8, generator=generator, requires_grad=True)
    view_ids = torch.tensor([[0, 2], [2, 5]])  # the second episode lacks contract view 0
    loss = consistency_loss(sketch, tokens, compact_visual, view_ids)
    terms, count = [], 0
    for b, ids in enumerate(view_ids.tolist()):
        for slot, v in enumerate(sketch.view_ids):
            if v in ids:
                coords = sketch.transform_slot(tokens[b, :, ids.index(v)], slot)
                terms.append((coords - compact_visual[b, :, slot * 4 : slot * 4 + 4]).abs().sum())
                count += coords.numel()
    assert count == (3 * 4) * 3
    assert torch.allclose(loss, sum(terms) / count)
    loss.backward()
    assert compact_visual.grad is None, "the compact target is stopped"
    assert tokens.grad[1, :, 1].abs().sum() == 0, "a view outside the contract carries no loss"
    assert consistency_loss(sketch, tokens, compact_visual, torch.tensor([[5, 6], [5, 6]])) is None


# ---------------------------------------------------------------- caching


@pytest.mark.parametrize("context,horizon", [(3, 4), (1, 5), (6, 6)])
def test_kv_generation_matches_full_forward_with_belief(context, horizon):
    compact = make_compact()
    sketch = make_sketch()
    model = b4_model(compact=compact, nonzero_belief=True)
    batch = make_batch(frames=context + horizon)
    cond = conditioning(compact, sketch, batch, context, horizon)
    with torch.no_grad():
        cached = model.rollout(batch, context, horizon, motion=cond.motion, belief=cond.belief, use_kv_generation=True)
        full = model.rollout(batch, context, horizon, motion=cond.motion, belief=cond.belief, use_kv_generation=False)
    assert torch.allclose(cached, full, atol=2e-4), (cached - full).abs().max()


def test_step_cached_with_belief_matches_rollout():
    """The planner's per-step cached path, with per-position belief, equals rollout()."""
    compact = make_compact()
    sketch = make_sketch()
    model = b4_model(compact=compact, nonzero_belief=True)
    batch = make_batch()
    context, horizon = 3, 3
    cond = conditioning(compact, sketch, batch, context, horizon)
    actions, states = model.sequence_inputs(batch, context, horizon, cond.motion)
    command = model.encode_commands(batch, context + horizon - 1)
    prefix = model.prime_context(batch["features"][:, : context - 1], actions[:, : context - 1], states[:, : context - 1], batch["view_ids"])
    frame = batch["features"][:, context - 1 : context]
    for k in range(horizon):
        t = context - 1 + k
        frame, prefix = model.step_cached(
            frame, actions[:, t : t + 1], states[:, t : t + 1], batch["view_ids"], command[:, t : t + 1], prefix, t, cond.belief[:, t : t + 1]
        )
    with torch.no_grad():
        full = model.rollout(batch, context, horizon, motion=cond.motion, belief=cond.belief, use_kv_generation=False)
    assert torch.allclose(frame, full[:, -1:], atol=2e-4)


# ---------------------------------------------------------------- training


def test_offline_trainer_runs_the_b4_loss_and_lowers_it_on_a_fixed_batch(tmp_path):
    compact = make_compact(dtype=torch.float64)  # also exercises the dtype boundary
    for p in compact.parameters():
        p.requires_grad_(False)
    sketch = make_sketch()
    model = b4_model(compact=make_compact())
    config = OfflineConfig(
        sequence_frames=6,
        effective_batch=2,
        microbatch=2,
        max_updates=40,
        warmup_updates=1,
        curriculum=((40, 2),),
        lora_lr=1e-2,
        residual_lr=1e-2,
        validation_interval=1000,
    )
    loss_fn = partial(belief_visual_loss, compact=compact, sketch=sketch)
    trainer = OfflineTrainer(model, config, tmp_path, loss_fn=loss_fn)
    batch = make_batch(batch=2, frames=6)
    compact_before = {k: v.clone() for k, v in compact.state_dict().items()}

    def fixed_loss():
        with torch.no_grad():
            return float(loss_fn(model, batch, 3, 2, config)[0])

    before = fixed_loss()
    records = [trainer.train_step(lambda: batch) for _ in range(30)]
    after = fixed_loss()
    assert all(r["horizon"] == 2 and "consistency" in r for r in records)
    assert after < 0.85 * before, (before, after)
    assert all(torch.equal(compact_before[k], v) for k, v in compact.state_dict().items())
