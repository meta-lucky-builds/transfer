"""Sequence building and differentiable-filter training on a simulated arm.

The synthetic camera is a smooth function of joint angles, not images. A falling
validation NLL proves the training machinery works, nothing about a real robot.
"""

import random

import numpy as np
import pytest
import torch

from robojepa_ttt.belief.compact import CompactBelief, CompactConfig
from robojepa_ttt.belief.data import BeliefSequences, SequenceSpec, command_block, joint_loss_scale, residual_bound_from_data, sequence_starts
from robojepa_ttt.belief.sketch import ObservationSketch
from robojepa_ttt.belief.train import CompactTrainConfig, CompactTrainer, sequence_loss
from robojepa_ttt.evaluation.perturbation import SimulatedArm, SyntheticCamera
from robojepa_ttt.learning.windows import Episode
from robojepa_ttt.models.servo import ServoParams
from robojepa_ttt.robot_io.so101_units import So101Units

UNITS = So101Units(gripper_direction_verified=True)
SERVO = ServoParams((0.1,) * 6, (2,) * 6)
SPEC = SequenceSpec(scored=6, burn_in=3)


@pytest.fixture(autouse=True)
def attention_layout():
    yield None


def simulate_episode(name, ticks=240, seed=0, camera=None, boundary_at=None):
    arm = SimulatedArm(tau_s=0.1, seed=seed)
    rng = np.random.default_rng(seed)
    camera = camera or SyntheticCamera(views=2, height=3, width=4, channels=16)
    episode = Episode(name, "cal", "enc", 2, capacity=16)
    frames = {}
    target = np.zeros(6)
    for n in range(ticks):
        if boundary_at is not None and n == boundary_at:
            episode.mark_boundary()
        if n % 15 == 0:
            target = np.clip(target + rng.normal(0, 0.15, 6), -0.8, 0.8)
        measured = arm.step(target)
        frames[n] = camera.features(arm.position)
        episode.append(n / 30, measured, True, target.copy(), True, np.ones(2, bool))
    return episode, frames


def build(seeds, camera, spec=SPEC):
    episodes, frames = [], {}
    for s in seeds:
        e, f = simulate_episode(f"e{s}", seed=s, camera=camera)
        episodes.append(e)
        frames[e.episode_id] = f
    return episodes, frames


def small_model(seed=0, bound=(0.3,) * 6):
    torch.manual_seed(seed)
    config = CompactConfig(
        joint_sensor_std=(0.003,) * 6, residual_bound=bound, view_ids=(6, 1), visual_per_view=4, fast_dims=6, slow_dims=2,
        recurrent_hidden=24, command_hidden=8, regime_dim=4, transition_hidden=(32, 32), observation_hidden=(32,),
    )
    return CompactBelief(config, UNITS, SERVO, stride=6, tick_s=1 / 30, history_ticks=30)


def fitted_sketch(episodes, frames):
    sketch = ObservationSketch((6, 1), token_dim=16, components=4)
    stacks = {v: torch.stack([frames[e.episode_id][k][slot] for e in episodes for k in range(0, e.length, 6)]) for slot, v in enumerate((6, 1))}
    sketch.fit(stacks)
    return sketch


def test_command_block_is_causal_and_pads_before_the_segment():
    e, _ = simulate_episode("x", ticks=90, boundary_at=40)
    block = command_block(e, 48, SPEC)
    assert block["issued"].shape == (36, 6)
    # Ticks 12..39 are in the previous segment: held and invalid. 40..47 are commands.
    assert not block["command_valid"][:28].any() and block["command_valid"][28:].all()
    assert np.allclose(block["issued"][:28], e.position[40])
    assert np.allclose(block["issued"][28:], e.issued[40:48])
    e.issued[48:] = 99.0  # the future must not matter
    assert np.allclose(command_block(e, 48, SPEC)["issued"], block["issued"])


def test_sequence_starts_respect_segments_and_shorten_burn_in():
    e, _ = simulate_episode("x", ticks=200, boundary_at=60)
    starts = list(sequence_starts(e, SPEC))
    assert starts, "expected admissible sequences"
    for s, a in starts:
        assert s % 6 == 0 and a % 6 == 0 and 0 <= (a - s) // 6 <= SPEC.burn_in
        last = a + SPEC.scored * 6
        assert e.segment[s] == e.segment[last]
    assert (0, 0) in starts, "an episode start needs no burn-in"
    assert any(s == 60 and a == 60 for s, a in starts), "burn-in must stop at the reset"


def test_batches_pad_score_and_never_score_the_first_step():
    camera = SyntheticCamera(views=2, height=3, width=4, channels=16)
    episodes, frames = build([0], camera)
    sketch = fitted_sketch(episodes, frames)
    data = BeliefSequences(episodes, {e.episode_id: [6, 1] for e in episodes}, lambda e, k: frames[e.episode_id][k], sketch, UNITS, SPEC)
    batch = data.batch([("e0", 0, 0), ("e0", 0, 18)])
    assert batch["observation"].shape[:2] == (2, SPEC.burn_in + 1 + SPEC.scored)
    assert batch["step_valid"][0].sum() == 1 + SPEC.scored and batch["step_valid"][1].all()
    assert batch["score"][0].sum() == SPEC.scored and batch["score"][1].sum() == SPEC.scored
    assert not batch["score"][:, 0].any()
    assert not batch["obs_mask"][0, ~batch["step_valid"][0]].any()


def test_burn_in_steps_receive_no_gradient_through_the_state():
    """After the boundary the state is detached: a perturbation applied to burn-in
    observations changes the loss value but must not create gradient paths to them."""
    camera = SyntheticCamera(views=2, height=3, width=4, channels=16)
    episodes, frames = build([1], camera)
    sketch = fitted_sketch(episodes, frames)
    data = BeliefSequences(episodes, {"e1": [6, 1]}, lambda e, k: frames[e.episode_id][k], sketch, UNITS, SPEC)
    batch = data.batch([("e1", 0, 18)])
    batch["observation"].requires_grad_(True)
    model = small_model()
    loss, _ = sequence_loss(model, batch, torch.full((6,), 0.01), CompactTrainConfig(regimes=(0,)))
    loss.backward()
    grad = batch["observation"].grad[0].abs().sum(-1)
    burn = int(batch["burn_in"][0])
    assert grad[: burn + 1].sum() == 0, "burn-in observations leaked gradient past the boundary"
    assert grad[burn + 1 :].sum() > 0


def test_loss_scale_and_residual_bound_helpers():
    episodes, _ = build([2, 3], SyntheticCamera(views=2, height=3, width=4, channels=16))
    scale = joint_loss_scale(episodes, UNITS, 6)
    bound = residual_bound_from_data(episodes, UNITS, 6)
    assert scale.shape == (6,) and (scale >= 1e-3).all()
    assert len(bound) == 6 and all(b >= s for b, s in zip(bound, scale.tolist()))


def test_training_lowers_validation_nll_and_resumes(tmp_path):
    camera = SyntheticCamera(views=2, height=3, width=4, channels=16)
    train_eps, train_frames = build([10, 11, 12, 13], camera)
    val_eps, val_frames = build([20, 21], camera)
    frames = {**train_frames, **val_frames}
    sketch = fitted_sketch(train_eps, train_frames)
    get = lambda e, k: frames[e.episode_id][k]
    views = {e.episode_id: [6, 1] for e in train_eps + val_eps}
    train = BeliefSequences(train_eps, views, get, sketch, UNITS, SPEC, step_every=12)
    val = BeliefSequences(val_eps, views, get, sketch, UNITS, SPEC, step_every=30)
    validation = val.fixed(16, 8)
    model = small_model(bound=residual_bound_from_data(train_eps, UNITS, 6))
    config = CompactTrainConfig(effective_batch=8, microbatch=8, max_updates=150, warmup_updates=10, lr=3e-3, final_lr=3e-4,
                                validation_interval=50, patience_evaluations=100)
    trainer = CompactTrainer(model, config, joint_loss_scale(train_eps, UNITS, 6), tmp_path)
    before = trainer.evaluate(validation)
    rng = random.Random(0)
    result = trainer.fit(lambda: train.sample(8, rng), validation)
    after = trainer.evaluate(validation)
    assert result["updates"] == 150
    assert after < before - 0.5, (before, after)
    assert model.repairs.count == 0, "covariance repairs indicate a numerical bug, not normal operation"
    fresh = CompactTrainer(small_model(bound=model.config.residual_bound), config, trainer.joint_scale, tmp_path)
    assert fresh.resume() and fresh.update == 150
    assert fresh.evaluate(validation) == pytest.approx(after, rel=1e-6)
