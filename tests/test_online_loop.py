"""End-to-end online loop on the real tiny predictor and a simulated arm.

This checks the machinery in order: predict, resolve, admit, train, snapshot, gate,
promote, rollback. Features come from a synthetic camera, so nothing here says that
adaptation helps a real robot.
"""

import numpy as np
import pytest
import torch

from conftest import make_model
from robojepa_ttt.evaluation.perturbation import CommandShift, ShiftedChannel, SimulatedArm, SyntheticCamera
from robojepa_ttt.learning.online_builder import OnlineConfig
from robojepa_ttt.learning.online_runner import OnlineRunner, live_batch
from robojepa_ttt.learning.promotion import ModelHub, PromotionConfig, VersionStore
from robojepa_ttt.learning.replay import CreditLedger, FeatureCache, ReplayBuffer, ReplayConfig
from robojepa_ttt.learning.windows import Episode, WindowSpec

SPEC = WindowSpec(frames=5)


class World:
    """One stream of ticks with a fake clock."""

    def __init__(self, tmp_path, shadow=True, promotion=None, online=None, shift=None, seed=0):
        self.arm = SimulatedArm(seed=seed)
        self.camera = SyntheticCamera()
        self.rng = np.random.default_rng(seed)
        self.now = 0.0
        self.positions = {}
        self.episode = Episode("live", "cal", "enc", 2)
        self.channel = ShiftedChannel(shift or CommandShift(), np.zeros(6))
        self.target = np.zeros(6)
        cache = FeatureCache(lambda e, t: self.camera.features(self.positions[t]), byte_budget=10**9)
        promotion = promotion or PromotionConfig(
            minimum_snapshot_interval_s=0.0,
            minimum_updates_per_snapshot=25,
            future_macro_transitions=20,
            bootstrap_replicates=200,
        )
        online = online or OnlineConfig(lora_lr=2e-3, residual_lr=2e-3, warmup_updates=1, horizon=1, weight_prox=0.0)
        hub = ModelHub(make_model(), VersionStore(tmp_path), promotion, shadow=shadow)
        self.runner = OnlineRunner(
            hub,
            ReplayBuffer(ReplayConfig(), seed=seed),
            CreditLedger(minimum_windows=8, windows_per_credit=4, max_updates_per_s=30.0, max_duty=1.0),
            cache,
            SPEC,
            {"live": [0, 2]},
            online,
            promotion,
            batch_size=4,
            clock=lambda: self.now,
            seed=seed,
        )
        self.runner.register_episode(self.episode)

    def command_block(self):
        self.target = np.clip(self.target + self.rng.normal(0, 0.08, 6), -1.0, 1.0)
        return np.repeat(self.target[None], SPEC.stride, axis=0)

    def run(self, macro_steps):
        for _ in range(macro_steps):
            block = self.command_block()
            end_tick = self.episode.length - 1
            if end_tick >= SPEC.span_ticks + SPEC.stride:
                self.runner.predict(self.episode, end_tick, 2, block)
            for command in block:
                measured = self.arm.step(self.channel.apply(command, self.arm.position))
                tick = self.episode.append(self.now, measured, True, command, True, np.ones(2, bool))
                self.positions[tick] = measured
                self.now += SPEC.tick_s
            self.runner.resolve(self.episode, self.episode.length - 1)
            self.runner.admit_windows(self.episode)
            self.runner.maybe_train()
            self.runner.maybe_snapshot()


def test_loop_runs_in_causal_order_and_trains(tmp_path):
    world = World(tmp_path)
    world.run(30)
    runner = world.runner
    assert runner.admitted_total > 0 and runner.history
    assert runner.credits.spent == len(runner.history)
    assert runner.scored_total > 0, "prequential scoring ran"
    assert not runner.faults
    assert runner.hub.serving.version == "offline-reference"


def test_live_batch_reads_nothing_after_the_context():
    world_episode = Episode("e", "c", "h", 2)
    positions = {}
    camera = SyntheticCamera()
    for n in range(60):
        p = np.full(6, n / 100)
        positions[n] = p
        world_episode.append(n / 30, p, True, p, True, np.ones(2, bool))
    cache = FeatureCache(lambda e, t: camera.features(positions[t]), 10**9)
    block = np.zeros((6, 6))
    batch = live_batch(world_episode, cache, SPEC, 40, 2, block, [0, 2], "cpu")
    reference = batch["features"].clone()
    world_episode.position[41:] = 99.0
    world_episode.issued[41:] = 99.0
    positions_after = dict(positions)
    positions_after.update({n: np.full(6, 99.0) for n in range(41, 60)})
    again = live_batch(world_episode, cache, SPEC, 40, 2, block, [0, 2], "cpu")
    assert torch.equal(batch["issued"], again["issued"]) and torch.equal(reference, again["features"])
    assert batch["issued"].shape[1] == SPEC.history_ticks + SPEC.stride + SPEC.stride


def test_prediction_is_saved_before_its_target_exists(tmp_path):
    world = World(tmp_path)
    world.run(5)
    assert world.runner.pending == [] or all(p.target_tick >= world.episode.length for p in world.runner.pending)
    block = world.command_block()
    end_tick = world.episode.length - 1
    world.runner.predict(world.episode, end_tick, 2, block)
    assert world.runner.pending[-1].target_tick == world.episode.length - 1 + SPEC.stride
    assert world.runner.resolve(world.episode, world.episode.length - 1) is None
    assert len(world.runner.pending) == 1


def test_target_across_a_reset_is_not_scored(tmp_path):
    world = World(tmp_path)
    world.run(5)
    block = world.command_block()
    world.runner.predict(world.episode, world.episode.length - 1, 2, block)
    count = len(world.runner.scores.serving)
    world.episode.mark_boundary()
    for command in block:
        measured = world.arm.step(command)
        tick = world.episode.append(world.now, measured, True, command, True, np.ones(2, bool))
        world.positions[tick] = measured
        world.now += SPEC.tick_s
    world.runner.resolve(world.episode, world.episode.length - 1)
    assert len(world.runner.scores.serving) == count and not world.runner.pending


def test_gate_promotes_a_better_adapter_under_a_dynamics_shift(tmp_path):
    world = World(tmp_path, shadow=False, shift=CommandShift(gain=0.5, delay_ticks=3))
    world.run(60)
    runner = world.runner
    assert runner.gate_ready()
    decision = runner.evaluate_challenger()
    serving = np.mean(runner.scores.serving[-len(runner.scores.challenger) :])
    challenger = np.mean(runner.scores.challenger)
    assert challenger < serving, (challenger, serving)
    assert decision.accept, decision.reasons
    version = runner.planning_boundary(decision)
    assert runner.hub.serving.version == version and version.startswith("candidate-")
    world.run(10)
    assert runner.hub.previous is not None
    assert runner.hub.rollback("drill") == "offline-reference"
    assert runner.hub.serving.version == "offline-reference"


def test_shadow_mode_rejects_promotion_and_keeps_serving(tmp_path):
    world = World(tmp_path, shadow=True, shift=CommandShift(gain=0.5, delay_ticks=3))
    world.run(60)
    runner = world.runner
    assert runner.gate_ready()
    runner.planning_boundary(runner.evaluate_challenger())
    assert runner.hub.serving.version == "offline-reference" and runner.hub.challenger is None


def test_rollback_restores_the_builder_too(tmp_path):
    """Regression: serving was restored while the builder kept the rejected weights."""
    world = World(tmp_path, shadow=False, shift=CommandShift(gain=0.5, delay_ticks=3))
    world.run(60)
    runner = world.runner
    runner.planning_boundary(runner.evaluate_challenger())
    world.run(5)
    runner.rollback_due = "drill"
    assert runner.planning_boundary(None) == "offline-reference"
    assert runner.rollback_due is None
    serving, builder = runner.hub.serving.adapter_state(), runner.hub.builder.adapter_state()
    assert all(torch.equal(serving[k], builder[k]) for k in serving)
    assert runner.hub.builder.trainable_named(), "builder must keep training after a rollback"
    world.run(10)
    assert not runner.faults


def test_rollback_monitor_trigger_reaches_the_boundary(tmp_path):
    world = World(tmp_path, shadow=False)
    world.runner.rollback_monitor.observe = lambda *args: True
    world.run(60)
    runner = world.runner
    runner.planning_boundary(runner.evaluate_challenger())
    world.run(3)
    assert runner.rollback_due is not None


def test_no_prediction_from_a_context_across_a_reset(tmp_path):
    world = World(tmp_path)
    world.run(5)
    world.episode.mark_boundary()
    for command in world.command_block():
        measured = world.arm.step(command)
        tick = world.episode.append(world.now, measured, True, command, True, np.ones(2, bool))
        world.positions[tick] = measured
        world.now += SPEC.tick_s
    pending = len(world.runner.pending)
    assert not world.runner.predict(world.episode, world.episode.length - 1, 2, world.command_block())
    assert len(world.runner.pending) == pending
    assert world.runner.predict(world.episode, world.episode.length - 1, 1, world.command_block())


def test_gate_uses_anchors_and_action_probe(tmp_path):
    world = World(tmp_path, shadow=False, shift=CommandShift(gain=0.5, delay_ticks=3))
    world.run(45)
    runner = world.runner
    for start in range(0, 24, 6):
        runner.replay.add_anchor(("live", start))
    assert runner.hub.challenger is not None
    groups, probe = runner._anchor_scores()
    assert set(groups) == {"live"} and len(groups["live"]) == 2
    assert probe is not None and np.isfinite(probe).all()
    runner.anchor_groups = lambda serving, challenger: {"critical": (1.0, 2.0)}
    world.run(30)
    decision = runner.evaluate_challenger()
    assert not decision.accept and any("critical" in reason for reason in decision.reasons)


def test_shift_bounds_are_enforced():
    with pytest.raises(ValueError):
        CommandShift(gain=1.5)
    with pytest.raises(ValueError):
        CommandShift(delay_ticks=100)


def test_shift_delays_and_scales():
    channel = ShiftedChannel(CommandShift(gain=0.5, delay_ticks=2), np.zeros(6))
    position = np.zeros(6)
    outputs = [channel.apply(np.full(6, 1.0), position)[0] for _ in range(4)]
    assert outputs == [0.0, 0.0, 0.5, 0.5]
