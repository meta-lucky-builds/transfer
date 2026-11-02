"""Losses, windows, replay, credits, gate, offline training, the planner."""

import random

import numpy as np
import pytest
import torch

from conftest import make_batch, make_model, set_adapters_nonzero
from robojepa_ttt.learning.losses import discounted_rollout_loss, proximal_penalty, token_metric
from robojepa_ttt.learning.offline_train import OfflineConfig, OfflineTrainer, horizon_at, lr_factor, visual_loss
from robojepa_ttt.learning.online_builder import OnlineBuilder, OnlineConfig
from robojepa_ttt.learning.promotion import (
    GateInput,
    ModelHub,
    PromotionConfig,
    RollbackMonitor,
    VersionStore,
    action_probe,
    block_bootstrap_upper,
    evaluate_gate,
)
from robojepa_ttt.learning.replay import CreditLedger, FeatureCache, ReplayBuffer, ReplayConfig, Reservoir
from robojepa_ttt.learning.windows import Episode, WindowSpec, dense_starts, extract_window, sparse_starts, window_is_valid
from robojepa_ttt.models.adapted_predictor import StaleCacheError
from robojepa_ttt.planning.joint_cem import CemConfig, JointCem, PlanContext
from robojepa_ttt.robot_io.command_limiter import LimiterConfig

SPEC = WindowSpec(frames=4)


def episode(ticks=100, views=2):
    e = Episode("e", "cal", "enc", views, capacity=8)
    for n in range(ticks):
        e.append(n / 30, np.full(6, n / 100), True, np.full(6, n / 50), True, np.ones(views, bool))
    return e


def test_token_metric_skips_when_nothing_is_valid():
    z = torch.randn(1, 2, 2, 3, 4, 8)
    assert token_metric(z, z + 1, mask=torch.zeros(1, 2, 2, 3, 4)) is None


def test_token_metric_ignores_masked_tokens_and_is_view_balanced():
    a = torch.randn(2, 1, 2, 3, 4, 8)
    b = a.clone()
    b[:, :, 1] = torch.randn(2, 1, 3, 4, 8)
    mask = torch.ones(2, 1, 2, 3, 4)
    mask[:, :, 1] = 0
    assert token_metric(a, b, mask=mask) == pytest.approx(0.0, abs=1e-6)
    assert token_metric(a, b) > 0.1


def test_discounted_rollout_normalises_and_skips_invalid_steps():
    one, two = torch.tensor(1.0), torch.tensor(2.0)
    assert discounted_rollout_loss([one, one], 0.5) == pytest.approx(1.0)
    assert discounted_rollout_loss([one, None, two], 0.5) == pytest.approx((1 + 0.25 * 2) / 1.25)
    assert discounted_rollout_loss([None], 0.5) is None


def test_proximal_penalty_handles_zero_reference():
    value = {"b": torch.full((2, 2), 1e-3)}
    assert proximal_penalty(value, {"b": torch.zeros(2, 2)}) == pytest.approx(1.0)


def test_dense_windows_cover_every_raw_offset_with_correct_spacing():
    e = episode(40)
    starts = list(dense_starts(e, SPEC))
    assert starts == list(range(0, 40 - SPEC.span_ticks))
    assert list(sparse_starts(e, SPEC)) == starts[::6]
    window = extract_window(e, 7, SPEC)
    assert list(window["keyframes"]) == [7, 13, 19, 25]


def test_window_rejects_boundary_invalid_frame_and_timing_gap():
    e = episode(10)
    e.mark_boundary()
    for n in range(10, 40):
        e.append(n / 30, np.zeros(6), True, np.zeros(6), True, np.ones(2, bool))
    assert not window_is_valid(e, 5, SPEC) and window_is_valid(e, 10, SPEC)
    e.frame_valid[16] = False
    assert not window_is_valid(e, 10, SPEC)
    e = episode(40)
    e.time_s[15:] += 0.2
    assert not window_is_valid(e, 10, SPEC) and window_is_valid(e, 15, SPEC)


def test_window_history_holds_position_before_any_command():
    e = episode(30)
    window = extract_window(e, 3, SPEC)
    assert window["issued"].shape[0] == SPEC.history_ticks + SPEC.span_ticks
    assert not window["command_valid"][: 27].any() and window["command_valid"][27:].all()
    assert window["issued"][0] == pytest.approx(e.position[3])
    assert window["issued"][27] == pytest.approx(e.issued[0])


def test_history_stops_at_a_reset():
    e = episode(20)
    e.mark_boundary()
    for n in range(20, 50):
        e.append(n / 30, np.zeros(6), True, np.ones(6), True, np.ones(2, bool))
    window = extract_window(e, 22, SPEC)
    assert window["command_valid"].sum() == 2 + SPEC.span_ticks


def test_tick_times_must_increase():
    e = episode(3)
    with pytest.raises(ValueError, match="increase"):
        e.append(0.0, np.zeros(6), True, np.zeros(6), True, np.ones(2, bool))


def test_feature_cache_respects_budget_and_reuses_entries():
    calls = []

    def encode(episode_id, tick):
        calls.append(tick)
        return torch.zeros(10, dtype=torch.float32)

    cache = FeatureCache(encode, byte_budget=100)
    for tick in range(5):
        cache.get("h", "e", tick)
    assert cache.bytes <= 100
    cache.get("h", "e", 4)
    assert calls == [0, 1, 2, 3, 4]
    cache.get("other-encoder", "e", 4)
    assert len(calls) == 6


def test_reservoir_is_uniform():
    counts = np.zeros(100)
    for seed in range(600):
        reservoir = Reservoir(10, random.Random(seed))
        for i in range(100):
            reservoir.add(("e", i))
        for _, i in reservoir.items:
            counts[i] += 1
    assert counts.std() / counts.mean() < 0.2


def test_replay_mix_uniqueness_and_refcounts():
    replay = ReplayBuffer(ReplayConfig(recent_windows=50, reservoir_windows=20, anchor_windows=10))
    for i in range(200):
        replay.add(("e", i), 0)
    for i in range(10):
        assert replay.add_anchor(("anchor", i))
    assert not replay.add_anchor(("anchor", 99))
    windows, flags = replay.sample(16, 0)
    assert len(set(windows)) == 16
    assert sum(flags) == 4
    assert all(flag == (w[0] == "anchor") for w, flag in zip(windows, flags))
    assert replay.references["anchor"] == 10
    assert replay.references["e"] == len(replay.recent) + len(replay.reservoirs[0].items)


def test_replay_refuses_short_batches_and_quarantine_is_separate():
    replay = ReplayBuffer(ReplayConfig(quarantine_windows=2))
    replay.add(("e", 0), None)
    replay.add(("e", 1), None)
    replay.add(("e", 2), None)
    assert len(replay.quarantine) == 2 and not replay.recent
    with pytest.raises(ValueError, match="unique windows"):
        replay.sample(4, 0)


def test_credit_ledger_rules():
    ledger = CreditLedger(minimum_windows=32, windows_per_credit=16, max_backlog=3, max_duty=0.5)
    ledger.admit(16)
    assert not ledger.can_update(0.0), "below the minimum window count"
    ledger.admit(16)
    assert ledger.available == 2 and ledger.can_update(1.0)
    ledger.spend(1.0)
    assert not ledger.can_update(1.5), "rate limit"
    assert ledger.can_update(2.1)
    ledger.refund()
    assert ledger.available == 2
    ledger.admit(16 * 10)
    assert ledger.available == 3 and ledger.discarded == 9


def test_duty_limit_blocks_updates():
    ledger = CreditLedger(minimum_windows=16, windows_per_credit=16, max_duty=0.1)
    ledger.admit(160)
    assert ledger.can_update(0.0)
    ledger.spend(0.0)
    ledger.record_busy(5.0)
    assert not ledger.can_update(10.0)
    assert ledger.can_update(51.0)


PROMOTION = PromotionConfig(future_macro_transitions=128, bootstrap_replicates=500)


def gate(serving, challenger, **kwargs):
    return evaluate_gate(GateInput(serving, challenger, 1.0, 1.0, **kwargs), PROMOTION)


def noisy(n=200, mean=1.0, seed=0):
    return mean + np.random.default_rng(seed).normal(0, 0.02, n)


def test_gate_accepts_a_clear_improvement():
    decision = gate(noisy(), noisy() * 0.9 + np.random.default_rng(5).normal(0, 0.005, 200))
    assert decision.accept, decision.reasons


def test_gate_retains_serving_when_there_is_no_real_gain():
    assert not gate(noisy(), noisy(seed=1)).accept


def test_gate_needs_enough_future_data():
    decision = gate(noisy(50), noisy(50) * 0.8)
    assert not decision.accept and "future transitions" in decision.reasons[0]


def test_gate_treats_correlated_data_as_blocks():
    """Blocks of correlated scores must widen the bound over sample-level resampling."""
    difference = np.repeat(np.random.default_rng(0).normal(-0.01, 0.1, 20), 10)
    per_sample = np.random.default_rng(0).choice(difference, (2000, 200)).mean(1)
    assert block_bootstrap_upper(difference, 10, 2000, 0.9, 0) > np.quantile(per_sample, 0.9)
    assert block_bootstrap_upper(difference[:15], 10, 2000, 0.9, 0) == float("inf")


def test_gate_regressions_each_block_acceptance():
    good = (noisy(), noisy() * 0.9)
    assert not gate(*good, anchor_groups={"nominal": (1.0, 1.06)}).accept
    assert not gate(*good, task_probe=(1.0, 1.06)).accept
    assert not gate(*good, sensitivity_ratio=0.1).accept
    assert not gate(*good, response_cosine=-0.5).accept
    assert not gate(*good, numerics_ok=False).accept
    joint = evaluate_gate(GateInput(good[0], good[1], 1.0, 1.05), PROMOTION)
    assert not joint.accept


def test_action_probe_detects_flattened_and_reversed_challengers():
    serving = make_model()
    batch = make_batch()
    ratio, cosine = action_probe(serving, serving, batch, 3, 0.1)
    assert ratio == pytest.approx(1.0) and cosine == pytest.approx(1.0)

    flat = make_model()
    constant = serving.rollout(batch, 3, 1).detach()
    flat.rollout = lambda b, c, h: constant
    assert action_probe(serving, flat, batch, 3, 0.1)[0] == pytest.approx(0.0)

    reversed_model = make_model()
    original = serving.rollout
    reversed_model.rollout = lambda b, c, h: -original(b, c, h)
    assert action_probe(serving, reversed_model, batch, 3, 0.1)[1] == pytest.approx(-1.0)


def test_rollback_monitor_needs_a_sustained_regression():
    config = PromotionConfig(rollback_window_s=10.0, macro_hz=5.0)
    monitor = RollbackMonitor(config)
    assert not monitor.observe(0.0, 2.0, 1.0), "one sample proves nothing"
    triggered = [monitor.observe(t / 5, 2.0, 1.0) for t in range(1, 60)]
    assert any(triggered)
    monitor.reset()
    assert not any(monitor.observe(t / 5, 1.05, 1.0) for t in range(60))
    assert monitor.observe(100.0, float("nan"), 1.0)


def test_hub_shadow_never_changes_serving(tmp_path):
    hub = ModelHub(make_model(), VersionStore(tmp_path), PROMOTION, shadow=True)
    assert hub.maybe_snapshot(0.0, 10)
    with pytest.raises(RuntimeError, match="shadow"):
        hub.promote()


def test_hub_promotes_immutably_rolls_back_and_invalidates_caches(tmp_path):
    hub = ModelHub(make_model(), VersionStore(tmp_path), PROMOTION, shadow=False)
    batch = make_batch(batch=1)
    set_adapters_nonzero(hub.builder)
    original = hub.serving
    before = original.adapter_state()
    actions, states = original.sequence_inputs(batch, 3, 1)
    command = original.encode_commands(batch, 3)
    prefix = original.prime_context(batch["features"][:, :2], actions[:, :2], states[:, :2], batch["view_ids"])

    assert hub.maybe_snapshot(0.0, 10)
    assert not hub.maybe_snapshot(40.0, 10), "one challenger at a time"
    version = hub.promote()
    assert hub.serving is not original and hub.serving.version == version
    assert all(torch.equal(v, original.adapter_state()[k]) for k, v in before.items()), "plan's model must not mutate"
    with pytest.raises(StaleCacheError):
        hub.serving.step_cached(
            batch["features"][:, 2:3], actions[:, 2:3], states[:, 2:3], batch["view_ids"], command[:, 2:3], prefix, 2
        )
    target = hub.rollback("test")
    assert target == "offline-reference"
    with torch.no_grad():
        assert torch.equal(hub.serving.rollout(batch, 3, 1), original.rollout(batch, 3, 1))


def test_versions_are_immutable_and_files_survive_rollback(tmp_path):
    store = VersionStore(tmp_path)
    snapshot = make_model().snapshot("v1")
    store.save(snapshot)
    with pytest.raises(FileExistsError):
        store.save(snapshot)
    assert store.load("v1")["snapshot"].digest == snapshot.digest


def test_builder_updates_only_adapters_and_persists_moments():
    model = make_model()
    reference = model.clone_role()
    builder = OnlineBuilder(model, reference, OnlineConfig(lora_lr=1e-3, residual_lr=1e-3, warmup_updates=1))
    frozen = {n: p.clone() for n, p in model.named_parameters() if n not in model.trainable_named()}
    batch = make_batch(batch=4, frames=4)
    result = builder.update(batch, [True, False, False, True], context=2)
    assert result.status == "ok" and result.anchor is not None
    assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)
    b_o = next(p for n, p in model.trainable_named().items() if n.endswith("b_o.weight"))
    assert b_o.abs().sum() > 0 and builder.optimizer.state[b_o]["step"] == 1
    state = builder.state_dict()
    fresh = OnlineBuilder(make_model(), reference, builder.config)
    fresh.load_state_dict(state)
    assert fresh.updates == 1
    first = next(iter(fresh.optimizer.state.values()))
    assert first["step"] == 1


def test_microbatches_give_the_same_update_as_one_batch():
    """Spec 11.4: splitting changes memory use, never the objective."""
    config = OnlineConfig(lora_lr=1e-3, residual_lr=1e-3, warmup_updates=1)
    batch = make_batch(batch=4, frames=4)
    flags = [True, False, False, True]
    results = []
    for microbatch in (None, 1, 3):
        model = make_model()
        builder = OnlineBuilder(model, model.clone_role(), config)
        result = builder.update(batch, flags, context=2, microbatch=microbatch)
        results.append((result, model.adapter_state()))
    full, *split = results
    for result, state in split:
        assert result.loss == pytest.approx(full[0].loss, rel=1e-5)
        assert result.anchor == pytest.approx(full[0].anchor, rel=1e-4)
        assert result.grad_norm == pytest.approx(full[0].grad_norm, rel=1e-4)
        assert all(torch.allclose(state[k], full[1][k], atol=2e-6) for k in state)


def test_builder_reports_nonfinite_without_touching_weights():
    model = make_model()
    builder = OnlineBuilder(model, model.clone_role(), OnlineConfig(warmup_updates=1))
    before = model.adapter_state()
    batch = make_batch(batch=2, frames=4)
    batch["features"] = batch["features"] * float("nan")
    result = builder.update(batch, [False, False], context=2)
    assert result.status == "nonfinite"
    assert all(torch.equal(v, model.adapter_state()[k]) for k, v in before.items())


def test_repeated_updates_reduce_loss_on_a_fixed_batch():
    model = make_model()
    builder = OnlineBuilder(model, model.clone_role(), OnlineConfig(lora_lr=3e-3, residual_lr=3e-3, warmup_updates=1, weight_prox=0.0))
    batch = make_batch(batch=4, frames=4)
    losses = [builder.update(batch, [False] * 4, context=2).cmd_tf for _ in range(30)]
    assert losses[-1] < 0.8 * losses[0]


def test_schedule_and_curriculum():
    config = OfflineConfig()
    assert lr_factor(0, config) == pytest.approx(1 / 500)
    assert lr_factor(499, config) == pytest.approx(1.0)
    assert lr_factor(20000, config) == pytest.approx(0.1)
    assert [horizon_at(u, config) for u in (0, 4999, 5000, 14999, 15000, 25000)] == [2, 2, 5, 5, 10, 10]


def test_offline_step_saves_resumes_and_stops_early(tmp_path):
    config = OfflineConfig(
        sequence_frames=10,
        effective_batch=2,
        max_updates=6,
        warmup_updates=1,
        curriculum=((2, 2), (4, 5)),
        validation_interval=1,
        patience_evaluations=2,
        lora_lr=1e-3,
        residual_lr=1e-3,
    )
    model = make_model()
    trainer = OfflineTrainer(model, config, tmp_path)
    batch = make_batch(batch=1, frames=10)
    result = trainer.fit(lambda: batch, lambda m: 1.0)
    assert result["updates"] == 3, "stopped after two evaluations with no improvement"
    assert (tmp_path / "best.pt").exists()
    resumed = OfflineTrainer(make_model(), config, tmp_path)
    assert resumed.resume()
    assert resumed.update == 3
    assert all(torch.equal(a, b) for a, b in zip(model.adapter_state().values(), resumed.model.adapter_state().values()))


def test_realized_path_uses_only_recorded_motion():
    model = make_model()
    batch = make_batch(batch=2, frames=5)
    loss, parts = visual_loss(model, batch, 2, 2, OfflineConfig())
    assert set(parts) == {"cmd_tf", "rollout", "realized"} and torch.isfinite(loss)


LIMITS = LimiterConfig((-2.0,) * 6, (2.0,) * 6, (3.0,) * 6, (100.0,) * 6)
CEM = CemConfig(horizon=3, candidates=12, iterations=2, elites=4, microbatch=5, view_weights=(1.0, 0.0))


def plan_context(model, c=3):
    batch = make_batch(batch=1, frames=c + 3)
    ticks = 30 + (c - 1) * 6
    return PlanContext(
        features=batch["features"][:, :c],
        position=batch["position"][:, :c] * 0,
        issued_past=torch.zeros(1, ticks, 6),
        dt_past=torch.full((1, ticks), 1 / 30),
        valid_past=torch.ones(1, ticks, dtype=torch.bool),
        view_ids=batch["view_ids"],
        goal=batch["features"][:, -1:],
        reference=torch.zeros(1, 6),
        reference_velocity=torch.zeros(1, 6),
    )


def test_plan_commands_stay_within_limiter_and_are_reproducible():
    model = make_model()
    planner = JointCem(model, LIMITS, (0.2,) * 6, CEM)
    context = plan_context(model)
    plan = planner.plan(context, seed=3)
    again = planner.plan(context, seed=3)
    assert plan.commands.shape == (6, 6) and torch.equal(plan.commands, again.commands)
    assert plan.commands.abs().max() <= 2.0 and plan.version == model.version


def test_cached_candidate_scores_match_a_full_uncached_rollout():
    """The planner's cached path must rank candidates as rollout() would."""
    model = make_model()
    set_adapters_nonzero(model)
    planner = JointCem(model, LIMITS, (0.2,) * 6, CEM)
    context = plan_context(model)
    increments = torch.randn(3, 3, 6, generator=torch.Generator().manual_seed(0)).clamp(-1, 1)
    commands = planner.commands_for(increments, context)
    tokens, _ = planner._terminal_tokens(context, commands)
    batch = planner._batch(context, commands)
    batch["features"] = context.features.expand(3, -1, -1, -1, -1, -1)
    batch["view_ids"] = context.view_ids.expand(3, -1)
    with torch.no_grad():
        full = model.rollout(batch, 3, 3, use_kv_generation=False)
    assert torch.allclose(tokens, full[:, -1:], atol=2e-4)


def test_planner_works_from_a_single_context_frame():
    """c=1 has no shared prefix, so every candidate builds its own cache."""
    model = make_model()
    set_adapters_nonzero(model)
    planner = JointCem(model, LIMITS, (0.2,) * 6, CEM)
    context = plan_context(model, c=1)
    increments = torch.randn(3, 3, 6, generator=torch.Generator().manual_seed(0)).clamp(-1, 1)
    commands = planner.commands_for(increments, context)
    tokens, _ = planner._terminal_tokens(context, commands)
    batch = planner._batch(context, commands)
    batch["features"] = context.features.expand(3, -1, -1, -1, -1, -1)
    batch["view_ids"] = context.view_ids.expand(3, -1)
    with torch.no_grad():
        full = model.rollout(batch, 1, 3, use_kv_generation=False)
    assert torch.allclose(tokens, full[:, -1:], atol=2e-4)


def test_hold_is_always_available_and_infeasible_start_raises():
    model = make_model()
    planner = JointCem(model, LIMITS, (0.0,) * 6, CEM)
    plan = planner.plan(plan_context(model), seed=1)
    assert plan.commands.abs().max() == 0
    context = plan_context(model)
    context.position = context.position + 5.0
    context.reference = context.reference + 5.0
    context.issued_past = context.issued_past + 5.0
    with pytest.raises(RuntimeError, match="hold is infeasible"):
        JointCem(model, LIMITS, (0.2,) * 6, CEM).plan(context, seed=1)
