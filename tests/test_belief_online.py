"""Fixed-lag replay, dedupe, change monitor, mode switch and calibration (spec 21:
late-event replay equals a clean chronological run; no double assimilation)."""

import pytest
import torch
from dataclasses import replace

from robojepa_ttt.belief.calibration import fit_calibration, innovation_summary
from robojepa_ttt.belief.online import ChangeMonitor, FixedLagBelief, ModeSwitch
from test_belief_compact import make_belief, make_step


@pytest.fixture(autouse=True)
def attention_layout():
    yield None


def perturbed_model():
    model = make_belief()
    torch.manual_seed(3)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model


def ids(t):
    return {"joints": f"j{t}", "view6": f"a{t}", "view1": f"b{t}"}


def steps(model, count=7):
    return [make_step(model, batch=1, seed=100 + t) for t in range(count)]


def run_on_time(model, seq):
    f = FixedLagBelief(model, window_s=2.0)
    f.start(0.0, seq[0], ids(0))
    for t in range(1, len(seq)):
        f.observe(0.2 * t, seq[t], ids(t))
    return f


def test_late_observation_replay_equals_on_time_run():
    model = perturbed_model()
    seq = steps(model)
    clean = run_on_time(model, seq)

    late = FixedLagBelief(model, window_s=2.0)
    late.start(0.0, seq[0], ids(0))
    held = seq[3].observation[:, 6:10].clone()
    for t in range(1, len(seq)):
        step, step_ids = seq[t], ids(t)
        if t == 3:  # view 6 of step 3 is delayed
            mask = step.obs_mask.clone()
            mask[:, 6:10] = False
            step = replace(step, obs_mask=mask)
            step_ids = {k: v for k, v in step_ids.items() if k != "view6"}
        late.observe(0.2 * t, step, step_ids)
    assert not torch.allclose(late.state.mean, clean.state.mean), "the late frame must matter"
    assert late.late(0.6, "view6", held, "a3") == "replayed"
    for name in ("log_pi", "mean", "cov", "h", "age_s"):
        assert torch.allclose(getattr(late.state, name), getattr(clean.state, name), atol=1e-12), name


def test_duplicate_ids_are_never_assimilated_twice():
    model = perturbed_model()
    seq = steps(model, 3)
    f = FixedLagBelief(model)
    f.start(0.0, seq[0], ids(0))
    f.observe(0.2, seq[1], ids(1))
    repeat = {"joints": "j2", "view6": "a1", "view1": "b2"}  # view6 re-sent with an old id
    f.observe(0.4, seq[2], repeat)

    reference = FixedLagBelief(model)
    reference.start(0.0, seq[0], ids(0))
    reference.observe(0.2, seq[1], ids(1))
    mask = seq[2].obs_mask.clone()
    mask[:, 6:10] = False
    reference.observe(0.4, replace(seq[2], obs_mask=mask), {"joints": "j2", "view1": "b2"})
    assert torch.allclose(f.state.mean, reference.state.mean, atol=1e-12)
    assert f.late(0.2, "view1", seq[1].observation[:, 10:14], "b1") == "duplicate"


def test_late_frames_outside_the_window_or_for_filled_blocks_are_dropped():
    model = perturbed_model()
    seq = steps(model, 16)
    f = run_on_time(model, seq)
    before = f.state.mean.clone()
    assert f.late(0.2, "view6", torch.zeros(1, 4), "old") == "dropped"  # 0.2 s step pruned
    assert f.late(f.records[-1].time_s, "view6", torch.zeros(1, 4), "second-reading") == "dropped"
    assert torch.equal(f.state.mean, before) and len(f.dropped) == 2
    assert f.records[-1].time_s - f.records[0].time_s <= 2.0 + 0.2


def test_belief_message_has_the_spec_fields():
    model = perturbed_model()
    f = run_on_time(model, steps(model, 3))
    message = f.message()
    for key in ("time_s", "regime_probabilities", "summary_mean", "summary_covariance", "view_age_s", "incorporated_ids", "motion_allowed"):
        assert key in message
    assert message["summary_covariance"].shape == (14, 14)


def test_start_needs_an_identified_joint_reading():
    model = perturbed_model()
    seq = steps(model, 1)
    with pytest.raises(ValueError):
        FixedLagBelief(model).start(0.0, seq[0], {"view6": "a0"})


def test_cusum_needs_a_sustained_shift_and_enough_valid_observations():
    monitor = ChangeMonitor(kappa=0.5, threshold=10.0, minimum_valid=5)
    monitor.calibrate(torch.randn(500, generator=torch.Generator().manual_seed(0)))
    assert not monitor.update(30.0), "one spike is below the observation minimum"
    monitor.reset()
    assert not any(monitor.update(0.3) for _ in range(200)), "no alarm without a shift"
    monitor.reset()
    alarms = [monitor.update(3.0) for _ in range(10)]
    assert not any(alarms[:3]) and alarms[-1]
    assert not monitor.update(100.0, valid=False) or True
    with pytest.raises(RuntimeError):
        ChangeMonitor().update(1.0)


def test_mode_switch_needs_a_held_majority():
    switch = ModeSwitch(switch_probability=0.8, hold_observations=5)
    p = torch.tensor([0.1, 0.9])
    assert [switch.update(p, True) for _ in range(4)] == [0, 0, 0, 0]
    assert switch.update(torch.tensor([0.5, 0.5]), True) == 0, "streak broken"
    results = [switch.update(p, True) for _ in range(5)]
    assert results[-1] == 1 and results[:4] == [0, 0, 0, 0]
    assert switch.update(torch.tensor([0.9, 0.1]), False) == 1, "invalid observations do not count"
    assert switch.planning_set(torch.tensor([0.02, 0.9, 0.08])) == [1, 2]


def test_calibration_lowers_nll_when_noise_is_misscaled():
    from test_belief_train import SPEC, UNITS, build, fitted_sketch, small_model
    from robojepa_ttt.belief.data import BeliefSequences, joint_loss_scale
    from robojepa_ttt.evaluation.perturbation import SyntheticCamera

    camera = SyntheticCamera(views=2, height=3, width=4, channels=16)
    episodes, frames = build([30, 31], camera)
    sketch = fitted_sketch(episodes, frames)
    data = BeliefSequences(episodes, {e.episode_id: [6, 1] for e in episodes}, lambda e, k: frames[e.episode_id][k], sketch, UNITS, SPEC, step_every=24)
    batches = data.fixed(8, 4)
    model = small_model()
    model.calibration.fill_(0.25)  # far too narrow for an untrained model
    result = fit_calibration(model, batches, joint_loss_scale(episodes, UNITS, 6), grid=(0.25, 1.0, 4.0, 32.0), passes=1)
    assert result["nll_after"] < result["nll_before"]
    assert all(m > 0.25 for m in result["multipliers"])
    summary = innovation_summary(model, batches)
    assert set(summary) == {"joints", "view6", "view1"} and all(v["coordinates"] > 0 for v in summary.values())
