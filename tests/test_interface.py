"""Units, limiter, event log, servo fit, supervisor, data import."""

import json
import math

import numpy as np
import pytest
import torch

from robojepa_ttt.data.import_lerobot import check_dataset_info, episode_from_arrays, split_episodes
from robojepa_ttt.models.servo import ServoParams, fit_servo, rollout_servo
from robojepa_ttt.models.upstream import patch_joint_integration
from robojepa_ttt.robot_io.command_limiter import LimiterConfig, limit_reference
from robojepa_ttt.robot_io.event_log import EventLog, SessionHeader, read_events
from robojepa_ttt.robot_io.so101_units import JOINT_ORDER, So101Units, propagate_joint_state
from robojepa_ttt.robot_io.supervisor import PlanPacket, Supervisor, SupervisorConfig

UNITS = So101Units(gripper_direction_verified=True, center=(0.1, 0.2, -0.1, 0.0, 0.3, 0.5), scale=(1, 2, 1, 1, 1, 0.5))


def test_unit_round_trips():
    device = np.array([10.0, -20.0, 30.0, -40.0, 50.0, 25.0])
    physical = UNITS.device_to_physical(device)
    assert physical[0] == pytest.approx(math.radians(10))
    assert physical[5] == pytest.approx(0.25)
    assert UNITS.physical_to_device(physical) == pytest.approx(device)
    assert UNITS.compact_to_physical(UNITS.physical_to_compact(physical)) == pytest.approx(physical)
    assert UNITS.legacy_to_physical(UNITS.physical_to_legacy(physical)) == pytest.approx(physical)


def test_legacy_gripper_is_device_value_times_pi_over_180():
    device = np.array([0, 0, 0, 0, 0, 100.0])
    legacy = UNITS.physical_to_legacy(UNITS.device_to_physical(device))
    assert legacy[5] == pytest.approx(100 * math.pi / 180)
    assert legacy[5] > 1.74


def test_legacy_delta_has_no_gripper_offset():
    inverted = So101Units(gripper_device_closed=100.0, gripper_device_open=0.0, gripper_direction_verified=True)
    a = np.array([0, 0, 0, 0, 0, 0.2])
    b = np.array([0, 0, 0, 0, 0, 0.7])
    expected = inverted.physical_to_legacy(b) - inverted.physical_to_legacy(a)
    assert inverted.physical_delta_to_legacy(b - a) == pytest.approx(expected)


def test_torch_and_numpy_agree():
    device = np.array([10.0, -20.0, 30.0, -40.0, 50.0, 25.0], dtype=np.float32)
    a = UNITS.physical_to_legacy(UNITS.device_to_physical(device))
    b = UNITS.physical_to_legacy(UNITS.device_to_physical(torch.tensor(device)))
    assert b.numpy() == pytest.approx(a, abs=1e-5)


def test_unverified_gripper_is_refused():
    with pytest.raises(ValueError, match="gripper direction"):
        So101Units().device_to_physical(np.zeros(6))


def test_percentage_body_units_are_refused():
    units = So101Units(use_degrees=False, gripper_direction_verified=True)
    with pytest.raises(ValueError, match="RANGE_M100_100"):
        units.device_to_physical(np.zeros(6))


def test_joint_propagation_is_additive_unclipped_and_keeps_padding():
    state = torch.zeros(64)
    state[:6] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 1.7])
    delta = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.03])
    out = propagate_joint_state(state, delta)
    assert out[5] == pytest.approx(1.73)
    assert out[6:].abs().sum() == 0


def test_upstream_state_delta_patch_matches_additive_update():
    patch_joint_integration()
    from app.robo_jepa import world_model

    pose = torch.zeros(1, 1, 64)
    pose[0, 0, :6] = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0, 1.7])
    action = torch.zeros(1, 1, 64)
    action[0, 0, :6] = 0.1
    names = ["so-101_joints"]
    new = world_model.compute_new_pose(pose, action, torch.zeros(1, 1, dtype=torch.long), names)
    assert new[0, 0, :6] == pytest.approx(pose[0, 0, :6] + 0.1)


LIMITS = LimiterConfig((-1.0,) * 6, (1.0,) * 6, (2.0,) * 6, (50.0,) * 6)


def test_limiter_respects_position_velocity_and_acceleration():
    start = torch.zeros(1, 6)
    targets = torch.full((1, 40, 6), 5.0)
    out, velocity, clipped = limit_reference(start, torch.zeros(1, 6), targets, 1 / 30, LIMITS)
    full = torch.cat([start[:, None], out], dim=1)
    v = (full[:, 1:] - full[:, :-1]) * 30
    assert out.max() <= 1.0 and v.max() <= 2.0 + 1e-5
    assert ((v[:, 1:] - v[:, :-1]) * 30).abs().max() <= 50.0 + 1e-2
    assert clipped.all()


def test_limiter_passes_slow_targets_unchanged():
    targets = torch.linspace(0, 0.1, 30)[None, :, None].repeat(1, 1, 6)
    start = torch.zeros(1, 6)
    velocity = (targets[:, 1, :] - targets[:, 0, :]) * 30
    out, _, _ = limit_reference(start, velocity, targets, 1 / 30, LIMITS)
    assert torch.allclose(out, targets, atol=1e-5)


def test_limiter_refuses_a_start_outside_limits():
    """Regression: clipping such a start made a hold command swing the full range."""
    with pytest.raises(ValueError, match="outside the validated"):
        limit_reference(torch.full((1, 6), 1.5), torch.zeros(1, 6), torch.ones(1, 5, 6), 1 / 30, LIMITS)


def test_limiter_velocity_stays_bounded_when_stopping_at_a_limit():
    start = torch.full((1, 6), 0.9)
    out, velocity, _ = limit_reference(start, torch.full((1, 6), 2.0), torch.full((1, 30, 6), 5.0), 1 / 30, LIMITS)
    full = torch.cat([start[:, None], out], dim=1)
    assert ((full[:, 1:] - full[:, :-1]) * 30).abs().max() <= 2.0 + 1e-4
    assert out.max() <= 1.0 and velocity.abs().max() < 1e-4


def test_limiter_needs_measured_values():
    with pytest.raises(ValueError):
        LimiterConfig((-1.0,) * 5, (1.0,) * 6, (2.0,) * 6, (5.0,) * 6)


HEADER = SessionHeader("s", "e", "r", "cal", "abc", True)


def test_event_log_keeps_requested_and_issued_apart(tmp_path):
    path = tmp_path / "log.jsonl"
    log = EventLog(path, HEADER, clock=lambda: 1.0)
    log.joint([0.0] * 6, True)
    event = log.command([1.0] * 6, [0.5] * 6, "v1")
    log.close()
    assert event.clipped
    records = list(read_events(path))
    assert records[0]["kind"] == "header"
    assert records[2]["requested"][0] == 1.0 and records[2]["issued"][0] == 0.5


def test_event_log_refuses_overwrite_and_detects_gaps(tmp_path):
    path = tmp_path / "log.jsonl"
    log = EventLog(path, HEADER)
    log.joint([0.0] * 6, True)
    log.joint([0.0] * 6, True)
    log.close()
    with pytest.raises(FileExistsError):
        EventLog(path, HEADER)
    lines = path.read_text().splitlines()
    broken = tmp_path / "broken.jsonl"
    broken.write_text("\n".join([lines[0], lines[2]]))
    with pytest.raises(ValueError, match="sequence"):
        list(read_events(broken))


def simulate(tau, delay, n=1200, tick=1 / 30, seed=0):
    rng = np.random.default_rng(seed)
    u = np.cumsum(rng.normal(0, 0.05, (n, 6)), axis=0)
    u = np.repeat(u[::15], 15, axis=0)[:n]
    p = np.zeros((n, 6))
    a = 1 - np.exp(-tick / tau)
    for k in range(1, n):
        target = u[max(k - 1 - delay, 0)]
        p[k] = p[k - 1] + a * (target - p[k - 1])
    return p, u


def test_servo_fit_recovers_time_constant_and_delay():
    p, u = simulate(0.12, 4)
    params = fit_servo([p + np.random.default_rng(1).normal(0, 1e-4, p.shape)], [u], 1 / 30)
    assert params.delay_ticks == (4,) * 6
    assert params.tau_s == pytest.approx((0.12,) * 6, rel=0.1)


def test_servo_fit_refuses_data_with_no_excitation():
    still = np.zeros((300, 6))
    with pytest.raises(ValueError, match="no identifiable"):
        fit_servo([still], [still], 1 / 30)


def test_servo_rollout_delays_and_converges():
    params = ServoParams((0.1,) * 6, (3,) * 6)
    commands = torch.zeros(1, 40, 6)
    commands[:, 5:] = 1.0
    out = rollout_servo(torch.zeros(1, 6), commands, torch.full((1, 40), 1 / 30), params, 3)
    assert out[0, 4, 0] == 0.0, "no motion before the delayed step reaches the arm"
    assert out[0, 5, 0] > 0.0
    assert out[0, -1, 0] > 0.9


def test_servo_rollout_needs_history():
    params = ServoParams((0.1,) * 6, (3,) * 6)
    with pytest.raises(ValueError, match="history"):
        rollout_servo(torch.zeros(1, 6), torch.zeros(1, 5, 6), torch.ones(1, 5), params, 2)


CONFIG = SupervisorConfig("r", "cal", "lim", 0.5, (-1.0,) * 6, (1.0,) * 6)


def packet(**changes):
    values = dict(
        plan_id=1,
        robot_id="r",
        calibration_hash="cal",
        limits_version="lim",
        model_version="v",
        observation_time_s=10.0,
        incorporated_action_seq=7,
        issue_time_s=10.1,
        expiry_time_s=10.4,
        expected_start=(0.0,) * 6,
        start_tolerance=(0.05,) * 6,
        references=np.zeros((6, 6)),
    )
    values.update(changes)
    return PlanPacket(**values)


@pytest.mark.parametrize(
    "changes,now,seq,position,reason",
    [
        (dict(robot_id="x"), 10.2, 7, 0.0, "wrong robot"),
        (dict(calibration_hash="x"), 10.2, 7, 0.0, "calibration"),
        (dict(limits_version="x"), 10.2, 7, 0.0, "limits"),
        ({}, 10.5, 7, 0.0, "expired"),
        (dict(expiry_time_s=20.0), 11.0, 7, 0.0, "too old"),
        ({}, 10.2, 8, 0.0, "already issued"),
        ({}, 10.2, 7, 0.2, "planned start"),
        (dict(references=np.full((6, 6), 2.0)), 10.2, 7, 0.0, "outside limits"),
        (dict(references=np.full((6, 6), np.nan)), 10.2, 7, 0.0, "malformed"),
    ],
)
def test_supervisor_rejections(changes, now, seq, position, reason):
    assert reason in Supervisor(CONFIG).accept(packet(**changes), now, seq, np.full(6, position), True)


def test_supervisor_accepts_once_and_refuses_duplicates():
    supervisor = Supervisor(CONFIG)
    assert supervisor.accept(packet(), 10.2, 7, np.zeros(6), True) is None
    assert "duplicate" in supervisor.accept(packet(), 10.2, 7, np.zeros(6), True)
    assert "invalid" in supervisor.accept(packet(plan_id=2), 10.2, 7, np.zeros(6), False)


def test_dataset_info_checks_joint_order(tmp_path):
    names = [f"{j}.pos" for j in JOINT_ORDER]
    info = {"fps": 30, "robot_type": "so101_follower", "features": {"observation.state": {"names": names}, "action": {"names": names}}}
    path = tmp_path / "info.json"
    path.write_text(json.dumps(info))
    assert check_dataset_info(path).robot_type == "so101_follower"
    info["features"]["action"]["names"] = names[::-1]
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="joint order"):
        check_dataset_info(path)


def test_import_builds_valid_episode():
    state = np.zeros((10, 6))
    state[3, 0] = np.nan
    episode = episode_from_arrays("e", "dataset", UNITS, state, np.zeros((10, 6)), np.arange(10) / 30, "h", 2)
    assert episode.length == 10 and not episode.position_valid[3] and episode.position_valid[4]


def test_session_split_is_disjoint_and_complete():
    groups = {f"s{i}": [f"s{i}-e{j}" for j in range(3)] for i in range(10)}
    split = split_episodes(groups)
    assert sum(len(v) for v in split.values()) == 30
    sessions = [{e.split("-")[0] for e in v} for v in split.values()]
    assert all(not (a & b) for i, a in enumerate(sessions) for b in sessions[i + 1 :])
    assert all(split.values())
    with pytest.raises(ValueError):
        split_episodes({"a": ["1"], "b": ["2"]})


@pytest.mark.parametrize("sessions", range(4, 25))
def test_session_split_never_empties_or_shares_a_split(sessions):
    """Regression: 4 to 8 sessions left calibration empty, and at 4 or 5 the
    validation session was also the test session."""
    groups = {f"s{i}": [f"s{i}-e"] for i in range(sessions)}
    split = split_episodes(groups)
    assigned = [e for v in split.values() for e in v]
    assert all(split.values())
    assert len(assigned) == len(set(assigned)) == sessions
    assert len(split["train"]) >= max(len(split[k]) for k in ("validation", "calibration", "test"))
