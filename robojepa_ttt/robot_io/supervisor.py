"""Local acceptance of cloud plans (spec 15.2, 15.3).

Runs on the machine connected to the arm. It trusts no plan, and every rejection has a
named reason for the event log. Commands are never replayed after reconnect. What the
robot does on rejection is the commissioned hold or stop policy, supplied by the
operator, because disabling torque can drop an arm or an object (spec 15.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PlanPacket:
    plan_id: int
    robot_id: str
    calibration_hash: str
    limits_version: str
    model_version: str
    observation_time_s: float
    incorporated_action_seq: int
    issue_time_s: float
    expiry_time_s: float
    expected_start: tuple[float, ...]
    start_tolerance: tuple[float, ...]
    references: np.ndarray  # [ticks, 6] physical


@dataclass(frozen=True)
class SupervisorConfig:
    robot_id: str
    calibration_hash: str
    limits_version: str
    maximum_observation_age_s: float
    position_min: tuple[float, ...]
    position_max: tuple[float, ...]


class Supervisor:
    def __init__(self, config: SupervisorConfig):
        self.config = config
        self.last_plan_id = -1

    def rejection(
        self, plan: PlanPacket, now_s: float, action_seq: int, position: np.ndarray, position_valid: bool
    ) -> Optional[str]:
        """Return the reason to reject, or None to accept. Times are local monotonic."""
        c = self.config
        if plan.robot_id != c.robot_id:
            return "wrong robot"
        if plan.calibration_hash != c.calibration_hash:
            return "calibration mismatch"
        if plan.limits_version != c.limits_version:
            return "limits mismatch"
        if plan.plan_id <= self.last_plan_id:
            return "duplicate or reordered plan"
        if now_s >= plan.expiry_time_s:
            return "expired"
        if now_s - plan.observation_time_s > c.maximum_observation_age_s:
            return "observation too old"
        if plan.incorporated_action_seq != action_seq:
            return "plan ignores commands already issued"
        if not position_valid:
            return "joint feedback invalid"
        start_error = np.abs(np.asarray(position) - np.asarray(plan.expected_start))
        if (start_error > np.asarray(plan.start_tolerance)).any():
            return "arm is not at the planned start"
        references = np.asarray(plan.references)
        if references.ndim != 2 or references.shape[1] != 6 or not np.isfinite(references).all():
            return "malformed references"
        if (references < np.asarray(c.position_min)).any() or (references > np.asarray(c.position_max)).any():
            return "references outside limits"
        return None

    def accept(self, plan: PlanPacket, now_s, action_seq, position, position_valid) -> Optional[str]:
        reason = self.rejection(plan, now_s, action_seq, position, position_valid)
        if reason is None:
            self.last_plan_id = plan.plan_id
        return reason
