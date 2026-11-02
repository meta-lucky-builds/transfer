"""Single conversion point between device, physical, compact and legacy units.

Spec section 3. Four coordinate systems exist and none may be confused:

device   what LeRobot reads and writes. Body joints in degrees (use_degrees=True)
         or RANGE_M100_100. Gripper in RANGE_0_100.
physical body joints in radians, gripper aperture proxy g in [0, 1] (0 closed).
compact  (physical - center) / scale, used by the servo model and the command
         encoder.
legacy   what the so-101_joints checkpoint saw in training. Body joints in
         radians, gripper = raw LeRobot value * pi/180 (so101.md), i.e. about
         0 to 1.745 for a 0 to 100 source. Not a [0, 1] closedness.

All functions accept numpy arrays or torch tensors with the joint axis last and
keep the input type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
NUM_JOINTS = 6
GRIPPER_INDEX = 5
DEG2RAD = math.pi / 180.0


def _like(values: Sequence[float], ref):
    """Return values as a constant of the same kind, dtype and device as ref."""
    if isinstance(ref, np.ndarray):
        return np.asarray(values, dtype=ref.dtype)
    return ref.new_tensor(values)


def _copy(x):
    return x.copy() if isinstance(x, np.ndarray) else x.clone()


@dataclass(frozen=True)
class So101Units:
    """Verified unit contract for one calibrated arm.

    gripper_device_closed and gripper_device_open must come from a physical test
    on the arm. gripper_direction_verified guards every conversion that depends
    on them, because an inverted gripper silently reverses the behaviour.
    """

    use_degrees: bool = True
    gripper_device_closed: float = 0.0
    gripper_device_open: float = 100.0
    gripper_direction_verified: bool = False
    center: tuple[float, ...] = field(default=(0.0,) * 5 + (0.5,))
    scale: tuple[float, ...] = field(default=(1.0,) * 5 + (0.5,))

    def __post_init__(self):
        if len(self.center) != NUM_JOINTS or len(self.scale) != NUM_JOINTS:
            raise ValueError("center and scale need six values in JOINT_ORDER")
        if min(self.scale) <= 0:
            raise ValueError("compact scale must be positive")
        if self.gripper_device_open == self.gripper_device_closed:
            raise ValueError("gripper closed and open device values must differ")

    def _require_degrees(self):
        # RANGE_M100_100 is a per-joint calibrated percentage. Converting it as
        # though it were degrees is the failure the spec warns about in 3.4.
        if not self.use_degrees:
            raise ValueError("body joints are RANGE_M100_100; convert with calibration before using these units")

    def _require_gripper(self):
        if not self.gripper_direction_verified:
            raise ValueError("gripper direction is unverified; run the device direction test first")

    def device_to_physical(self, device):
        self._require_degrees()
        self._require_gripper()
        out = _copy(device) * 1.0
        out[..., :5] = device[..., :5] * DEG2RAD
        span = self.gripper_device_open - self.gripper_device_closed
        out[..., GRIPPER_INDEX] = (device[..., GRIPPER_INDEX] - self.gripper_device_closed) / span
        return out

    def physical_to_device(self, physical):
        self._require_degrees()
        self._require_gripper()
        out = _copy(physical) * 1.0
        out[..., :5] = physical[..., :5] / DEG2RAD
        span = self.gripper_device_open - self.gripper_device_closed
        out[..., GRIPPER_INDEX] = physical[..., GRIPPER_INDEX] * span + self.gripper_device_closed
        return out

    def physical_to_compact(self, physical):
        return (physical - _like(self.center, physical)) / _like(self.scale, physical)

    def compact_to_physical(self, compact):
        return compact * _like(self.scale, compact) + _like(self.center, compact)

    def physical_to_legacy(self, physical):
        """Legacy gripper is the device gripper value times pi/180, not g."""
        self._require_gripper()
        out = _copy(physical) * 1.0
        span = self.gripper_device_open - self.gripper_device_closed
        device_gripper = physical[..., GRIPPER_INDEX] * span + self.gripper_device_closed
        out[..., GRIPPER_INDEX] = device_gripper * DEG2RAD
        return out

    def legacy_to_physical(self, legacy):
        self._require_gripper()
        out = _copy(legacy) * 1.0
        span = self.gripper_device_open - self.gripper_device_closed
        device_gripper = legacy[..., GRIPPER_INDEX] / DEG2RAD
        out[..., GRIPPER_INDEX] = (device_gripper - self.gripper_device_closed) / span
        return out

    def physical_delta_to_legacy(self, delta):
        """Convert a displacement. The gripper offset must not be applied here."""
        out = _copy(delta) * 1.0
        span = self.gripper_device_open - self.gripper_device_closed
        out[..., GRIPPER_INDEX] = delta[..., GRIPPER_INDEX] * span * DEG2RAD
        return out


def propagate_joint_state(legacy_state, legacy_delta):
    """Spec 3.4 item 1: p' = p + delta on the six joints, padding stays zero.

    Accepts the padded [..., state_dim] state and either a padded or a
    six-dimensional delta. No clipping: upstream compute_new_pose clips slot 6
    to [0, 1], and legacy gripper values reach about 1.745.
    """
    out = _copy(legacy_state) * 1.0
    out[..., :NUM_JOINTS] = legacy_state[..., :NUM_JOINTS] + legacy_delta[..., :NUM_JOINTS]
    return out
