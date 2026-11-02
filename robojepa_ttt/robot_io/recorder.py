"""Record a teleoperated or planned SO-101 session with the spec's data contract.

LeRobot's recorder keeps the requested action and discards what send_action returns,
and it stores no per-camera capture times. This loop keeps both commands. read_latest
gives no capture time either, so frames carry their read time and are flagged as
estimated. For a measured capture time, patch the camera thread.

Hardware code. It has not been run against an arm.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from .event_log import EventLog
from .so101_units import JOINT_ORDER


def _vector(values: dict) -> np.ndarray:
    return np.array([values[f"{name}.pos"] for name in JOINT_ORDER], dtype=np.float64)


def record_session(
    robot,
    log: EventLog,
    next_action: Callable[[np.ndarray], dict],
    frame_directory: str | Path,
    duration_s: float,
    hz: float = 30.0,
    plan_version: str = "teleop",
) -> None:
    """robot: a connected LeRobot SOFollower. All logged values are device units.

    next_action receives the measured device position and returns a LeRobot action
    dict. Frames are written by the caller's choice of format: here, one npy per
    frame, which is simple and large. Swap for a video writer in long sessions.
    """
    frame_directory = Path(frame_directory)
    frame_directory.mkdir(parents=True, exist_ok=True)
    period = 1.0 / hz
    deadline = time.monotonic()
    end = deadline + duration_s
    tick = 0
    while time.monotonic() < end:
        before = time.monotonic()
        observation = robot.get_observation()
        after = time.monotonic()
        position = _vector(observation)
        log.joint(position, valid=bool(np.isfinite(position).all()), monotonic_s=before)
        for name in robot.cameras:
            reference = frame_directory / f"{name}_{tick:08d}.npy"
            np.save(reference, observation[name])
            log.frame(name, capture_s=after, receipt_s=after, frame_ref=reference.name, estimated=True)
        requested = next_action(position)
        sent = robot.send_action(requested)
        log.command(_vector(requested), _vector(sent), plan_version)
        tick += 1
        deadline += period
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            log.marker("overrun", late_s=-delay, tick=tick)
