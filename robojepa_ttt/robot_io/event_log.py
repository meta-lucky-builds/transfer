"""Append-only event log, the authoritative record of what happened (spec 4, 10.1).

Requested target, issued target and measured position are separate fields of
separate events. Nothing in this module can overwrite one with another. A
sequence number proves ordering; monotonic time proves spacing. Each line is
flushed so a crash loses at most the line being written.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SessionHeader:
    session_id: str
    episode_id: str
    robot_id: str
    calibration_hash: str
    software_commit: str
    use_degrees: bool
    commands_source: str = "issued"  # "issued" or "requested_only" for imported data
    timing_source: str = "monotonic_capture"  # or "nominal_fps" for imported data


@dataclass(frozen=True)
class JointEvent:
    seq: int
    monotonic_s: float
    position: tuple[float, ...]
    valid: bool
    kind: str = "joint"


@dataclass(frozen=True)
class CommandEvent:
    seq: int
    monotonic_s: float
    requested: tuple[float, ...]
    issued: tuple[float, ...]
    clipped: bool
    plan_version: str
    acknowledged: Optional[bool] = None
    kind: str = "command"


@dataclass(frozen=True)
class FrameEvent:
    seq: int
    camera: str
    capture_s: float
    receipt_s: float
    frame_ref: str
    capture_is_estimated: bool
    kind: str = "frame"


@dataclass(frozen=True)
class MarkerEvent:
    """Physical reset, camera move, power cycle, intervention label, task change."""

    seq: int
    monotonic_s: float
    marker: str
    detail: dict = field(default_factory=dict)
    kind: str = "marker"


class EventLog:
    def __init__(self, path: str | os.PathLike, header: SessionHeader, clock=time.monotonic):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"refusing to append to an existing session log: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")
        self._clock = clock
        self._seq = 0
        self._write({"kind": "header", "schema": SCHEMA_VERSION, **asdict(header)})

    def _write(self, record: dict):
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._file.flush()

    def _next(self) -> int:
        self._seq += 1
        return self._seq

    def joint(self, position, valid: bool, monotonic_s: Optional[float] = None) -> JointEvent:
        t = self._clock() if monotonic_s is None else monotonic_s
        event = JointEvent(self._next(), t, tuple(float(v) for v in position), bool(valid))
        self._write(asdict(event))
        return event

    def command(self, requested, issued, plan_version: str, acknowledged=None, tolerance=1e-9) -> CommandEvent:
        requested = tuple(float(v) for v in requested)
        issued = tuple(float(v) for v in issued)
        clipped = any(abs(r - i) > tolerance for r, i in zip(requested, issued))
        event = CommandEvent(self._next(), self._clock(), requested, issued, clipped, plan_version, acknowledged)
        self._write(asdict(event))
        return event

    def frame(self, camera: str, capture_s: float, receipt_s: float, frame_ref: str, estimated: bool) -> FrameEvent:
        event = FrameEvent(self._next(), camera, capture_s, receipt_s, frame_ref, estimated)
        self._write(asdict(event))
        return event

    def marker(self, marker: str, **detail) -> MarkerEvent:
        event = MarkerEvent(self._next(), self._clock(), marker, detail)
        self._write(asdict(event))
        return event

    def close(self):
        self._file.close()


def read_events(path: str | os.PathLike) -> Iterator[dict]:
    """Yield events in file order, rejecting a broken sequence."""
    expected = 1
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["kind"] != "header":
                if record["seq"] != expected:
                    raise ValueError(f"event sequence broken at {record['seq']}, expected {expected}")
                expected += 1
            yield record
