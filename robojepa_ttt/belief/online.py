"""Online belief: fixed-lag replay, observation dedupe, change monitor, mode switching
(spec 7.7, 8.1, 8.2).

FixedLagBelief keeps, for the last `window_s` seconds, every macro step's inputs and
the belief before that step. A late observation for a past step is merged into that
step's record, the belief is restored to the snapshot before it, and every later
step is recomputed once, in order. The result equals a run in which the observation
had arrived on time (a test checks this). Observations older than the window are
not applied to the live belief; they are returned as dropped for the caller to keep
for training (spec 7.7: never apply a late measurement directly to the present).

Every observation carries an id per block ("joints", "view6", ...). An id already
assimilated is masked out of any later step, so a duplicated or retransmitted frame
is never counted twice. Replay recomputes from snapshots, so evidence already fused
into a snapshot is never reapplied either.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Optional

import torch

from .compact import BeliefState, CompactBelief, StepInput, StepOutput
from .imm import mixture_moments


@dataclass
class _Record:
    time_s: float
    step: StepInput  # batch of one
    ids: dict[str, str]  # block -> observation id actually assimilated in this record
    predict: bool


class FixedLagBelief:
    def __init__(self, model: CompactBelief, window_s: float = 2.0):
        self.model = model
        self.window_s = window_s
        self.blocks = model.config.blocks()
        self.records: deque[_Record] = deque()
        self.snapshots: deque[BeliefState] = deque()  # state before each record
        self.seen: set[str] = set()
        self.state: Optional[BeliefState] = None
        self.last_output: Optional[StepOutput] = None
        self.dropped: list[tuple[float, str]] = []

    # ------------------------------------------------------------ helpers

    def _strip_seen(self, step: StepInput, ids: dict[str, str]) -> tuple[StepInput, dict[str, str]]:
        """Mask blocks whose id was already assimilated, or that repeat within this step."""
        mask = step.obs_mask.clone()
        kept = {}
        for block, obs_id in ids.items():
            start, end = self.blocks[block]
            if obs_id in self.seen or obs_id in kept.values():
                mask[:, start:end] = False
            elif bool(mask[:, start:end].any()):
                kept[block] = obs_id
        return replace(step, obs_mask=mask), kept

    @torch.no_grad()
    def _apply(self, state: BeliefState, record: _Record) -> tuple[BeliefState, StepOutput]:
        return self.model.step(state, record.step, predict=record.predict)

    def _prune(self) -> None:
        if not self.records:
            return
        horizon = self.records[-1].time_s - self.window_s
        while len(self.records) > 1 and self.records[0].time_s < horizon:
            self.records.popleft()
            self.snapshots.popleft()

    # ---------------------------------------------------------------- api

    @torch.no_grad()
    def start(self, time_s: float, step: StepInput, ids: dict[str, str], regimes=(0,)) -> StepOutput:
        """Initialise from the first frame's joints, then assimilate its visual block."""
        if "joints" not in ids or not bool(step.obs_mask[:, :6].all()):
            raise ValueError("the first frame needs a valid joint reading with an id")
        state = self.model.initial_state(step.observation[:, :6], torch.ones(1, dtype=torch.bool), regimes)
        self.seen.add(ids["joints"])  # initialisation consumed it
        visual_ids = {k: v for k, v in ids.items() if k != "joints"}
        stripped, kept = self._strip_seen(step, visual_ids)
        record = _Record(time_s, stripped, {"joints": ids["joints"], **kept}, predict=False)
        self.snapshots.append(state)
        self.records.append(record)
        self.state, self.last_output = self._apply(state, record)
        self.seen.update(kept.values())
        return self.last_output

    @torch.no_grad()
    def observe(self, time_s: float, step: StepInput, ids: dict[str, str]) -> StepOutput:
        """A new macro step, later than every step so far."""
        if self.state is None:
            raise RuntimeError("call start first")
        if time_s <= self.records[-1].time_s:
            raise ValueError("observe is for new steps; use late() for a past step")
        stripped, kept = self._strip_seen(step, ids)
        record = _Record(time_s, stripped, kept, predict=True)
        self.snapshots.append(self.state)
        self.records.append(record)
        self.state, self.last_output = self._apply(self.state, record)
        self.seen.update(kept.values())
        self._prune()
        return self.last_output

    @torch.no_grad()
    def late(self, time_s: float, block: str, values: torch.Tensor, obs_id: str) -> str:
        """Merge an observation block that belongs to a past step and replay.

        Returns "replayed", "duplicate" or "dropped" (older than the window, or no
        step at that time). A block that step had already assimilated under another
        id is not replaced: two readings of one block at one time would be double
        evidence, so the new one is dropped.
        """
        if obs_id in self.seen:
            return "duplicate"
        index = next((i for i, r in enumerate(self.records) if abs(r.time_s - time_s) < 1e-9), None)
        if index is None:
            self.dropped.append((time_s, obs_id))
            return "dropped"
        record = self.records[index]
        if block in record.ids:
            self.dropped.append((time_s, obs_id))
            return "dropped"
        if block == "joints" and not record.predict:
            raise ValueError("the initialising joint reading cannot arrive late")
        start, end = self.blocks[block]
        observation = record.step.observation.clone()
        mask = record.step.obs_mask.clone()
        observation[:, start:end] = values.to(observation.dtype)
        mask[:, start:end] = True
        self.records[index] = replace(record, step=replace(record.step, observation=observation, obs_mask=mask), ids={**record.ids, block: obs_id})
        state = self.snapshots[index]
        for i in range(index, len(self.records)):
            self.snapshots[i] = state
            state, output = self._apply(state, self.records[i])
        self.state, self.last_output = state, output
        self.seen.add(obs_id)
        return "replayed"

    def message(self) -> dict:
        """Belief message of spec 7.7, as plain tensors for publishing."""
        if self.state is None:
            raise RuntimeError("no belief yet")
        mean, cov = mixture_moments(self.state.log_pi, self.state.mean, self.state.cov)
        return {
            "time_s": self.records[-1].time_s,
            "regimes": self.state.regimes,
            "regime_probabilities": self.state.log_pi.exp()[0],
            "component_means": self.state.mean[0],
            "component_covariances": self.state.cov[0],
            "summary_mean": mean[0],
            "summary_covariance": cov[0],
            "view_age_s": self.state.age_s[0],
            "incorporated_ids": sorted(self.seen),
            "motion_allowed": bool(self.last_output.motion_allowed[0]) if self.last_output is not None else False,
            "covariance_repairs": self.model.repairs.count,
        }


# ------------------------------------------------------------------ change


@dataclass
class ChangeMonitor:
    """CUSUM on a calibrated pre-update discrepancy, eq. 21.

    v is standardised with no-shift calibration statistics, C = max(0, C + a - kappa),
    and an alarm needs C > threshold after at least minimum_valid valid observations.
    These are calibration candidates, not false-alarm guarantees (spec 8.1). A
    detector identifies inconsistency, not its cause.
    """

    kappa: float = 0.5
    threshold: float = 10.0
    minimum_valid: int = 5
    calibration_mean: Optional[float] = None
    calibration_std: Optional[float] = None
    statistic: float = 0.0
    valid_count: int = 0

    def calibrate(self, values: torch.Tensor) -> None:
        values = torch.as_tensor(values, dtype=torch.float64)
        if values.numel() < 2:
            raise ValueError("calibration needs no-shift values")
        self.calibration_mean = float(values.mean())
        self.calibration_std = float(values.std())

    def update(self, value: float, valid: bool = True) -> bool:
        if self.calibration_mean is None:
            raise RuntimeError("calibrate on no-shift data first")
        if not valid:
            return False
        a = (value - self.calibration_mean) / max(self.calibration_std, 1e-3)
        self.statistic = max(0.0, self.statistic + a - self.kappa)
        self.valid_count += 1
        return self.valid_count >= self.minimum_valid and self.statistic > self.threshold

    def reset(self) -> None:
        self.statistic = 0.0
        self.valid_count = 0


@dataclass
class ModeSwitch:
    """Dominant-regime selection of spec 8.2.

    The planner's dominant regime changes only after another regime's posterior
    exceeds switch_probability for hold_observations consecutive valid observations.
    Regimes above planning_cutoff stay in the planning set meanwhile.
    """

    switch_probability: float = 0.8
    hold_observations: int = 5
    planning_cutoff: float = 0.05
    dominant: int = 0
    candidate: Optional[int] = None
    streak: int = 0

    def update(self, probabilities: torch.Tensor, valid: bool) -> int:
        if not valid:
            return self.dominant
        best = int(torch.argmax(probabilities))
        if best != self.dominant and float(probabilities[best]) > self.switch_probability:
            self.streak = self.streak + 1 if self.candidate == best else 1
            self.candidate = best
            if self.streak >= self.hold_observations:
                self.dominant, self.candidate, self.streak = best, None, 0
        else:
            self.candidate, self.streak = None, 0
        return self.dominant

    def planning_set(self, probabilities: torch.Tensor) -> list[int]:
        keep = [i for i, p in enumerate(probabilities.tolist()) if p > self.planning_cutoff]
        return keep if self.dominant in keep else [self.dominant, *keep]
