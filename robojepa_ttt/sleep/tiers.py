"""Multi-frequency adapter tiers with masked pre-allocated slots: the memory substrate of Sleep.

Normative source: docs/SLEEP_SPECIFICATION.md ("sleep spec"), section 0 contract items C1, C2,
C3, C6 and C11, sections 3.3, 4, 6.1-6.2 and 9.1-9.4. "Spec" alone means SPECIFICATION.md.
Paper basis: "Language Models Need Sleep" (Behrouz et al., arXiv 2606.03979v2, "the paper")
and "Nested Learning" (arXiv 2512.24695, "NL").

What the paper says, and what is used here
------------------------------------------
* CMS (paper 2.2 eq. 1-2, NL 7.1 eq. 70-71). Memory is a chain of blocks with update
  frequencies f_1 >= ... >= f_k. Block l is updated every C(l) steps with the error it
  accumulated over its chunk. The paper's requirement is "before each update of a memory
  block, a mechanism consolidates the abstracted knowledge of that block to more stable
  parameters" (2.2). Here every tier reads the same input and the tier outputs are summed.
  That is NL's *independent* CMS variant (eq. 74) with Agg = sum, sleep spec eq. (S1):
      W^(r) = W_0 + dW_offline^(r) + sum_{k active} dW_slow^(r,k) + dW_fast^(r).
  DEVIATION from eq. 2: slower tiers are not written by chunk-accumulated gradients. They
  are written only by gated sleep consolidation (sleep spec 3.2, 12).
* Parameter expansion (paper 3.2). The receiving block is a set of low-rank experts. It
  gains one new expert {A, B} per consolidation, so transferred knowledge does not overwrite
  what that block already stores. Implementation note (paper 3.3): experts are pre-allocated
  and "masked ... in the forward and backward pass, before their initial activation". Here
  such an expert is a *slot*. An inactive slot is skipped in the forward pass, so it gets no
  gradient (its .grad stays None) and cannot change any output. No tensor shape ever
  changes (sleep spec I2).
* Synaptic pruning (paper 3.3 step (c)(b)). After consolidation the sender's experts are
  reset. `reset_tier` sets B <- 0, so the tier contributes exactly zero. It re-initialises
  A from a seeded stream that does not use the global RNG.
* Fig. 7 (high -> mid -> low frequency FFNs, 1k -> 5k -> 10k) is the tier chain. Fig. 8 (a
  router updates a few experts per sleep cycle, the others stay inactive) is the slot mask.
  DEVIATION: there is no learned router. Every active slot of the current regime is always
  on, and the regime index (spec 8.2) chooses the slot bank (sleep spec 12: a soft router
  would let weights depend on context at fast timescales, which spec 12.3 forbids).
* Appendix B: the paper uses 5 extra width-64 MLP blocks, LoRA rank 64 and alpha 128, and
  keeps the active parameter count unchanged. None of these numbers carry over. The ranks
  here (8, alpha 16) are inherited from spec 6.6, and the slot count (8) is a sleep spec
  14 hypothesis.

Tier roles (sleep spec 3.1, 4.1)
--------------------------------
``tiers[0]`` is the *wake* tier ("fast"). It has exactly 1 slot, is always active, and is
trained online. The middle tiers ("slow" by default) are written only in sleep and have N_S
pre-allocated masked slots per regime. The last tier, ``offline``, has 1 slot, holds the
stage-3 adapters and is frozen. If the caller's tiers do not end in an ``offline`` tier,
`convert_to_tiered` appends one with the legacy LoRA rank and alpha, so the stage-3 copy is
bit-exact.

Layout and names (C1, C2, C3)
-----------------------------
LoRA, per wrapped projection (Q slice, V slice and output; K is untouched, spec 6.6):

    <proj>.deltas.<r>.<i>.lora_[ab]                          fast tier, legacy names (C2)
    <proj>.tiers.<tier>.deltas.<r>.slots.<k>.<i>.lora_[ab]   slow and offline slots
    <proj>.tiers.<tier>.deltas.<r>.slot_active               0/1 float vector (C1)

`TieredLoRALinear` subclasses `LoRALinear` and its ``deltas`` is *exactly* the fast tier
(a ModuleList[regime][slice] of LoRADelta). The unchanged `lora_parameters`, and with it
`AdaptedPredictor.set_trainable("online")` and `OnlineBuilder`, therefore select only the
fast tier. `set_lora_regime` keeps working.

Token residual, per regime (C3). The trunk (a_o, the modulation hidden layers, the regime
and view embeddings, and the command encoder) is shared between tiers and frozen:

    z_hat = LN0[ z0 + sum_k b_o,k tanh(a_o z0 + m_k(M(c))) ]
    residual.regimes.<r>.modulation_out / b_o                     fast slot (legacy names)
    residual.regimes.<r>.deltas.<tier>.slots.<k>.{modulation_out,b_o}.*   slow and offline
    residual.regimes.<r>.deltas.<tier>.slot_{active,generation,birth}     slot metadata
    residual.regimes.<r>.deltas.<fast>.slot_generation                    fast reset counter
    residual.tier_schema                                     schema tag (C6), int64 bytes

Every tier tensor's name contains ".deltas.", and the residual lies outside the predictor.
`AdaptedPredictor.clone_role` therefore never shares a tier tensor, a mask or a counter
between roles. It shares only the frozen base. Every one of them is a *parameter*, so
`adapter_state`, `state_digest`, `load_adapter_state`, `ModelHub` and `VersionStore` carry
the slot map, the reset counters and the schema together with the weights. Masks are float
parameters with requires_grad=False. Counters and the schema tag are int64 parameters with
requires_grad=False, which keeps them exact under `model.to(torch.bfloat16)`. A registered
buffer would be wrong: clone_role shares every predictor buffer (C1).

Invariants (sleep spec 4.3), checked by `check_invariants`
----------------------------------------------------------
I1  An inactive slot has B == 0 exactly (LoRA lora_b, residual b_o weight and bias),
    slot_active == 0, and its A (LoRA lora_a, residual modulation_out) equals the seeded
    initialisation recorded by its generation counter.
I2  No tier operation changes a tensor shape or the set of parameter names.
I3  Every activation, reset, transplant, merge, compaction or tier-state load changes
    ``model.version``, so version-checked KV caches (spec 13.3) are rebuilt.
Masks agree across every container of a (tier, regime): every projection and the residual.

A converted model's class is switched to `TieredAdaptedPredictor`, a subclass of
`AdaptedPredictor` that adds no state. It refuses `set_trainable("offline")` (sleep spec
3.3) and refuses `load_adapter_state` of a snapshot with another schema (C6), including
every legacy snapshot. A legacy snapshot is migrated explicitly with
`migrate_legacy_state`, which applies the same mapping as `convert_to_tiered`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn

from ..learning.online_builder import build_optimizer
from ..models.adapted_predictor import AdaptedPredictor
from ..models.lora import LoRADelta, LoRALinear
from ..models.token_residual import TokenResidual, _RegimeResidual

SCHEMA_NAME = "tiered-v1"
LEGACY_SCHEMA = "legacy-v1"
OFFLINE = "offline"
SCHEMA_KEY = "residual.tier_schema"

_B_ROLES = ("lora_b", "b_o.weight", "b_o.bias")
_A_ROLES = ("lora_a", "modulation_out.weight", "modulation_out.bias")


# =============================================================================== errors


class TierError(ValueError):
    """A tier operation was refused. Nothing was modified."""


class TierFullError(TierError):
    """No free slot. Slots are never evicted silently (sleep spec 4.4)."""


class TierStateError(TierError):
    """An invariant (I1, mask agreement) is violated, in the model or in a state to load."""


class TierSchemaError(TierError):
    """A snapshot of another parameterisation schema was offered (C6)."""


# =============================================================================== config

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class TierConfig:
    """One memory tier, i.e. one CMS frequency level (paper 2.2, NL 7.1).

    name               identifier. It must not shadow an nn.ModuleDict attribute.
    rank, alpha        rank k and alpha of every slot in the tier. scale = alpha / k (spec eq. 11).
    slots              number of pre-allocated experts per regime (paper 3.3 implementation note).
    consolidate_every  U_max (C11): the maximum number of builder updates between
                       consolidations *out of* this tier. It is a bound the scheduler enforces
                       (sleep spec 10.1 trigger T-a), not a clock. Sleep is event-triggered.
                       None means the tier is never consolidated at runtime (the offline
                       tier, and by default the slow tier, which only operator deep
                       consolidation empties, sleep spec 6.5).
    """

    name: str
    rank: int
    alpha: float
    slots: int
    consolidate_every: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not _NAME.match(self.name) or hasattr(nn.ModuleDict, self.name):
            raise TierError(f"tier name {self.name!r} must match {_NAME.pattern} and not shadow an nn.ModuleDict attribute")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise TierError(f"tier {self.name}: rank must be an int >= 1")
        if isinstance(self.slots, bool) or not isinstance(self.slots, int) or self.slots < 1:
            raise TierError(f"tier {self.name}: slots must be an int >= 1")
        if not (isinstance(self.alpha, (int, float)) and math.isfinite(self.alpha) and self.alpha > 0):
            raise TierError(f"tier {self.name}: alpha must be finite and > 0")
        ce = self.consolidate_every
        if ce is not None and (isinstance(ce, bool) or not isinstance(ce, int) or ce < 1):
            raise TierError(f"tier {self.name}: consolidate_every must be None or an int >= 1")

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


# Sleep spec 14 (hypotheses except where marked): the fast rank and alpha are inherited from
# spec 6.6. Fast U_max = 200 updates (about 2 h at 2B). Slow: 8 slots per regime, rank 8,
# alpha 16, emptied only by compaction or deep consolidation. The offline tier is derived from
# the stage-3 adapters at conversion.
DEFAULT_TIERS: tuple[TierConfig, ...] = (
    TierConfig("fast", rank=8, alpha=16.0, slots=1, consolidate_every=200),
    TierConfig("slow", rank=8, alpha=16.0, slots=8, consolidate_every=None),
)


def resolve_tiers(tiers: Sequence[TierConfig], legacy_rank: int, legacy_alpha: float) -> tuple[TierConfig, ...]:
    """Validate a tier list and append the offline tier when it is absent.

    Rules: names are unique; tiers[0] is the wake tier with exactly one slot; at least one
    sleep tier lies between wake and offline; ``offline`` may appear only last and then
    has 1 slot, consolidate_every None and rank >= the legacy rank; non-None
    consolidate_every values strictly increase from fast to slow (f_1 >= ... >= f_k, paper
    2.2). DEVIATION: the paper's divisibility of chunk sizes is not required, because sleep
    is event-triggered here (C11).
    """
    tiers = tuple(tiers)
    if not tiers or not all(isinstance(t, TierConfig) for t in tiers):
        raise TierError("tiers must be a non-empty sequence of TierConfig")
    if tiers[-1].name != OFFLINE:
        tiers = tiers + (TierConfig(OFFLINE, rank=legacy_rank, alpha=float(legacy_alpha), slots=1),)
    names = [t.name for t in tiers]
    if len(set(names)) != len(names):
        raise TierError(f"duplicate tier names {names}")
    if OFFLINE in names[:-1]:
        raise TierError("the offline tier must be last")
    if len(tiers) < 3:
        raise TierError("need a wake tier, at least one sleep tier and the offline tier")
    if tiers[0].slots != 1:
        raise TierError(f"wake tier {tiers[0].name} must have exactly 1 slot")
    offline = tiers[-1]
    if offline.slots != 1 or offline.consolidate_every is not None:
        raise TierError("the offline tier has 1 slot and consolidate_every=None")
    if offline.rank < legacy_rank:
        raise TierError(f"offline rank {offline.rank} < stage-3 rank {legacy_rank}: the copy could not be exact")
    periods = [t.consolidate_every for t in tiers if t.consolidate_every is not None]
    if any(b <= a for a, b in zip(periods, periods[1:])):
        raise TierError(f"consolidate_every must strictly increase from fast to slow: {periods}")
    return tiers


@dataclass(frozen=True)
class TierSchema:
    """Parameterisation schema of a converted model (C6).

    The tag hashes everything that gives the tier tensors their meaning: tier names, order,
    ranks, alphas, slot counts, the number of regimes, the legacy rank and alpha, and the
    init seed (I1 records seeded A). consolidate_every is scheduling, not parameterisation,
    and is excluded, so changing U_max does not orphan snapshots.
    """

    tiers: tuple[TierConfig, ...]
    regimes: int
    legacy_rank: int
    legacy_alpha: float
    seed: int
    name: str = SCHEMA_NAME

    def canonical(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "tiers": [[t.name, t.rank, float(t.alpha), t.slots] for t in self.tiers],
                "regimes": self.regimes,
                "legacy": [self.legacy_rank, float(self.legacy_alpha)],
                "seed": self.seed,
            },
            sort_keys=True,
        )

    def tag(self) -> torch.Tensor:
        return torch.tensor(list(hashlib.sha256(self.canonical().encode()).digest()), dtype=torch.int64)

    def identifier(self) -> str:
        return f"{self.name}:{hashlib.sha256(self.canonical().encode()).hexdigest()[:16]}"

    @property
    def fast(self) -> TierConfig:
        return self.tiers[0]

    @property
    def slotted(self) -> tuple[TierConfig, ...]:
        return self.tiers[1:]

    def tier(self, name: str) -> TierConfig:
        for t in self.tiers:
            if t.name == name:
                return t
        raise TierError(f"unknown tier {name!r}; tiers are {[t.name for t in self.tiers]}")


# =============================================================================== seeded init


def _seed_of(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") & ((1 << 63) - 1)


def _seeded(shape: Sequence[int], key: str, kind: str, fan_in: int) -> torch.Tensor:
    """Deterministic CPU float32 values; never touches the global RNG.

    kind "kaiming": LoRA A / nn.Linear weight default (kaiming_uniform, a=sqrt(5)).
    kind "bias":    nn.Linear bias default, U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
    """
    generator = torch.Generator().manual_seed(_seed_of(key))
    values = torch.empty(tuple(shape), dtype=torch.float32)
    if kind == "kaiming":
        nn.init.kaiming_uniform_(values, a=math.sqrt(5), generator=generator)
    else:
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(values, -bound, bound, generator=generator)
    return values


def _no_rng(factory):
    """Build a module with its usual constructor without consuming the global CPU RNG."""
    with torch.random.fork_rng(devices=[]):
        return factory()


def _delta(in_features: int, out_features: int, rank: int, alpha: float, device, dtype) -> LoRADelta:
    return _no_rng(lambda: LoRADelta(in_features, out_features, rank, alpha)).to(device=device, dtype=dtype)


def _linear(in_features: int, out_features: int, device, dtype) -> nn.Linear:
    return _no_rng(lambda: nn.Linear(in_features, out_features)).to(device=device, dtype=dtype)


def _flag(n: int, device, dtype=torch.float32) -> nn.Parameter:
    return nn.Parameter(torch.zeros(n, device=device, dtype=dtype), requires_grad=False)


# =============================================================================== LoRA


class _LoRASlots(nn.Module):
    """One regime of one slotted tier in one projection: slots[k][slice] + slot_active."""

    def __init__(self, slots: int, in_features: int, widths: Sequence[int], rank: int, alpha: float, device, dtype):
        super().__init__()
        self.slots = nn.ModuleList(
            nn.ModuleList(_delta(in_features, w, rank, alpha, device, dtype) for w in widths) for _ in range(slots)
        )
        self.slot_active = _flag(slots, device)


class _LoRATierBank(nn.Module):
    """One slotted tier of one projection: deltas[regime] -> _LoRASlots."""

    def __init__(self, tier: TierConfig, regimes: int, in_features: int, widths, device, dtype):
        super().__init__()
        self.deltas = nn.ModuleList(
            _LoRASlots(tier.slots, in_features, widths, tier.rank, tier.alpha, device, dtype) for _ in range(regimes)
        )


def _read_mask(mask: torch.Tensor, where: str) -> tuple[int, ...]:
    values = mask.detach().float().cpu().tolist()
    if any(v not in (0.0, 1.0) for v in values):
        raise TierStateError(f"{where}: slot mask holds non-binary values {values}")
    return tuple(i for i, v in enumerate(values) if v == 1.0)


def _cached_active(owner: nn.Module, cache_key, masks: Sequence[tuple[str, torch.Tensor]]) -> dict[str, tuple[int, ...]]:
    """Active slots per tier, re-read only when a mask may have changed.

    The key is (id, version counter, data pointer) of every mask: in-place writes
    (copy_ in load_adapter_state, fill_ here) bump the counter, and deepcopy or a device
    move changes id or pointer. One host read per change, none per forward.
    """
    key = tuple((id(m), m._version, m.data_ptr()) for _, m in masks)
    cache = owner.__dict__.setdefault("_tier_mask_cache", {})
    hit = cache.get(cache_key)
    if hit is not None and hit[0] == key:
        return hit[1]
    result = {name: _read_mask(m, f"{type(owner).__name__}[{cache_key}].{name}") for name, m in masks}
    cache[cache_key] = (key, result)
    return result


class TieredLoRALinear(LoRALinear):
    """A LoRALinear whose ``deltas`` is exactly the fast tier (C2), plus slotted tiers.

    Forward order (fixed, so adding a zero slot is bit-identical, sleep spec Theorem 9.1):
    offline, then slower to faster sleep tiers (slots in index order), then fast. Each slot
    runs the same op sequence as LoRADelta.forward and is accumulated separately.
    """

    def __init__(self, base: nn.Linear, slices: Sequence[tuple[int, int]], schema: TierSchema, device=None, dtype=None):
        nn.Module.__init__(self)
        ordered = sorted(slices)
        for (_, end), (start, _) in zip(ordered, ordered[1:]):
            if end > start:
                raise TierError("LoRA slices must not overlap")
        if ordered[0][0] < 0 or ordered[-1][1] > base.out_features:
            raise TierError(f"slices {ordered} exceed output width {base.out_features}")
        self.base = base
        self.slices = ordered
        self.active_regime = 0
        widths = [end - start for start, end in ordered]
        fast = schema.fast
        self.deltas = nn.ModuleList(
            nn.ModuleList(_delta(base.in_features, w, fast.rank, fast.alpha, device, dtype) for w in widths)
            for _ in range(schema.regimes)
        )
        self.tiers = nn.ModuleDict(
            {t.name: _LoRATierBank(t, schema.regimes, base.in_features, widths, device, dtype) for t in schema.slotted}
        )

    def active_plan(self, regime: int) -> list[tuple[str, tuple[int, ...]]]:
        """[(tier, active slots)] in summation order (slowest first), slotted tiers only."""
        masks = [(name, bank.deltas[regime].slot_active) for name, bank in self.tiers.items()]
        active = _cached_active(self, regime, masks)
        return [(name, active[name]) for name, _ in reversed(masks) if active[name]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        regime = self.active_regime
        plan = self.active_plan(regime)
        fast = self.deltas[regime]
        pieces, cursor = [], 0
        for index, (start, end) in enumerate(self.slices):
            delta = None
            for tier, slots in plan:
                bank = self.tiers[tier].deltas[regime].slots
                for slot in slots:
                    term = bank[slot][index](x)
                    delta = term if delta is None else delta + term
            term = fast[index](x)
            delta = term if delta is None else delta + term
            if start > cursor:
                pieces.append(out[..., cursor:start])
            pieces.append(out[..., start:end] + delta.to(out.dtype))
            cursor = end
        if cursor < out.shape[-1]:
            pieces.append(out[..., cursor:])
        return torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]


# =============================================================================== residual


class _ResidualSlot(nn.Module):
    """One residual expert: m_k = modulation_out (trunk -> bottleneck), b_o,k (bottleneck -> token)."""

    def __init__(self, trunk: int, bottleneck: int, token: int, device, dtype):
        super().__init__()
        self.modulation_out = _linear(trunk, bottleneck, device, dtype)
        self.b_o = _linear(bottleneck, token, device, dtype)


class _ResidualSlots(nn.Module):
    """One slotted tier of one regime's residual, with the slot metadata of that (tier, regime).

    slot_generation: number of resets of each slot (seeds the A re-init, I1).
    slot_birth: activation order within the bank (0 = inactive), for "oldest slot" policies
    (sleep spec 4.4) and attribution bookkeeping (4.2 iii).
    """

    def __init__(self, slots: int, trunk: int, bottleneck: int, token: int, device, dtype):
        super().__init__()
        self.slots = nn.ModuleList(_ResidualSlot(trunk, bottleneck, token, device, dtype) for _ in range(slots))
        self.slot_active = _flag(slots, device)
        self.slot_generation = _flag(slots, device, torch.int64)
        self.slot_birth = _flag(slots, device, torch.int64)


class _FastMeta(nn.Module):
    """Reset counter of the fast tier of one regime (LoRA and residual share it)."""

    def __init__(self, device):
        super().__init__()
        self.slot_generation = _flag(1, device, torch.int64)


class TieredRegimeResidual(_RegimeResidual):
    """A regime residual with a shared frozen trunk and one expert per tier slot (C3).

    ``modulation_out``/``b_o`` keep their legacy names and are the fast slot, so the unchanged
    `TokenResidual.online_parameters(r)` returns the fast tier. ``a_o`` and ``modulation``
    are the shared trunk (the same objects as before conversion, so belief code that edits
    ``modulation[0]`` keeps working).
    """

    def __init__(self, *args, **kwargs):  # pragma: no cover - built by from_legacy only
        raise TierError("build with TieredRegimeResidual.from_legacy")

    @classmethod
    def from_legacy(cls, legacy: _RegimeResidual, schema: TierSchema) -> "TieredRegimeResidual":
        if type(legacy) is not _RegimeResidual:
            raise TierError(f"unsupported residual regime class {type(legacy).__name__}")
        if set(legacy._modules) != {"a_o", "modulation", "modulation_out", "b_o"} or legacy._parameters or legacy._buffers:
            raise TierError(f"unrecognised residual regime structure {sorted(legacy._modules)}")
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.training = legacy.training
        device, dtype = legacy.b_o.weight.device, legacy.b_o.weight.dtype
        trunk, bottleneck = legacy.modulation_out.in_features, legacy.modulation_out.out_features
        token = legacy.b_o.out_features
        if legacy.a_o.out_features != bottleneck or legacy.b_o.in_features != bottleneck:
            raise TierError("residual a_o, modulation_out and b_o disagree on the bottleneck width")
        obj.a_o = legacy.a_o
        obj.modulation = legacy.modulation
        obj.modulation_out = _linear(trunk, bottleneck, device, dtype)
        obj.b_o = _linear(bottleneck, token, device, dtype)
        modules = {schema.fast.name: _FastMeta(device)}
        for t in schema.slotted:
            modules[t.name] = _ResidualSlots(t.slots, trunk, bottleneck, token, device, dtype)
        obj.deltas = nn.ModuleDict(modules)
        obj.slotted_names = tuple(t.name for t in schema.slotted)
        return obj

    def active_plan(self) -> list[tuple[str, tuple[int, ...]]]:
        masks = [(name, self.deltas[name].slot_active) for name in self.slotted_names]
        active = _cached_active(self, 0, masks)
        return [(name, active[name]) for name, _ in reversed(masks) if active[name]]

    def forward(self, z0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        trunk = self.modulation(cond)
        hidden = self.a_o(z0)
        out = None
        for tier, slots in self.active_plan():
            bank = self.deltas[tier].slots
            for slot in slots:
                expert = bank[slot]
                term = expert.b_o(torch.tanh(hidden + expert.modulation_out(trunk)))
                out = term if out is None else out + term
        term = self.b_o(torch.tanh(hidden + self.modulation_out(trunk)))
        return term if out is None else out + term


# =============================================================================== tiered model


class TieredAdaptedPredictor(AdaptedPredictor):
    """AdaptedPredictor after `convert_to_tiered`. It adds no state; the class is switched in place.

    * set_trainable("offline") raises: offline stage-3 training runs on unconverted models
      only (sleep spec 3.3). "online" (fast tier only, C2) and "none" are unchanged.
    * load_adapter_state refuses a state of another schema (C6), a legacy state, non-binary
      or disagreeing masks, and inactive slots with non-zero B (I1), before copying anything.
    """

    def set_trainable(self, mode: str):
        if mode == "offline":
            raise TierError("set_trainable('offline') is not valid on a tiered model (sleep spec 3.3); "
                            "new offline knowledge enters by deep consolidation (6.5)")
        return super().set_trainable(mode)

    def load_adapter_state(self, state: dict, version: str) -> None:
        validate_state(self, state)
        super().load_adapter_state(state, version)


def is_tiered(model: nn.Module) -> bool:
    return isinstance(model, TieredAdaptedPredictor)


def model_schema(model: nn.Module) -> TierSchema:
    if not is_tiered(model):
        raise TierError("model is not tiered; call convert_to_tiered first")
    return model.sleep_schema


def model_tiers(model: nn.Module) -> tuple[TierConfig, ...]:
    """All tiers of a converted model, fastest first, offline last."""
    return model_schema(model).tiers


def _lora_modules(model: nn.Module) -> list[tuple[str, TieredLoRALinear]]:
    return [(n, m) for n, m in model.predictor.named_modules() if isinstance(m, TieredLoRALinear)]


def _regimes(model: nn.Module) -> int:
    return model_schema(model).regimes


def _resolve_regimes(model: nn.Module, regime) -> list[int]:
    count = _regimes(model)
    if regime is None:
        regime = getattr(model, "regime", 0)
    if regime == "all":
        return list(range(count))
    if isinstance(regime, bool) or not isinstance(regime, int) or not 0 <= regime < count:
        raise TierError(f"regime {regime!r} out of range 0..{count - 1}")
    return [regime]


def _one_regime(model: nn.Module, regime) -> int:
    if regime == "all":
        raise TierError("this operation takes one regime")
    return _resolve_regimes(model, regime)[0]


def _slotted(model: nn.Module, tier: str) -> TierConfig:
    schema = model_schema(model)
    config = schema.tier(tier)
    if config.name == schema.fast.name:
        raise TierError(f"{tier} is the wake tier: it has one permanently active slot")
    return config


# =============================================================================== entries


@dataclass
class SlotEntry:
    """One tier tensor with its model-level name and coordinates.

    component "lora" | "residual" | "meta"; slot is None for metadata; path is the
    projection's module path inside the predictor (LoRA) or "residual.regimes.<r>".
    """

    name: str
    parameter: nn.Parameter
    component: str
    tier: str
    regime: Optional[int]
    slot: Optional[int]
    role: str
    path: str
    index: Optional[int] = None
    fan_in: Optional[int] = None  # residual modulation_out bias: in_features of its weight


def slot_entries(model: nn.Module) -> list[SlotEntry]:
    """Every tier tensor of the model (all slots, active or not, and all metadata), in model order."""
    schema = model_schema(model)
    names = {id(p): n for n, p in model.named_parameters()}
    fast = schema.fast.name
    entries: list[SlotEntry] = []
    for path, module in _lora_modules(model):
        for regime in range(schema.regimes):
            for index, delta in enumerate(module.deltas[regime]):
                for role in ("lora_a", "lora_b"):
                    p = getattr(delta, role)
                    entries.append(SlotEntry(names[id(p)], p, "lora", fast, regime, 0, role, path, index))
        for tier in schema.slotted:
            for regime in range(schema.regimes):
                bank = module.tiers[tier.name].deltas[regime]
                p = bank.slot_active
                entries.append(SlotEntry(names[id(p)], p, "meta", tier.name, regime, None, "slot_active", path))
                for slot, deltas in enumerate(bank.slots):
                    for index, delta in enumerate(deltas):
                        for role in ("lora_a", "lora_b"):
                            p = getattr(delta, role)
                            entries.append(SlotEntry(names[id(p)], p, "lora", tier.name, regime, slot, role, path, index))
    residual = model.residual
    for regime, module in enumerate(residual.regimes):
        path = f"residual.regimes.{regime}"
        for part in ("modulation_out", "b_o"):
            for kind in ("weight", "bias"):
                linear = getattr(module, part)
                p = getattr(linear, kind)
                entries.append(SlotEntry(names[id(p)], p, "residual", fast, regime, 0, f"{part}.{kind}", path,
                                         fan_in=linear.in_features))
        p = module.deltas[fast].slot_generation
        entries.append(SlotEntry(names[id(p)], p, "meta", fast, regime, None, "slot_generation", path))
        for tier in schema.slotted:
            bank = module.deltas[tier.name]
            for role in ("slot_active", "slot_generation", "slot_birth"):
                p = getattr(bank, role)
                entries.append(SlotEntry(names[id(p)], p, "meta", tier.name, regime, None, role, path))
            for slot, expert in enumerate(bank.slots):
                for part in ("modulation_out", "b_o"):
                    for kind in ("weight", "bias"):
                        linear = getattr(expert, part)
                        p = getattr(linear, kind)
                        entries.append(SlotEntry(names[id(p)], p, "residual", tier.name, regime, slot, f"{part}.{kind}",
                                                 path, fan_in=linear.in_features))
    p = residual.tier_schema
    entries.append(SlotEntry(names[id(p)], p, "meta", "", None, None, "schema", "residual"))
    return entries


def _bank(model: nn.Module, tier: str, regime: int) -> _ResidualSlots:
    return model.residual.regimes[regime].deltas[tier]


def _generation(model: nn.Module, tier: str, regime: int, slot: int) -> int:
    if tier == model_schema(model).fast.name:
        return int(model.residual.regimes[regime].deltas[tier].slot_generation[0])
    return int(_bank(model, tier, regime).slot_generation[slot])


def _init_key(schema: TierSchema, entry: SlotEntry, generation: int) -> str:
    return (f"{schema.seed}|{entry.component}|{entry.path}|{entry.tier}|r{entry.regime}|s{entry.slot}|"
            f"i{entry.index}|{entry.role}|g{generation}")


def _expected_init(schema: TierSchema, entry: SlotEntry, generation: int) -> torch.Tensor:
    """The seeded value an A-role tensor takes at the given generation (zeros for B roles)."""
    shape = tuple(entry.parameter.shape)
    if entry.role in _B_ROLES:
        return torch.zeros(shape)
    if entry.role == "modulation_out.bias":
        return _seeded(shape, _init_key(schema, entry, generation), "bias", entry.fan_in)
    return _seeded(shape, _init_key(schema, entry, generation), "kaiming", shape[1])


def _write_init(schema: TierSchema, entries: Iterable[SlotEntry], generation: int) -> None:
    with torch.no_grad():
        for e in entries:
            if e.component == "meta":
                continue
            value = _expected_init(schema, e, generation)
            e.parameter.copy_(value.to(e.parameter.device, e.parameter.dtype))


def _slot_tensors(entries: list[SlotEntry], tier: str, regime: int, slot: int) -> list[SlotEntry]:
    return [e for e in entries if e.component != "meta" and e.tier == tier and e.regime == regime and e.slot == slot]


def _masks(entries: list[SlotEntry], tier: str, regime: int) -> list[SlotEntry]:
    return [e for e in entries if e.role == "slot_active" and e.tier == tier and e.regime == regime]


def _b_is_zero(entries: list[SlotEntry]) -> bool:
    return not any(bool(e.parameter.detach().any()) for e in entries if e.role in _B_ROLES)


def _bump_version(model: nn.Module, operation: str, version: Optional[str] = None) -> None:
    """Invariant I3 (sleep spec 4.3): every slot event yields a new version string."""
    if version is not None:
        model.version = version
        return
    count = getattr(model, "_tier_edit_count", 0) + 1
    model._tier_edit_count = count
    root = str(getattr(model, "version", "base")).split("~")[0]
    model.version = f"{root}~{operation}.{count}"


# =============================================================================== conversion


@dataclass
class ConversionReport:
    modules: list[str]
    regimes: int
    tiers: tuple[TierConfig, ...]
    schema: str
    legacy_rank: int
    legacy_alpha: float
    offline_padded: bool  # offline rank > stage-3 rank (exact in exact arithmetic)
    max_abs_weight_error: float  # of the offline LoRA delta, from rescaling B (0 when scales match)
    probe_relative_error: Optional[float] = None  # mean|after - before| / mean|before| on the probe


def convert_to_tiered(
    model: AdaptedPredictor,
    tiers: Sequence[TierConfig] = DEFAULT_TIERS,
    *,
    seed: int = 0,
    probe: Optional[tuple[dict, int, int]] = None,
    tolerance: float = 1e-6,
) -> ConversionReport:
    """One-way migration of a stage-3 model to the tiered parameterisation (sleep spec 3.3), in place.

    1. Stage-3 LoRA of each regime -> the offline slot; the stage-3 residual (modulation_out,
       b_o) -> the offline residual slot. Both are exact copies (B rescaled only when the
       offline scale differs, reported).
    2. The fast tier starts at zero: B <- 0, A <- seeded init; the fast residual slot has
       b_o = 0 and a seeded modulation_out.
    3. Every sleep-tier slot is inactive with B = 0 and seeded A (I1).
    4. With ``probe=(batch, context, horizon)``, function equality is asserted: the relative
       mean absolute difference of the rollout must be <= tolerance (1e-6 fp32 by default;
       sleep spec 3.3 suggests 1e-2 for bf16). Otherwise the conversion is undone and
       TierError raised. In fp32 the conversion is in fact bit-exact: the offline slot runs
       the legacy op sequence and each zero tier adds exact zeros (Theorem 9.1).
    5. The schema (C6) is tagged on the residual, and ``model.version`` changes.

    Atomic: everything is built beside the model; the base and the residual trunk are shared,
    never written. Optimizers built before conversion hold the old (now detached) tensors and
    must be rebuilt. Apply belief.visual.expand_residual_for_belief *before* conversion when
    the residual has no hidden layers (with hidden layers the shared trunk makes it safe).
    """
    if not isinstance(model, AdaptedPredictor):
        raise TierError("convert_to_tiered needs an AdaptedPredictor")
    if is_tiered(model):
        raise TierError("model is already tiered")
    legacy = [(n, m) for n, m in model.predictor.named_modules() if isinstance(m, LoRALinear)]
    if not legacy:
        raise TierError("no LoRALinear modules found")
    if any(type(m) is not LoRALinear for _, m in legacy):
        raise TierError("unexpected LoRALinear subclass in the predictor")
    expected = getattr(model, "lora_modules", None)
    if expected is not None and sorted(expected) != sorted(n for n, _ in legacy):
        raise TierError("LoRA modules differ from model.lora_modules")
    regime_counts = {len(m.deltas) for _, m in legacy}
    if len(regime_counts) != 1:
        raise TierError(f"LoRA modules disagree on the regime count {regime_counts}")
    regimes = regime_counts.pop()
    residual = model.residual
    if not isinstance(residual, TokenResidual) or len(residual.regimes) != regimes:
        raise TierError("the residual is not a TokenResidual with one module per LoRA regime")
    if "tier_schema" in residual._parameters:
        raise TierError("the residual already carries a tier schema")
    ranks = {d.lora_a.shape[0] for _, m in legacy for r in m.deltas for d in r}
    scales = {d.scale for _, m in legacy for r in m.deltas for d in r}
    if len(ranks) != 1 or len(scales) != 1:
        raise TierError(f"stage-3 LoRA has mixed ranks {ranks} or scales {scales}")
    legacy_rank, legacy_scale = ranks.pop(), scales.pop()
    legacy_alpha = legacy_scale * legacy_rank
    resolved = resolve_tiers(tiers, legacy_rank, legacy_alpha)
    schema = TierSchema(resolved, regimes, legacy_rank, float(legacy_alpha), int(seed))
    offline = schema.tier(OFFLINE)
    ratio = legacy_scale / offline.scale

    reference = None
    if probe is not None:
        batch, context, horizon = probe
        with torch.no_grad():
            reference = model.rollout(batch, context, horizon).detach().clone()

    # ---- build beside the model
    replacements = []
    for name, module in legacy:
        sample = module.deltas[0][0].lora_a
        tiered = TieredLoRALinear(module.base, module.slices, schema, device=sample.device, dtype=sample.dtype)
        tiered.active_regime = module.active_regime
        tiered.train(module.training)
        replacements.append((name, module, tiered))
    new_regimes = [TieredRegimeResidual.from_legacy(r, schema) for r in residual.regimes]
    old_class, old_version = model.__class__, model.version
    old_regimes = list(residual.regimes)

    def install():
        for name, _, tiered in replacements:
            parent, _, attribute = name.rpartition(".")
            setattr(model.predictor.get_submodule(parent) if parent else model.predictor, attribute, tiered)
        for index, module in enumerate(new_regimes):
            residual.regimes[index] = module
        residual.register_parameter("tier_schema", nn.Parameter(schema.tag().to(new_regimes[0].b_o.weight.device), requires_grad=False))
        model.__class__ = TieredAdaptedPredictor
        model.sleep_schema = schema

    def uninstall():
        for name, module, _ in replacements:
            parent, _, attribute = name.rpartition(".")
            setattr(model.predictor.get_submodule(parent) if parent else model.predictor, attribute, module)
        for index, module in enumerate(old_regimes):
            residual.regimes[index] = module
        del residual._parameters["tier_schema"]
        model.__class__ = old_class
        del model.sleep_schema
        model.version = old_version

    install()
    try:
        entries = slot_entries(model)
        _write_init(schema, entries, 0)  # every slot, every tier: B = 0, seeded A (generation 0)
        error = 0.0
        with torch.no_grad():
            for (name, module, tiered) in replacements:
                for regime in range(regimes):
                    target = tiered.tiers[OFFLINE].deltas[regime]
                    for index, old in enumerate(module.deltas[regime]):
                        new = target.slots[0][index]
                        new.lora_a[:legacy_rank].copy_(old.lora_a)
                        if ratio == 1.0:
                            new.lora_b[:, :legacy_rank].copy_(old.lora_b)
                        else:
                            scaled = (old.lora_b.double() * ratio).to(new.lora_b.dtype)
                            new.lora_b[:, :legacy_rank].copy_(scaled)
                            exact = old.lora_b.double() * legacy_scale
                            error = max(error, float((scaled.double() * offline.scale - exact).abs().max()))
                    target.slot_active[0] = 1.0
            for regime, (old, new) in enumerate(zip(old_regimes, new_regimes)):
                bank = new.deltas[OFFLINE]
                bank.slots[0].modulation_out.load_state_dict(old.modulation_out.state_dict())
                bank.slots[0].b_o.load_state_dict(old.b_o.state_dict())
                bank.slot_active[0] = 1.0
                bank.slot_birth[0] = 1
        _bump_version(model, "tiered")
        problems = check_invariants(model)
        if problems:
            raise TierStateError(f"conversion produced an inconsistent model: {problems[:3]}")
        relative = None
        if reference is not None:
            with torch.no_grad():
                after = model.rollout(batch, context, horizon)
            relative = float((after - reference).abs().mean() / reference.abs().mean().clamp_min(1e-30))
            if not relative <= tolerance:
                raise TierError(f"conversion changed the function: relative error {relative:.3e} > {tolerance}")
    except BaseException:
        uninstall()
        raise
    return ConversionReport(
        [n for n, _, _ in replacements], regimes, resolved, schema.identifier(), legacy_rank, float(legacy_alpha),
        offline.rank > legacy_rank, error, relative,
    )


# =============================================================================== schema checks


def state_schema(state: Mapping[str, torch.Tensor]) -> str:
    """'legacy-v1' for an untiered adapter state, else '<name>:<hash prefix>' read from its tag."""
    tag = state.get(SCHEMA_KEY)
    if tag is None:
        return LEGACY_SCHEMA
    digest = bytes(int(v) for v in tag.detach().cpu().flatten().tolist())
    return f"{SCHEMA_NAME}:{digest.hex()[:16]}"


def schema_payload(model: nn.Module) -> dict:
    """JSON-able schema record for VersionStore metadata (C6). The tag inside every
    adapter_state already carries the hash; this adds the human-readable configs."""
    schema = model_schema(model)
    return {"parameterisation_schema": schema.identifier(), "canonical": json.loads(schema.canonical()),
            "consolidate_every": {t.name: t.consolidate_every for t in schema.tiers}}


def validate_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Refuse a state that is not a consistent snapshot of this model's schema (C6, C1, I1).

    Checks: tag present and equal; every mask binary and equal across containers of its
    (tier, regime); every inactive slot has B == 0. Key-set and shape agreement are checked
    by AdaptedPredictor.load_adapter_state / load_tier_state.
    """
    schema = model_schema(model)
    if SCHEMA_KEY not in state:
        raise TierSchemaError(f"cross-schema load refused: the state is {LEGACY_SCHEMA}, the model is "
                              f"{schema.identifier()}; migrate it with migrate_legacy_state")
    if not torch.equal(state[SCHEMA_KEY].detach().cpu().to(torch.int64).flatten(), schema.tag()):
        raise TierSchemaError(f"cross-schema load refused: the state is {state_schema(state)}, the model is "
                              f"{schema.identifier()}")
    entries = slot_entries(model)
    for tier in schema.slotted:
        for regime in range(schema.regimes):
            values = set()
            for e in _masks(entries, tier.name, regime):
                if e.name not in state:
                    raise TierSchemaError(f"state lacks {e.name}")
                value = state[e.name].detach().cpu().float()
                if value.shape != e.parameter.shape or not bool(((value == 0) | (value == 1)).all()):
                    raise TierStateError(f"{e.name}: mask is not a binary vector of shape {tuple(e.parameter.shape)}")
                values.add(tuple(value.tolist()))
            if len(values) != 1:
                raise TierStateError(f"state masks disagree for tier {tier.name} regime {regime}")
            mask = values.pop()
            for slot in range(tier.slots):
                if mask[slot] == 1.0:
                    continue
                for e in _slot_tensors(entries, tier.name, regime, slot):
                    if e.role in _B_ROLES and e.name in state and bool(state[e.name].detach().any()):
                        raise TierStateError(f"{e.name}: inactive slot with non-zero B in the state (I1)")


def check_invariants(model: nn.Module, check_init: bool = True) -> list[str]:
    """Violations of I1 and mask agreement (sleep spec 4.3, 10.3 (d)); empty when consistent.

    check_init also verifies that the A of every inactive slot equals its seeded init at its
    recorded generation (CPU work proportional to the inactive slot parameters).
    """
    schema = model_schema(model)
    problems: list[str] = []
    entries = slot_entries(model)
    if not torch.equal(model.residual.tier_schema.detach().cpu(), schema.tag()):
        problems.append("schema tag differs from the model's schema")
    for tier in schema.slotted:
        for regime in range(schema.regimes):
            maps = set()
            for e in _masks(entries, tier.name, regime):
                try:
                    maps.add(_read_mask(e.parameter, e.name))
                except TierStateError as error:
                    problems.append(str(error))
            if len(maps) > 1:
                problems.append(f"masks disagree for tier {tier.name} regime {regime}: {sorted(maps)}")
                continue
            if not maps:
                continue
            active = maps.pop()
            bank = _bank(model, tier.name, regime)
            for slot in range(tier.slots):
                if slot in active:
                    if int(bank.slot_birth[slot]) <= 0:
                        problems.append(f"{tier.name}[{slot}] r{regime}: active slot without a birth index")
                    continue
                if int(bank.slot_birth[slot]) != 0:
                    problems.append(f"{tier.name}[{slot}] r{regime}: inactive slot with a birth index")
                tensors = _slot_tensors(entries, tier.name, regime, slot)
                if not _b_is_zero(tensors):
                    problems.append(f"{tier.name}[{slot}] r{regime}: inactive slot with non-zero B (I1)")
                if check_init:
                    generation = int(bank.slot_generation[slot])
                    for e in tensors:
                        if e.role in _A_ROLES:
                            want = _expected_init(schema, e, generation).to(e.parameter.dtype)
                            if not torch.equal(e.parameter.detach().cpu(), want):
                                problems.append(f"{e.name}: inactive A differs from its seeded init (I1)")
    return problems


def assert_invariants(model: nn.Module, check_init: bool = True) -> None:
    problems = check_invariants(model, check_init)
    if problems:
        raise TierStateError("; ".join(problems[:5]))


# =============================================================================== slot map


def active_slots(model: nn.Module, regime: Optional[int] = None):
    """Active slots (sleep spec section 0 contract).

    regime None: {tier: {regime: [slots]}} for every tier and regime. The wake tier is always
    [0]. regime int: {tier: [slots]} for that regime. Raises TierStateError if containers
    disagree (C1).
    """
    schema = model_schema(model)
    entries = slot_entries(model)
    result: dict[str, dict[int, list[int]]] = {schema.fast.name: {r: [0] for r in range(schema.regimes)}}
    for tier in schema.slotted:
        result[tier.name] = {}
        for r in range(schema.regimes):
            maps = {_read_mask(e.parameter, e.name) for e in _masks(entries, tier.name, r)}
            if len(maps) != 1:
                raise TierStateError(f"slot masks disagree for tier {tier.name} regime {r}: {sorted(maps)}")
            result[tier.name][r] = list(maps.pop())
    if regime is None:
        return result
    r = _one_regime(model, regime)
    return {name: list(per[r]) for name, per in result.items()}


def slots_by_age(model: nn.Module, tier: str, regime: Optional[int] = None) -> list[int]:
    """Active slots of a sleep tier, oldest activation first (for sleep spec 4.4 compaction)."""
    _slotted(model, tier)
    r = _one_regime(model, regime)
    bank = _bank(model, tier, r)
    active = active_slots(model, r)[tier]
    return sorted(active, key=lambda s: int(bank.slot_birth[s]))


def _set_active(model: nn.Module, entries: list[SlotEntry], tier: str, regime: int, slot: int, value: bool) -> None:
    with torch.no_grad():
        for e in _masks(entries, tier, regime):
            e.parameter[slot] = 1.0 if value else 0.0
        bank = _bank(model, tier, regime)
        if value:
            bank.slot_birth[slot] = int(bank.slot_birth.max()) + 1
        else:
            bank.slot_birth[slot] = 0


def _reinit_slot(model: nn.Module, entries: list[SlotEntry], tier: str, regime: int, slot: int) -> None:
    """B <- 0 and A <- seeded init at generation + 1 (all projections, all slices, residual)."""
    schema = model_schema(model)
    generation = _generation(model, tier, regime, slot) + 1
    with torch.no_grad():
        if tier == schema.fast.name:
            model.residual.regimes[regime].deltas[tier].slot_generation[0] = generation
        else:
            _bank(model, tier, regime).slot_generation[slot] = generation
    _write_init(schema, _slot_tensors(entries, tier, regime, slot), generation)


def activate_slot(model: nn.Module, tier: str, regime: Optional[int] = None, *, version: Optional[str] = None) -> int:
    """Activate the lowest free slot of a sleep tier and return its index (sleep spec 4.3, 5.1 (b)).

    regime None means model.regime. The slot has B == 0 (checked; I1), so the function is
    unchanged bit for bit (Theorem 9.1); the version still changes (I3). Raises TierFullError
    when every slot is active: nothing is ever evicted silently (4.4). A newly activated slot
    is trainable only through an optimizer built afterwards (spec 12.1).
    """
    config = _slotted(model, tier)
    r = _one_regime(model, regime)
    entries = slot_entries(model)
    current = active_slots(model, r)[tier]
    free = [s for s in range(config.slots) if s not in current]
    if not free:
        raise TierFullError(f"tier {tier} has all {config.slots} slots active in regime {r}; "
                            "compact (sleep spec 4.4) or reset explicitly")
    slot = free[0]
    if not _b_is_zero(_slot_tensors(entries, tier, r, slot)):
        raise TierStateError(f"free slot {tier}[{slot}] r{r} has non-zero B (I1); activating it would change "
                             "the function. reset_tier it or load a consistent state")
    _set_active(model, entries, tier, r, slot, True)
    _bump_version(model, f"activate-{tier}{slot}-r{r}", version)
    return slot


def reset_tier(
    model: nn.Module,
    tier: str,
    regime: Union[None, int, str] = None,
    *,
    slots: Optional[Sequence[int]] = None,
    allow_offline: bool = False,
    version: Optional[str] = None,
) -> None:
    """Synaptic pruning (paper 3.3 step (c)(b), sleep spec 5.1 (c)): zero contribution, exactly.

    For every selected slot: B <- 0 (LoRA lora_b, residual b_o weight and bias), A <- seeded
    init at generation + 1 (LoRA lora_a, residual modulation_out), never from the global
    RNG. Sleep-tier slots become inactive. The wake tier stays active (it is the builder's
    plastic memory) and contributes zero until trained again. regime None means model.regime,
    "all" means every regime; slots None means every slot of the tier. Resetting the offline
    tier erases the stage-3 knowledge and needs allow_offline=True. Optimizer moments of the
    reset tensors are stale: rebuild the optimizer (spec 12.1, sleep spec F8).
    """
    schema = model_schema(model)
    config = schema.tier(tier)
    if tier == OFFLINE and not allow_offline:
        raise TierError("resetting the offline tier erases the stage-3 adapters; pass allow_offline=True")
    chosen = list(range(config.slots)) if slots is None else sorted(set(slots))
    if any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < config.slots for s in chosen):
        raise TierError(f"slots {slots} out of range for tier {tier}")
    _reset_slots(model, tier, _resolve_regimes(model, regime), chosen)
    _bump_version(model, f"reset-{tier}", version)


def _reset_slots(model: nn.Module, tier: str, regimes: Sequence[int], slots: Sequence[int]) -> None:
    schema = model_schema(model)
    entries = slot_entries(model)
    if tier != schema.fast.name:
        active_slots(model)  # refuse inconsistent masks before writing
    for r in regimes:
        for slot in slots:
            _reinit_slot(model, entries, tier, r, slot)
            if tier != schema.fast.name:
                _set_active(model, entries, tier, r, slot, False)


# =============================================================================== parameter selection

_COMPONENTS = ("all", "lora", "residual")


def tier_named_parameters(
    model: nn.Module,
    tier: str,
    regime: int = 0,
    slot: Optional[int] = None,
    component: str = "all",
    *,
    include_inactive: bool = False,
) -> dict[str, nn.Parameter]:
    """Name -> trainable tensor of the selected slots; metadata is never included.

    The wake tier: all its LoRA tensors and its residual (modulation_out, b_o), in the order
    `AdaptedPredictor.set_trainable("online")` returns them (C2). Sleep tiers: slot None
    selects the active slots (sleep spec 4.3: an inactive slot is never handed to an
    optimizer), or every slot with include_inactive=True; an explicit slot index returns
    that slot's tensors whether or not it is active (e.g. for dropping optimizer moments).
    """
    schema = model_schema(model)
    config = schema.tier(tier)
    r = _one_regime(model, regime)
    if component not in _COMPONENTS:
        raise TierError(f"component must be one of {_COMPONENTS}")
    if tier == schema.fast.name:
        if slot not in (None, 0):
            raise TierError("the wake tier has only slot 0")
        chosen = {0}
    elif slot is None:
        chosen = set(range(config.slots)) if include_inactive else set(active_slots(model, r)[tier])
    else:
        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < config.slots:
            raise TierError(f"slot {slot!r} out of range for tier {tier}")
        chosen = {slot}
    result = {}
    for e in slot_entries(model):
        if e.component == "meta" or e.tier != tier or e.regime != r or e.slot not in chosen:
            continue
        if component != "all" and e.component != component:
            continue
        result[e.name] = e.parameter
    return result


def tier_parameters(model: nn.Module, tier: str, regime: int = 0, slot: Optional[int] = None,
                    component: str = "all", *, include_inactive: bool = False) -> list[nn.Parameter]:
    """List form of tier_named_parameters (the section 0 contract signature)."""
    return list(tier_named_parameters(model, tier, regime, slot, component, include_inactive=include_inactive).values())


def set_tier_trainable(model: nn.Module, tiers: Sequence[str], regime: Optional[int] = None, *,
                       components: Sequence[str] = ("lora", "residual"), online: bool = True) -> dict[str, list[nn.Parameter]]:
    """Freeze every parameter, then unfreeze the active slots of ``tiers`` only.

    Returns groups keyed "<tier>.<component>" for learning/online_builder.build_optimizer
    (see tier_learning_rates and build_tier_optimizer). Every tier slot's residual part is
    (modulation_out, b_o), which is already the spec 9.8 online subset, so ``online`` is
    accepted for compatibility and changes nothing. Refuses when nothing would be trainable.
    """
    del online
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    r = _one_regime(model, regime)
    groups: dict[str, list[nn.Parameter]] = {}
    for tier in tiers:
        for component in components:
            if component not in ("lora", "residual"):
                raise TierError(f"component must be 'lora' or 'residual', got {component!r}")
            params = tier_parameters(model, tier, r, None, component)
            if params:
                groups[f"{tier}.{component}"] = params
    if not groups:
        raise TierError(f"tiers {list(tiers)} have no active slot in regime {r}: nothing to train")
    for params in groups.values():
        for parameter in params:
            parameter.requires_grad_(True)
    return groups


def tier_learning_rates(groups: Mapping[str, list], lora_lr: Union[float, Mapping[str, float]],
                        residual_lr: Union[float, Mapping[str, float]]) -> dict[str, float]:
    """Learning rates for set_tier_trainable groups; mappings are keyed by tier name."""
    rates = {}
    for key in groups:
        tier, component = key.rsplit(".", 1)
        source = lora_lr if component == "lora" else residual_lr
        if isinstance(source, Mapping):
            if tier not in source:
                raise TierError(f"no {component} learning rate for tier {tier}")
            rates[key] = float(source[tier])
        else:
            rates[key] = float(source)
    return rates


def build_tier_optimizer(model: nn.Module, tiers: Sequence[str], lora_lr, residual_lr, regime: Optional[int] = None, *,
                         weight_decay: float = 0.0, betas=(0.9, 0.95), eps: float = 1e-8):
    """AdamW over the active slots of ``tiers`` via learning/online_builder.build_optimizer.

    Defaults follow sleep spec 5.4 (betas 0.9/0.95) and 5.3.6 (no weight decay on a slot:
    decay toward zero would pull a transplanted slot off the function it carries). Returns
    (optimizer, groups).
    """
    groups = set_tier_trainable(model, tiers, regime)
    rates = tier_learning_rates(groups, lora_lr, residual_lr)
    return build_optimizer(groups, rates, weight_decay, betas, eps), groups


# =============================================================================== tier state


@dataclass(eq=False)
class TierState:
    """Self-contained copy of every tier tensor and its metadata (CPU), without the version.

    Equality compares tensors by value (torch.equal), so ``==`` and dataclass-recursive
    comparisons work. Restore with load_tier_state. The same tensors are part of
    adapter_state, which VersionStore persists.
    """

    schema: str
    tiers: tuple[TierConfig, ...]
    active: dict
    tensors: dict = field(default_factory=dict)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TierState):
            return NotImplemented
        return (self.schema == other.schema and self.tiers == other.tiers and self.active == other.active
                and set(self.tensors) == set(other.tensors)
                and all(self.tensors[k].shape == other.tensors[k].shape and self.tensors[k].dtype == other.tensors[k].dtype
                        and torch.equal(self.tensors[k], other.tensors[k]) for k in self.tensors))

    __hash__ = None  # type: ignore[assignment]


def tier_state(model: nn.Module) -> TierState:
    schema = model_schema(model)
    return TierState(
        schema=schema.identifier(),
        tiers=schema.tiers,
        active=active_slots(model),
        tensors={e.name: e.parameter.detach().cpu().clone() for e in slot_entries(model)},
    )


def load_tier_state(model: nn.Module, state: TierState, version: Optional[str] = None) -> None:
    """Restore a TierState. Atomic: validates schema, keys, shapes and invariants before copying.

    Changes model.version (I3) unless ``version`` is given.
    """
    if not isinstance(state, TierState):
        raise TierError("load_tier_state needs a TierState from tier_state()")
    schema = model_schema(model)
    if state.schema != schema.identifier() or tuple(state.tiers) != schema.tiers:
        raise TierSchemaError(f"tier state of schema {state.schema} cannot load into {schema.identifier()}")
    entries = slot_entries(model)
    expected = {e.name: e for e in entries}
    if set(state.tensors) != set(expected):
        raise TierError(f"tier tensor key mismatch: {sorted(set(state.tensors) ^ set(expected))[:5]}")
    for name, e in expected.items():
        if state.tensors[name].shape != e.parameter.shape:
            raise TierError(f"shape mismatch for {name}")
    validate_state(model, state.tensors)
    with torch.no_grad():
        for name, e in expected.items():
            e.parameter.copy_(state.tensors[name].to(e.parameter.device, e.parameter.dtype))
    active_slots(model)
    _bump_version(model, "load", version)


# =============================================================================== legacy migration


def migrate_legacy_state(model: nn.Module, legacy_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map an untiered (legacy-v1) adapter_state onto this model's schema, as convert_to_tiered does.

    Stage-3 LoRA -> offline slot (padded/rescaled exactly as at conversion); stage-3 residual
    modulation_out/b_o -> offline residual slot; trunk, embeddings and command encoder copied;
    the fast tier and every sleep slot reset (B = 0, A seeded at generation + 1, inactive).
    Every legacy key must be consumed and every non-generated key provided, else TierError.
    The result loads with load_adapter_state (C6: this is the only legal cross-schema path).
    """
    schema = model_schema(model)
    if SCHEMA_KEY in legacy_state:
        raise TierSchemaError("the state is already tiered")
    offline = schema.tier(OFFLINE)
    ratio = (schema.legacy_alpha / schema.legacy_rank) / offline.scale
    state = model.adapter_state()
    entries = slot_entries(model)
    by_name = {e.name: e for e in entries}
    for e in entries:
        if e.component == "meta":
            continue
        if e.tier == OFFLINE:
            continue
        generation = _generation(model, e.tier, e.regime, e.slot) + 1
        state[e.name] = _expected_init(schema, e, generation).to(state[e.name].dtype)
    for e in entries:
        if e.component != "meta":
            continue
        if e.role == "slot_active":
            value = torch.zeros_like(state[e.name])
            if e.tier == OFFLINE:
                value[0] = 1.0
            state[e.name] = value
        elif e.role == "slot_birth":
            value = torch.zeros_like(state[e.name])
            if e.tier == OFFLINE:
                value[0] = 1
            state[e.name] = value
        elif e.role == "slot_generation" and e.tier != OFFLINE:
            state[e.name] = state[e.name] + 1
    pattern = re.compile(r"^predictor\.(.+)\.deltas\.(\d+)\.(\d+)\.(lora_[ab])$")
    residual_pattern = re.compile(r"^residual\.regimes\.(\d+)\.(modulation_out|b_o)\.(weight|bias)$")
    covered = set()
    for key, value in legacy_state.items():
        value = value.detach().cpu()
        match, rmatch = pattern.match(key), residual_pattern.match(key)
        if match is not None:
            path, regime, index, role = match.groups()
            target = f"predictor.{path}.tiers.{OFFLINE}.deltas.{regime}.slots.0.{index}.{role}"
            if target not in state:
                raise TierError(f"legacy key {key!r} maps to missing {target!r}")
            out = state[target].clone()
            rank = schema.legacy_rank
            if role == "lora_a":
                if value.shape != (rank, out.shape[1]):
                    raise TierError(f"shape mismatch for {key}")
                out[:rank] = value.to(out.dtype)
            else:
                if value.shape != (out.shape[0], rank):
                    raise TierError(f"shape mismatch for {key}")
                out[:, :rank] = value.to(out.dtype) if ratio == 1.0 else (value.double() * ratio).to(out.dtype)
                out[:, rank:] = 0
            state[target] = out
            covered.add(target)
        elif rmatch is not None:
            regime, part, kind = rmatch.groups()
            target = f"residual.regimes.{regime}.deltas.{OFFLINE}.slots.0.{part}.{kind}"
            if target not in state or state[target].shape != value.shape:
                raise TierError(f"legacy key {key!r} has no matching offline residual tensor")
            state[target] = value.to(state[target].dtype).clone()
            covered.add(target)
        else:
            if key not in state or key in by_name:
                raise TierError(f"legacy key {key!r} has no counterpart in the tiered model")
            if state[key].shape != value.shape:
                raise TierError(f"shape mismatch for {key}")
            state[key] = value.to(state[key].dtype).clone()
            covered.add(key)
    required = {k for k in state if k not in by_name}
    required |= {e.name for e in entries if e.tier == OFFLINE and e.component != "meta" and e.slot == 0
                 and not (e.role == "lora_a" and offline.rank > schema.legacy_rank)}
    missing = required - covered
    if missing:
        raise TierError(f"legacy state lacks {len(missing)} keys, e.g. {sorted(missing)[:3]}")
    return state


# =============================================================================== low-rank algebra


def _lowrank_fro(left: torch.Tensor, right: torch.Tensor) -> float:
    """||left @ right||_F without forming the product (left [m, K], right [K, n])."""
    if left.shape[1] == 0:
        return 0.0
    _, r_left = torch.linalg.qr(left)
    _, r_right = torch.linalg.qr(right.T)
    return float(torch.linalg.matrix_norm(r_left @ r_right.T))


def _slot_deltas(module: TieredLoRALinear, tier: str, fast_name: str, regime: int, slot: int):
    return module.deltas[regime] if tier == fast_name else module.tiers[tier].deltas[regime].slots[slot]


def _residual_slot(model: nn.Module, tier: str, regime: int, slot: int) -> nn.Module:
    module = model.residual.regimes[regime]
    return module if tier == model_schema(model).fast.name else module.deltas[tier].slots[slot]


def _nonzero_columns(b: torch.Tensor) -> list[int]:
    return torch.nonzero(b.detach().ne(0).any(dim=0), as_tuple=False).flatten().tolist()


def tier_delta_norms(model: nn.Module, regime: Optional[int] = None) -> dict[str, float]:
    """Frobenius norm of each tier's total LoRA delta over all projections (diagnostic).

    Useful for the sleep spec 10.1 skip condition (fast tier functionally about zero).
    """
    schema = model_schema(model)
    r = _one_regime(model, regime)
    active = active_slots(model, r)
    totals = {t.name: 0.0 for t in schema.tiers}
    for _, module in _lora_modules(model):
        for tier in schema.tiers:
            for index in range(len(module.slices)):
                bs, as_ = [], []
                for slot in active[tier.name]:
                    delta = _slot_deltas(module, tier.name, schema.fast.name, r, slot)[index]
                    bs.append(delta.lora_b.detach().double() * delta.scale)
                    as_.append(delta.lora_a.detach().double())
                if bs:
                    totals[tier.name] += _lowrank_fro(torch.cat(bs, 1), torch.cat(as_, 0)) ** 2
    return {k: math.sqrt(v) for k, v in totals.items()}


@dataclass
class TransplantReport:
    src: tuple
    dst: tuple
    max_relative_error: float  # max over projections/slices of ||dW_dst - dW_src||_F / ||dW_src||_F (float64)
    scale_ratio: float  # B multiplier scale_src / scale_dst (exact when a power of two)
    residual_copied: bool


def _parse_src(model, src) -> tuple[str, int, int]:
    schema = model_schema(model)
    if not isinstance(src, tuple) or len(src) not in (2, 3):
        raise TierError("src must be (tier, regime) for the wake tier or (tier, regime, slot)")
    tier, regime = src[0], src[1]
    schema.tier(tier)
    r = _one_regime(model, regime)
    if tier == schema.fast.name:
        if len(src) == 3 and src[2] != 0:
            raise TierError("the wake tier has only slot 0")
        return tier, r, 0
    if len(src) != 3:
        raise TierError(f"src of sleep tier {tier} needs a slot")
    if not 0 <= src[2] < schema.tier(tier).slots:
        raise TierError(f"slot {src[2]} out of range for {tier}")
    return tier, r, src[2]


def transplant(model: nn.Module, src: tuple, dst: tuple, *, version: Optional[str] = None) -> TransplantReport:
    """Exact copy of one slot's function into an empty active slot (sleep spec Theorem 9.2, mode M1).

    src = (tier, regime) for the wake tier, or (tier, regime, slot); dst = (tier, regime, slot)
    of a sleep tier, which must be active and empty (B == 0), and of rank >= the source rank.
    A_dst = [A_src; R] keeps the slot's own seeded rows R (random, not zero, so the padded
    rank stays trainable, Theorem 9.2 remark), B_dst = (scale_src / scale_dst) [B_src 0].
    Residual expert (modulation_out, b_o) copied exactly. The source is not modified: reset
    it separately (compute -> consolidate -> update, paper 3.3). Changes the version (I3).
    """
    schema = model_schema(model)
    s_tier, s_regime, s_slot = _parse_src(model, src)
    if not isinstance(dst, tuple) or len(dst) != 3:
        raise TierError("dst must be (tier, regime, slot)")
    d_cfg = _slotted(model, dst[0])
    d_tier, d_regime = dst[0], _one_regime(model, dst[1])
    d_slot = dst[2]
    if isinstance(d_slot, bool) or not isinstance(d_slot, int) or not 0 <= d_slot < d_cfg.slots:
        raise TierError(f"slot {d_slot!r} out of range for {d_tier}")
    if (s_tier, s_regime, s_slot) == (d_tier, d_regime, d_slot):
        raise TierError("source and destination are the same slot")
    s_cfg = schema.tier(s_tier)
    if d_cfg.rank < s_cfg.rank:
        raise TierError(f"destination rank {d_cfg.rank} < source rank {s_cfg.rank}: not exact; use whitened_compaction")
    if d_slot not in active_slots(model, d_regime)[d_tier]:
        raise TierError(f"destination {d_tier}[{d_slot}] r{d_regime} is not active; activate_slot first")
    entries = slot_entries(model)
    if not _b_is_zero(_slot_tensors(entries, d_tier, d_regime, d_slot)):
        raise TierError(f"destination {d_tier}[{d_slot}] r{d_regime} is not empty; transplant never overwrites")
    ratio = s_cfg.scale / d_cfg.scale
    worst = 0.0
    with torch.no_grad():
        for _, module in _lora_modules(model):
            sources = _slot_deltas(module, s_tier, schema.fast.name, s_regime, s_slot)
            targets = _slot_deltas(module, d_tier, schema.fast.name, d_regime, d_slot)
            for source, target in zip(sources, targets):
                k = s_cfg.rank
                target.lora_a[:k].copy_(source.lora_a)
                if ratio == 1.0:
                    target.lora_b[:, :k].copy_(source.lora_b)
                else:
                    target.lora_b[:, :k].copy_((source.lora_b.double() * ratio).to(target.lora_b.dtype))
                target.lora_b[:, k:].zero_()
                left = torch.cat([target.lora_b.double() * target.scale, -source.lora_b.double() * source.scale], 1)
                right = torch.cat([target.lora_a.double(), source.lora_a.double()], 0)
                reference = _lowrank_fro(source.lora_b.double() * source.scale, source.lora_a.double())
                error = _lowrank_fro(left, right)
                worst = max(worst, error / reference if reference > 0 else (0.0 if error == 0 else math.inf))
        s_res = _residual_slot(model, s_tier, s_regime, s_slot)
        d_res = _residual_slot(model, d_tier, d_regime, d_slot)
        for part in ("modulation_out", "b_o"):
            for kind in ("weight", "bias"):
                getattr(getattr(d_res, part), kind).copy_(getattr(getattr(s_res, part), kind))
    _bump_version(model, f"transplant-{s_tier}-{d_tier}{d_slot}", version)
    return TransplantReport((s_tier, s_regime, s_slot), (d_tier, d_regime, d_slot), worst, ratio, True)


@dataclass
class SliceReport:
    module: str
    slice_index: int
    method: str  # "concat" | "whitened-svd" | "empty"
    source_rank: int  # stacked components (before truncation)
    target_rank: int
    retained_energy: float  # eta of (S5); 1.0 for concat / empty
    whitened_error: float  # sum_{i>r} sigma_i^2 of Delta C_eps^{1/2} (0 for concat)
    data_error: float  # tr(E C E^T) with E = Delta' - Delta and the undamped C (or ||E||_F^2 without C)
    relative_error: float  # data_error / tr(Delta C Delta^T)


@dataclass
class MergeReport:
    tier: str
    regime: int
    target: tuple[str, int]
    sources: tuple[int, ...]
    method: str
    exact: bool
    residual: str  # "exact" (units moved) | "zero" (target residual reset; distil it, sleep spec 6.2) | "none"
    residual_units_moved: int
    min_retained_energy: float
    max_relative_error: float
    lossy_projections: int  # retained energy below energy_threshold
    slices: list[SliceReport] = field(default_factory=list)


def _residual_merge_plan(model, tier: str, regime: int, sources: Sequence[int], target: tuple[str, int], mode: str):
    """Exact residual merge: a source unit j moves to unit j of the target (a_o row j is shared),
    so it needs unit j free in the target and unused by every other source; b_o biases add."""
    if mode not in ("exact", "zero"):
        raise TierError("residual must be 'exact' or 'zero'")
    t_res = _residual_slot(model, target[0], regime, target[1])
    if mode == "zero":
        return None
    used = set(_nonzero_columns(t_res.b_o.weight))
    moves = []
    for s in sources:
        units = _nonzero_columns(_residual_slot(model, tier, regime, s).b_o.weight)
        clash = used.intersection(units)
        if clash:
            raise TierError(f"residual units {sorted(clash)[:5]} are used by more than one merged slot: no exact "
                            "residual merge exists (shared a_o). Pass residual='zero' and distil (sleep spec 6.2)")
        used.update(units)
        moves.append((s, units))
    return moves


def _apply_residual_merge(model, tier, regime, target, moves) -> int:
    t_res = _residual_slot(model, target[0], regime, target[1])
    count = 0
    with torch.no_grad():
        if moves is None:
            return 0
        for s, units in moves:
            s_res = _residual_slot(model, tier, regime, s)
            for j in units:
                t_res.modulation_out.weight[j] = s_res.modulation_out.weight[j]
                t_res.modulation_out.bias[j] = s_res.modulation_out.bias[j]
                t_res.b_o.weight[:, j] = s_res.b_o.weight[:, j]
            t_res.b_o.bias.add_(s_res.b_o.bias)
            count += len(units)
    return count


def _merge_common(model, tier, regime, slots, into):
    schema = model_schema(model)
    config = schema.tier(tier)
    r = _one_regime(model, regime)
    active = active_slots(model, r)
    sources = tuple(sorted(set(slots)))
    if not sources:
        raise TierError("no slots to merge")
    if tier == schema.fast.name:
        if sources != (0,):
            raise TierError("the wake tier has only slot 0")
    elif any(s not in active[tier] for s in sources):
        raise TierError(f"slots {sources} must be active slots of {tier} in regime {r}")
    t_tier, t_slot = into
    t_cfg = _slotted(model, t_tier)
    if not 0 <= t_slot < t_cfg.slots or t_slot not in active[t_tier]:
        raise TierError(f"target {t_tier}[{t_slot}] must be an active slot")
    return schema, config, r, sources, t_cfg


def merge_slots(model: nn.Module, tier: str, regime: Optional[int], slots: Sequence[int], into: tuple[str, int], *,
                residual: str = "exact", release: bool = True, version: Optional[str] = None) -> MergeReport:
    """Exact merge by concatenation (sleep spec Proposition 9.3, 6.1).

    The non-zero components (columns of B, with their rows of A) of each source slot are
    stacked into free components (zero columns of B) of the active target slot
    ``into=(tier, slot)``, B rescaled by scale_src/scale_tgt. Lossless; refused (nothing
    written) when the components do not fit, i.e. when the target rank is below the stacked
    rank (with uniform slot ranks this frees capacity only into a higher-rank slot). Residual:
    see _residual_merge_plan; residual='zero' resets the target's residual for distillation.
    release=True resets the source slots afterwards (wake source: zeroed, stays active).
    """
    schema, config, r, sources, t_cfg = _merge_common(model, tier, regime, slots, into)
    t_tier, t_slot = into
    if t_tier == tier and t_slot in sources:
        raise TierError("the target slot cannot be one of the sources for an exact merge")
    moves = _residual_merge_plan(model, tier, r, sources, into, residual)
    writes, reports = [], []
    for path, module in _lora_modules(model):
        targets = _slot_deltas(module, t_tier, schema.fast.name, r, t_slot)
        for index, target in enumerate(targets):
            free = [j for j in range(t_cfg.rank) if j not in set(_nonzero_columns(target.lora_b))]
            new_a, new_b = target.lora_a.detach().clone(), target.lora_b.detach().clone()
            placed = 0
            for s in sources:
                source = _slot_deltas(module, tier, schema.fast.name, r, s)[index]
                for column in _nonzero_columns(source.lora_b):
                    if placed >= len(free):
                        raise TierError(f"{path}[{index}]: stacked components exceed the {len(free)} free components "
                                        f"of {t_tier}[{t_slot}]; use whitened_compaction")
                    j = free[placed]
                    placed += 1
                    new_a[j] = source.lora_a.detach()[column]
                    new_b[:, j] = (source.lora_b.detach()[:, column].double() * (source.scale / target.scale)).to(new_b.dtype)
            writes += [(target.lora_a, new_a), (target.lora_b, new_b)]
            reports.append(SliceReport(path, index, "concat" if placed else "empty", placed, t_cfg.rank, 1.0, 0.0, 0.0, 0.0))
    with torch.no_grad():
        for parameter, value in writes:
            parameter.copy_(value)
    if residual == "zero":
        entries = slot_entries(model)
        _write_init(schema, [e for e in _slot_tensors(entries, t_tier, r, t_slot) if e.component == "residual"],
                    _generation(model, t_tier, r, t_slot))
        moved = 0
    else:
        moved = _apply_residual_merge(model, tier, r, into, moves)
    if release:
        _reset_slots(model, tier, [r], list(sources))
    _bump_version(model, f"merge-{tier}-{t_tier}{t_slot}", version)
    return MergeReport(tier, r, into, sources, "concat", True, residual, moved, 1.0, 0.0, 0, reports)


class InputCovariance:
    """Input second moments C = E[x x^T] of every tiered projection (sleep spec 6.2).

    Forward pre-hooks accumulate x^T x in float64 (default) per projection path, over every
    token of every forward while the context is open. Run it under torch.no_grad on the
    sleep training windows. Q and V of the split layout see the same input but are kept
    separately (simplicity over memory).
    """

    def __init__(self, model: nn.Module, dtype: torch.dtype = torch.float64):
        self.model = model
        self.dtype = dtype
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self._handles = []

    def __enter__(self) -> "InputCovariance":
        for path, module in _lora_modules(self.model):
            self._handles.append(module.register_forward_pre_hook(self._hook(path)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _hook(self, path: str):
        def hook(_module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(self.dtype)
            gram = x.T @ x
            if path in self.sums:
                self.sums[path] += gram
            else:
                self.sums[path] = gram
            self.counts[path] = self.counts.get(path, 0) + x.shape[0]
        return hook

    def covariances(self) -> dict[str, torch.Tensor]:
        return {path: (total / self.counts[path]).cpu() for path, total in self.sums.items()}


def whitened_compaction(model: nn.Module, tier: str, regime: Optional[int], slots: Sequence[int],
                        covariances: Optional[Mapping[str, torch.Tensor]], *, into: Optional[int] = None,
                        residual: str = "exact", damping: float = 1e-4, energy_threshold: float = 0.98,
                        release: bool = True, version: Optional[str] = None) -> MergeReport:
    """Lossy compaction of several slots of one sleep tier into one (sleep spec 6.2, Theorem 9.4, eq. S5).

    Per projection slice, with Delta = sum_k s_k B_k A_k over ``slots`` (float64) and
    C_eps = C + eps I, eps = damping * tr(C)/d:
        M = Delta C_eps^{1/2} = U S V^T,  B' = (1/s) U_r S_r^{1/2},  A' = S_r^{1/2} V_r^T C_eps^{-1/2},
    written into slot ``into`` (default: the first of ``slots``). This minimises
    E||(Delta - Delta') x||^2 under C_eps over rank <= r (Eckart-Young after whitening). The
    report gives per slice the retained energy eta, the discarded whitened energy, the error
    under the undamped C and its relative value; slices with eta < energy_threshold are
    counted as lossy (the refinement and the non-inferiority gate decide, 4.4). Unused
    components keep B = 0 and the slot's seeded A rows (alive for later refinement).
    covariances None means C = I (plain truncated SVD, Frobenius-optimal).
    Residual experts are nonlinear: merged exactly when unit supports are disjoint, else
    refused, or reset for distillation with residual='zero' (sleep spec 6.2).
    The other slots are reset and released when release=True. Atomic until the writes.
    """
    _slotted(model, tier)
    target = min(slots) if into is None else into
    schema, config, r, sources, _ = _merge_common(model, tier, regime, slots, (tier, target))
    if target not in sources:
        raise TierError("the compaction target must be one of the merged slots")
    if not damping >= 0:
        raise TierError("damping must be >= 0")
    others = [s for s in sources if s != target]
    moves = _residual_merge_plan(model, tier, r, others, (tier, target), residual)
    entries = slot_entries(model)
    generation = _generation(model, tier, r, target)
    seeded_rows = {(e.path, e.index): _expected_init(schema, e, generation)
                   for e in _slot_tensors(entries, tier, r, target) if e.role == "lora_a"}
    writes, reports = [], []
    for path, module in _lora_modules(model):
        cov = None
        if covariances is not None:
            if path not in covariances:
                raise TierError(f"no input covariance for {path}")
            cov = covariances[path].to(torch.float64)
        for index in range(len(module.slices)):
            deltas = [module.tiers[tier].deltas[r].slots[s][index] for s in sources]
            tgt = deltas[sources.index(target)]
            bs = [d.lora_b.detach().double().cpu() * d.scale for d in deltas]
            as_ = [d.lora_a.detach().double().cpu() for d in deltas]
            delta = sum(b @ a for b, a in zip(bs, as_))
            d_in = delta.shape[1]
            if cov is None:
                c = torch.eye(d_in, dtype=torch.float64)
                eps = 0.0
            else:
                if cov.shape != (d_in, d_in):
                    raise TierError(f"covariance of {path} has shape {tuple(cov.shape)}, expected {(d_in, d_in)}")
                c = (cov + cov.T) / 2
                eps = damping * float(torch.trace(c)) / d_in
            c_eps = c + eps * torch.eye(d_in, dtype=torch.float64)
            evals, evecs = torch.linalg.eigh(c_eps)
            if float(evals.min()) <= 0:
                raise TierError(f"{path}: C_eps is not positive definite; increase damping")
            root = (evecs * evals.sqrt()) @ evecs.T
            inv_root = (evecs * evals.rsqrt()) @ evecs.T
            u, s_values, vh = torch.linalg.svd(delta @ root, full_matrices=False)
            k = config.rank
            total = float((s_values ** 2).sum())
            keep = min(k, int((s_values > 0).sum()))
            retained = float((s_values[:keep] ** 2).sum())
            eta = retained / total if total > 0 else 1.0
            half = s_values[:keep].sqrt()
            new_b = torch.zeros_like(tgt.lora_b, dtype=torch.float64, device="cpu")
            new_a = seeded_rows[(path, index)].double().clone()
            new_b[:, :keep] = u[:, :keep] * half / tgt.scale
            new_a[:keep] = half[:, None] * (vh[:keep] @ inv_root)
            approx = (new_b * tgt.scale) @ new_a
            err = approx - delta
            data_error = float(((err @ c) * err).sum())
            signal = float(((delta @ c) * delta).sum())
            reports.append(SliceReport(path, index, "whitened-svd" if total > 0 else "empty", sum(config.rank for _ in sources),
                                       k, eta, total - retained, data_error,
                                       data_error / signal if signal > 0 else 0.0))
            writes += [(tgt.lora_a, new_a), (tgt.lora_b, new_b)]
    with torch.no_grad():
        for parameter, value in writes:
            parameter.copy_(value.to(parameter.device, parameter.dtype))
    if residual == "zero":
        _write_init(schema, [e for e in _slot_tensors(entries, tier, r, target) if e.component == "residual"], generation)
        moved = 0
    else:
        moved = _apply_residual_merge(model, tier, r, (tier, target), moves)
    if release and others:
        _reset_slots(model, tier, [r], others)
    _bump_version(model, f"compact-{tier}{target}", version)
    min_eta = min((x.retained_energy for x in reports), default=1.0)
    lossy = sum(1 for x in reports if x.retained_energy < energy_threshold)
    worst = max((x.relative_error for x in reports), default=0.0)
    return MergeReport(tier, r, (tier, target), tuple(sources), "whitened-svd", lossy == 0 and worst <= 1e-9,
                       residual, moved, min_eta, worst, lossy, reports)


__all__ = [
    "DEFAULT_TIERS",
    "LEGACY_SCHEMA",
    "OFFLINE",
    "SCHEMA_KEY",
    "SCHEMA_NAME",
    "ConversionReport",
    "InputCovariance",
    "MergeReport",
    "SliceReport",
    "SlotEntry",
    "TierConfig",
    "TierError",
    "TierFullError",
    "TierSchema",
    "TierSchemaError",
    "TierState",
    "TierStateError",
    "TieredAdaptedPredictor",
    "TieredLoRALinear",
    "TieredRegimeResidual",
    "TransplantReport",
    "activate_slot",
    "active_slots",
    "assert_invariants",
    "build_tier_optimizer",
    "check_invariants",
    "convert_to_tiered",
    "is_tiered",
    "load_tier_state",
    "merge_slots",
    "migrate_legacy_state",
    "model_schema",
    "model_tiers",
    "reset_tier",
    "resolve_tiers",
    "schema_payload",
    "set_tier_trainable",
    "slot_entries",
    "slots_by_age",
    "state_schema",
    "tier_delta_norms",
    "tier_learning_rates",
    "tier_named_parameters",
    "tier_parameters",
    "tier_state",
    "transplant",
    "validate_state",
    "whitened_compaction",
]
