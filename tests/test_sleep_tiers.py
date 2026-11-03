"""Tiered multi-frequency adapters (robojepa_ttt/sleep/tiers.py, docs/SLEEP_SPECIFICATION.md).

Every test runs on both attention layouts through conftest's autouse fixture: fused qkv
(300M) and split q / kv (1B, 2B). The fixture model has LoRA rank 4, alpha 8 (scale 2) and a
residual bottleneck of 8.
"""

import copy

import pytest
import torch

from conftest import SERVO, UNITS, make_batch, make_model, set_adapters_nonzero, tiny_config
from robojepa_ttt.learning.online_builder import OnlineBuilder, OnlineConfig
from robojepa_ttt.learning.promotion import ModelHub, PromotionConfig, VersionStore
from robojepa_ttt.models.adapted_predictor import AdaptedPredictor, StaleCacheError, state_digest
from robojepa_ttt.models.lora import LoRALinear, lora_parameters
from robojepa_ttt.models.token_residual import TokenResidual
from robojepa_ttt.models.upstream import build_predictor
from robojepa_ttt.sleep.tiers import (
    OFFLINE,
    SCHEMA_KEY,
    DEFAULT_TIERS,
    InputCovariance,
    TierConfig,
    TierError,
    TierFullError,
    TierSchemaError,
    TierStateError,
    TieredAdaptedPredictor,
    TieredLoRALinear,
    TieredOnlineBuilder,
    activate_slot,
    active_slots,
    build_tier_optimizer,
    check_invariants,
    convert_to_tiered,
    free_slots,
    is_tiered,
    load_tier_state,
    merge_slots,
    migrate_legacy_state,
    rank_compaction_pairs,
    model_tiers,
    refresh_slot_cache,
    reset_tier,
    resolve_tiers,
    schema_payload,
    set_tier_trainable,
    slot_entries,
    slots_by_age,
    state_schema,
    tier_delta_norms,
    tier_named_parameters,
    tier_parameters,
    tier_state,
    transplant,
    validate_state,
    whitened_compaction,
)

TIERS = (TierConfig("fast", 4, 8.0, 1, 3), TierConfig("slow", 8, 16.0, 3, None))
ARCHIVE = (TierConfig("fast", 4, 8.0, 1, 2), TierConfig("slow", 4, 8.0, 3, 5), TierConfig("archive", 8, 16.0, 1, None))


def run(model, batch, context=3, horizon=2, use_kv=True):
    with torch.no_grad():
        return model.rollout(batch, context, horizon, use_kv_generation=use_kv)


def relative(a, b):
    return float((a - b).abs().mean() / b.abs().mean())


def tiered(tiers=TIERS, nonzero=True, **kwargs):
    model = make_model(**kwargs)
    if nonzero:
        set_adapters_nonzero(model)
    convert_to_tiered(model, tiers)
    return model


def two_regime_model():
    torch.manual_seed(0)
    predictor = build_predictor(tiny_config(*__import__("conftest").ACTIVE["heads"])).eval()
    residual = TokenResidual(32, 8, (32, 32), 16, 4, 4, regimes=2)
    return AdaptedPredictor(predictor, UNITS, SERVO, 2, 2, 4, 8.0, 2, command_hidden=16, residual=residual)


def fill(model, tier, slot=0, regime=0, seed=0, scale=0.05, lora=True, residual=True):
    """Give one slot a non-zero contribution through its B tensors."""
    generator = torch.Generator().manual_seed(seed)
    named = tier_named_parameters(model, tier, regime, slot)
    with torch.no_grad():
        for name, p in named.items():
            if (lora and name.endswith("lora_b")) or (residual and ".b_o." in name):
                p.add_(torch.randn(p.shape, generator=generator) * scale)


def tier_tensor_names(model):
    return {e.name for e in slot_entries(model)}


# ---------------------------------------------------------------------------- configuration


def test_tier_configuration_is_validated():
    with pytest.raises(TierError):
        TierConfig("Fast", 4, 8.0, 1)
    with pytest.raises(TierError):
        TierConfig("keys", 4, 8.0, 1)  # shadows nn.ModuleDict.keys
    with pytest.raises(TierError):
        TierConfig("slow", 0, 8.0, 1)
    with pytest.raises(TierError):
        TierConfig("slow", 4, 8.0, 1, consolidate_every=True)
    with pytest.raises(TierError):
        resolve_tiers([TierConfig("fast", 4, 8.0, 2), TierConfig("slow", 4, 8.0, 2)], 4, 8.0)  # wake has 1 slot
    with pytest.raises(TierError):
        resolve_tiers([TierConfig("fast", 4, 8.0, 1)], 4, 8.0)  # no sleep tier
    with pytest.raises(TierError):
        resolve_tiers([TierConfig("fast", 4, 8.0, 1, 10), TierConfig("slow", 4, 8.0, 2, 5)], 4, 8.0)
    with pytest.raises(TierError):
        resolve_tiers([TierConfig("fast", 4, 8.0, 1), TierConfig(OFFLINE, 4, 8.0, 1), TierConfig("slow", 4, 8.0, 2)], 4, 8.0)
    with pytest.raises(TierError):
        resolve_tiers([TierConfig("fast", 4, 8.0, 1), TierConfig("slow", 4, 8.0, 2), TierConfig(OFFLINE, 2, 4.0, 1)], 4, 8.0)
    resolved = resolve_tiers(TIERS, 4, 8.0)
    assert [t.name for t in resolved] == ["fast", "slow", OFFLINE]
    assert (resolved[-1].rank, resolved[-1].alpha, resolved[-1].slots, resolved[-1].consolidate_every) == (4, 8.0, 1, None)


# ---------------------------------------------------------------------------- conversion (sleep spec 3.3)


def test_conversion_is_bit_exact_and_moves_stage3_into_offline(batch):
    model = make_model()
    set_adapters_nonzero(model)
    legacy = {n: p.detach().clone() for n, p in model.named_parameters()}
    expected = {kv: run(model, batch, use_kv=kv) for kv in (True, False)}
    version = model.version
    report = convert_to_tiered(model, TIERS, probe=(batch, 3, 2))

    assert is_tiered(model) and isinstance(model, TieredAdaptedPredictor) and isinstance(model, AdaptedPredictor)
    assert report.probe_relative_error == 0.0 and report.max_abs_weight_error == 0.0 and not report.offline_padded
    for kv in (True, False):
        assert torch.equal(run(model, batch, use_kv=kv), expected[kv])
    assert model.version != version and check_invariants(model) == []
    assert active_slots(model) == {"fast": {0: [0]}, "slow": {0: []}, OFFLINE: {0: [0]}}
    params = dict(model.named_parameters())
    for name in model.lora_modules:
        for index in range(2 if name.endswith("qkv") else 1):
            for role in ("lora_a", "lora_b"):
                old = legacy[f"predictor.{name}.deltas.0.{index}.{role}"]
                assert torch.equal(params[f"predictor.{name}.tiers.{OFFLINE}.deltas.0.slots.0.{index}.{role}"], old)
            assert not params[f"predictor.{name}.deltas.0.{index}.lora_b"].any(), "fast tier starts at zero"
    for part in ("modulation_out", "b_o"):
        for kind in ("weight", "bias"):
            old = legacy[f"residual.regimes.0.{part}.{kind}"]
            assert torch.equal(params[f"residual.regimes.0.deltas.{OFFLINE}.slots.0.{part}.{kind}"], old)
    assert not params["residual.regimes.0.b_o.weight"].any() and not params["residual.regimes.0.b_o.bias"].any()
    # the trunk and the base are the same objects, untouched
    assert torch.equal(params["residual.regimes.0.a_o.weight"], legacy["residual.regimes.0.a_o.weight"])
    assert all(isinstance(m, TieredLoRALinear) for m in model.modules() if isinstance(m, LoRALinear))
    assert model_tiers(model)[-1].name == OFFLINE
    payload = schema_payload(model)
    assert payload["parameterisation_schema"] == report.schema == state_schema(model.adapter_state())


def test_conversion_with_padded_and_rescaled_offline_preserves_function(batch):
    for offline in (TierConfig(OFFLINE, 8, 16.0, 1), TierConfig(OFFLINE, 8, 8.0, 1)):
        model = make_model()
        set_adapters_nonzero(model)
        before = run(model, batch)
        report = convert_to_tiered(model, TIERS + (offline,), probe=(batch, 3, 2))
        assert report.offline_padded and report.probe_relative_error <= 1e-6
        assert relative(run(model, batch), before) <= 1e-6
        assert check_invariants(model) == []


def test_conversion_refusals_leave_the_model_untouched(batch):
    model = make_model()
    set_adapters_nonzero(model)
    names = sorted(n for n, _ in model.named_parameters())
    before = run(model, batch)
    with pytest.raises(TierError):
        convert_to_tiered(model, TIERS + (TierConfig(OFFLINE, 2, 4.0, 1),))
    with pytest.raises(TierError):  # probe failure: the swap is undone
        convert_to_tiered(model, TIERS, probe=(batch, 3, 2), tolerance=-1.0)
    assert type(model) is AdaptedPredictor and not hasattr(model, "sleep_schema")
    assert sorted(n for n, _ in model.named_parameters()) == names
    assert torch.equal(run(model, batch), before) and model.version == "base"
    convert_to_tiered(model, TIERS)
    with pytest.raises(TierError):
        convert_to_tiered(model, TIERS)


def test_global_rng_is_never_consumed():
    model = make_model()
    state = torch.random.get_rng_state()
    convert_to_tiered(model, TIERS)
    activate_slot(model, "slow")
    reset_tier(model, "slow")
    reset_tier(model, "fast")
    assert torch.equal(torch.random.get_rng_state(), state)


# ---------------------------------------------------------------------------- C1, C2, C3 contracts


def test_c2_online_training_reaches_exactly_the_fast_tier():
    model = tiered()
    activate_slot(model, "slow")
    fill(model, "slow")
    groups = model.set_trainable("online")
    online = [p for g in groups.values() for p in g]
    assert [id(p) for p in online] == [id(p) for p in tier_parameters(model, "fast", 0)]
    assert [id(p) for p in groups["lora"]] == [id(p) for p in lora_parameters(model.predictor, 0)]
    assert [id(p) for p in groups["lora"]] == [id(p) for p in tier_parameters(model, "fast", 0, component="lora")]
    assert [id(p) for p in groups["residual"]] == [id(p) for p in model.residual.online_parameters(0)]
    assert [id(p) for p in groups["residual"]] == [id(p) for p in tier_parameters(model, "fast", 0, component="residual")]
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == set(tier_named_parameters(model, "fast", 0))
    with pytest.raises(TierError):
        model.set_trainable("offline")


def test_unchanged_online_builder_updates_only_the_fast_tier():
    model = tiered()
    activate_slot(model, "slow")
    fill(model, "slow")
    builder = OnlineBuilder(model, model.clone_role(), OnlineConfig(warmup_updates=1, lora_lr=1e-2, residual_lr=1e-2))
    fast = set(tier_named_parameters(model, "fast", 0))
    fast_b = {n for n in fast if n.endswith("lora_b") or ".b_o." in n}
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    result = builder.update(make_batch(frames=6), [True, False], context=3)
    assert result.status == "ok"
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed and changed <= fast and changed >= fast_b


def test_c1_slot_state_is_parameters_named_deltas_and_part_of_adapter_state():
    model = tiered()
    names = set(model.adapter_names())
    for e in slot_entries(model):
        assert e.name in names, e.name
        assert not e.parameter.requires_grad or e.component != "meta"
        if e.name.startswith("predictor."):
            assert ".deltas." in e.name, e.name
    masks = [e for e in slot_entries(model) if e.role == "slot_active"]
    assert masks and all(not m.parameter.requires_grad and m.parameter.dtype.is_floating_point for m in masks)
    counters = [e for e in slot_entries(model) if e.role in ("slot_generation", "slot_birth", "schema")]
    assert all(c.parameter.dtype == torch.int64 for c in counters)
    buffers = {id(b) for b in model.buffers()}
    assert not any(id(e.parameter) in buffers for e in slot_entries(model))
    digest = state_digest(model.adapter_state())
    activate_slot(model, "slow")
    assert state_digest(model.adapter_state()) != digest, "the slot map is in the digest"


def test_c3_residual_tiers_share_the_frozen_trunk(batch):
    model = tiered()
    regime = model.residual.regimes[0]
    assert model.residual.online_parameters(0) == [*regime.modulation_out.parameters(), *regime.b_o.parameters()]
    slot = activate_slot(model, "slow")
    residual_params = tier_named_parameters(model, "slow", 0, slot, component="residual")
    assert sorted(n.rsplit("slots.", 1)[1] for n in residual_params) == sorted(
        f"{slot}.{part}.{kind}" for part in ("modulation_out", "b_o") for kind in ("weight", "bias"))
    before = run(model, batch)
    fill(model, "slow", slot, lora=False)
    assert not torch.equal(run(model, batch), before), "a residual slot contributes when B is non-zero"
    # residual slot k computes b_o,k tanh(a_o z0 + m_k(M(c))): one expert per slot, shared a_o and trunk
    z0 = torch.nn.functional.layer_norm(torch.randn(1, 1, 2, 3, 4, 32), (32,))
    cond = torch.randn(1, 1, 2, 3, 4, regime.modulation[0].in_features)
    trunk, hidden = regime.modulation(cond), regime.a_o(z0)
    expected = sum(e.b_o(torch.tanh(hidden + e.modulation_out(trunk)))
                   for e in (regime.deltas[OFFLINE].slots[0], regime.deltas["slow"].slots[slot], regime))
    assert torch.allclose(regime(z0, cond), expected, atol=1e-6)


# ---------------------------------------------------------------------------- roles and snapshots


def test_clone_role_shares_the_base_and_isolates_every_tier_tensor(batch):
    model = tiered()
    clone = model.clone_role()
    base = {n: p for n, p in model.named_parameters() if n.endswith("base.weight")}
    clone_params = dict(clone.named_parameters())
    assert base and all(clone_params[n].data_ptr() == p.data_ptr() for n, p in base.items())
    ptrs = {e.parameter.data_ptr() for e in slot_entries(model)}
    assert not ptrs & {e.parameter.data_ptr() for e in slot_entries(clone)}
    before = (run(model, batch), tier_state(model), active_slots(model))
    slot = activate_slot(clone, "slow")
    fill(clone, "slow", slot)
    reset_tier(clone, "fast")
    assert active_slots(model) == before[2] and tier_state(model) == before[1]
    assert torch.equal(run(model, batch), before[0])
    assert active_slots(clone, 0)["slow"] == [slot]


def test_adapter_state_and_version_store_round_trip(tmp_path, batch):
    model = tiered()
    slot = activate_slot(model, "slow")
    fill(model, "slow", slot)
    store = VersionStore(tmp_path)
    store.save(model.snapshot("v1"))
    other = tiered(nonzero=False)
    assert active_slots(other, 0)["slow"] == []
    other.load_adapter_state(store.load("v1")["snapshot"].state, "v1")
    assert tier_state(other) == tier_state(model) and active_slots(other) == active_slots(model)
    assert torch.equal(run(other, batch), run(model, batch))
    assert state_digest(other.adapter_state()) == state_digest(model.adapter_state())


def test_model_hub_challenger_reproduces_a_builder_with_new_slots(tmp_path, batch):
    model = tiered()
    hub = ModelHub(model, VersionStore(tmp_path), PromotionConfig(), shadow=False)
    slot = activate_slot(hub.builder, "slow")
    fill(hub.builder, "slow", slot)
    reset_tier(hub.builder, "fast")
    assert hub.maybe_snapshot(0.0, 0)
    assert active_slots(hub.challenger) == active_slots(hub.builder)
    assert torch.equal(run(hub.challenger, batch), run(hub.builder, batch))
    assert active_slots(hub.serving, 0)["slow"] == []
    hub.promote()
    restored = hub.rollback("test")
    assert restored == "offline-reference" and active_slots(hub.serving, 0)["slow"] == []


def test_cross_schema_loads_are_refused():
    legacy = make_model()
    legacy_state = legacy.adapter_state()
    model = tiered()
    assert state_schema(legacy_state) == "legacy-v1"
    with pytest.raises(TierSchemaError):
        model.load_adapter_state(legacy_state, "old")
    with pytest.raises(ValueError):  # tiered state into a legacy model: keys differ
        legacy.load_adapter_state(model.adapter_state(), "new")
    other_alpha = tiered((TierConfig("fast", 4, 8.0, 1, 3), TierConfig("slow", 8, 8.0, 3, None)))
    assert set(other_alpha.adapter_state()) == set(model.adapter_state())
    with pytest.raises(TierSchemaError):  # same keys and shapes, different meaning
        model.load_adapter_state(other_alpha.adapter_state(), "x")
    with pytest.raises(TierSchemaError):
        load_tier_state(model, tier_state(other_alpha))


def test_inconsistent_states_are_refused_before_anything_is_copied():
    model = tiered()
    reference = tier_state(model)
    masks = [e.name for e in slot_entries(model) if e.role == "slot_active" and e.tier == "slow"]
    state = model.adapter_state()
    state[masks[0]] = torch.tensor([1.0, 0.0, 0.0])
    with pytest.raises(TierStateError):  # containers disagree
        model.load_adapter_state(state, "bad")
    state = model.adapter_state()
    state[masks[0]] = torch.tensor([0.5, 0.0, 0.0])
    with pytest.raises(TierStateError):
        validate_state(model, state)
    state = model.adapter_state()
    b = next(e.name for e in slot_entries(model) if e.tier == "slow" and e.role == "lora_b")
    state[b] = torch.ones_like(state[b])
    with pytest.raises(TierStateError):  # I1: inactive slot with non-zero B
        model.load_adapter_state(state, "bad")
    assert tier_state(model) == reference


def test_migrate_legacy_state_matches_conversion(batch):
    legacy = make_model()
    set_adapters_nonzero(legacy)
    legacy_state = legacy.adapter_state()
    converted = make_model()
    set_adapters_nonzero(converted, seed=9)  # different stage-3 values, overwritten by the migration
    convert_to_tiered(converted, TIERS)
    slot = activate_slot(converted, "slow")
    fill(converted, "slow", slot)
    migrated = migrate_legacy_state(converted, legacy_state)
    converted.load_adapter_state(migrated, "migrated")
    convert_to_tiered(legacy, TIERS)
    assert check_invariants(converted) == []
    assert active_slots(converted) == active_slots(legacy)
    assert torch.equal(run(converted, batch), run(legacy, batch))
    with pytest.raises(TierSchemaError):
        migrate_legacy_state(converted, converted.adapter_state())
    broken = dict(legacy_state)
    broken.pop(next(k for k in broken if k.startswith("residual.regimes.0.a_o")))
    with pytest.raises(TierError):
        migrate_legacy_state(converted, broken)


def test_tier_state_round_trip():
    model = tiered()
    saved = tier_state(model)
    assert saved == copy.deepcopy(saved) and saved == tier_state(model.clone_role())
    slot = activate_slot(model, "slow")
    fill(model, "slow", slot)
    reset_tier(model, "fast")
    assert tier_state(model) != saved
    version = model.version
    load_tier_state(model, saved)
    assert tier_state(model) == saved and model.version != version
    load_tier_state(model, saved, version="restored")
    assert model.version == "restored"


# ---------------------------------------------------------------------------- masking, activation, reset


def test_activation_preserves_function_bit_for_bit_and_never_evicts(batch):
    model = tiered()
    fill(model, "fast")
    before = run(model, batch)
    versions = {model.version}
    for expected in range(3):
        assert activate_slot(model, "slow") == expected
        assert torch.equal(run(model, batch), before)  # Theorem 9.1
        versions.add(model.version)
    assert len(versions) == 4, "every activation is a new version (I3)"
    state = tier_state(model)
    with pytest.raises(TierFullError):
        activate_slot(model, "slow")
    assert tier_state(model) == state
    with pytest.raises(TierError):
        activate_slot(model, "fast")
    with pytest.raises(TierFullError):
        activate_slot(model, OFFLINE)


def test_version_change_invalidates_kv_caches():
    model = tiered()
    batch = make_batch()
    actions, states = model.sequence_inputs(batch, 3, 1)
    prefix = model.prime_context(batch["features"][:, :2], actions[:, :2], states[:, :2], batch["view_ids"])
    activate_slot(model, "slow")
    with pytest.raises(StaleCacheError):
        model.step_cached(batch["features"][:, 2:3], actions[:, 2:3], states[:, 2:3], batch["view_ids"],
                          model.encode_commands(batch, 3)[:, 2:3], prefix, 2)


def test_masked_slots_change_nothing_and_receive_no_gradient(batch):
    model = tiered()
    before = run(model, batch)
    inactive = tier_named_parameters(model, "slow", 0, 1)
    with torch.no_grad():  # violate I1 on purpose: a masked slot must still be invisible
        for name, p in inactive.items():
            if name.endswith("lora_b") or ".b_o." in name:
                p.add_(1.0)
    assert torch.equal(run(model, batch), before)
    for p in inactive.values():
        p.requires_grad_(True)
    model.rollout(batch, 3, 2).sum().backward()
    assert all(p.grad is None for p in inactive.values())
    assert any("I1" in problem for problem in check_invariants(model))
    activate_slot(model, "slow")  # slot 0 is clean
    with pytest.raises(TierStateError):
        activate_slot(model, "slow")  # slot 1 is dirty: refused
    assert active_slots(model, 0)["slow"] == [0]


def test_gradients_are_isolated_per_tier(batch):
    model = tiered()
    slot = activate_slot(model, "slow")
    fill(model, "fast")
    fill(model, "slow", slot, seed=1)
    with pytest.raises(TierError):  # written slow slots are frozen: the slot must be named
        set_tier_trainable(model, ["slow"])
    groups = set_tier_trainable(model, ["slow"], slots={"slow": [slot]})
    assert set(groups) == {"slow.lora", "slow.residual"}
    model.rollout(batch, 3, 2).sum().backward()
    with_grad = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert with_grad and with_grad <= set(tier_named_parameters(model, "slow", 0, slot))
    model.zero_grad(set_to_none=True)
    model.set_trainable("online")
    model.rollout(batch, 3, 2).sum().backward()
    with_grad = {n for n, p in model.named_parameters() if p.grad is not None}
    assert with_grad and with_grad <= set(tier_named_parameters(model, "fast", 0))


def test_reset_is_exact_seeded_and_refuses_the_offline_tier_by_default(batch):
    model = tiered()
    zero_fast = run(model, batch)
    fill(model, "fast")
    reference = model.clone_role()
    slot = activate_slot(model, "slow")
    fill(model, "slow", slot)
    a_before = tier_named_parameters(model, "slow", 0, slot, component="lora")
    a_before = {n: p.detach().clone() for n, p in a_before.items() if n.endswith("lora_a")}
    twin = model.clone_role()
    reset_tier(model, "slow")
    reset_tier(twin, "slow")
    assert torch.equal(run(model, batch), run(reference, batch)), "a reset slot contributes exactly zero"
    assert check_invariants(model) == [], "inactive, B = 0, A = seeded init of the new generation"
    after = tier_named_parameters(model, "slow", 0, slot)
    assert all(not torch.equal(after[n], v) for n, v in a_before.items()), "A re-initialised"
    assert tier_state(model) == tier_state(twin), "the re-init is deterministic"
    reset_tier(model, "fast")
    assert active_slots(model, 0)["fast"] == [0]
    assert torch.equal(run(model, batch), zero_fast)
    with pytest.raises(TierError):
        reset_tier(model, OFFLINE)
    reset_tier(model, OFFLINE, allow_offline=True)
    assert active_slots(model, 0)[OFFLINE] == [] and check_invariants(model) == []
    assert activate_slot(model, OFFLINE) == 0


def test_slots_by_age_tracks_activation_order():
    model = tiered()
    for _ in range(3):
        activate_slot(model, "slow")
    reset_tier(model, "slow", slots=[0])
    assert activate_slot(model, "slow") == 0
    assert slots_by_age(model, "slow") == [1, 2, 0]
    assert check_invariants(model) == []


# ---------------------------------------------------------------------------- regimes


def test_regimes_are_independent():
    model = two_regime_model()
    set_adapters_nonzero(model)
    convert_to_tiered(model, TIERS)
    batch = make_batch()
    before = run(model, batch)
    slot = activate_slot(model, "slow", regime=1)
    fill(model, "slow", slot, regime=1)
    assert active_slots(model)["slow"] == {0: [], 1: [slot]}
    assert torch.equal(run(model, batch), before), "regime 0 is unaffected"
    model.set_regime(1)
    regime1 = run(model, batch)
    assert not torch.equal(regime1, before)
    model.set_regime(0)
    model.set_trainable("online")
    model.rollout(batch, 3, 2).sum().backward()
    assert all(p.grad is None for p in tier_parameters(model, "slow", 1, slot))
    reset_tier(model, "fast", "all")
    reset_tier(model, "slow", regime=0)
    assert active_slots(model)["slow"] == {0: [], 1: [slot]}
    assert check_invariants(model) == []


# ---------------------------------------------------------------------------- exact moves (Theorems 9.2, Prop. 9.3)


@pytest.mark.parametrize("slow", [TierConfig("slow", 8, 16.0, 3), TierConfig("slow", 8, 8.0, 3), TierConfig("slow", 4, 8.0, 3)],
                         ids=["pad-same-scale", "pad-scale-ratio-2", "same-rank"])
def test_transplant_then_reset_preserves_function(batch, slow):
    model = tiered((TIERS[0], slow))
    fill(model, "fast", seed=4)
    before = {kv: run(model, batch, use_kv=kv) for kv in (True, False)}
    slot = activate_slot(model, "slow")
    report = transplant(model, ("fast", 0), ("slow", 0, slot))
    reset_tier(model, "fast")
    assert report.max_relative_error <= 1e-12
    for kv in (True, False):
        assert relative(run(model, batch, use_kv=kv), before[kv]) <= 1e-6
    lora_a = [p for n, p in tier_named_parameters(model, "slow", 0, slot).items() if n.endswith("lora_a")]
    if slow.rank > 4:
        assert all(p[4:].abs().sum() > 0 for p in lora_a), "padded rows of A are random, not dead zeros"
    assert check_invariants(model) == []


def test_transplant_refusals():
    model = tiered()
    fill(model, "fast")
    with pytest.raises(TierError):
        transplant(model, ("fast", 0), ("slow", 0, 0))  # not active
    slot = activate_slot(model, "slow")
    transplant(model, ("fast", 0), ("slow", 0, slot))
    with pytest.raises(TierError):
        transplant(model, ("fast", 0), ("slow", 0, slot))  # not empty: never overwrites
    small = tiered((TierConfig("fast", 4, 8.0, 1), TierConfig("slow", 2, 4.0, 2)))
    activate_slot(small, "slow")
    with pytest.raises(TierError):
        transplant(small, ("fast", 0), ("slow", 0, 0))  # rank too small


def test_merge_slots_by_concatenation_is_exact(batch):
    model = tiered(ARCHIVE)
    for seed in (1, 2):
        slot = activate_slot(model, "slow")
        fill(model, "slow", slot, seed=seed, residual=False)
    before = run(model, batch)
    target = activate_slot(model, "archive")
    report = merge_slots(model, "slow", 0, [0, 1], ("archive", target))
    assert report.exact and report.method == "concat"
    assert active_slots(model, 0)["slow"] == [] and active_slots(model, 0)["archive"] == [target]
    assert relative(run(model, batch), before) <= 1e-6
    assert check_invariants(model) == []


def test_merge_refuses_overlapping_residuals_unless_zeroed(batch):
    model = tiered(ARCHIVE)
    for seed in (1, 2):
        fill(model, "slow", activate_slot(model, "slow"), seed=seed)
    target = activate_slot(model, "archive")
    state = tier_state(model)
    with pytest.raises(TierError):
        merge_slots(model, "slow", 0, [0, 1], ("archive", target))
    assert tier_state(model) == state, "refusal writes nothing"
    report = merge_slots(model, "slow", 0, [0, 1], ("archive", target), residual="zero")
    assert report.residual == "zero" and report.residual_units_moved == 0
    archive_bo = [p for n, p in tier_named_parameters(model, "archive", 0, target).items() if ".b_o." in n]
    assert all(not p.any() for p in archive_bo)
    # a merge whose components do not fit is refused
    full = tiered(ARCHIVE)
    for seed in (1, 2, 3):
        fill(full, "slow", activate_slot(full, "slow"), seed=seed, residual=False)
    t = activate_slot(full, "archive")
    with pytest.raises(TierError):
        merge_slots(full, "slow", 0, [0, 1, 2], ("archive", t))


# ---------------------------------------------------------------------------- whitened compaction (Theorem 9.4, eq. S5)


def collect(model, batches):
    with InputCovariance(model) as collector, torch.no_grad():
        for b in batches:
            model.rollout(b, 3, 2)
    return collector.covariances()


def test_whitened_compaction_is_optimal_and_reports_its_error():
    model = tiered(ARCHIVE)
    for seed in (5, 6):
        fill(model, "slow", activate_slot(model, "slow"), seed=seed, residual=False)
    covariances = collect(model, [make_batch(seed=s) for s in range(3)])
    plain_twin = model.clone_role()
    entry = next(e for e in slot_entries(model) if e.tier == "slow" and e.slot == 0 and e.role == "lora_b")
    deltas = [model.predictor.get_submodule(entry.path).tiers["slow"].deltas[0].slots[s][entry.index] for s in (0, 1)]
    delta = sum(d.lora_b.detach().double() * d.scale @ d.lora_a.detach().double() for d in deltas)
    c = covariances[entry.path].double()
    c = (c + c.T) / 2
    c_eps = c + 1e-4 * torch.trace(c) / c.shape[0] * torch.eye(c.shape[0], dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(c_eps)
    sigma = torch.linalg.svdvals(delta @ ((evecs * evals.sqrt()) @ evecs.T))

    report = whitened_compaction(model, "slow", 0, [0, 1], covariances)
    plain = whitened_compaction(plain_twin, "slow", 0, [0, 1], None)
    assert active_slots(model, 0)["slow"] == [0] and check_invariants(model) == []
    first = next(s for s in report.slices if s.module == entry.path and s.slice_index == entry.index)
    assert first.whitened_error == pytest.approx(float((sigma[4:] ** 2).sum()), rel=1e-6)
    assert first.retained_energy == pytest.approx(float((sigma[:4] ** 2).sum() / (sigma ** 2).sum()), rel=1e-9)
    assert 0 < report.min_retained_energy < 1 and not report.exact
    # the data-weighted error is no larger than the error of plain (unwhitened) truncation
    merged = model.predictor.get_submodule(entry.path).tiers["slow"].deltas[0].slots[0][entry.index]
    twin = plain_twin.predictor.get_submodule(entry.path).tiers["slow"].deltas[0].slots[0][entry.index]
    for d, expect in ((merged, first.data_error), (twin, None)):
        err = d.lora_b.detach().double() * d.scale @ d.lora_a.detach().double() - delta
        value = float(((err @ c) * err).sum())
        if expect is not None:
            assert value == pytest.approx(expect, rel=1e-5, abs=1e-12)
        else:
            assert first.data_error <= value * (1 + 1e-6)


def test_whitened_compaction_is_exact_when_the_rank_fits(batch):
    model = tiered(ARCHIVE)
    fill(model, "slow", activate_slot(model, "slow"), seed=7)
    activate_slot(model, "slow")  # active but empty
    before = run(model, batch)
    covariances = collect(model, [batch])
    report = whitened_compaction(model, "slow", 0, [0, 1], covariances)
    assert report.min_retained_energy == pytest.approx(1.0) and report.max_relative_error < 1e-12
    assert report.residual_units_moved == 0 and active_slots(model, 0)["slow"] == [0]
    assert relative(run(model, batch), before) <= 1e-6


# ---------------------------------------------------------------------------- parity, optimizers, dtypes


@pytest.mark.parametrize("context,horizon", [(3, 4), (6, 7), (1, 12)])
def test_kv_generation_matches_full_forward_with_several_active_tiers(context, horizon):
    model = tiered()
    fill(model, "fast", seed=1)
    for seed in (2, 3):
        fill(model, "slow", activate_slot(model, "slow"), seed=seed)
    batch = make_batch(frames=context + horizon)
    cached = run(model, batch, context, horizon, use_kv=True)
    full = run(model, batch, context, horizon, use_kv=False)
    assert torch.allclose(cached, full, atol=2e-4), (cached - full).abs().max()


def test_tier_optimizer_steps_only_the_selected_slot():
    model = tiered()
    slot = activate_slot(model, "slow")
    activate_slot(model, "slow")  # an older active slot that must stay frozen
    optimizer, groups = build_tier_optimizer(model, ["slow"], 1e-2, 1e-2, slots={"slow": [slot]})
    assert all(g["weight_decay"] == 0.0 for g in optimizer.param_groups)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.rollout(make_batch(), 3, 2).sum().backward()
    optimizer.step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed and changed <= set(tier_named_parameters(model, "slow", 0, slot))
    with pytest.raises(TierError):
        set_tier_trainable(tiered(), ["slow"], slots={"slow": [0]})  # inactive slot: never trained
    with pytest.raises(TierError):
        set_tier_trainable(tiered(), ["slow"], all_active=True)  # no active slot: nothing to train


def test_low_precision_keeps_counters_exact(batch):
    model = tiered()
    for _ in range(2):
        activate_slot(model, "slow")
    for _ in range(300):  # more resets than bf16 can count exactly
        reset_tier(model, "fast")
    model.to(torch.bfloat16)
    entries = slot_entries(model)
    assert all(e.parameter.dtype == torch.int64 for e in entries if e.role in ("slot_generation", "slot_birth", "schema"))
    fast_generation = next(e for e in entries if e.role == "slot_generation" and e.tier == "fast")
    assert int(fast_generation.parameter[0]) == 300
    assert active_slots(model, 0)["slow"] == [0, 1]


def test_belief_conditioned_residual_converts_exactly():
    from robojepa_ttt.belief.visual import expand_residual_for_belief

    model = make_model()
    set_adapters_nonzero(model)
    model.residual = expand_residual_for_belief(model.residual, 5)
    batch = make_batch()
    belief = torch.randn(2, 4, 5)
    with torch.no_grad():
        before = model.rollout(batch, 3, 2, belief=belief)
    convert_to_tiered(model, TIERS)
    with torch.no_grad():
        assert torch.equal(model.rollout(batch, 3, 2, belief=belief), before)


def test_delta_norms_follow_contributions():
    model = tiered()
    norms = tier_delta_norms(model)
    assert norms["fast"] == 0.0 and norms["slow"] == 0.0 and norms[OFFLINE] > 0
    fill(model, "fast")
    assert tier_delta_norms(model)["fast"] > 0
    assert SCHEMA_KEY in model.adapter_state()


# ---------------------------------------------------------------------------- review additions


def shapes(model):
    return {n: (tuple(p.shape), p.dtype) for n, p in model.named_parameters()}


def test_i2_no_operation_changes_a_name_shape_or_dtype(batch):
    model = tiered(ARCHIVE)
    reference = shapes(model)
    fill(model, "fast")
    slot = activate_slot(model, "slow")
    transplant(model, ("fast", 0), ("slow", 0, slot))
    reset_tier(model, "fast")
    fill(model, "slow", activate_slot(model, "slow"), seed=2, residual=False)
    assert shapes(model) == reference
    whitened_compaction(model, "slow", 0, [0, 1], collect(model, [batch]), residual="zero")
    assert shapes(model) == reference
    merge_slots(model, "slow", 0, [0], ("archive", activate_slot(model, "archive")), residual="zero")
    load_tier_state(model, tier_state(model))
    assert shapes(model) == reference and check_invariants(model) == []
    assert set(model.adapter_state()) == set(tiered(ARCHIVE).adapter_state())


def test_cross_regime_transplant_is_refused_because_the_residual_trunk_is_per_regime():
    model = two_regime_model()
    convert_to_tiered(model, TIERS)
    fill(model, "fast")
    assert not torch.equal(model.residual.regimes[0].a_o.weight, model.residual.regimes[1].a_o.weight)
    slot = activate_slot(model, "slow", regime=1)
    state = tier_state(model)
    with pytest.raises(TierError, match="across regimes"):
        transplant(model, ("fast", 0), ("slow", 1, slot))
    assert tier_state(model) == state


def test_versions_are_unique_across_clones_so_caches_never_cross(batch):
    """I3 with roles: identical edits on two clones still yield different versions."""
    model = tiered()
    a, b = model.clone_role(), model.clone_role()
    activate_slot(a, "slow")
    activate_slot(b, "slow")
    assert a.version != b.version and tier_state(a) == tier_state(b)
    actions, states = a.sequence_inputs(batch, 3, 1)
    prefix = a.prime_context(batch["features"][:, :2], actions[:, :2], states[:, :2], batch["view_ids"])
    with pytest.raises(StaleCacheError):
        b.step_cached(batch["features"][:, 2:3], actions[:, 2:3], states[:, 2:3], batch["view_ids"],
                      b.encode_commands(batch, 3)[:, 2:3], prefix, 2)
    with pytest.raises(TierError):
        activate_slot(a, "slow", version="")


def test_masks_are_the_only_authority_even_when_written_through_data(batch):
    """C1: the forward reads the mask parameters; no stale plan survives any write path."""
    model = tiered()
    slot = activate_slot(model, "slow")
    fill(model, "slow", slot)
    with_slot = run(model, batch)
    masks = [e.parameter for e in slot_entries(model) if e.role == "slot_active" and e.tier == "slow"]
    for mask in masks:
        mask.data[slot] = 0.0  # bypasses the autograd version counter
    without = run(model, batch)
    assert not torch.equal(without, with_slot)
    for mask in masks:
        mask.data[slot] = 1.0
    refresh_slot_cache(model)
    assert torch.equal(run(model, batch), with_slot)
    clone = model.clone_role()  # a deepcopied plan cache must not leak into the clone
    reset_tier(clone, "slow")
    assert not torch.equal(run(clone, batch), with_slot) and torch.equal(run(model, batch), with_slot)


def test_free_slots_and_reset_expectations():
    model = tiered()
    assert free_slots(model, "slow") == [0, 1, 2]
    activate_slot(model, "slow")
    activate_slot(model, "slow")
    assert free_slots(model, "slow") == [2]
    state = tier_state(model)
    with pytest.raises(TierError):
        reset_tier(model, "slow", slots=[0], active_after=0)  # one slot would stay active
    assert tier_state(model) == state
    reset_tier(model, "slow", slots=[0], active_after=1)
    reset_tier(model, "fast", active_after=1)
    with pytest.raises(TierError):
        reset_tier(model, "fast", active_after=0)  # the wake tier never deactivates
    with pytest.raises(TierError):
        free_slots(model, "fast")
    assert free_slots(model, "slow") == [0, 2]


def test_spec_default_tiers_convert_exactly(batch):
    """Sleep spec 4.1 / 14: fast rank 8 alpha 16, slow 8 slots of rank 8 alpha 16 per regime.
    The fixture's stage-3 rank is 4, so the offline tier takes rank 4 and the fast rank differs
    from the legacy rank: harmless, because the fast tier starts at zero."""
    slow = DEFAULT_TIERS[1]
    assert (DEFAULT_TIERS[0].rank, DEFAULT_TIERS[0].alpha, DEFAULT_TIERS[0].slots) == (8, 16.0, 1)
    assert (slow.name, slow.rank, slow.alpha, slow.slots) == ("slow", 8, 16.0, 8)
    model = make_model()
    set_adapters_nonzero(model)
    before = run(model, batch)
    report = convert_to_tiered(model, probe=(batch, 3, 2))
    assert report.probe_relative_error == 0.0 and torch.equal(run(model, batch), before)
    assert [(t.name, t.rank) for t in model_tiers(model)] == [("fast", 8), ("slow", 8), (OFFLINE, 4)]
    assert model.sleep_tiers == model_tiers(model)
    for expected in range(8):
        assert activate_slot(model, "slow") == expected
    with pytest.raises(TierFullError):
        activate_slot(model, "slow")
    assert torch.equal(run(model, batch), before)


def test_conversion_is_exact_under_bf16_autocast():
    if not torch.amp.autocast_mode.is_autocast_available("cpu"):
        pytest.skip("no cpu autocast")
    model = make_model(amp_dtype=torch.bfloat16)
    set_adapters_nonzero(model)
    batch = make_batch()
    before = run(model, batch)
    report = convert_to_tiered(model, TIERS, probe=(batch, 3, 2), tolerance=1e-2)
    assert report.probe_relative_error == 0.0 and torch.equal(run(model, batch), before)
    slot = activate_slot(model, "slow")
    assert torch.equal(run(model, batch), before)
    fill(model, "fast", seed=3)
    with_fast = run(model, batch)
    transplant(model, ("fast", 0), ("slow", 0, slot))
    reset_tier(model, "fast")
    assert relative(run(model, batch), with_fast) <= 1e-2  # sleep spec 9.2 bf16 tolerance


def test_merge_from_the_wake_tier_into_an_archive_slot_is_exact(batch):
    model = tiered(ARCHIVE)
    fill(model, "fast", seed=8)
    before = run(model, batch)
    target = activate_slot(model, "archive")
    report = merge_slots(model, "fast", 0, [0], ("archive", target))
    assert report.exact and report.residual == "exact"
    assert active_slots(model, 0)["fast"] == [0] and not any(
        p.any() for n, p in tier_named_parameters(model, "fast", 0).items() if n.endswith("lora_b") or ".b_o." in n)
    assert relative(run(model, batch), before) <= 1e-6 and check_invariants(model) == []


def test_tiered_online_builder_refuses_anything_but_the_fast_tier():
    legacy = make_model()
    with pytest.raises(TierError):
        TieredOnlineBuilder(legacy, legacy.clone_role(), OnlineConfig())
    model = tiered()
    builder = TieredOnlineBuilder(model, model.clone_role(), OnlineConfig())
    trained = {id(p) for g in builder.optimizer.param_groups for p in g["params"]}
    assert trained == {id(p) for p in tier_parameters(model, "fast", 0)}
    builder.reset_optimizer()


def test_compute_consolidate_update_cycle_on_roles(tmp_path, batch):
    """Paper 3.3 / sleep spec 5.1 with M1 at the tier level: the teacher (serving) is never
    written; the student gets the fast tier transplanted into a fresh slot and the fast tier
    reset; a fresh builder then trains only the new fast tier; rollback restores the slot map."""
    model = tiered()
    hub = ModelHub(model, VersionStore(tmp_path), PromotionConfig(), shadow=False)
    fill(hub.serving, "fast", seed=11)  # stands in for validated fast knowledge
    teacher_state = tier_state(hub.serving)
    expected = run(hub.serving, batch)

    student = hub.serving.clone_role()
    slot = activate_slot(student, "slow")
    report = transplant(student, ("fast", 0), ("slow", 0, slot))
    reset_tier(student, "fast")
    assert report.max_relative_error <= 1e-12 and relative(run(student, batch), expected) <= 1e-6
    assert tier_state(hub.serving) == teacher_state, "the teacher is untouched"

    hub.challenger_snapshot = student.snapshot("sleep-00001", parent=hub.serving.version)
    hub.challenger = hub.serving.clone_role()
    hub.challenger.load_adapter_state(hub.challenger_snapshot.state, "sleep-00001")
    hub.promote()
    assert active_slots(hub.serving, 0)["slow"] == [slot]
    builder = TieredOnlineBuilder(hub.serving.clone_role(), hub.serving.clone_role(),
                                  OnlineConfig(warmup_updates=1, lora_lr=1e-2, residual_lr=1e-2))
    frozen = {n: p.detach().clone() for n, p in builder.model.named_parameters() if n not in
              tier_named_parameters(builder.model, "fast", 0)}
    assert builder.update(make_batch(frames=6), [True, False], context=3).status == "ok"
    assert all(torch.equal(p, frozen[n]) for n, p in builder.model.named_parameters() if n in frozen)
    hub.rollback("test")
    assert active_slots(hub.serving, 0)["slow"] == [] and check_invariants(hub.serving) == []


def test_compaction_dry_run_writes_nothing_and_ranks_pairs_by_discarded_energy(batch):
    """Sleep spec 4.4 step 2: prefer the pair whose merge discards the least whitened energy."""
    model = tiered(ARCHIVE)
    fill(model, "slow", activate_slot(model, "slow"), seed=1, residual=False)
    duplicate = activate_slot(model, "slow")
    transplant(model, ("slow", 0, 0), ("slow", 0, duplicate))  # same function: merging 0 and 1 is lossless
    fill(model, "slow", activate_slot(model, "slow"), seed=2, residual=False)
    covariances = collect(model, [batch])
    state, version = tier_state(model), model.version
    dry = whitened_compaction(model, "slow", 0, [0, 2], covariances, residual="zero", dry_run=True)
    assert tier_state(model) == state and model.version == version
    ranked = rank_compaction_pairs(model, "slow", 0, covariances)
    assert tier_state(model) == state and model.version == version
    assert [pair for _, pair, _ in ranked][0] == (0, 1) and ranked[0][0] == pytest.approx(0.0, abs=1e-12)
    assert all(score > 0 for score, pair, _ in ranked[1:])
    real = whitened_compaction(model.clone_role(), "slow", 0, [0, 2], covariances, residual="zero")
    assert [x.whitened_error for x in dry.slices] == pytest.approx([x.whitened_error for x in real.slices])
    before = run(model, batch)
    whitened_compaction(model, "slow", 0, list(ranked[0][1]), covariances, residual="exact")
    assert relative(run(model, batch), before) <= 1e-6 and free_slots(model, "slow") == [1]
