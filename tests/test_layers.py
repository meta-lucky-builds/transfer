"""Layer correctness against the real upstream predictor."""

import pytest
import torch

from conftest import make_batch, make_model, set_adapters_nonzero
from robojepa_ttt.models.adapted_predictor import StaleCacheError, repeat_cache, state_digest
from robojepa_ttt.models.lora import LoRALinear, attach_lora, lora_parameters
from robojepa_ttt.models.token_residual import layer_norm_0
from robojepa_ttt.models.upstream import build_predictor
from conftest import tiny_config


def base_output(seed, batch, context=4, num_heads=4, num_kv_heads=4):
    """The unmodified upstream predictor, built from the same seed."""
    torch.manual_seed(seed)
    predictor = build_predictor(tiny_config(num_heads, num_kv_heads)).eval()
    probe = make_model(seed, num_heads, num_kv_heads)
    actions, states = probe.sequence_inputs(batch, context, 1)
    pad = torch.nn.functional.pad
    with torch.no_grad():
        out, *_ = predictor(
            batch["features"][:, :context],
            pad(actions, (0, 58)),
            states=pad(states, (0, 58)),
            view_map=batch["view_ids"],
            embodiment_ids=torch.full((2,), 2),
        )
    return out


@pytest.mark.parametrize("heads,kv", [(4, 4), (4, 2)], ids=["fused-qkv", "split-q-kv"])
def test_zero_adapters_match_base_model(heads, kv):
    batch = make_batch()
    model = make_model(0, heads, kv)
    with torch.no_grad():
        adapted = model.rollout(batch, 4, 1)
    expected = base_output(0, batch, 4, heads, kv)[:, -1:]
    assert torch.allclose(adapted, layer_norm_0(expected), atol=1e-5)


@pytest.mark.parametrize("heads,kv", [(4, 4), (4, 2)], ids=["fused-qkv", "split-q-kv"])
def test_lora_covers_q_v_output_and_not_k(heads, kv):
    predictor = build_predictor(tiny_config(heads, kv))
    names = attach_lora(predictor, 2, rank=4, alpha=8.0)
    assert len(names) == (4 if kv == heads else 6)
    attn = predictor.predictor_blocks[-1].attn
    wrapper = attn.qkv if kv == heads else attn.kv
    dim = wrapper.base.in_features
    x = torch.randn(1, 5, dim)
    with torch.no_grad():
        for delta in wrapper.deltas[0]:
            delta.lora_b.add_(1.0)
    change = wrapper(x) - wrapper.base(x)
    if kv == heads:
        assert change[..., dim : 2 * dim].abs().max() == 0, "K slice must stay untouched"
        assert change[..., :dim].abs().max() > 0 and change[..., 2 * dim :].abs().max() > 0
    else:
        width = wrapper.base.out_features
        assert change[..., : width // 2].abs().max() == 0, "K slice must stay untouched"
        assert change[..., width // 2 :].abs().max() > 0


def test_attach_fails_on_unrecognised_layout():
    predictor = build_predictor(tiny_config())
    del predictor.predictor_blocks[-1].attn.qkv
    with pytest.raises(ValueError, match="no recognised"):
        attach_lora(predictor, 2, rank=4, alpha=8.0)


def test_gradients_reach_only_adapters_and_b_first(model, batch):
    model.set_trainable("online")
    predicted = model.rollout(batch, 4, 2)
    predicted.abs().mean().backward()
    trainable = model.trainable_named()
    assert all(p.grad is not None for n, p in trainable.items() if not n.endswith("lora_a")), "missing gradient"
    assert all(p.grad is None for n, p in model.named_parameters() if n not in trainable)
    assert any(p.grad.abs().sum() > 0 for n, p in trainable.items() if n.endswith("lora_b"))


def test_gradient_survives_autocast_cache_on_second_pass(model, batch):
    """The bug upstream guards against at world_model.py:1431."""
    if not torch.amp.autocast_mode.is_autocast_available("cpu"):
        pytest.skip("no cpu autocast")
    model.amp_dtype = torch.bfloat16
    model.set_trainable("online")
    model.rollout(batch, 4, 3).float().abs().mean().backward()
    assert sum(p.grad.abs().sum() for n, p in model.trainable_named().items() if n.endswith("lora_b")) > 0


@pytest.mark.parametrize("use_kv", [False, True], ids=["full-forward", "kv-cache"])
def test_fused_rollout_equals_stepwise_rollout(use_kv):
    """The single grad forward must reproduce per-step autoregression exactly."""
    model = make_model()
    set_adapters_nonzero(model)
    batch = make_batch()
    context, horizon = 3, 5
    with torch.no_grad():
        fused = model.rollout(batch, context, horizon, use_kv_generation=use_kv)
        actions, states = model.sequence_inputs(batch, context, horizon)
        command = model.encode_commands(batch, context + horizon - 1)
        sequence = batch["features"][:, :context]
        outputs = []
        for k in range(horizon):
            T = context + k
            out = model.forward_tokens(sequence, actions[:, :T], states[:, :T], batch["view_ids"], command[:, :T])
            outputs.append(out[:, -1:])
            sequence = torch.cat([sequence, out[:, -1:]], dim=1)
    assert torch.allclose(fused, torch.cat(outputs, dim=1), atol=2e-5)


@pytest.mark.parametrize("context,horizon", [(6, 7), (8, 6), (1, 12)])
def test_kv_generation_matches_full_forward_past_the_attention_window(context, horizon):
    """Sequences longer than the 8-frame window, where the rolling cache evicts."""
    model = make_model()
    set_adapters_nonzero(model)
    batch = make_batch(frames=context + horizon)
    with torch.no_grad():
        cached = model.rollout(batch, context, horizon, use_kv_generation=True)
        full = model.rollout(batch, context, horizon, use_kv_generation=False)
    assert torch.allclose(cached, full, atol=2e-4), (cached - full).abs().max()


def test_activation_checkpointing_changes_nothing():
    """Offline training runs with checkpointing on, in generation and in the grad pass."""
    model = make_model()
    set_adapters_nonzero(model)
    batch = make_batch()
    with torch.no_grad():
        plain = model.rollout(batch, 3, 4)
    model.predictor.use_activation_checkpointing = True
    with torch.no_grad():
        checkpointed = model.rollout(batch, 3, 4)
    assert torch.allclose(plain, checkpointed, atol=1e-6)
    model.set_trainable("online")
    model.rollout(batch, 3, 4).abs().mean().backward()
    assert sum(p.grad.abs().sum() for n, p in model.trainable_named().items() if n.endswith("lora_b")) > 0


def test_rollout_is_causal(model, batch):
    """Changing anything measured after the context must not change predictions."""
    set_adapters_nonzero(model)
    context, horizon = 3, 3
    with torch.no_grad():
        reference = model.rollout(batch, context, horizon)
        tampered = {k: v.clone() for k, v in batch.items()}
        tampered["features"][:, context:] = 99.0
        tampered["position"][:, context:] = 99.0
        again = model.rollout(tampered, context, horizon)
    assert torch.equal(reference, again)


def test_future_commands_do_not_reach_earlier_predictions(model, batch):
    set_adapters_nonzero(model)
    with torch.no_grad():
        reference = model.rollout(batch, 3, 2)
        tampered = {k: v.clone() for k, v in batch.items()}
        cut = 30 + 3 * 6  # first tick after the block that drives frame index 3
        tampered["issued"][:, cut:] += 5.0
        again = model.rollout(tampered, 3, 2)
    assert torch.allclose(reference[:, 0], again[:, 0], atol=1e-6)
    assert not torch.allclose(reference[:, 1], again[:, 1], atol=1e-6)


def test_residual_sees_commands(model, batch):
    set_adapters_nonzero(model)
    with torch.no_grad():
        reference = model.rollout(batch, 3, 1)
        tampered = dict(batch)
        tampered["issued"] = batch["issued"] + 0.5
        again = model.rollout(tampered, 3, 1)
    assert not torch.allclose(reference, again)


def test_roles_share_base_and_isolate_adapters(model):
    other = model.clone_role()
    base = dict(model.predictor.named_parameters())
    copy = dict(other.predictor.named_parameters())
    frozen = next(n for n in base if "deltas" not in n)
    assert base[frozen] is copy[frozen]
    lora = next(n for n in base if n.endswith("lora_b"))
    assert base[lora] is not copy[lora]
    with torch.no_grad():
        copy[lora].add_(1.0)
    assert base[lora].abs().sum() == 0


def test_adapter_state_round_trip_and_digest(model, batch):
    set_adapters_nonzero(model)
    state = model.adapter_state()
    other = make_model()
    assert state_digest(state) != state_digest(other.adapter_state())
    other.load_adapter_state(state, "v1")
    assert other.version == "v1"
    with torch.no_grad():
        assert torch.equal(model.rollout(batch, 3, 1), other.rollout(batch, 3, 1))
    with pytest.raises(ValueError, match="key mismatch"):
        other.load_adapter_state({"x": torch.zeros(1)}, "bad")


def test_cached_candidates_match_uncached_for_different_actions(model):
    """Spec 13.3: only the action-independent prefix may be shared."""
    set_adapters_nonzero(model)
    batch = make_batch(batch=1)
    context = 4
    actions, states = model.sequence_inputs(batch, context, 1)
    command = model.encode_commands(batch, context)
    command = command.repeat(3, 1, 1)
    candidates = torch.cat([actions[:, -1:] + d for d in (0.0, 0.1, -0.2)], dim=0)
    view_ids = batch["view_ids"].repeat(3, 1)
    prefix = model.prime_context(
        batch["features"][:, : context - 1], actions[:, : context - 1], states[:, : context - 1], batch["view_ids"]
    )
    prefix.cache = repeat_cache(prefix.cache, 3)
    frame = batch["features"][:, context - 1 : context].repeat(3, 1, 1, 1, 1, 1)
    with torch.no_grad():
        cached, _ = model.step_cached(
            frame, candidates, states[:, -1:].repeat(3, 1, 1), view_ids, command[:, -1:], prefix, context - 1
        )
        full_actions = torch.cat([actions[:, :-1].repeat(3, 1, 1), candidates], dim=1)
        uncached = model.forward_tokens(
            batch["features"][:, :context].repeat(3, 1, 1, 1, 1, 1), full_actions, states.repeat(3, 1, 1), view_ids, command
        )
    assert torch.allclose(cached, uncached[:, -1:], atol=2e-5)
    assert not torch.allclose(cached[0], cached[1], atol=1e-5)


def test_stale_cache_is_refused(model):
    batch = make_batch(batch=1)
    actions, states = model.sequence_inputs(batch, 3, 1)
    command = model.encode_commands(batch, 3)
    prefix = model.prime_context(batch["features"][:, :2], actions[:, :2], states[:, :2], batch["view_ids"])
    model.version = "newer"
    with pytest.raises(StaleCacheError):
        model.step_cached(
            batch["features"][:, 2:3], actions[:, 2:3], states[:, 2:3], batch["view_ids"], command[:, 2:3], prefix, 2
        )


def test_online_mode_leaves_encoder_and_lower_residual_frozen(model):
    model.set_trainable("online")
    names = set(model.trainable_named())
    assert not any("command_encoder" in n or "a_o" in n or "modulation.0" in n for n in names)
    assert any("b_o" in n for n in names) and any("modulation_out" in n for n in names)
    assert sum(1 for n in names if ".deltas." in n) == len(lora_parameters(model.predictor, 0))


def test_one_view_path():
    model = make_model()
    set_adapters_nonzero(model)
    batch = make_batch(views=1)
    with torch.no_grad():
        output = model.rollout(batch, 3, 2)
    assert output.shape == (2, 2, 1, 3, 4, 32)


def test_regime_adapters_are_independent():
    from robojepa_ttt.models.token_residual import TokenResidual
    from robojepa_ttt.models.adapted_predictor import AdaptedPredictor
    from conftest import UNITS, SERVO

    torch.manual_seed(0)
    predictor = build_predictor(tiny_config()).eval()
    residual = TokenResidual(32, 8, (32, 32), 16, 4, 4, regimes=2)
    model = AdaptedPredictor(predictor, UNITS, SERVO, 2, 2, 4, 8.0, 2, command_hidden=16, residual=residual)
    assert any(isinstance(m, LoRALinear) for m in model.modules())
    with torch.no_grad():
        for parameter in lora_parameters(model.predictor, 1):
            parameter.add_(0.1)
    batch = make_batch()
    with torch.no_grad():
        first = model.rollout(batch, 3, 1)
        model.set_regime(1)
        second = model.rollout(batch, 3, 1)
        model.set_regime(0)
        third = model.rollout(batch, 3, 1)
    assert torch.equal(first, third) and not torch.allclose(first, second)
