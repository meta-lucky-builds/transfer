"""RoboJEPA predictor + LoRA + token residual + nominal command bridge.

This talks to VisionTransformerPredictor directly and does not use
WorldModel.rollout, because that path feeds measured future states and needs a
train_config that load_pretrained never sets.

Token layout of one sequence, as in upstream: at position t the model reads frame
z_t, action a_t and state s_t, and its output at t predicts z_{t+1}.

Causality contract for a context of c frames and a horizon H:
  frames   z_0..z_{c-1} are observed, later frames are the model's own detached output.
  actions  a_0..a_{c-2} are realized past motion. a_{c-1} onward come from the
           servo model, which reads only position at c-1 and issued commands.
  states   s_0..s_{c-1} are measured, later states integrate predicted motion.
No measured quantity after frame c-1 can reach a prediction.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..robot_io.so101_units import So101Units, propagate_joint_state
from .command_encoder import CommandEncoder, command_features
from .lora import LoRALinear, attach_lora, lora_parameters, set_lora_regime
from .servo import ServoParams, rollout_servo
from .token_residual import TokenResidual


class StaleCacheError(RuntimeError):
    """A KV cache built with one parameter version was used with another (spec 13.3)."""


@dataclass
class ContextCache:
    """KV cache and the version that built it. Valid for that version only."""

    cache: object
    version: str


@dataclass(frozen=True)
class AdapterSnapshot:
    """An immutable set of adapter tensors with its lineage."""

    version: str
    parent: Optional[str]
    state: dict
    digest: str
    data_cutoff: int = -1  # global admission count of the last window used to fit


def state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().float().numpy().tobytes())
    return digest.hexdigest()


def repeat_cache(cache, times: int):
    """Tile a KV cache across candidates. Lists are copied, so tiles never alias."""
    return type(cache)(
        layers=cache.layers,
        batch=cache.batch * times,
        frame_tokens=cache.frame_tokens,
        cache_len=cache.cache_len,
        num_heads=cache.num_heads,
        head_dim=cache.head_dim,
        dtype=cache.dtype,
        device=cache.device,
        pointer=list(cache.pointer),
        evicted=list(cache.evicted),
        keys=[k.repeat(times, 1, 1, 1, 1) for k in cache.keys],
        values=[v.repeat(times, 1, 1, 1, 1) for v in cache.values],
    )


class AdaptedPredictor(nn.Module):
    def __init__(
        self,
        predictor: nn.Module,
        units: So101Units,
        servo: ServoParams,
        embodiment_id: int,
        last_blocks: int = 8,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        regimes: int = 1,
        stride: int = 6,
        tick_s: float = 1.0 / 30.0,
        history_ticks: int = 30,
        command_hidden: int = 64,
        residual: Optional[TokenResidual] = None,
        amp_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.predictor = predictor
        self.units = units
        self.servo = servo
        self.embodiment_id = embodiment_id
        self.stride = stride
        self.tick_s = tick_s
        self.history_ticks = history_ticks
        self.amp_dtype = amp_dtype
        self.version = "base"
        self.regime = 0
        self.lora_modules = attach_lora(predictor, last_blocks, lora_rank, lora_alpha, regimes)
        self.command_encoder = CommandEncoder(command_hidden)
        token_dim = predictor.predictor_proj.out_features
        view_ids = len(predictor.view_map) if predictor.view_map else 7
        self.residual = residual or TokenResidual(
            token_dim=token_dim, command_dim=command_hidden, num_view_ids=view_ids, regimes=regimes
        )
        predictor.eval()

    # ------------------------------------------------------------------ roles

    def train(self, mode: bool = True):
        """Dropout stays off and running statistics stay fixed (spec 12.1).

        Gradients flow regardless of this flag. JEPA-TTT also adapts in eval mode.
        """
        return super().train(False)

    def _shared_base_tensors(self) -> list[torch.Tensor]:
        shared = [p for n, p in self.predictor.named_parameters() if "deltas" not in n]
        shared += list(self.predictor.buffers())
        for mask in getattr(self.predictor, "attn_mask_cache", {}).values():
            shared += list(mask) if isinstance(mask, tuple) else [mask]
        return shared

    def clone_role(self) -> "AdaptedPredictor":
        """Copy for another role. Frozen base storage is shared, adapters are not."""
        memo = {id(t): t for t in self._shared_base_tensors()}
        return copy.deepcopy(self, memo)

    # --------------------------------------------------------------- adapters

    def adapter_names(self) -> list[str]:
        return [
            name
            for name, _ in self.named_parameters()
            if ".deltas." in name or name.startswith(("residual.", "command_encoder."))
        ]

    def adapter_state(self) -> dict[str, torch.Tensor]:
        parameters = dict(self.named_parameters())
        return {name: parameters[name].detach().cpu().clone() for name in self.adapter_names()}

    def load_adapter_state(self, state: dict[str, torch.Tensor], version: str) -> None:
        parameters = dict(self.named_parameters())
        expected = set(self.adapter_names())
        if set(state) != expected:
            raise ValueError(f"adapter key mismatch: {sorted(set(state) ^ expected)[:10]}")
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value.to(parameters[name].device))
        self.version = version

    def snapshot(self, version: str, parent: Optional[str] = None, data_cutoff: int = -1) -> AdapterSnapshot:
        state = self.adapter_state()
        return AdapterSnapshot(version, parent, state, state_digest(state), data_cutoff)

    def set_regime(self, regime: int) -> None:
        self.regime = regime
        set_lora_regime(self.predictor, regime)

    def set_trainable(self, mode: str) -> dict[str, list[nn.Parameter]]:
        """Select parameters for 'offline', 'online' or 'none', spec 9.8."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        groups: dict[str, list[nn.Parameter]] = {"lora": [], "residual": []}
        if mode == "offline":
            groups["lora"] = lora_parameters(self.predictor, self.regime)
            groups["residual"] = [
                *self.residual.regimes[self.regime].parameters(),
                *self.residual.regime_embedding.parameters(),
                *self.residual.view_embedding.parameters(),
                *self.command_encoder.parameters(),
            ]
        elif mode == "online":
            groups["lora"] = lora_parameters(self.predictor, self.regime)
            groups["residual"] = self.residual.online_parameters(self.regime)
        elif mode != "none":
            raise ValueError(f"unknown mode {mode!r}")
        for group in groups.values():
            for parameter in group:
                parameter.requires_grad_(True)
        return groups

    def trainable_named(self) -> dict[str, nn.Parameter]:
        return {name: p for name, p in self.named_parameters() if p.requires_grad}

    # ---------------------------------------------------------------- forward

    def amp(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        device_type = next(self.parameters()).device.type
        return torch.autocast(device_type=device_type, dtype=self.amp_dtype)

    def _call_predictor(self, features, actions, states, view_ids, **kwargs):
        B = features.shape[0]
        ids = torch.full((B,), self.embodiment_id, device=features.device, dtype=torch.long)
        return self.predictor(
            features,
            F.pad(actions, (0, self.predictor.action_dim - actions.shape[-1])),
            states=F.pad(states, (0, self.predictor.state_dim - states.shape[-1])),
            view_map=view_ids,
            embodiment_ids=ids,
            **kwargs,
        )

    def forward_tokens(self, features, actions, states, view_ids, command, belief=None) -> torch.Tensor:
        """All next-frame predictions for a sequence. Output position t predicts z_{t+1}."""
        z0, *_ = self._call_predictor(features, actions, states, view_ids)
        return self.residual(z0, command, view_ids, self.regime, belief)

    def encode_commands(self, batch: dict, blocks: int) -> torch.Tensor:
        """e(U_j) for blocks j = 0..blocks-1, each block with its one second of history."""
        history, stride = self.history_ticks, self.stride
        compact = self.units.physical_to_compact(batch["issued"])
        features = command_features(compact, batch["dt"], batch["command_valid"], self.tick_s)
        windows = [features[:, j * stride : history + (j + 1) * stride] for j in range(blocks)]
        # Blocks share one length: pad on the left with zeros (validity 0).
        length = history + blocks * stride
        padded = [F.pad(w, (0, 0, length - w.shape[1], 0)) for w in windows]
        return self.command_encoder(torch.stack(padded, dim=1))

    def predicted_positions(self, batch: dict, context: int, horizon: int) -> torch.Tensor:
        """Servo prediction of keyframes context..context+horizon-1, [B, H, 6] physical."""
        start_tick = self.history_ticks + (context - 1) * self.stride
        end_tick = start_tick + horizon * self.stride
        ticks = rollout_servo(
            batch["position"][:, context - 1],
            batch["issued"][:, :end_tick],
            batch["dt"][:, :end_tick],
            self.servo,
            start_tick,
        )
        return ticks[:, self.stride - 1 :: self.stride]

    def sequence_inputs(self, batch: dict, context: int, horizon: int, motion: Optional[torch.Tensor] = None):
        """Legacy actions and states for positions 0..context+horizon-2.

        Implements the causality contract in the module docstring. motion [B, H, 6],
        physical joints of keyframes context..context+horizon-1, replaces the servo
        prediction: the belief's predicted joints in B4. It must itself be causal
        (built from data up to frame context-1 and issued commands only).
        """
        measured = batch["position"][:, :context]
        predicted = self.predicted_positions(batch, context, horizon) if motion is None else motion
        if predicted.shape[1] != horizon:
            raise ValueError(f"motion covers {predicted.shape[1]} keyframes, horizon is {horizon}")
        trajectory = torch.cat([measured, predicted], dim=1)
        legacy = self.units.physical_to_legacy(trajectory)
        actions = legacy[:, 1:] - legacy[:, :-1]
        return actions, legacy[:, :-1]

    def rollout(
        self,
        batch: dict,
        context: int,
        horizon: int,
        use_kv_generation: bool = True,
        motion: Optional[torch.Tensor] = None,
        belief: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autonomous command rollout. Returns [B, H, V, H, W, C], predictions of frames
        context..context+horizon-1, with gradient through the current parameters.

        Generation runs under no_grad, then one grad forward runs over the full
        sequence. Outputs at positions c-1.. equal the per-step predictions because
        fed-back frames are detached and attention is frame-causal. This matches
        upstream _rollout_fused and spec 9.2, where later losses do not backpropagate
        through earlier visual predictions.

        motion [B, H, 6] and belief [B, context + horizon - 1, belief_dim] are the B4
        inputs: belief-predicted joints and the residual's [h_t, x_{t+1}] per position.
        Both must arrive detached, so visual losses cannot train the belief (spec 9.4).
        """
        length = context + horizon - 1
        if belief is not None and belief.shape[1] != length:
            raise ValueError(f"belief covers {belief.shape[1]} positions, the sequence has {length}")
        if length > self.predictor.num_frames:
            raise ValueError(f"sequence of {length} frames exceeds the model's {self.predictor.num_frames}")
        actions, states = self.sequence_inputs(batch, context, horizon, motion)
        command = self.encode_commands(batch, length)
        observed = batch["features"][:, :context]
        view_ids = batch["view_ids"]

        generated = []
        if horizon > 1:
            with torch.no_grad(), self.amp():
                generated = self._generate(
                    observed, actions, states, view_ids, command, context, horizon, use_kv_generation, belief
                )

        # Autocast caches detached casts of leaf parameters during no_grad passes.
        # Reusing them here would silently cut the gradient to the LoRA matrices.
        torch.clear_autocast_cache()
        with self.amp():
            sequence = torch.cat([observed, *generated], dim=1) if generated else observed
            output = self.forward_tokens(sequence.detach(), actions, states, view_ids, command, belief)
        return output[:, context - 1 :]

    def _generate(self, observed, actions, states, view_ids, command, context, horizon, use_kv, belief=None):
        frames: list[torch.Tensor] = []

        def cond(start, end):
            return None if belief is None else belief[:, start:end]

        if use_kv:
            z0, _, _, cache = self._call_predictor(
                observed,
                actions[:, :context],
                states[:, :context],
                view_ids,
                use_cache=True,
                cache=None,
                cache_size=0,
                input_position=0,
            )
            pred = self.residual(z0[:, -1:], command[:, context - 1 : context], view_ids, self.regime, cond(context - 1, context))
            frames.append(pred)
            for k in range(1, horizon - 1):
                t = context - 1 + k
                z0, _, _, cache = self._call_predictor(
                    pred,
                    actions[:, t : t + 1],
                    states[:, t : t + 1],
                    view_ids,
                    use_cache=True,
                    cache=cache,
                    cache_size=cache.pointer[0],
                    input_position=t,
                )
                pred = self.residual(z0, command[:, t : t + 1], view_ids, self.regime, cond(t, t + 1))
                frames.append(pred)
            return frames
        sequence = observed
        for k in range(horizon - 1):
            T = context + k
            out = self.forward_tokens(sequence, actions[:, :T], states[:, :T], view_ids, command[:, :T], cond(0, T))
            frames.append(out[:, -1:])
            sequence = torch.cat([sequence, frames[-1]], dim=1)
        return frames

    def realized_forward(self, batch: dict, frames: int, belief: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Eq. 23: teacher forcing on realized motion. Diagnostic and pretraining only.

        Deployable prediction cannot use this path, as it reads the realized motion.
        """
        legacy = self.units.physical_to_legacy(batch["position"][:, :frames])
        actions = legacy[:, 1:] - legacy[:, :-1]
        command = self.encode_commands(batch, frames - 1)
        with self.amp():
            return self.forward_tokens(
                batch["features"][:, : frames - 1], actions, legacy[:, :-1], batch["view_ids"], command, belief
            )

    # ---------------------------------------------------------------- caching

    @torch.no_grad()
    def prime_context(self, features, actions, states, view_ids) -> Optional[ContextCache]:
        """Cache the action-independent prefix: frames whose action is already known.

        The current frame shares attention with its action, so it must not enter a
        cache shared between candidates (spec 13.3). Pass only frames 0..c-2 here.
        """
        if features.shape[1] == 0:
            return None
        with self.amp():
            *_, cache = self._call_predictor(
                features, actions, states, view_ids, use_cache=True, cache=None, cache_size=0, input_position=0
            )
        return ContextCache(cache, self.version)

    @torch.no_grad()
    def step_cached(
        self, frame, action, state, view_ids, command, context: Optional[ContextCache], position: int, belief=None
    ):
        """One prediction from a single new frame. Returns (prediction, updated cache)."""
        if context is not None and context.version != self.version:
            raise StaleCacheError(f"cache built by {context.version}, model is {self.version}")
        cache = context.cache if context is not None else None
        with self.amp():
            z0, _, _, cache = self._call_predictor(
                frame,
                action,
                state,
                view_ids,
                use_cache=True,
                cache=cache,
                cache_size=cache.pointer[0] if cache is not None else 0,
                input_position=position,
            )
            prediction = self.residual(z0, command, view_ids, self.regime, belief)
        return prediction, ContextCache(cache, self.version)


__all__ = [
    "AdaptedPredictor",
    "AdapterSnapshot",
    "ContextCache",
    "LoRALinear",
    "StaleCacheError",
    "propagate_joint_state",
    "repeat_cache",
    "state_digest",
]
