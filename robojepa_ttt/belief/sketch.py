"""Frozen compact observation sketch psi (spec 6.2, eq. 4).

    tokens [15, 20, 1664] per view
      -> LN0 per token
      -> adaptive average pool to 3 x 4 cells
      -> flatten cells, 12 x 1664 = 19,968 values
      -> 64 PCA coordinates, fitted per semantic camera on training frames only
      -> standardised with fixed training statistics and a variance floor

The sketch is a linear map after LN0 and pooling, so it is differentiable and can
also summarise predicted tokens for the compact/full consistency loss (eq. 28).

Camera contract. The sketch is fitted for named semantic view ids (RoboJEPA's
view_map: left 0, right 1, top 5, front 6, never wrist 2 here). Slot i of the
observation always means contract view i. An episode that lacks a contract view
gets that slot masked, never zeros presented as an observation, and an episode
view outside the contract is ignored. A moved or changed camera is a new contract
and needs a new fit (spec 6.2, 18).

Statistics are buffers, never parameters: they are frozen after fitting and no
optimiser can move them.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

SKETCH_VERSION = "ln0_pool3x4_pca_std_v1"


def pooled_tokens(tokens: torch.Tensor, rows: int = 3, cols: int = 4) -> torch.Tensor:
    """[..., H, W, C] -> [..., rows * cols * C] after LN0 and adaptive average pooling.

    Cells are row-major and channels vary fastest inside a cell, a fixed order that
    the fitted components depend on.
    """
    *lead, h, w, c = tokens.shape
    x = F.layer_norm(tokens.float(), (c,))
    x = x.reshape(-1, h, w, c).permute(0, 3, 1, 2)
    x = F.adaptive_avg_pool2d(x, (rows, cols))
    return x.permute(0, 2, 3, 1).reshape(*lead, rows * cols * c)


class ObservationSketch(nn.Module):
    def __init__(
        self,
        view_ids: Sequence[int],
        token_dim: int = 1664,
        components: int = 64,
        rows: int = 3,
        cols: int = 4,
    ):
        super().__init__()
        if len(set(view_ids)) != len(view_ids) or not view_ids:
            raise ValueError("contract view ids must be distinct and non-empty")
        self.view_ids = tuple(int(v) for v in view_ids)
        self.token_dim = token_dim
        self.components = components
        self.rows, self.cols = rows, cols
        width = rows * cols * token_dim
        views = len(self.view_ids)
        self.register_buffer("token_mean", torch.zeros(views, width))
        self.register_buffer("basis", torch.zeros(views, width, components))
        self.register_buffer("coord_mean", torch.zeros(views, components))
        self.register_buffer("coord_std", torch.ones(views, components))
        self.register_buffer("explained", torch.zeros(views, components))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @property
    def dim(self) -> int:
        return len(self.view_ids) * self.components

    def slot(self, view_id: int) -> int:
        return self.view_ids.index(int(view_id))

    # ------------------------------------------------------------------ fit

    @torch.no_grad()
    def fit(
        self,
        frames: Mapping[int, torch.Tensor | Iterable[torch.Tensor]],
        max_samples: int = 20_000,
        seed: int = 0,
        relative_floor: float = 1e-3,
        absolute_floor: float = 1e-6,
        oversample: int = 16,
        power_iterations: int = 6,
    ) -> dict:
        """Fit one PCA per contract view from TRAINING frames only.

        frames maps view id -> tokens [N, H, W, C] (or an iterable of such chunks).
        At most max_samples frames per view are used, drawn without replacement with
        a fixed seed, so the fit is reproducible. Components are sign-normalised so
        their largest-magnitude loading is positive. Coordinate scales are floored at
        max(relative_floor * largest scale, absolute_floor).
        Returns explained-variance fractions per view for the caller to inspect.
        """
        report = {}
        generator = torch.Generator().manual_seed(seed)
        for view_id in self.view_ids:
            if view_id not in frames:
                raise ValueError(f"no training frames for contract view {view_id}")
            source = frames[view_id]
            chunks = [source] if isinstance(source, torch.Tensor) else list(source)
            pooled = torch.cat([pooled_tokens(c.cpu(), self.rows, self.cols) for c in chunks]).double()
            if pooled.shape[0] > max_samples:
                pooled = pooled[torch.randperm(pooled.shape[0], generator=generator)[:max_samples]]
            count = pooled.shape[0]
            if count <= self.components:
                raise ValueError(f"view {view_id}: {count} frames cannot fit {self.components} components")
            mean = pooled.mean(0)
            centred = pooled - mean
            rank = min(self.components + oversample, count, centred.shape[1])
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                _, singular, vectors = torch.svd_lowrank(centred, q=rank, niter=power_iterations)
            basis = vectors[:, : self.components]
            flip = torch.sign(basis.gather(0, basis.abs().argmax(0, keepdim=True)))
            basis = basis * torch.where(flip == 0, torch.ones_like(flip), flip)
            coords = centred @ basis
            scale = coords.std(0, unbiased=True)
            floor = max(relative_floor * float(scale.max()), absolute_floor)
            total_variance = centred.pow(2).sum() / (count - 1)
            slot = self.slot(view_id)
            self.token_mean[slot] = mean.float()
            self.basis[slot] = basis.float()
            self.coord_mean[slot] = coords.mean(0).float()
            self.coord_std[slot] = scale.clamp_min(floor).float()
            self.explained[slot] = (singular[: self.components].pow(2) / (count - 1) / total_variance).float()
            report[view_id] = {
                "frames": count,
                "explained_total": float(self.explained[slot].sum()),
                "floored_coordinates": int((scale < floor).sum()),
            }
        self.fitted.fill_(True)
        return report

    def digest(self) -> str:
        """Identity of the fitted contract, for compatibility checks between models."""
        h = hashlib.sha256(SKETCH_VERSION.encode())
        h.update(repr(self.view_ids).encode())
        for name in ("token_mean", "basis", "coord_mean", "coord_std"):
            h.update(getattr(self, name).detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------ transform

    def transform_slot(self, tokens: torch.Tensor, slot: int) -> torch.Tensor:
        """[..., H, W, C] tokens of contract slot -> [..., components] standardised."""
        if not bool(self.fitted):
            raise RuntimeError("sketch is not fitted")
        pooled = pooled_tokens(tokens, self.rows, self.cols)
        coords = (pooled - self.token_mean[slot]) @ self.basis[slot]
        return (coords - self.coord_mean[slot]) / self.coord_std[slot]

    def observe(
        self,
        tokens: torch.Tensor,
        episode_view_ids: Sequence[int],
        frame_valid: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map an episode's views onto the contract.

        tokens [..., V_ep, H, W, C], frame_valid [..., V_ep] (default all valid).
        Returns (visual [..., dim], view_valid [..., n_contract_views]). Missing slots
        hold zeros and are False in view_valid; the filter must mask them.
        """
        lead = tokens.shape[:-4]
        visual = tokens.new_zeros(*lead, self.dim, dtype=torch.float32)
        valid = torch.zeros(*lead, len(self.view_ids), dtype=torch.bool, device=tokens.device)
        episode_view_ids = [int(v) for v in episode_view_ids]
        for slot, view_id in enumerate(self.view_ids):
            if view_id not in episode_view_ids:
                continue
            index = episode_view_ids.index(view_id)
            ok = (
                torch.ones(lead, dtype=torch.bool, device=tokens.device)
                if frame_valid is None
                else frame_valid[..., index].bool()
            )
            coords = self.transform_slot(tokens[..., index, :, :, :], slot)
            start = slot * self.components
            visual[..., start : start + self.components] = torch.where(ok[..., None], coords, torch.zeros_like(coords))
            valid[..., slot] = ok
        return visual, valid
