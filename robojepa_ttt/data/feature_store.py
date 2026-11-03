"""Encode frames with the frozen V-JEPA 2.1 encoder and store features.

Follows upstream training: one image per encoder call, ImageNet statistics, bilinear
resize to 240 x 320, parameter-free layer norm on channels. No random crop or flip.
A centre crop to 4:3 precedes the resize, so wide cameras keep undistorted geometry.

Storage is per episode at keyframe rate, as spec 10.4 suggests for development. At
30 Hz, dense features cost about 216 GB per hour for two views, so offline training
uses keyframe starts and online replay re-encodes on demand.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PREPROCESSING_ID = "center-4x3_bilinear-240x320_antialias-over-2x_imagenet_ln0_v2"


def encoder_hash(checkpoint_path: str | os.PathLike) -> str:
    digest = hashlib.sha256(PREPROCESSING_ID.encode())
    with open(checkpoint_path, "rb") as handle:
        digest.update(handle.read(1 << 24))
    return digest.hexdigest()[:16]


def preprocess(frames: torch.Tensor, height: int = 240, width: int = 320) -> torch.Tensor:
    """frames: [N, H, W, 3] uint8. Returns [N, 3, height, width] normalised."""
    x = frames.permute(0, 3, 1, 2).float() / 255.0
    _, _, h, w = x.shape
    target_w = min(w, h * width // height)
    target_h = min(h, w * height // width)
    top, left = (h - target_h) // 2, (w - target_w) // 2
    x = x[:, :, top : top + target_h, left : left + target_w]
    # Training frames shrank by up to about 2x, where plain bilinear is what the model
    # saw. A 1080p phone frame shrinks 4.5x, and plain bilinear would alias fine detail.
    antialias = target_h > 2 * height
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False, antialias=antialias)
    mean = x.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = x.new_tensor(IMAGENET_STD)[None, :, None, None]
    return (x - mean) / std


@torch.no_grad()
def encode_frames(encoder, frames: torch.Tensor, device, dtype=torch.bfloat16, batch: int = 32) -> torch.Tensor:
    """Returns [N, 15, 20, 1664] after LN0, in bf16 on CPU."""
    outputs = []
    patch = encoder.vit.patch_size
    for start in range(0, frames.shape[0], batch):
        # Resize and normalise on the device: on CPU, shrinking 576x1024 frames dominated encoding.
        pixels = preprocess(frames[start : start + batch].to(device, non_blocking=True))
        grid_h, grid_w = pixels.shape[-2] // patch, pixels.shape[-1] // patch
        with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
            tokens = encoder(pixels[:, :, None])  # one frame per image, as in V-JEPA 2.1
        tokens = tokens.reshape(tokens.shape[0], grid_h, grid_w, -1)
        outputs.append(F.layer_norm(tokens.float(), (tokens.size(-1),)).to(torch.bfloat16).cpu())
    return torch.cat(outputs)


class DiskFeatures:
    """Per-episode keyframe tensors [K, V, H, W, C], memory-mapped on read."""

    def __init__(self, directory: str | os.PathLike, stride: int = 6):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stride = stride
        self._open: dict[str, torch.Tensor] = {}

    def path(self, episode_id: str) -> Path:
        safe = hashlib.sha256(episode_id.encode()).hexdigest()[:24]
        return self.directory / f"{safe}.pt"

    def save(self, episode_id: str, features: torch.Tensor) -> None:
        path = self.path(episode_id)
        temporary = path.with_suffix(".tmp")
        torch.save(features, temporary)
        os.replace(temporary, path)

    def get(self, episode_id: str, tick: int) -> torch.Tensor:
        if tick % self.stride:
            raise KeyError(f"tick {tick} is not a stored keyframe; use keyframe starts or re-encode")
        if episode_id not in self._open:
            self._open[episode_id] = torch.load(self.path(episode_id), map_location="cpu", mmap=True)
        return self._open[episode_id][tick // self.stride].float()
