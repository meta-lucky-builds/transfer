"""Loading the pinned RoboJEPA predictor without changing its source.

Weights must load before LoRA wrapping, because wrapping renames attention layers.
Upstream load_pretrained uses strict=False, which would drop weights silently. This
loader reports every mismatch and raises unless the caller has listed it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch

PINNED_COMMIT = "f93106f5f4a451fd526d51778aa8ab7c852eb824"
SO101_EMBODIMENT = "so-101_joints"
DEFAULT_REPO = Path(__file__).resolve().parents[2] / "third_party" / "robo_jepa"


def add_upstream_to_path(repo: Optional[str | os.PathLike] = None) -> Path:
    root = Path(repo) if repo is not None else DEFAULT_REPO
    if not (root / "app" / "robo_jepa").is_dir():
        raise FileNotFoundError(f"RoboJEPA checkout not found at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def build_predictor(config: dict, repo: Optional[str | os.PathLike] = None) -> torch.nn.Module:
    add_upstream_to_path(repo)
    from app.robo_jepa.robo_jepa.models.predictor import vit_predictor

    settings = {k: v for k, v in config.items() if k != "encoder_version"}
    if "grid_size" in settings:
        raise ValueError("use grid_height and grid_width")
    return vit_predictor(**settings)


def load_predictor_weights(
    predictor: torch.nn.Module,
    checkpoint_path: str | os.PathLike,
    allowed_missing: Iterable[str] = (),
    allowed_unexpected: Iterable[str] = (),
) -> None:
    # mmap keeps the optimizer state stored in these files out of RAM. Its moments are
    # about twice the size of the weights, which is tens of GB at 2B.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    weights = state["predictor"] if "predictor" in state else state["model"]
    weights = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in weights.items()}
    result = predictor.load_state_dict(weights, strict=False)
    missing = sorted(set(result.missing_keys) - set(allowed_missing))
    unexpected = sorted(set(result.unexpected_keys) - set(allowed_unexpected))
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch. missing={missing[:20]} unexpected={unexpected[:20]}")


def load_json_config(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_embodiment_id(config: dict, name: str = SO101_EMBODIMENT) -> int:
    """Resolve by name, because table positions in the paper are not array indices."""
    names = config.get("embodiment_list") or []
    if name not in names:
        raise ValueError(f"embodiment {name!r} is not in this checkpoint")
    return names.index(name)


def patch_joint_integration() -> None:
    """Make upstream WorldModel.step integrate SO-101 joints additively.

    Registering the embodiment as state-delta bypasses both the Euler composition
    and the [0, 1] gripper clip in compute_new_pose. This package's own rollouts do
    not need the patch. It exists for any caller still using WorldModel.step.
    """
    add_upstream_to_path()
    from app.robo_jepa import world_model

    world_model.STATE_DELTA_EMBODIMENTS.add(SO101_EMBODIMENT)
