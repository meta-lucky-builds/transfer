"""Image preprocessing against the upstream path, and SO-101 data semantics."""

import numpy as np
import pytest
import torch

from conftest import UNITS
from robojepa_ttt.data.feature_store import preprocess
from robojepa_ttt.models.upstream import add_upstream_to_path


def test_preprocess_matches_upstream_world_model_for_a_4x3_frame():
    add_upstream_to_path()
    from app.robo_jepa.world_model import WorldModel

    frames = torch.randint(0, 256, (2, 480, 640, 3), dtype=torch.uint8)
    upstream = WorldModel.preprocess_frames(None, frames.numpy(), (240, 320)).permute(1, 0, 2, 3)
    ours = preprocess(frames)
    # Upstream rounds through uint8 after resizing, so at most half a grey level differs.
    assert ours.shape == upstream.shape == (2, 3, 240, 320)
    assert (ours - upstream).abs().max() < 2.0 / 255 / 0.224


def test_wide_camera_is_centre_cropped_not_stretched():
    frames = torch.zeros(1, 90, 160, 3, dtype=torch.uint8)
    frames[:, :, :20] = 255  # a band that a 4:3 centre crop removes
    frames[:, :, 140:] = 255
    # Per channel: ImageNet normalisation maps black to a different value per channel.
    assert preprocess(frames)[0].flatten(1).std(dim=1).max() < 1e-6


def test_large_phone_frames_are_antialiased():
    """A one-pixel grid averages to mid grey. Aliasing would keep strong contrast."""
    frames = torch.zeros(1, 1080, 1440, 3, dtype=torch.uint8)
    frames[:, ::2, ::2] = 255
    assert preprocess(frames).std() < 0.3


def test_legacy_conversion_matches_the_training_action_definition():
    """Upstream: actions = states[1:] - states[:-1], joints + gripper, no normalisation."""
    device = np.array([[10.0, 0, 0, 0, 0, 20.0], [12.0, 1, 0, 0, 0, 30.0]])
    legacy = UNITS.physical_to_legacy(UNITS.device_to_physical(device))
    assert legacy[1] - legacy[0] == pytest.approx(np.radians(device[1] - device[0]))
