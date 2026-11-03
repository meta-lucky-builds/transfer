"""The whole run_pipeline.sh on CPU, with stand-ins only where hardware or downloads are needed.

Real: LeRobot v3 reader and video decoding, prepare(), split, units, servo fit, B1 offline
training under a 2-process torchrun (data parallel, gloo), belief training with
calibration, B0 versus B1 evaluation, all through the shipped scripts.
Stand-ins: a tiny RoboJEPA predictor with random weights saved as a real checkpoint, a
fixed function instead of the V-JEPA encoder, and a settings file sized for CPU.

Slow (a few minutes), so it runs only with RUN_PIPELINE_TEST=1.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
pytestmark = pytest.mark.skipif(os.environ.get("RUN_PIPELINE_TEST") != "1", reason="slow; set RUN_PIPELINE_TEST=1")


@pytest.fixture(autouse=True)
def attention_layout():
    yield None


def fake_encoder(seed=0):
    """Deterministic stand-in for V-JEPA: 3x4 grid of mean colours -> 32 channels."""
    projection = torch.randn(3, 32, generator=torch.Generator().manual_seed(seed))

    def encode(frames: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(frames).float() / 255.0  # [K, H, W, 3]
        x = torch.nn.functional.adaptive_avg_pool2d(x.permute(0, 3, 1, 2), (3, 4)).permute(0, 2, 3, 1)
        tokens = torch.sin(4 * x @ projection + torch.linspace(0, 3, 32))
        return torch.nn.functional.layer_norm(tokens, (32,)).to(torch.bfloat16)

    return encode


def test_full_pipeline_on_cpu(tmp_path):
    from conftest import tiny_config
    from prepare_lerobot import prepare
    from test_lerobot_v3 import build_dataset
    from robojepa_ttt.data.dataset_meta import DatasetMeta
    from robojepa_ttt.data.lerobot_v3 import LeRobotV3Source
    from robojepa_ttt.models.upstream import build_predictor
    from robojepa_ttt.robot_io.so101_units import So101Units

    work = tmp_path
    # 1. A v3 dataset with real video: 40 episodes of 6 s at 20 fps, 2 tasks.
    raw = work / "raw"
    build_dataset(raw, lengths=tuple(120 + (e % 5) * 4 for e in range(40)), per_file=8)
    source = LeRobotV3Source("user/synthetic", root=raw)
    cameras = {"front": 6, "top": 5}
    units = So101Units(gripper_device_closed=0.0, gripper_device_open=100.0, gripper_direction_verified=True)
    data = work / "data"
    DatasetMeta("user/synthetic", 20.0, cameras, True, 0.0, 100.0, "fake-encoder").save(data)
    manifest = prepare(source, list(range(40)), cameras, units, fake_encoder(), "fake-encoder", data, log=lambda m: None)
    (data / "manifest-0-of-1.json").write_text(json.dumps(manifest))

    # 2. A tiny predictor checkpoint in the released format.
    torch.manual_seed(0)
    config = tiny_config(4, 4)
    predictor = build_predictor(config)
    weights = work / "tiny.pth.tar"
    torch.save({"predictor": predictor.state_dict()}, weights)
    pred_config = work / "tiny_config.json"
    pred_config.write_text(json.dumps(config))

    # 3. Settings sized for CPU; the YAML's own dimension checks still apply.
    settings = yaml.safe_load((ROOT / "config_so101.yaml").read_text())
    settings["predictor"]["lora"]["last_blocks"] = 2
    settings["offline"]["compact"].update(effective_batch=4, sequence_steps=8, burn_in_steps=4)
    settings_path = work / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(settings))

    env = dict(
        os.environ, WORK=str(work), DATA=str(data), RUN=str(work / "run"), DEVICE="cpu", NGPU="2", SKIP_PREP="1",
        STAGES="split units servo smoke train belief eval", PRED_CONFIG=str(pred_config), WEIGHTS=str(weights),
        SETTINGS=str(settings_path), B1_MAX_UPDATES="6", B1_WARMUP="2", B1_CURRICULUM="3:2,3:5", B1_VAL_INTERVAL="3",
        BELIEF_MAX_UPDATES="4", BELIEF_VAL_INTERVAL="2", OMP_NUM_THREADS="2",
    )
    result = subprocess.run(["bash", str(ROOT / "scripts" / "run_pipeline.sh")], env=env, capture_output=True, text=True, timeout=1800)
    log = result.stdout + result.stderr
    assert result.returncode == 0, log[-6000:]
    run = work / "run"
    for stage in ("split", "units", "servo", "smoke", "train", "belief", "eval"):
        assert (run / f".done_{stage}").exists(), stage

    b1_log = (run / "logs" / "b1.log").read_text()
    assert "ranks 2" in b1_log, "B1 must have trained data parallel on two ranks"
    assert (run / "b1" / "best.pt").exists() and (run / "b1" / "latest.pt").exists()
    records = [json.loads(line) for line in b1_log.splitlines() if line.startswith('{"update"')]
    assert records and all(np.isfinite(r.get("loss", r.get("validation", 0.0))) for r in records)
    report = json.loads((run / "b0_vs_b1.json").read_text())
    assert {"h1", "h2", "h5"} <= set(report)
    belief = json.loads((run / "belief" / "report.json").read_text())
    assert (run / "sketch.pt").exists() and (run / "belief_seed1" / "belief.pt").exists()
    assert json.loads((run / "belief_seeds.json").read_text())["selected_seed"] == 1
    assert np.isfinite(belief["validation_nll"]) and "calibration" in belief
    units_json = json.loads((data / "units.json").read_text())
    assert units_json["use_degrees"] and len(units_json["center"]) == 6
