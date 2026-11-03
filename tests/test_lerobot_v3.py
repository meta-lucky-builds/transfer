"""The LeRobot v3 reader against a synthetic dataset in the exact v3 layout with real
AV1 video (ArmNetBench uses AV1 at 20 fps), and prepare() end to end with a stand-in
encoder."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

av = pytest.importorskip("av")

from robojepa_ttt.data.lerobot_v3 import LeRobotV3Source, select_blocks
from robojepa_ttt.robot_io.so101_units import JOINT_ORDER, So101Units

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FPS = 20


@pytest.fixture(autouse=True)
def attention_layout():
    yield None


def encoder_name():
    for name in ("libsvtav1", "libaom-av1", "mpeg4"):
        if name in av.codecs_available:
            return name
    pytest.skip("no usable video encoder")


def brightness(episode, frame):
    """Distinct, recoverable grey level per (episode, frame)."""
    return (37 * episode + 3 * frame) % 200 + 20


def write_video(path, episodes_frames, height=64, width=96):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream(encoder_name(), rate=FPS)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        pts = 0
        for episode, count in episodes_frames:
            for k in range(count):
                image = np.full((height, width, 3), brightness(episode, k), dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts = pts
                pts += 1
                for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def build_dataset(root, lengths=(30, 25, 40, 22, 35, 28), per_file=3, tasks=("pick", "place")):
    names = [f"{j}.pos" for j in JOINT_ORDER]
    info = {
        "codebase_version": "v3.0", "robot_type": "so101_follower", "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6], "names": names},
            "action": {"dtype": "float32", "shape": [6], "names": names},
            "observation.images.front": {"dtype": "video", "shape": [64, 96, 3]},
            "observation.images.top": {"dtype": "video", "shape": [64, 96, 3]},
            "observation.images.wrist": {"dtype": "video", "shape": [64, 96, 3]},
        },
    }
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    pd.DataFrame({"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task")).to_parquet(root / "meta" / "tasks.parquet")
    episodes, rows, offset = [], {}, 0
    for e, length in enumerate(lengths):
        file_index = e // per_file
        start_in_file = sum(lengths[file_index * per_file : e]) / FPS
        task = e % len(tasks)
        record = {"episode_index": e, "tasks": [tasks[task]], "length": length,
                  "data/chunk_index": 0, "data/file_index": file_index,
                  "dataset_from_index": offset, "dataset_to_index": offset + length}
        for camera in ("front", "top", "wrist"):
            key = f"observation.images.{camera}"
            record[f"videos/{key}/chunk_index"] = 0
            record[f"videos/{key}/file_index"] = file_index
            record[f"videos/{key}/from_timestamp"] = start_in_file
            record[f"videos/{key}/to_timestamp"] = start_in_file + length / FPS
        episodes.append(record)
        state = np.cumsum(np.full((length, 6), 0.5), 0) + 10 * e  # degrees
        rows.setdefault(file_index, []).append(pd.DataFrame({
            "observation.state": list(state.astype(np.float32)),
            "action": list((state + 1).astype(np.float32)),
            "timestamp": np.arange(length, dtype=np.float32) / FPS,
            "frame_index": np.arange(length), "episode_index": e, "task_index": task,
            "index": np.arange(offset, offset + length),
        }))
        offset += length
    pd.DataFrame(episodes).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    for file_index, parts in rows.items():
        path = root / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(parts[::-1], ignore_index=True).to_parquet(path)  # shuffled order inside the file
        members = [(e, lengths[e]) for e in range(len(lengths)) if e // per_file == file_index]
        for camera in ("front", "top", "wrist"):
            write_video(root / "videos" / f"observation.images.{camera}" / "chunk-000" / f"file-{file_index:03d}.mp4", members)
    return lengths


def test_rows_and_frames_come_from_the_right_episode_and_time(tmp_path):
    lengths = build_dataset(tmp_path)
    source = LeRobotV3Source("user/synthetic", root=tmp_path)
    assert source.fps == 20 and set(source.camera_keys()) == {"front", "top", "wrist"}
    for e in (0, 4, 5):  # first in a file, middle of a later file, last episode
        rows = source.rows(e)
        assert len(rows.timestamp) == lengths[e] and rows.task_index == e % 2
        assert rows.state[0, 0] == pytest.approx(0.5 + 10 * e)
        keyframes = np.arange(0, lengths[e], 4)
        frames = source.frames(e, "front", rows.timestamp[keyframes])
        assert frames.shape == (len(keyframes), 64, 96, 3) and frames.dtype == np.uint8
        observed = frames.reshape(len(keyframes), -1).mean(1)
        expected = np.array([brightness(e, k) for k in keyframes])
        assert np.abs(observed - expected).max() < 6, (e, observed[:5], expected[:5])


def test_bad_timestamps_and_versions_are_refused(tmp_path):
    build_dataset(tmp_path)
    source = LeRobotV3Source("user/synthetic", root=tmp_path)
    with pytest.raises(ValueError, match="no frame"):
        source.frames(0, "front", [100.0])
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    info["codebase_version"] = "v2.1"
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="v3"):
        LeRobotV3Source("user/synthetic", root=tmp_path)


def test_block_selection_is_balanced_contiguous_and_deterministic(tmp_path):
    build_dataset(tmp_path, lengths=tuple([20] * 40), per_file=10)
    source = LeRobotV3Source("user/synthetic", root=tmp_path)
    chosen = select_blocks(source, episodes_per_task=6, block=3, seed=1)
    assert chosen == select_blocks(source, episodes_per_task=6, block=3, seed=1)
    tasks = [source.episode_task(e) for e in chosen]
    assert tasks.count(0) == 6 and tasks.count(1) == 6


def test_prepare_writes_episodes_features_and_meta(tmp_path):
    from prepare_lerobot import prepare, shard_of
    from robojepa_ttt.data.dataset_meta import timing_for
    from robojepa_ttt.data.feature_store import DiskFeatures
    from robojepa_ttt.data.offline_dataset import load_episode

    data = tmp_path / "dataset"
    lengths = build_dataset(data)
    source = LeRobotV3Source("user/synthetic", root=data)
    units = So101Units(gripper_device_closed=0.0, gripper_device_open=100.0, gripper_direction_verified=True)
    calls = []

    def encode(frames):
        calls.append(frames.shape)
        return torch.from_numpy(frames[..., :1].astype(np.float32)).mean((1, 2), keepdim=True).expand(-1, 3, 4, 8).contiguous()

    out = tmp_path / "prepared"
    assignment = shard_of(source, list(range(6)), 2)
    assert set(assignment.values()) == {0, 1}
    manifest = prepare(source, [0, 4], {"front": 6, "top": 5}, units, encode, "enc", out, log=lambda m: None)
    assert [m["source_index"] for m in manifest] == [0, 4]
    episodes = sorted((out / "episodes").glob("*.npz"))
    assert len(episodes) == 2
    episode, views = load_episode(episodes[0])
    assert views in ([6, 5],)
    stride = timing_for(20).stride
    features = DiskFeatures(out / "features", stride)
    e = episode.episode_id
    keyframes = len(range(0, episode.length, stride))
    tensor = torch.load(features.path(e))
    assert tensor.shape[:2] == (keyframes, 2)
    assert calls[0][0] == len(range(0, lengths[0], stride)), "only keyframes are decoded and encoded"
    # Physical radians from degrees: first state 0.5 deg (+10 deg per episode index).
    index = int(e.split("/")[-1])
    assert episode.position[0, 0] == pytest.approx(np.radians(0.5 + 10 * index))
