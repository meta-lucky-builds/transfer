"""Dataset timing, the camera contract, episode selection and splitting."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from robojepa_ttt.data.dataset_meta import DatasetMeta, common_timing, load_meta, parse_cameras, timing_for
from robojepa_ttt.data.offline_dataset import episode_filename, save_episode
from robojepa_ttt.learning.windows import Episode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(autouse=True)
def attention_layout():
    yield None


def test_timing_follows_the_frame_rate():
    assert (timing_for(30).stride, timing_for(30).history_ticks) == (6, 30)
    assert (timing_for(20).stride, timing_for(20).history_ticks) == (4, 20)
    assert timing_for(25).stride == 5 and timing_for(20).macro_dt_s == pytest.approx(0.2)
    with pytest.raises(ValueError, match="whole number"):
        timing_for(7)


def test_wrist_view_is_refused_for_a_phone_rig():
    assert parse_cameras(["front=6", "top=5"]) == {"front": 6, "top": 5}
    with pytest.raises(ValueError, match="wrist"):
        parse_cameras(["front=6", "wrist=2"])
    assert parse_cameras(["front=6", "wrist=2"], allow_wrist=True)["wrist"] == 2
    with pytest.raises(ValueError, match="unique"):
        parse_cameras(["a=6", "b=6"])


def meta(fps=20.0):
    return DatasetMeta("user/data", fps, {"front": 6, "top": 5}, True, 0.0, 100.0, "enc")


def test_meta_round_trip_and_mixed_rates_are_refused(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    meta(20).save(a)
    meta(30).save(b)
    (a / "episodes").mkdir()
    assert load_meta(a / "episodes").view_ids == [6, 5], "found from the episodes folder"
    assert common_timing(a, a / "episodes").stride == 4
    with pytest.raises(ValueError, match="disagree"):
        common_timing(a, b)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    assert common_timing(legacy, default=timing_for(30)).stride == 6
    with pytest.raises(ValueError, match="no dataset.json"):
        common_timing(legacy)


def test_episode_selection_is_balanced_deterministic_and_shard_disjoint():
    from prepare_lerobot import select_episodes

    # 3 tasks with 60 / 30 / 10 episodes, 20 frames each.
    sizes = [60, 30, 10]
    episode_index, task_index, e = [], [], 0
    for task, count in enumerate(sizes):
        for _ in range(count):
            episode_index += [e] * 20
            task_index += [task] * 20
            e += 1
    episode_index, task_index = np.array(episode_index), np.array(task_index)
    chosen = select_episodes(episode_index, task_index, 30, seed=1)
    assert chosen == select_episodes(episode_index, task_index, 30, seed=1)
    per_task = np.bincount([task_index[np.flatnonzero(episode_index == c)[0]] for c in chosen], minlength=3)
    assert per_task.tolist() == [10, 10, 10]
    shards = [chosen[i::4] for i in range(4)]
    assert sorted(sum(shards, [])) == chosen and len(set(sum(shards, []))) == 30
    assert len(select_episodes(episode_index, task_index, 95, seed=1)) == 95  # small task exhausted, others fill
    assert select_episodes(episode_index, None, None, seed=1) == list(range(100))


def test_episode_files_are_named_by_id_and_rewrites_replace(tmp_path):
    episode = Episode("user/data/7", "cal", "enc", 2, capacity=4)
    for n in range(10):
        episode.append(n / 20, np.zeros(6), True, np.zeros(6), True, np.ones(2, bool))
    first = save_episode(episode, tmp_path, [6, 5])
    again = save_episode(episode, tmp_path, [6, 5])
    assert first == again and first.name == episode_filename("user/data/7")
    assert [p.name for p in tmp_path.iterdir()] == [first.name], "no temporary or duplicate files remain"


def test_split_script_groups_neighbours_and_links_files(tmp_path):
    root = tmp_path / "prepared"
    meta(20).save(root)
    manifest = []
    for index in range(60):
        episode = Episode(f"user/data/{index}", "cal", "enc", 2, capacity=4)
        for n in range(8):
            episode.append(n / 20, np.zeros(6), True, np.zeros(6), True, np.ones(2, bool))
        save_episode(episode, root / "episodes", [6, 5])
        manifest.append({"episode_id": episode.episode_id, "source_index": index, "task_index": index % 3, "ticks": 8})
    (root / "manifest-0-of-2.json").write_text(json.dumps(manifest[::2]))
    (root / "manifest-1-of-2.json").write_text(json.dumps(manifest[1::2]))
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "split_dataset.py"), "--root", str(root), "--group-size", "10"],
                            capture_output=True, text=True, check=True)
    assert "train" in result.stdout
    seen = {}
    for name in ("train", "validation", "calibration", "test"):
        folder = root / "splits" / name
        assert load_meta(folder / "episodes").fps == 20.0
        for link in (folder / "episodes").iterdir():
            assert link.is_symlink() and link.resolve().exists()
            seen.setdefault(link.name, []).append(name)
    assert len(seen) == 60 and all(len(v) == 1 for v in seen.values()), "every episode in exactly one split"
    # Neighbours (one pseudo-session of 10) share a split.
    by_index = {episode_filename(f"user/data/{i}"): i for i in range(60)}
    for start in range(0, 60, 10):
        splits = {seen[name][0] for name, i in by_index.items() if start <= i < start + 10}
        assert len(splits) == 1


def test_belief_trains_at_twenty_frames_per_second():
    """Stride 4 and 20 history ticks end to end through sequences and the filter."""
    from robojepa_ttt.belief.compact import CompactBelief, CompactConfig
    from robojepa_ttt.belief.data import BeliefSequences, SequenceSpec
    from robojepa_ttt.belief.sketch import ObservationSketch
    from robojepa_ttt.belief.train import CompactTrainConfig, sequence_loss
    from robojepa_ttt.evaluation.perturbation import SimulatedArm, SyntheticCamera
    from robojepa_ttt.models.servo import ServoParams
    from robojepa_ttt.robot_io.so101_units import So101Units

    timing = timing_for(20)
    units = So101Units(gripper_direction_verified=True)
    camera = SyntheticCamera(views=2, height=3, width=4, channels=16)
    arm = SimulatedArm(tau_s=0.1, tick_s=timing.tick_s, seed=0)
    episode = Episode("e20", "cal", "enc", 2, capacity=16)
    frames, target, rng = {}, np.zeros(6), np.random.default_rng(0)
    for n in range(200):
        if n % 10 == 0:
            target = np.clip(target + rng.normal(0, 0.15, 6), -0.8, 0.8)
        measured = arm.step(target)
        frames[n] = camera.features(arm.position)
        episode.append(n * timing.tick_s, measured, True, target.copy(), True, np.ones(2, bool))
    sketch = ObservationSketch((6, 5), token_dim=16, components=4)
    sketch.fit({v: torch.stack([frames[k][s] for k in range(0, 200, timing.stride)]) for s, v in enumerate((6, 5))})
    spec = SequenceSpec(scored=6, burn_in=3, stride=timing.stride, history_ticks=timing.history_ticks, tick_s=timing.tick_s)
    data = BeliefSequences([episode], {"e20": [6, 5]}, lambda e, k: frames[k], sketch, units, spec)
    assert len(data) > 0
    batch = data.batch(data.index[:2])
    assert batch["issued"].shape[2] == timing.history_ticks + timing.stride
    torch.manual_seed(0)
    config = CompactConfig(joint_sensor_std=(0.003,) * 6, residual_bound=(0.3,) * 6, view_ids=(6, 5), visual_per_view=4,
                           fast_dims=6, slow_dims=2, recurrent_hidden=16, command_hidden=8, regime_dim=4,
                           transition_hidden=(16,), observation_hidden=(16,))
    model = CompactBelief(config, units, ServoParams((0.1,) * 6, (1,) * 6), timing.stride, timing.tick_s, timing.history_ticks)
    loss, parts = sequence_loss(model, batch, torch.full((6,), 0.01), CompactTrainConfig())
    loss.backward()
    assert torch.isfinite(loss) and parts["scored_steps"] > 0
