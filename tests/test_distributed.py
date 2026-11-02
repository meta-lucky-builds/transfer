"""Data-parallel offline training gives the single-process update (gloo, CPU)."""

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from conftest import make_batch, make_model
from robojepa_ttt.learning.distributed import DistContext
from robojepa_ttt.learning.offline_train import OfflineConfig, OfflineTrainer

CONFIG = OfflineConfig(
    sequence_frames=6,
    effective_batch=4,
    microbatch=1,
    max_updates=10,
    warmup_updates=1,
    curriculum=((10, 2),),
    lora_lr=1e-3,
    residual_lr=1e-3,
)
BATCHES = [make_batch(batch=1, frames=6, seed=10 + i) for i in range(4)]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _train(trainer: OfflineTrainer, batches: list, steps: int = 2) -> dict:
    trainer._context = lambda frames, horizon: 2  # same contexts on every layout
    cursor = {"i": 0}

    def next_batch():
        batch = batches[cursor["i"] % len(batches)]
        cursor["i"] += 1
        return batch

    records, adapters, grads = [], [], []
    for _ in range(steps):
        records.append(trainer.train_step(next_batch))
        adapters.append(trainer.model.adapter_state())
        # train_step leaves the reduced, clipped gradients in place until the next step.
        grads.append({n: p.grad.detach().clone() for n, p in trainer.model.trainable_named().items()})
    return {"adapters": adapters, "grads": grads, "records": records}


def _worker(rank: int, world: int, port: int, out: str) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        # Same frozen base on every rank, as when all ranks load one checkpoint. Rank 1's
        # adapters are scrambled: the trainer's broadcast must realign them.
        model = make_model(seed=0, num_heads=4, num_kv_heads=4)
        if rank:
            model.set_trainable("offline")
            with torch.no_grad():
                for parameter in model.trainable_named().values():
                    parameter.add_(torch.randn_like(parameter))
        trainer = OfflineTrainer(model, CONFIG, out, DistContext(rank, world, rank))
        share = len(BATCHES) // world
        result = _train(trainer, BATCHES[rank * share : (rank + 1) * share])
        torch.save(result, os.path.join(out, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def attention_layout():
    """Override the conftest fixture: one layout is enough to test the reduction."""
    yield "fused-300m"


def test_two_ranks_match_one_process(tmp_path):
    single = _train(OfflineTrainer(make_model(seed=0, num_heads=4, num_kv_heads=4), CONFIG, tmp_path / "single"), BATCHES)

    mp.spawn(_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    ranks = [torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in range(2)]

    # Ranks must stay bit-identical, or they would drift apart over training.
    for step in range(2):
        for name, value in ranks[0]["adapters"][step].items():
            assert torch.equal(value, ranks[1]["adapters"][step][name]), f"ranks diverged at {name}"
    # After step 1 the parameters equal the single-process ones.
    for name, value in single["adapters"][0].items():
        assert torch.allclose(ranks[0]["adapters"][0][name], value, atol=1e-6), f"step 1 differs at {name}"
    # Gradients equal the single-process ones at every step. Parameters after later
    # steps are not compared: Adam turns summation-order noise in near-zero gradients
    # into steps of size lr, which is numerics, not a reduction error.
    for step in range(2):
        for name, value in single["grads"][step].items():
            got = ranks[0]["grads"][step][name]
            assert torch.allclose(got, value, rtol=1e-4, atol=1e-7 * (1 + value.abs().max())), f"step {step + 1} grad {name}"
    for a, b in zip(single["records"], ranks[0]["records"]):
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-5)
        assert a["grad_norm"] == pytest.approx(b["grad_norm"], rel=1e-4)


def test_batch_must_split_evenly_over_ranks(tmp_path):
    with pytest.raises(ValueError, match="evenly"):
        OfflineTrainer(make_model(num_heads=4, num_kv_heads=4), CONFIG, tmp_path, DistContext(0, 3, 0))
