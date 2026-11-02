"""Data-parallel helpers for offline adapter training.

Every rank holds the full frozen predictor and an identical copy of the adapters.
Ranks draw different data, accumulate gradients locally, then sum them once per
optimizer step. Only adapter gradients cross the network, which is a few MB even at
2B, so plain all-reduce is enough and no DistributedDataParallel wrapper is needed.
That also keeps the model's rollout() entry point, which DDP's forward hooks would
not see.

Launch with torchrun, which sets RANK, LOCAL_RANK and WORLD_SIZE:

    torchrun --nproc_per_node 8 scripts/train_offline.py ...

Without those variables everything runs as a single process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


SINGLE = DistContext()


def init_distributed(backend: str | None = None) -> DistContext:
    """Join the process group torchrun describes, or return a single-process context."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return SINGLE
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistContext(rank, world_size, local_rank)


def _device_of(tensors: list[torch.Tensor]) -> torch.device:
    return tensors[0].device if tensors else torch.device("cpu")


def all_reduce_gradients(parameters: Iterable[torch.nn.Parameter], context: DistContext) -> None:
    """Sum gradients over ranks in one flat buffer.

    The caller has already divided each rank's loss by the global number of
    microbatches, so the sum equals the single-process gradient. A missing
    gradient counts as zero, so ranks never disagree on the buffer layout.
    """
    if not context.enabled:
        return
    parameters = list(parameters)
    for p in parameters:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    flat = torch.cat([p.grad.reshape(-1).float() for p in parameters])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for p in parameters:
        count = p.numel()
        p.grad.copy_(flat[offset : offset + count].view_as(p.grad))
        offset += count


def broadcast_parameters(parameters: Iterable[torch.nn.Parameter], context: DistContext) -> None:
    """Make every rank start from rank 0's values, e.g. its random LoRA A matrices."""
    if not context.enabled:
        return
    with torch.no_grad():
        for p in parameters:
            dist.broadcast(p.data, src=0)


def all_reduce_sum(values: dict[str, float], context: DistContext, device) -> dict[str, float]:
    """Sum scalar logs over ranks. Keys must match on every rank."""
    if not context.enabled or not values:
        return values
    keys = sorted(values)
    buffer = torch.tensor([values[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    return dict(zip(keys, buffer.tolist()))


def broadcast_object(value: Any, context: DistContext) -> Any:
    """Send a picklable value from rank 0 to every rank."""
    if not context.enabled:
        return value
    holder = [value if context.is_main else None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def barrier(context: DistContext) -> None:
    if context.enabled:
        dist.barrier()


def cleanup(context: DistContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()
