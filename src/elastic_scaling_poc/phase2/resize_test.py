"""Stub job for Phase 2 resize tests. Runs long enough to attempt a patch."""


def train_func(**parameters):
    import os
    import time

    import torch
    import torch.distributed as dist

    p = {
        "duration_minutes": 10,
        **parameters,
    }

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"world_size={world_size}", flush=True)
        print(f"PET_NNODES={os.environ.get('PET_NNODES', 'unset')}", flush=True)
        if torch.cuda.is_available():
            print(f"gpu={torch.cuda.get_device_name(local_rank)}", flush=True)

    # Light GPU work so the job looks real to the scheduler
    x = torch.randn(1024, 1024, device=device)
    duration_s = p["duration_minutes"] * 60
    for i in range(duration_s):
        x = x @ x.T
        x = x / x.norm()
        if world_size > 1:
            dist.all_reduce(x)
        time.sleep(1)
        if i % 60 == 0 and rank == 0:
            print(f"minute {i // 60}/{p['duration_minutes']} world_size={world_size}", flush=True)

    if rank == 0:
        print("done", flush=True)

    if world_size > 1:
        dist.destroy_process_group()
