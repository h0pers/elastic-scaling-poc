"""LoRA fine-tuning with startup timing for measuring restart cost.

Logs a timestamp at each stage of startup so we can see where the time
goes when a job restarts from a checkpoint. Resumes from the latest
checkpoint automatically if one exists.
"""


def train_func(**parameters):
    import json
    import os
    import threading
    import time
    from pathlib import Path
    from types import SimpleNamespace

    startup_time = time.perf_counter()
    timeline = []

    def log(label):
        timeline.append({"label": label, "elapsed_s": round(time.perf_counter() - startup_time, 3)})

    log("script_started")

    import torch
    import torch.distributed as dist
    log("torch_imported")

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer
    log("libraries_imported")

    p = SimpleNamespace(**{
        "model_id": "meta-llama/Llama-3.1-8B",
        "dataset_id": "tatsu-lab/alpaca",
        "output_dir": "/mnt/kubeflow-checkpoints",
        "max_steps": 200,
        "save_steps": 100,
        "seq_length": 1024,
        "seed": 42,
        "global_batch_size": 128,
        "per_device_batch_size": 4,
        "lora_r": 16,
        "lora_alpha": 32,
        "warmup_steps_excluded": 20,
        **parameters,
    })

    hf_cache = f"{p.output_dir}/hf-cache"
    os.environ.setdefault("HF_HOME", hf_cache)

    hf_token_path = Path("/mnt/hf-token/HF_TOKEN")
    if hf_token_path.exists() and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = hf_token_path.read_text().strip()

    def world_size():
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return int(os.environ.get("WORLD_SIZE", 1))

    def global_rank():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return int(os.environ.get("RANK", 0))

    def grad_accum_steps(ws):
        denom = p.per_device_batch_size * ws
        if p.global_batch_size % denom != 0:
            raise ValueError(
                f"global_batch_size={p.global_batch_size} not divisible by "
                f"per_device_batch_size={p.per_device_batch_size} x world_size={ws}"
            )
        return p.global_batch_size // denom

    class GpuSampler:
        """Background thread that samples GPU utilization and memory via NVML."""

        def __init__(self, device_index, interval_s=0.5):
            self.device_index = device_index
            self.interval_s = interval_s
            self._samples = []
            self._stop = threading.Event()
            self._thread = None
            self._handle = None
            self._nvml = None

        def start(self):
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvml = pynvml
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            except Exception as exc:
                print(f"[gpu-sampler] disabled: {exc}", flush=True)
                return
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        def _loop(self):
            while not self._stop.wait(self.interval_s):
                try:
                    util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self._samples.append(util.gpu)
                except Exception:
                    break

        def reset(self):
            """Drop samples collected before training started.

            The sampler runs from process start, so without this the mean is
            averaged over the model load and checkpoint restore, when the GPUs
            are idle.
            """
            self._samples.clear()

        def stop(self):
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)

        def snapshot(self):
            samples = list(self._samples)
            out = {
                "gpu_util_samples": len(samples),
                "gpu_util_mean_pct": round(sum(samples) / len(samples), 2) if samples else None,
                "gpu_util_max_pct": max(samples) if samples else None,
            }
            if torch.cuda.is_available():
                out["gpu_mem_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
                out["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
                out["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            return out

    class ThroughputCallback(TrainerCallback):
        """Records per-step timing, loss, and startup milestones."""

        def __init__(self, run_name, tokens_per_step, metrics_path, sampler):
            self.run_name = run_name
            self.tokens_per_step = tokens_per_step
            self.metrics_path = metrics_path
            self.sampler = sampler
            self.step_start = None
            self.train_start = None
            self.first_step_logged = False
            self.records = []

        def _write(self, record):
            self.records.append(record)
            with self.metrics_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")

        def on_train_begin(self, args, state, control, **kwargs):
            if not resumed:
                self.metrics_path.write_text("")
            self.sampler.reset()
            self.train_start = time.perf_counter()
            if resumed:
                log("checkpoint_restored")

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_start = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            if self.step_start is None or global_rank() != 0:
                return
            elapsed = time.perf_counter() - self.step_start
            step = state.global_step
            if not self.first_step_logged:
                self.first_step_logged = True
                log("first_step_complete")
            record = {
                "type": "step",
                "run_name": self.run_name,
                "world_size": world_size(),
                "step": step,
                "step_time_s": round(elapsed, 4),
                "tokens_per_step": self.tokens_per_step,
                "tokens_per_s": round(self.tokens_per_step / elapsed, 1) if elapsed > 0 else None,
                "warmup": step <= p.warmup_steps_excluded,
                "wall_clock_s": round(time.perf_counter() - self.train_start, 3),
            }
            record.update(self.sampler.snapshot())
            self._write(record)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or global_rank() != 0 or "loss" not in logs:
                return
            self._write({
                "type": "loss",
                "run_name": self.run_name,
                "world_size": world_size(),
                "step": state.global_step,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
                "epoch": logs.get("epoch"),
            })

    def build_dataset():
        ds = load_dataset(p.dataset_id, split="train")

        def to_text(row):
            instruction = row.get("instruction", "")
            context = row.get("input", "")
            response = row.get("output", "")
            if context:
                prompt = (
                    "Below is an instruction that describes a task, paired with an input "
                    "that provides further context. Write a response that appropriately "
                    f"completes the request.\n\n### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{context}\n\n### Response:\n{response}"
                )
            else:
                prompt = (
                    "Below is an instruction that describes a task. Write a response that "
                    f"appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n"
                    f"### Response:\n{response}"
                )
            return {"text": prompt}

        return ds.map(to_text, remove_columns=ds.column_names)

    ws = world_size()
    rank = global_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    accum = grad_accum_steps(ws)
    tokens_per_step = p.per_device_batch_size * p.seq_length * accum * ws
    run_name = f"phase3-restart-{ws}gpu"

    out_dir = Path(p.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics-rank{rank}.jsonl"

    if rank == 0:
        print(
            f"[phase3] run={run_name} world_size={ws} "
            f"per_device_batch={p.per_device_batch_size} grad_accum={accum} "
            f"global_batch={p.per_device_batch_size * accum * ws} "
            f"seq_len={p.seq_length} tokens/step={tokens_per_step}",
            flush=True,
        )

    sampler = GpuSampler(device_index=local_rank)
    sampler.start()

    torch.cuda.set_device(local_rank)
    torch.cuda.init()
    log("gpu_initialized")

    tokenizer = AutoTokenizer.from_pretrained(p.model_id)
    assert tokenizer is not None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    dataset = build_dataset()
    log("dataset_loaded")

    peft_config = LoraConfig(
        r=p.lora_r,
        lora_alpha=p.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    sft_config = SFTConfig(
        output_dir=str(checkpoint_dir),
        max_steps=p.max_steps,
        per_device_train_batch_size=p.per_device_batch_size,
        gradient_accumulation_steps=accum,
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_steps=1,
        save_steps=p.save_steps,
        save_strategy="steps",
        bf16=True,
        tf32=True,
        max_length=p.seq_length,
        packing=True,
        seed=p.seed,
        data_seed=p.seed,
        report_to=[],
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        model_init_kwargs={"dtype": torch.bfloat16},
    )

    trainer = SFTTrainer(
        model=p.model_id,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    log("model_loaded")
    log("rendezvous_complete")

    callback = ThroughputCallback(run_name, tokens_per_step, metrics_path, sampler)
    trainer.add_callback(callback)

    last_checkpoint = get_last_checkpoint(str(checkpoint_dir))
    resumed = last_checkpoint is not None
    if rank == 0:
        if resumed:
            print(f"[phase3] resuming from {last_checkpoint}", flush=True)
        else:
            print("[phase3] no checkpoint found, training from scratch", flush=True)

    start = time.perf_counter()
    trainer.train(resume_from_checkpoint=last_checkpoint or False)
    total_s = time.perf_counter() - start

    log("training_complete")
    sampler.stop()

    if rank == 0:
        losses = [record for record in callback.records if record["type"] == "loss"]
        result = {
            "run_name": run_name,
            "world_size": ws,
            "model_id": p.model_id,
            "dataset_id": p.dataset_id,
            "resumed_from": last_checkpoint if resumed else None,
            "total_train_time_s": round(total_s, 2),
            "timeline": timeline,
            "losses": losses,
        }
        result.update(sampler.snapshot())
        (out_dir / "startup_timing.json").write_text(json.dumps(result, indent=2))
        print("[phase3] startup timeline:", flush=True)
        for entry in timeline:
            print(f"  {entry['elapsed_s']:7.3f}s  {entry['label']}", flush=True)
