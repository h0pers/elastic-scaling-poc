"""LoRA fine-tuning of Llama-3.1-8B on Alpaca with per-step throughput recording.

Global batch size is held constant across GPU counts by scaling gradient
accumulation inversely with world size. Sequence packing keeps tokens-per-step
exact. The first WARMUP_STEPS_EXCLUDED steps are flagged so analysis can
exclude CUDA warm-up.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

MODEL_ID = "meta-llama/Llama-3.1-8B"
DATASET_ID = "tatsu-lab/alpaca"
OUTPUT_DIR = "/mnt/output"

MAX_STEPS = 200
SAVE_STEPS = 100
SEQ_LENGTH = 1024
SEED = 42

GLOBAL_BATCH_SIZE = 128
PER_DEVICE_BATCH_SIZE = 4

LORA_R = 16
LORA_ALPHA = 32

WARMUP_STEPS_EXCLUDED = 20


def world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def grad_accum_steps(ws: int) -> int:
    """Accumulation that keeps global batch constant at this world size."""
    denom = PER_DEVICE_BATCH_SIZE * ws
    if GLOBAL_BATCH_SIZE % denom != 0:
        raise ValueError(
            f"GLOBAL_BATCH_SIZE={GLOBAL_BATCH_SIZE} is not divisible by "
            f"PER_DEVICE_BATCH_SIZE={PER_DEVICE_BATCH_SIZE} x world_size={ws}. "
            "Pick a global batch that divides cleanly at every GPU count in the sweep."
        )
    return GLOBAL_BATCH_SIZE // denom


class GpuSampler:
    """Background thread that samples GPU utilization and memory via NVML."""

    def __init__(self, device_index: int, interval_s: float = 0.5):
        self.device_index = device_index
        self.interval_s = interval_s
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._nvml = None

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:  # nvml missing or no permission - degrade, don't fail the run
            print(f"[gpu-sampler] disabled: {exc}", flush=True)
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._samples.append(util.gpu)
            except Exception:
                break

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def snapshot(self) -> dict[str, Any]:
        samples = list(self._samples)
        out: dict[str, Any] = {
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
    """Records per-step wall time and derived throughput to a JSONL file.

    Only rank 0 writes. Tokens per step is computed from the fixed packing
    geometry rather than measured, so the number is exact across runs.
    """

    def __init__(self, run_name: str, tokens_per_step: int, metrics_path: Path, sampler: GpuSampler):
        self.run_name = run_name
        self.tokens_per_step = tokens_per_step
        self.metrics_path = metrics_path
        self.sampler = sampler
        self.step_start: float | None = None
        self.train_start: float | None = None
        self.records: list[dict[str, Any]] = []

    def _write(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        with self.metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self.metrics_path.write_text("")
        self.train_start = time.perf_counter()

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start is None or global_rank() != 0:
            return
        elapsed = time.perf_counter() - self.step_start
        step = state.global_step
        record = {
            "type": "step",
            "run_name": self.run_name,
            "world_size": world_size(),
            "step": step,
            "step_time_s": round(elapsed, 4),
            "tokens_per_step": self.tokens_per_step,
            "tokens_per_s": round(self.tokens_per_step / elapsed, 1) if elapsed > 0 else None,
            "warmup": step <= WARMUP_STEPS_EXCLUDED,
            "wall_clock_s": round(time.perf_counter() - self.train_start, 3),
        }
        record.update(self.sampler.snapshot())
        self._write(record)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or global_rank() != 0 or "loss" not in logs:
            return
        self._write(
            {
                "type": "loss",
                "run_name": self.run_name,
                "world_size": world_size(),
                "step": state.global_step,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
                "epoch": logs.get("epoch"),
            }
        )


def build_dataset() -> Dataset:
    """Alpaca formatted as a single text field for packed SFT."""
    ds = load_dataset(DATASET_ID, split="train")

    def to_text(row: dict[str, Any]) -> dict[str, str]:
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


def main() -> None:
    ws = world_size()
    rank = global_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    accum = grad_accum_steps(ws)
    tokens_per_step = PER_DEVICE_BATCH_SIZE * SEQ_LENGTH * accum * ws
    run_name = f"phase1-baseline-{ws}gpu"

    out_dir = Path(OUTPUT_DIR) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics-rank{rank}.jsonl"

    if rank == 0:
        print(
            f"[phase1] run={run_name} world_size={ws} "
            f"per_device_batch={PER_DEVICE_BATCH_SIZE} grad_accum={accum} "
            f"global_batch={PER_DEVICE_BATCH_SIZE * accum * ws} "
            f"seq_len={SEQ_LENGTH} tokens/step={tokens_per_step}",
            flush=True,
        )

    sampler = GpuSampler(device_index=local_rank)
    sampler.start()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    assert tokenizer is not None
    if tokenizer.pad_token is None:
        # Llama 3.1 includes a dedicated pad token for fine-tuning
        # Using eos_token as pad causes the model to ignore EOS during generation.
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    dataset = build_dataset()

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    sft_config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=accum,
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_steps=1,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        bf16=True,
        max_length=SEQ_LENGTH,
        packing=True,
        seed=SEED,
        data_seed=SEED,
        report_to=[],
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        model_init_kwargs={"dtype": torch.bfloat16},
    )

    trainer = SFTTrainer(
        model=MODEL_ID,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    callback = ThroughputCallback(run_name, tokens_per_step, metrics_path, sampler)
    trainer.add_callback(callback)

    start = time.perf_counter()
    trainer.train()
    total_s = time.perf_counter() - start

    sampler.stop()

    if rank == 0:
        steps = [r for r in callback.records if r["type"] == "step" and not r["warmup"]]
        times = [r["step_time_s"] for r in steps]
        summary = {
            "type": "summary",
            "run_name": run_name,
            "world_size": ws,
            "model_id": MODEL_ID,
            "dataset_id": DATASET_ID,
            "max_steps": MAX_STEPS,
            "seq_length": SEQ_LENGTH,
            "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
            "grad_accum_steps": accum,
            "global_batch_size": PER_DEVICE_BATCH_SIZE * accum * ws,
            "tokens_per_step": tokens_per_step,
            "warmup_steps_excluded": WARMUP_STEPS_EXCLUDED,
            "steady_state_steps": len(times),
            "total_train_time_s": round(total_s, 2),
            "mean_step_time_s": round(sum(times) / len(times), 4) if times else None,
            "median_step_time_s": round(sorted(times)[len(times) // 2], 4) if times else None,
            "mean_tokens_per_s": round(tokens_per_step / (sum(times) / len(times)), 1) if times else None,
        }
        summary.update(sampler.snapshot())
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(summary) + "\n")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[phase1] summary: {json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
