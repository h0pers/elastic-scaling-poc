"""Charts for the docs, rendered from results/ at build time.

conf.py calls render_all() on builder-inited, writing SVGs to docs/_generated/.
Standard library and matplotlib only: Read the Docs installs the docs group alone.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"

BLUE = "#2b6cb0"
RUST = "#c05621"
GREY = "#8a8a8a"
PALE = "#d9d9d9"


def _style():
    plt.rcParams.update({
        "figure.figsize": (7.0, 3.8),
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e9e9e9",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",  # keep text selectable in the SVG
    })


# Loaders


def load_phase1():
    """Phase 1 scaling sweep, one row per GPU count, ascending."""
    numeric = {
        "world_size": int,
        "mean_step_time_s": float,
        "mean_tokens_per_s": float,
        "gpu_util_mean_pct": float,
        "total_train_time_s": float,
        "speedup": float,
        "efficiency_pct": float,
    }
    with (RESULTS / "phase1" / "scaling.csv").open() as handle:
        rows = [
            {key: cast(row[key]) for key, cast in numeric.items()}
            for row in csv.DictReader(handle)
        ]
    return sorted(rows, key=lambda row: row["world_size"])


def load_phase5():
    """Phase 5 FSDP scaling sweep, one row per GPU count, ascending."""
    numeric = {
        "world_size": int,
        "mean_step_time_s": float,
        "mean_tokens_per_s": float,
        "gpu_util_mean_pct": float,
        "total_train_time_s": float,
        "speedup": float,
        "efficiency_pct": float,
    }
    with (RESULTS / "phase5" / "scaling.csv").open() as handle:
        rows = [
            {key: cast(row[key]) for key, cast in numeric.items()}
            for row in csv.DictReader(handle)
        ]
    return sorted(rows, key=lambda row: row["world_size"])


def load_phase3():
    """Phase 3 restart record: timeline plus per-step losses."""
    return json.loads((RESULTS / "phase3" / "startup_timing.json").read_text())


def phase3_breakdown():
    """Restart stages as (label, seconds), in timeline order.

    rendezvous_complete is excluded: phase3/train.py logs it immediately after
    model_loaded, so it always reads 0.000s. See the limits page.
    """
    marks = {e["label"]: e["elapsed_s"] for e in load_phase3()["timeline"]}
    stages = [
        ("Python + libraries", 0.0, marks["libraries_imported"]),
        ("GPU init", marks["libraries_imported"], marks["gpu_initialized"]),
        ("Dataset load", marks["gpu_initialized"], marks["dataset_loaded"]),
        ("Base model load", marks["dataset_loaded"], marks["model_loaded"]),
        ("Checkpoint restore", marks["model_loaded"], marks["checkpoint_restored"]),
        ("First training step", marks["checkpoint_restored"], marks["first_step_complete"]),
    ]
    return [(label, end - start) for label, start, end in stages]


def load_events(version):
    """Phase 4 event log for "a" or "b", as {label: unix timestamp}."""
    path = RESULTS / "phase4" / f"version_{version}" / "events.json"
    return {e["label"]: e["time"] for e in json.loads(path.read_text())["events"]}


def phase4_segments(version):
    """GPU allocation as (role, gpus, start_offset_s, duration_s).

    Read from event timestamps, so this reflects what actually ran.
    """
    events = load_events(version)
    origin = events["low_priority_submitted"]
    if version == "a":
        spans = [
            ("low", 8, events["low_priority_submitted"], events["low_priority_killed"]),
            ("high", 4, events["high_priority_submitted"], events["high_priority_complete"]),
            ("low", 8, events["low_priority_restarted"], events["low_priority_complete"]),
        ]
    else:
        # The shrunk job ends at the scale-back, or at completion if it never scaled back.
        shrunk_end = events.get("low_priority_scale_back_started",
                                events["low_priority_complete"])
        spans = [
            ("low", 8, events["low_priority_submitted"], events["low_priority_shrink_started"]),
            ("low", 4, events["low_priority_shrunk_submitted"], shrunk_end),
            ("high", 4, events["high_priority_submitted"], events["high_priority_complete"]),
        ]
        if "low_priority_scaled_back_submitted" in events:
            spans.append(("low", 8, events["low_priority_scaled_back_submitted"],
                          events["low_priority_complete"]))
    return [(role, gpus, start - origin, end - start) for role, gpus, start, end in spans]


def _last_run(relative_path, record_type):
    """Records of one type from the final run in a metrics file.

    A metrics file can hold more than one run appended together. A step number
    that drops marks the start of a later run, so everything before the last drop
    is discarded. A rising gap, such as 101 to 201, is a resume within the same
    arm and is kept.
    """
    records = [json.loads(line) for line in
               (RESULTS / relative_path).read_text().strip().splitlines() if line]
    records = [r for r in records if r.get("type") == record_type]
    start = 0
    for i in range(1, len(records)):
        if records[i]["step"] <= records[i - 1]["step"]:
            start = i
    return records[start:]


def load_steps(relative_path):
    """Steady-state step times in seconds, warm-up dropped."""
    return [r["step_time_s"] for r in _last_run(relative_path, "step")
            if not r.get("warmup")]


def load_losses(relative_path):
    """Per-step training loss as {step: loss}."""
    return {r["step"]: r["loss"] for r in _last_run(relative_path, "loss")}


# Figures


def _speedup_figure(rows, color, title):
    """Speedup against GPU count on log-log axes, with the ideal for reference.

    Both axes share log base 2, so linear scaling is a straight line of slope 1
    and any departure from it reads as curvature rather than a widening gap.
    """
    gpus = [r["world_size"] for r in rows]
    speedup = [r["speedup"] for r in rows]

    fig, ax = plt.subplots()
    ax.plot(gpus, gpus, linestyle="--", color=GREY, linewidth=1.2, label="Ideal (linear)")
    ax.plot(gpus, speedup, marker="o", color=color, linewidth=2, label="Measured")
    for x, y in zip(gpus, speedup):
        ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                    xytext=(8, -10), fontsize=9, color=color)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(gpus)
    ax.set_xticklabels(gpus)
    ax.set_yticks(gpus)
    ax.set_yticklabels(gpus)
    ax.minorticks_off()
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Speedup versus 1 GPU")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_scaling():
    return _speedup_figure(load_phase1(), BLUE,
                           "LoRA fine-tuning scales almost linearly to 8 GPUs")


def fig_fsdp_scaling():
    return _speedup_figure(load_phase5(), RUST,
                           "FSDP gives up some scaling to shard the model")


def fig_restart_breakdown():
    stages = phase3_breakdown()
    colors = ["#efefef", "#e0e0e0", "#cfcfcf", BLUE, RUST, GREY]

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    left = 0.0
    for (label, value), color in zip(stages, colors):
        ax.barh(0, value, left=left, color=color, edgecolor="white", linewidth=1.0,
                label=f"{label}, {value:.1f}s")
        if value > 25:
            ax.text(left + value / 2, 0, f"{label}\n{value:.1f}s", ha="center", va="center",
                    fontsize=9, color="white")
        left += value

    ax.set_xlim(0, left)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Seconds from process start")
    ax.set_title(f"Restart to first training step: {left:.1f}s, mostly the base model")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3, fontsize=8.5)
    fig.tight_layout()
    return fig


def fig_break_even():
    steps = {r["world_size"]: r["mean_step_time_s"] for r in load_phase1()}
    step_4, step_8 = steps[4], steps[8]
    restart = 95.442
    remaining = np.arange(0, 61)
    crossover = restart / (step_4 - step_8)

    fig, ax = plt.subplots()
    ax.plot(remaining, remaining * step_4 / 60, color=RUST, linewidth=2,
            label="Stay on 4 GPUs")
    ax.plot(remaining, (restart + remaining * step_8) / 60, color=BLUE, linewidth=2,
            label="Restart onto 8 GPUs")
    ax.axvline(crossover, color=GREY, linestyle="--", linewidth=1.2)
    ax.annotate(f"break-even\n{crossover:.1f} steps", (crossover, 6),
                textcoords="offset points", xytext=(10, 0), fontsize=9, color="#333333")
    ax.set_xlabel("Training steps remaining")
    ax.set_ylabel("Time to finish (minutes)")
    ax.set_title("Growing pays back after about 13 steps of remaining work")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_contention():
    """Node sharing versus running alone, both at 4 GPUs.

    The view excludes the epoch-boundary steps near 11.3s, which recur every 42
    steps in every run including the control, and so cancel in the comparison.
    """
    alone = load_steps("phase4/version_a/high-priority_metrics-4gpu-rank0.jsonl")
    low = load_steps("phase4/version_b/low-priority_metrics-4gpu-rank0.jsonl")
    high = load_steps("phase4/version_b/high-priority_metrics-4gpu-rank0.jsonl")

    def main_mode(values):
        centre = float(np.median(values))
        return [v for v in values if abs(v - centre) < 1.0]

    alone, low, high = main_mode(alone), main_mode(low), main_mode(high)
    control = float(np.median(alone))

    fig, ax = plt.subplots()
    bins = np.linspace(14.8, 15.8, 34)
    ax.hist(low, bins=bins, color=BLUE, alpha=0.6,
            label=f"Sharing, low-priority (n={len(low)})")
    ax.hist(high, bins=bins, color=RUST, alpha=0.6,
            label=f"Sharing, high-priority (n={len(high)})")
    ax.axvline(control, color=GREY, linestyle="--", linewidth=1.5,
               label=f"Alone on the node: {control:.3f}s median")
    ax.set_xlabel("Step time (seconds)")
    ax.set_ylabel("Steps")
    ax.set_title("Two 4-GPU jobs sharing a node match one 4-GPU job running alone")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def fig_utilisation_vs_throughput():
    rows = load_phase1()
    gpus = [r["world_size"] for r in rows]
    util = [r["gpu_util_mean_pct"] for r in rows]
    tokens = [r["mean_tokens_per_s"] for r in rows]

    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 5.0))

    top.plot(gpus, util, marker="o", color=RUST, linewidth=2)
    top.axhline(75, color=GREY, linestyle="--", linewidth=1.2)
    top.annotate("upstream 75% threshold", (1, 75), textcoords="offset points",
                 xytext=(4, 6), fontsize=9, color="#333333")
    for x, y in zip(gpus, util):
        top.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(6, -12), fontsize=9, color=RUST)
    top.set_ylim(60, 105)
    top.set_ylabel("GPU utilisation (%)")
    top.set_title("GPU utilisation is flat while throughput scales with GPU count")

    bottom.plot(gpus, tokens, marker="o", color=BLUE, linewidth=2)
    for x, y in zip(gpus, tokens):
        bottom.annotate(f"{y:,.0f}", (x, y), textcoords="offset points",
                        xytext=(6, -12), fontsize=9, color=BLUE)
    bottom.set_xscale("log", base=2)
    bottom.set_xticks(gpus)
    bottom.set_xticklabels(gpus)
    bottom.set_xlabel("GPUs")
    bottom.set_ylabel("Tokens/sec")
    fig.tight_layout()
    return fig


def fig_convergence():
    eight = load_losses("phase4/version_a/low-priority_metrics-8gpu-rank0.jsonl")
    four = load_losses("phase4/version_b/low-priority_metrics-4gpu-rank0.jsonl")
    steps = sorted(set(eight) & set(four))

    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(7.0, 4.6), gridspec_kw={"height_ratios": [3, 1]})

    top.plot(steps, [eight[s] for s in steps], color=BLUE, linewidth=1.6,
             label="Continued on 8 GPUs")
    top.plot(steps, [four[s] for s in steps], color=RUST, linewidth=1.6, linestyle="--",
             label="Continued on 4 GPUs")
    top.set_ylabel("Training loss")
    top.set_title("Halving the GPU count does not change what the model learns")
    top.legend(loc="upper right")

    bottom.plot(steps, [eight[s] - four[s] for s in steps], color=GREY, linewidth=1.2)
    bottom.axhline(0, color="#cccccc", linewidth=0.8)
    bottom.set_ylim(-0.004, 0.004)
    bottom.set_xlabel("Training step, both resumed from checkpoint 100")
    bottom.set_ylabel("Difference")
    fig.tight_layout()
    return fig


def fig_phase4_timeline():
    titles = {
        "a": "Kill: 4 GPUs sit idle while the high-priority job runs",
        "b": "Shrink: both jobs share the node, then it scales back up",
    }
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.0, 4.4))

    for ax, version in zip(axes, ("a", "b")):
        segments = phase4_segments(version)
        wall = max(start + dur for _, _, start, dur in segments)

        # Allocation is piecewise constant, so sample at segment boundaries only.
        bounds = sorted({0.0, wall}
                        | {start for _, _, start, _ in segments}
                        | {start + dur for _, _, start, dur in segments})
        times, low, high = [], [], []
        for t0, t1 in zip(bounds[:-1], bounds[1:]):
            mid = (t0 + t1) / 2
            active = [g for r, g, s, d in segments if s <= mid < s + d and r == "low"]
            waiting = [g for r, g, s, d in segments if s <= mid < s + d and r == "high"]
            times += [t0 / 60, t1 / 60]
            low += [sum(active)] * 2
            high += [sum(waiting)] * 2

        ax.stackplot(times, low, high, colors=[BLUE, RUST],
                     labels=["Low-priority", "High-priority"])
        ax.axhline(8, color=GREY, linestyle="--", linewidth=1.0)
        ax.set_ylim(0, 9)
        ax.set_yticks([0, 4, 8])
        ax.set_ylabel("GPUs in use")
        ax.set_title(titles[version], fontsize=10)

    axes[1].set_xlabel("Minutes from first submission")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


FIGURES = {
    "scaling": fig_scaling,
    "fsdp-scaling": fig_fsdp_scaling,
    "restart-breakdown": fig_restart_breakdown,
    "break-even": fig_break_even,
    "contention": fig_contention,
    "utilisation-vs-throughput": fig_utilisation_vs_throughput,
    "convergence": fig_convergence,
    "phase4-timeline": fig_phase4_timeline,
}


def render_all(out_dir):
    """Render every figure as SVG into out_dir. Returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _style()
    written = []
    for name, builder in FIGURES.items():
        figure = builder()
        path = out / f"{name}.svg"
        figure.savefig(path, format="svg", bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


if __name__ == "__main__":
    for path in render_all(Path(__file__).parent / "_generated"):
        print(f"wrote {path}")
