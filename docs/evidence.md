# The measurements

Every number quoted elsewhere on this site, with its source file and sample size.

## Phase 1: static scaling sweep

Source: `results/phase1/scaling.csv`. One run per configuration, 200 steps each, first 20
excluded as warm-up, so n=180 steady-state steps.

```{figure} _generated/scaling.svg
:alt: Speedup against GPU count, tracking just below the ideal linear line and reaching 7.76x at 8 GPUs

Speedup against the ideal, from the Speedup column below.
```

```{figure} _generated/utilisation-vs-throughput.svg
:alt: Top panel shows GPU utilisation flat near 99% across 1 to 8 GPUs, well above a dashed 75% threshold line; bottom panel shows tokens per second rising from 2,219 to 17,226

The GPU util and Tokens/sec columns plotted against each other. Only one responds to GPU
count.
```

| GPUs | Gradient accumulation | Mean step time | Median step time | Tokens per second | GPU utilisation | Speedup | Efficiency | Total training time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 59.078s | 59.578s | 2,219 | 99.06% | 1.00x | 100.0% | 11,839s |
| 2 | 16 | 29.920s | 30.111s | 4,381 | 98.96% | 1.97x | 98.7% | 6,001s |
| 4 | 8 | 15.071s | 15.073s | 8,697 | 98.94% | 3.92x | 98.0% | 3,022s |
| 8 | 4 | 7.609s | 7.577s | 17,226 | 98.80% | 7.76x | 97.1% | 1,528s |

Gradient accumulation scales inversely with GPU count, holding the global batch at 128 and
tokens per step at 131,072 in every row. GPU memory was flat at 20.8 GB peak across all
four.

## Phase 3: restart cost

Source: `results/phase3/startup_timing.json`. One restart, 8 GPUs, resumed from
`checkpoint-100`.

```{figure} _generated/restart-breakdown.svg
:alt: Horizontal bar splitting the 95.4 second restart into stages, dominated by base model load at 69.3 seconds

The stages to scale. Loading the base model is nearly three quarters of the restart.
```

| Stage | Duration | Share |
|---|---:|---:|
| Python + libraries | 1.59s | 1.7% |
| GPU init | 0.80s | 0.8% |
| Dataset load | 5.07s | 5.3% |
| Base model load | 69.25s | 72.6% |
| Checkpoint restore | 10.66s | 11.2% |
| First training step | 8.07s | 8.5% |
| **Total to first step** | **95.44s** | **100%** |

## Phase 4: what actually ran

Source: `results/phase4/version_a/events.json`, `version_b/events.json` and `phase4_summary.json`.

```{figure} _generated/phase4-timeline.svg
:alt: Two timelines of GPU allocation. Killing drops the node to 4 GPUs in use for 27 minutes. Shrinking keeps all 8 in use and finishes sooner.

GPU allocation over time in both arms.
```

| | Kill | Shrink |
|---|---:|---:|
| Wall time, first submit to low-priority done | 4,119s | 3,379s |
| Allocated GPU-hours | 7.33 | 7.47 |
| Idle GPU-hours | 1.82 | 0.03 |
| Fleet utilisation | 80.1% | 99.5% |

## Phase 4: node sharing

Two 4-GPU jobs sharing one 8-GPU node, compared with a 4-GPU job that had the node to
itself in the same experiment. The short step every run makes at an epoch boundary is
excluded, so the medians cover normal steps only.

```{figure} _generated/contention.svg
:alt: Step time distributions for two concurrent four-GPU jobs straddling the dashed line marking a four-GPU job running alone

The two shared jobs straddle the solo baseline.
```

| Run | Steps measured | Median step time | Difference |
|---|---:|---:|---:|
| 4 GPUs alone, from the kill arm | 78 | 15.169s | baseline |
| Sharing, low-priority | 79 | 15.061s | -0.71% |
| Sharing, high-priority | 78 | 15.138s | -0.20% |

## Phase 4: training loss after a resize

Two runs resumed from the same `checkpoint-100`, continuing on different GPU counts, over
101 steps of overlap.

```{figure} _generated/convergence.svg
:alt: Two training loss curves lying on top of each other, with a difference panel showing a gap of about 0.0005

The two loss curves, with their difference below at a 0.005 scale.
```

| | Value |
|---|---:|
| Mean loss, continued on 8 GPUs | 0.99761 |
| Mean loss, continued on 4 GPUs | 0.99725 |
| Mean absolute difference | 0.00037 |
| Maximum absolute difference | 0.00086 |
