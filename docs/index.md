# Elastic Scaling PoC

A Kubeflow Trainer job holds a fixed number of GPUs from submission to completion. It
cannot take more when capacity frees up, and it cannot hand any back when a
higher-priority job needs them. The only way to reclaim GPUs from a running job today is
to kill it, discarding everything since its last checkpoint.

This research asks whether Kubeflow Trainer should be able to change a job's GPU count
while it runs, and what building that would involve.

## Why it matters

GPUs are expensive and usually shared between teams. In a multi-tenant cluster the thing
worth optimising is how busy the whole fleet stays, not how fast any single job
finishes. Fixed allocations work against that: GPUs sit idle waiting for a large enough
block to come free, and reclaiming capacity costs an entire job's progress.

## Where the answers are

| | Question | Page |
|---|---|---|
| 1 | Is elastic scaling valuable? | [Is it valuable?](is-it-valuable.md) |
| 2 | What signal should drive a scaling decision? | [When should we scale?](when-to-scale.md) |
| 3 | Can existing tools automate it? | [Can we automate it?](can-we-automate.md) |

Every number quoted on those pages is tabulated in
[the measurements](evidence.md). All code and results are on
[GitHub](https://github.com/h0pers/elastic-scaling-poc).

## Setup

Every benchmark on this site was run with the setup below.

| | |
|---|---|
| Model | `meta-llama/Llama-3.1-8B` |
| Dataset | `tatsu-lab/alpaca`, sequence packing on, sequence length 1024 |
| Method | LoRA, r=16, alpha=32, on the q/k/v/o projections |
| Batch | 4 per device, global batch 128 held constant across GPU counts by scaling gradient accumulation inversely with world size |
| Hardware | One node, 8x A100 80GB |
| Stack | Kubeflow Trainer v2 |

```{toctree}
:maxdepth: 1
:caption: Start here
:hidden:

Overview <self>
```

```{toctree}
:maxdepth: 1
:caption: Findings
:hidden:

is-it-valuable
when-to-scale
can-we-automate
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

evidence
```
