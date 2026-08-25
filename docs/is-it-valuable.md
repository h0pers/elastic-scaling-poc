# Is it valuable?

**Yes. Under preemption, shrinking finished 18% sooner than killing and left the node
almost never idle.**

**Preemption.** A job is training on all 8 GPUs of a node when an urgent job arrives
needing 4 of them. Today the only option is to kill it. The urgent job gets its 4 GPUs,
and the other 4 sit doing nothing for the 27 minutes it runs.

Shrinking instead: the first job gives up 4 GPUs, carries on training with the 4 it
keeps, and takes the other 4 back when the urgent job finishes. Both jobs completed the
same work 12 minutes sooner, and no GPU on the node sat idle.

**Spare capacity.** Growing pays for itself on any run long enough to be worth
scheduling. The restart is a one-off cost, the speedup applies to every step after it.

**When the answer is no.** A job near the end of its run, where the restart costs more
than the remaining speedup gives back.

## Scaling is near-linear

**Measured.** Doubling the GPUs almost exactly doubles throughput, all the way to 8.

| GPUs | Step time | Tokens/sec | Speedup | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 59.08s | 2,219 | 1.00x | 100% |
| 2 | 29.92s | 4,381 | 1.97x | 98.7% |
| 4 | 15.07s | 8,697 | 3.92x | 98.0% |
| 8 | 7.61s | 17,226 | 7.76x | 97.1% |

n=180 steady-state steps per configuration, first 20 dropped as warm-up.
Source: `results/phase1/scaling.csv`.

```{figure} _generated/scaling.svg
:alt: Speedup against GPU count, tracking just below the ideal linear line and reaching 7.76x at 8 GPUs

Measured speedup against the ideal. The gap at 8 GPUs is 3%.
```

LoRA only synchronises the adapter weights, a small fraction of the 8B-parameter model,
so the all-reduce between GPUs costs little against a 7.6 second step. Expect this to
weaken for full fine-tuning, which synchronises everything.

## The restart timeline

Every resize restarts every worker, so a resize costs whatever a cold start costs. The
total is not a fixed figure. It moves with the model and the dataset, because the two
largest stages are pulling the base model into GPU memory and preparing the input
pipeline.

**Measured**, for the setup on the overview page: 95.4s from process start to the first
completed training step.

```{figure} _generated/restart-breakdown.svg
:alt: Horizontal bar splitting the 95.4 second restart into stages, dominated by base model load at 69.3 seconds

Where the restart time goes. Source: `results/phase3/startup_timing.json`.
```

Loading the base model dominates at 69.3s of the 95.4s, so the total tracks model size
rather than anything about the resize itself.

Freeing GPUs today means killing the job, and that job still has to restart from its last checkpoint afterwards,
paying the same stages. The restart happens either way. What changes is what the job
does in between.

## When is it worth taking more GPUs

Taking more GPUs is a trade. The job pays one restart up front, and every step after
that runs faster. Break-even is the moment those faster steps have saved back more time
than the restart cost.

**Derived**, for a job on 4 GPUs taking 8. The restart costs 95 seconds. Each step after
it saves about 7.5 seconds. So after roughly 13 steps the job has earned back what the
restart cost, and everything past that is time gained.

So a job with fewer than 13 steps to go finishes later for having scaled up. A job with
more finishes sooner, and the more it has left, the bigger the win.

```{figure} _generated/break-even.svg
:alt: Two lines showing time to finish against remaining steps, crossing at about 13 steps, after which restarting onto 8 GPUs wins

Break-even for growing 4 GPUs to 8. Derived from `results/phase1` and `results/phase3`.
```

13 steps is specific to this setup. Any other job needs two numbers: what a restart costs
it, and how much faster each of its steps becomes on the new GPU count.

```text
break-even steps = restart cost / time saved per step
```

The step time on the new GPU count has to be benchmarked once or estimated from the GPU
ratio, since the job is not running at that size yet.

## Two jobs on one node do not slow each other down

**Measured.** Two 4-GPU jobs running side by side on one 8-GPU node ran at the same speed
as a 4-GPU job with the node to itself.

| Run 4 GPUs            | Steps measured | Median step time | Difference |
|-----------------------|---:|---:|---:|
| Alone                 | 78 | 15.169s | baseline |
| Sharing, low-priority | 79 | 15.061s | -0.71% |
| Sharing, high-priority | 78 | 15.138s | -0.20% |

The baseline is the high-priority job from the kill arm, which had the node to itself.
Both shared jobs came out slightly faster. That is noise rather than a gain, and either
way the gap is under 1%.

```{figure} _generated/contention.svg
:alt: Step time distributions for two concurrent four-GPU jobs straddling the dashed line marking a four-GPU job running alone

Sharing the node did not slow either job down. Sources: `results/phase4/version_a` and
`results/phase4/version_b`.
```

:::{note}
Medians, not means. Every run has a short step at each epoch boundary, around 11.3s,
which drags the mean.
:::

That matters for shrinking. If handing back half a node made both jobs slower, shrinking
would be paying twice: once for the restart, and again for the contention.

## Changing GPU count does not change what the model learns

**Measured.** Two runs resumed from the same checkpoint, one continuing on 8 GPUs and one
on 4, produced near-identical loss curves over 101 steps: mean absolute difference
0.00037, maximum 0.00086, against a loss around 0.99.

```{figure} _generated/convergence.svg
:alt: Two training loss curves lying on top of each other, with a difference panel showing a gap of about 0.001

Loss after resuming at 4 GPUs versus 8 GPUs. Sources:
`results/phase4/version_a` and `results/phase4/version_b`.
```

Fewer GPUs means fewer examples per step, which would change how the model trains. This
job avoids that by raising gradient accumulation as GPUs drop, so the batch stays at 128
either way.

This is one pair of runs on LoRA

## A shrunk job keeps training while a killed one waits

**Measured.** A low-priority job on 8 GPUs is preempted by a high-priority job needing 4.
Two arms, one run each.

| | Kill | Shrink |
|---|---:|---:|
| High-priority time to GPUs | 15.3s | 5.4s |
| Low-priority steps during the contested window | 0 | 101 |
| Low-priority idle time | 1,625s | 5s |
| Restarts paid by the low-priority job | 1 | 2 |

The contested window is about 27 minutes in both arms, and the high-priority job starts
at least as fast under shrink. The difference is what the low-priority job does with that
window: nothing, or 101 steps of real progress on the half of the node it kept.

Shrink pays for a second restart, once to hand the GPUs over and once to take them back.
It still comes out ahead.

:::{note}
Each resize was simulated by killing the job and resubmitting it from its checkpoint at the new size
:::

## The whole run finishes 18% sooner

**Measured.** End to end, killing took 4,119s and shrinking took 3,379s. Shrinking saved
740s, 18.0%, on the same total work.

| | Kill | Shrink |
|---|---:|---:|
| Wall time, first submit to low-priority done | 4,119s | 3,379s |
| Allocated GPU-hours | 7.33 | 7.47 |
| Idle GPU-hours | 1.82 | 0.03 |
| Fleet utilisation | 80.1% | 99.5% |

```{figure} _generated/phase4-timeline.svg
:alt: Two timelines of GPU allocation. Killing drops the node to 4 GPUs in use for 27 minutes. Shrinking keeps all 8 in use throughout and finishes sooner.

GPU allocation over time in both arms, from the raw event timestamps.
Source: `results/phase4/version_*/events.json`.
```

The two arms allocate almost the same GPU-hours, so this is not a saving in hardware
consumed. It is the same hardware kept busy: killing leaves half the node doing nothing
for 27 minutes, and that is where the extra 12 minutes come from.
