# When should we scale?

Not on GPU utilisation. It reads about 99% whether the job has 1 GPU or 8, so it cannot
tell you anything about how many GPUs a job should have. Trigger on cluster state
instead, gate the decision on how much work is left, and use tokens/sec to check
afterwards that the change helped.

## GPU utilisation carries no signal

**Measured.** Across an 8x change in real throughput, utilisation moved by 0.26
percentage points.

| GPUs | GPU utilisation | Tokens/sec |
|---:|---:|---:|
| 1 | 99.06% | 2,219 |
| 2 | 98.96% | 4,381 |
| 4 | 98.94% | 8,697 |
| 8 | 98.80% | 17,226 |

Sampled every 0.5s through training via NVML, from 3,047 to 23,649 samples per run.
Source: `results/phase1/scaling.csv`.

```{figure} _generated/utilisation-vs-throughput.svg
:alt: Top panel shows GPU utilisation flat near 99% across 1 to 8 GPUs, well above a dashed 75% threshold line; bottom panel shows tokens per second rising from 2,219 to 17,226

The same four runs, measured two ways. Only one of them responds to GPU count.
```

The consequence is concrete. The elastic training proposal in
[kubeflow/trainer#2903](https://github.com/kubeflow/trainer/pull/2903) scales on a 75%
GPU-utilisation threshold. Every configuration here sits above that threshold at all
times, so the rule fires on every job at every size and never distinguishes a job that
would benefit from more GPUs from one that would not.

Utilisation measures whether the GPU is busy, not whether the work is useful or whether
more GPUs would help. It stays worth collecting for spotting a genuinely idle GPU, where
zero means zero. It is not a scaling signal.

## Tokens/sec is the number that moves

**Measured.** 2,219/s to 17,226/s across the same runs, tracking the actual speedup.

It is also comparable across configurations in a way step time is not. Changing GPU count
changes gradient accumulation, so step time measures a different quantity of work at each
size, while tokens/sec normalises that away.

## Ray has the same problem Kubeflow Trainer does

Ray is the closest working reference, and it decides on one thing: what hardware is free.

A Ray job declares a range instead of a fixed worker count,
`ScalingConfig(num_workers=(4, 8))`, and re-checks every 60 seconds. The cluster
autoscaler beside it compares requested resources against available ones and adds or
removes nodes, reading logical resource requests rather than GPU utilisation. Neither
loop looks at the training itself.

That gives two triggers:

- **Grow** when spare GPUs appear on the cluster, up to the maximum.
- **Shrink** when the job loses GPUs it was already using, such as a reclaimed spot node
  or a failed machine. Ray carries on with the workers that survive rather than failing
  the job.

The second one is not a choice. Ray shrinks because capacity was taken away from it,
never because another job needs it more. Nothing tells Ray that a high-priority workload
is waiting, and nothing tells Kubeflow Trainer either. That signal has to come from the
scheduler, and Kueue can only preempt a whole workload rather than reclaim part of one,
tracked in [kueue#975](https://github.com/kubernetes-sigs/kueue/issues/975).

So a Ray-style autoscaler would cover growing today. Priority-driven shrinking is the
half nobody can build yet.

## Knowing when is not the hardest part

None of this is buildable today, but the missing pieces differ enormously in cost.

**Most of the plumbing already exists.** Kueue's elastic workloads already resize an
admitted job's quota through workload slices, with no suspend and no requeue, and the
feature is beta and already covers RayJob and RayCluster. Extending it to JobSet is
extending something that works rather than designing something new. The layer below is
in the same state, since JobSet can already change a running job's worker count and
torchrun already handles the membership change.

**The trigger is genuinely new.** Neither Ray nor Kubeflow Trainer can learn that a
higher-priority job is waiting for its GPUs. Kueue can stop a workload but cannot ask it
to hand part of itself back, so there is nothing to subscribe to. That logic exists
nowhere and has to be designed from scratch.

**Ray needs it as much as Kubeflow Trainer does.**
[kueue#975](https://github.com/kubernetes-sigs/kueue/issues/975) is filed against Kueue
rather than against either training framework, and its description names Ray clusters as
the motivating case. Whoever builds the trigger unblocks both projects.

Once that signal exists, the decision splits in two, and the halves are not symmetric.

**Spare capacity is an offer**, and the job is free to decline it.

```text
Grow when:
    the cluster has idle GPUs
    AND remaining work clears break-even for the target size
```

**A reclaim is an instruction.** A higher-priority job cannot be asked to wait while a
low-priority one finishes, so break-even does not enter into it. The GPUs go back either
way, and all the job decides is what to do with what is left.

That turns on the minimum it declared up front, the `4` in `num_workers=(4, 8)`. If
enough GPUs remain to meet that minimum, the job keeps training at the smaller size,
which is the whole point of shrinking. If the reclaim takes it below, there are not
enough GPUs left to run on, so it saves its progress and exits cleanly instead of
crashing.

```text
On reclaim:
    release the GPUs immediately
    THEN keep training at the smaller size, if the remaining GPUs meet the minimum
    ELSE checkpoint and stop
```

Break-even governs the first case and not the second. The upstream proposal collapses
both into a single utilisation threshold, which misses that they are different decisions
with different authority behind them.

Two of the three inputs the grow rule needs already exist in Kubeflow Trainer's
`TrainJobStatus`: `estimatedRemainingSeconds` and `progressPercentage`. What is missing is
a component allowed to read them and act, which is the subject of
[can we automate it](can-we-automate.md).
