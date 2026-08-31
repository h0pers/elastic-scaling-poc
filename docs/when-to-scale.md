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

## Ray scales on capacity, not on the queue

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

## What is missing, and what it costs

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

**Ray could use it as much as Kubeflow Trainer does.**
[kueue#975](https://github.com/kubernetes-sigs/kueue/issues/975) is filed against Kueue
rather than against either training framework, and its description names Ray clusters as
the motivating case. Whoever builds the trigger unblocks both projects.

### Level 1: elastic training, no queue knowledge

Trainer watches the cluster itself, the way Ray's autoscaler does. It never learns who is
waiting or how important they are.

Growing, when capacity appears:

```{mermaid}
flowchart TB
    u1["TrainJob running on 4 GPUs<br/>declared range 4 to 8"]
    u2["Trainer polls the cluster<br/>on an interval"]
    u3{"Are there free GPUs?"}
    u4{"Is there enough work left<br/>to be worth the restart?"}
    u5["Trainer patches numNodes 4 to 8"]
    u6["Stay at 4, poll again later"]
    u7["JobSet creates 4 pods"]
    u8["torchrun restarts all workers,<br/>resumes from checkpoint"]
    u9["Kueue quota follows the resize"]

    u1 --> u2 --> u3
    u3 -- no --> u6
    u3 -- yes --> u4
    u4 -- no --> u6
    u4 -- yes --> u5 --> u7 --> u8 --> u9

    classDef kueue fill:#fdf3e0,stroke:#a3822a,color:#4a3a0c
    classDef trainer fill:#eaf0f8,stroke:#2b6cb0,color:#12304f
    classDef exec fill:#e6f2e6,stroke:#4a7a4a,color:#1c3a1c
    classDef missing fill:#fdeaea,stroke:#a34a4a,stroke-dasharray: 5 5,color:#4a1c1c

    class u1 kueue
    class u2,u3,u4,u5,u6 trainer
    class u7,u8 exec
    class u9 missing
```

Shrinking, which is not a decision at this level:

```{mermaid}
flowchart TB
    d1["TrainJob running on 8 GPUs"]
    d2["A node is lost<br/>spot reclaim or failure"]
    d3["torchrun sees the<br/>membership change"]
    d4{"Do the remaining GPUs<br/>meet the declared minimum?"}
    d5["Restart the survivors,<br/>carry on at the smaller size"]
    d6["Checkpoint and stop"]

    d1 --> d2 --> d3 --> d4
    d4 -- yes --> d5
    d4 -- no --> d6

    classDef exec fill:#e6f2e6,stroke:#4a7a4a,color:#1c3a1c
    classDef infra fill:#fdf3e0,stroke:#a3822a,color:#4a3a0c

    class d1,d2 infra
    class d3,d4,d5,d6 exec
```

Nothing chooses to shrink here. Capacity is taken away and the job survives on what is
left.

### Level 2: queue preemption

Everything above still applies. What is added is a second reason to shrink, one the job
does not choose.

```{mermaid}
flowchart TB
    s1["TrainJob running on 8 GPUs<br/>Kueue holds its quota"]
    s2["Higher-priority workload submitted,<br/>needs 4 GPUs, cluster is full"]
    s3["Kueue reclaims 4 of our 8<br/>instead of evicting the whole job"]
    s4["Kueue tells Trainer<br/>to release 4 GPUs"]
    s5{"Do the remaining 4<br/>meet the declared minimum?"}
    s6["Trainer patches numNodes 8 to 4"]
    s7["Trainer checkpoints<br/>and lets the job stop"]
    s8["JobSet deletes 4 pods"]
    s9["torchrun restarts the survivors,<br/>resumes from checkpoint"]
    s10["Kueue admits the waiting workload"]

    s1 --> s2 --> s3 --> s4 --> s5
    s5 -- yes --> s6 --> s8 --> s9 --> s10
    s5 -- no --> s7 --> s10

    classDef kueue fill:#fdf3e0,stroke:#a3822a,color:#4a3a0c
    classDef trainer fill:#eaf0f8,stroke:#2b6cb0,color:#12304f
    classDef exec fill:#e6f2e6,stroke:#4a7a4a,color:#1c3a1c
    classDef missing fill:#fdeaea,stroke:#a34a4a,stroke-dasharray: 5 5,color:#4a1c1c

    class s1,s2,s10 kueue
    class s3,s4 missing
    class s5,s6,s7 trainer
    class s8,s9 exec
```

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
