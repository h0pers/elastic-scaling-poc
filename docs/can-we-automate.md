# Can we automate it?

**Yes, build the scaling logic into the Trainer controller.** Five steps from where
upstream already is, three of them extending things that already work.

## The path

The chain already starts upstream. JobSet shipped Elastic support for `JobSet`, which makes a running
job's worker count mutable, and everything below builds on it.

```{mermaid}
flowchart TB
    s0["JobSet: resize a running job"]
    s1["Kueue: support elastic JobSet"]
    s2["Trainer: apply numNodes changes"]
    s3["Trainer: autoscale on free GPUs"]
    s4["Kueue: partial preemption"]
    s5["Trainer: shrink on reclaim"]

    s0 --> s1 --> s2 --> s3 --> s4 --> s5

    classDef done fill:#e6f2e6,stroke:#4a7a4a,color:#1c3a1c
    classDef extend fill:#eaf0f8,stroke:#2b6cb0,color:#12304f
    classDef new fill:#fdf0e8,stroke:#c05621,color:#5a2a0c

    class s0 done
    class s1,s2,s3 extend
    class s4,s5 new
```

| | Step | Owner | Upstream | Status |
|---|---|---|---|---|
| Done | Resize a running JobSet | JobSet | [jobset#463](https://github.com/kubernetes-sigs/jobset/issues/463) | Closed, ships behind the `ElasticJobSet` gate |
| 1 | Track a JobSet resize in quota | Kueue | [kueue#9411](https://github.com/kubernetes-sigs/kueue/issues/9411) | Elastic workloads are beta, JobSet not covered |
| 2 | Let `numNodes` change on a live job and reach the JobSet | Trainer | [trainer#2903](https://github.com/kubeflow/trainer/issues/2903), [trainer#3559](https://github.com/kubeflow/trainer/pull/3559) | Blocked on `PET_NNODES`, see below |
| 3 | Grow into free GPUs within a declared range | Trainer | [trainer#3654](https://github.com/kubeflow/trainer/pull/3654) | KEP proposed. Its trigger is the 75% utilisation threshold this research argues against |
| 4 | Ask a running job for part of its allocation back | Kueue | [kueue#975](https://github.com/kubernetes-sigs/kueue/issues/975) | Open since 2023, no design yet |
| 5 | Shrink on reclaim, or checkpoint and stop | Trainer | Not filed | Needs 3 and 4 |

Trainer copies `numNodes` into the `PET_NNODES` env var. JobSet allows `parallelism` to
change but not env vars, so the whole patch is rejected
([trainer#3559](https://github.com/kubeflow/trainer/pull/3559)).

**Fix: set `PET_NNODES` to a `min:max` range at creation.** torchrun's elastic mode
accepts a range, so the value never has to change again.

**What that buys.** Step 3 is the first point at which a job grows into GPUs that come
free. Step 5 is the first point at which a preempted job keeps running on fewer GPUs
instead of being stopped.

## Why it belongs in the controller

A resize decision needs two things at once: what the cluster has spare or wants back, and
what the job can usefully do about it. Kueue holds the first. Only the controller that
owns the TrainJob holds the second. Ray built its autoscaler the same way, inside Ray
rather than beside it.

KEDA would move that logic out of Kubeflow Trainer and onto the customer, who would have
to run and configure another component for behaviour that should work out of the box.

## What happens if you try it today

Two paths, both tested on a cluster: the resize a user would attempt through Kubeflow
Trainer, and the same resize applied directly to the JobSet underneath.

**Measured.** Patching a live TrainJob is accepted and then ignored:

```console
$ oc patch trainjob phase2-resize-trainer --type=merge \
    -p '{"spec": {"trainer": {"numNodes": 2}}}'
trainjob.trainer.kubeflow.org/phase2-resize-trainer patched

$ oc get trainjob phase2-resize-trainer -o jsonpath='{.spec.trainer.numNodes}'
2

$ oc get jobset phase2-resize-trainer \
    -o jsonpath='{.spec.replicatedJobs[0].template.spec.parallelism}'
1
```

The TrainJob reports the new value, the JobSet keeps the old one, no pod is created and
no error is raised.

**Measured.** Patching an Elastic JobSet directly, on upstream JobSet v0.12.0 with the
`ElasticJobSet` gate on, works in both directions. Scaling up creates a second worker
pod, scaling down terminates it while the first keeps running.

Source: `results/phase2/phase2_resize_test_executed.ipynb`.

## Partial preemption is what the autoscaler logic actually needs

Steps 1 to 3 connect parts that already function. Step 4 does not exist anywhere: no
scheduler in this ecosystem can ask a running job to hand back part of its allocation, so
the behaviour has to be designed rather than wired up. Nothing downstream of it can be
built either, since step 5 has no signal to act on.

[kueue#975](https://github.com/kubernetes-sigs/kueue/issues/975) is filed against Kueue
rather than against any training framework, and its description names Ray clusters as the
motivating case. Ray is blocked on the same thing Kubeflow Trainer are, so whoever builds it unblocks
both.
