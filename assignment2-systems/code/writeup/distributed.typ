#import "lib.typ": gen

All experiments in this section use 2 processes on 2 GPUs of one node with the NCCL backend. Two machine-specific facts shape every number: (i) NCCL's direct GPU-to-GPU transport hangs on this machine (the first collective never completes), so `NCCL_P2P_DISABLE=1` is set and data travels GPU → host shared memory → GPU; (ii) both cards were running other jobs. Communication is therefore one to two orders of magnitude slower relative to compute than on NVLink-connected B200s.

== Problem `distributed_communication_single_node`

`cs336_systems/bench/allreduce_bench.py` all-reduces a float32 tensor of 1 MB – 1 GB, with 5 warm-up calls, 20 timed calls (10 for Gloo), `torch.cuda.synchronize()` around each call, and the per-rank means collected with `all_gather_object`. Only two GPUs were available, so the GPU sweep has 2 processes; to still show the effect of the process count, the same script was run with the Gloo backend on CPU for 2, 4 and 6 processes.

#grid(columns: (1.15fr, 1fr), gutter: 10pt,
  [#figure(gen("allreduce"), caption: [All-reduce latency.])],
  [#figure(image("generated/allreduce.png", width: 100%), caption: [Same data, log–log.])])

For large payloads the time is proportional to the size — the link is saturated at ≈ 0.85–0.9 GB/s for NCCL on this machine (1.13 s for 1 GB) and ≈ 1.4 GB/s for Gloo with 2 processes — while small payloads are dominated by a fixed latency (≈ 9 ms for NCCL, under 1 ms for Gloo), so the effective throughput of a 1 MB all-reduce is eight times lower than that of a 1 GB one. Adding processes makes each all-reduce slower even though the payload per process is unchanged (Gloo: 0.71 s, 1.03 s and 1.40 s for 1 GB with 2, 4 and 6 processes), because more data has to cross the same shared memory bus and the reduction takes more hops. The two factors interact as expected: many small messages are latency-bound and should be batched, large messages are bandwidth-bound and cost the same whether batched or not. That NCCL on GPUs is _slower_ than Gloo on CPUs here is an artefact of the disabled peer-to-peer path, not a property of NCCL.

== Problem `naive_ddp`

`cs336_systems/ddp.py` implements `DDP(module, mode)`: the constructor broadcasts every tensor of rank 0's `state_dict`, and in `mode="naive"` `finish_gradient_synchronization()` issues one blocking all-reduce per parameter gradient after the backward pass and divides by the world size. The adapters return this class (in its overlapping mode, below) and `tests/test_ddp.py` passes.

== Problems `naive_ddp_benchmarking`, `minimal_ddp_flat_benchmarking` and `ddp_overlap_individual_parameters_benchmarking` (a)

*Setup* (`cs336_systems/bench/ddp_bench.py`): the xl model cannot be trained unsharded on these cards (≥ 50.8 GiB per GPU), so the benchmarks use the *large* model (969 M parameters, 3.6 GiB of gradients to synchronise per step) and, for comparison, the *medium* model (423 M, 1.6 GiB). Global batch 4 (2 sequences per GPU), context 512, FP32, AdamW, 5 warm-up and 10 timed steps, averaged over both ranks. "Blocked on communication" is the wall time spent inside `finish_gradient_synchronization()` after the backward pass has been synchronised.

#figure(gen("ddp"), caption: [DDP training step with three ways of synchronising gradients (2 GPUs).]) <tab-ddp>

- *Naive* (one all-reduce per parameter tensor, after backward): 3.98 s per step for the large model, of which 2.82 s (71%) is gradient communication; 3.22 s and 73% for medium. The communication time is what the all-reduce benchmark predicts for 3.6 GiB at ≈ 1 GB/s — on this machine data-parallel training is communication-bound by a wide margin.
- *Flat* (one all-reduce of all gradients concatenated): 3.84 s per step and 2.69 s of communication for large (3.03 s / 2.16 s for medium), i.e. 4–8% less communication time. Batching removes the per-call latency of 327 (large) or 219 (medium) separate collectives, but most parameters are large matrices whose transfer is bandwidth-bound, so the saving is small, and it costs an extra copy of all gradients into and out of the flat buffer.
- *Overlapped* (asynchronous all-reduce per parameter, issued from a post-accumulate-grad hook as soon as each gradient is ready): 3.29 s for large (−17% vs naive) and 2.82 s for medium (−12%). The gradients of the last layers are already travelling while autograd is still working on the earlier ones, so the part of the communication that fits inside the backward pass becomes free. Because communication (≈ 2.7 s) is much longer than the backward pass (≈ 0.9 s) here, most of it still sticks out after the backward pass; with a fast interconnect, where communication is shorter than the backward pass, the same mechanism hides it almost completely. (The "blocked" column is not comparable for this mode: the remaining wait happens inside the `synchronize` that precedes the timer.)

== Problem `ddp_overlap_individual_parameters`

`DDP(mode="overlap")` registers `register_post_accumulate_grad_hook` on every parameter that requires a gradient; the hook starts `dist.all_reduce(param.grad, async_op=True)` and stores the handle, and `finish_gradient_synchronization()` waits for all handles and divides by the world size. Tied weights are handled naturally (the hook fires once per parameter tensor), and parameters without gradients are skipped. `tests/test_ddp.py` passed in 5 consecutive runs on GPU and 5 on CPU.

== Problem `ddp_overlap_individual_parameters_benchmarking` (b)

Instead of Nsight screenshots, the two timelines below are drawn from a `torch.profiler` (Kineto/CUPTI) trace of one training step on rank 0 — the same kernel-level data Nsight shows in its "CUDA HW" row — separating NCCL kernels from all other kernels (`cs336_systems/bench/dist_utils.py::dump_gpu_trace`, medium model).

#figure(image("generated/trace_ddp_naive.png", width: 100%), caption: [Naive DDP: all compute kernels (forward, backward) finish first; the NCCL all-reduce kernels only start afterwards, and the optimizer step follows them.])
#figure(image("generated/trace_ddp_overlap.png", width: 100%), caption: [Overlapped DDP: NCCL kernels start as soon as the backward pass produces its first gradients and run concurrently with the remaining compute kernels.])

In the naive trace communication and computation are strictly sequential; in the overlapped trace the NCCL kernels begin about 0.25 s into the step, while the forward/backward compute kernels are still running (they finish at ≈ 1.0 s), and the step ends ≈ 0.5 s earlier.

== Problem `optimizer_state_sharding`

`cs336_systems/sharded_optimizer.py` implements `ShardedOptimizer(params, optimizer_cls, **kwargs)`. It calls the `torch.optim.Optimizer` constructor (so `param_groups`, `zero_grad` etc. see all parameters); `add_param_group` assigns each parameter tensor to the rank with the smallest number of elements so far (deterministic, so all ranks agree without communicating) and adds the locally-owned ones to a wrapped local optimizer; `step` copies hyper-parameter changes into the local optimizer, steps it, and then broadcasts every parameter from its owner. `tests/test_sharded_optimizer.py` passed 5 times on GPU and 5 times on CPU.

== Problem `optimizer_state_sharding_accounting`

`cs336_systems/bench/sharded_bench.py`, same setup as the DDP benchmark (overlapped DDP, reference AdamW, FP32), large and medium models in place of xl. Peaks are `max_memory_allocated` up to the named point, in GiB.

#figure(gen("sharded"), caption: [Memory (GiB) and time with and without optimizer state sharding, per rank.])

*(a)* For the large model (3.61 GiB of parameters): after initialisation both variants hold just the parameters (3.7 GiB). Directly before the optimizer step the peak is 19.4 GiB without sharding and 15.7 GiB with it; after the step the process holds 14.7 GiB vs 11.0 GiB. The breakdown matches expectations exactly: without sharding the steady state is parameters + gradients + two AdamW moments $= 4 times 3.61 = 14.4$ GiB (7.22 GiB of optimizer state); with sharding each rank keeps moments for half of the parameters (3.59 and 3.64 GiB), so it holds $3 times 3.61 = 10.8$ GiB — a saving of $2P (1 - 1\/N) = 3.6$ GiB, 25% of the non-activation memory. The peak before the step is ≈ 4.7 GiB higher in both cases, which is activation memory of the forward/backward pass and is unaffected. The medium model shows the same pattern (3.15 → 1.55/1.60 GiB of state). For xl the same arithmetic gives 50.8 → 38.1 GiB, which is why sharding the optimizer state alone is not enough to fit xl on these cards.

*(b)* Sharding makes an iteration considerably slower here: 3.35 → 5.30 s for large (2.75 → 4.71 s for medium), entirely because `optimizer.step()` grows from 0.25–0.44 s to 2.4 s. After the local update every parameter is broadcast from its owner — 3.6 GiB through a ≈ 1 GB/s link, issued sequentially and blocking — which more than cancels the halved update work. With a fast interconnect this overhead would be a small fraction of the step.

*(c)* ZeRO stage 1 ($P_"os"$) partitions the _flattened_ optimizer state into $N$ equal contiguous shards, reduce-scatters the gradients so that each rank receives only the reduced gradient of its shard, and all-gathers the updated parameters; its communication volume per step equals that of plain data parallelism (a reduce-scatter plus an all-gather is one all-reduce, about $2 Psi$). Our version differs in three ways: (1) it assigns whole tensors to ranks, so shards are only approximately equal (3.59 vs 3.64 GiB above) and a single huge tensor cannot be split; (2) gradients are still fully all-reduced by DDP and the parameters are then _additionally_ broadcast, so the communication volume is roughly $3 Psi$ instead of $2 Psi$, sent as hundreds of small sequential collectives rather than one; (3) the memory saving on optimizer state is the same ($2 Psi -> 2 Psi \/ N$ for Adam in FP32 units of parameter size), and like ZeRO-1 both keep full gradients and full parameters on every rank.

== Problem `fsdp`

`cs336_systems/fsdp.py` implements `FSDP(module, compute_dtype=None)`:

- Every `Linear` and `Embedding` weight is replaced by an `nn.Parameter` holding this rank's $1\/N$ slice along dimension 0 (zero-padded if the size is not divisible), so parameters, gradients and optimizer state are all sharded and the shard is what the optimizer (e.g. the Assignment 1 AdamW) sees; `named_parameters()` keeps the original names. RMSNorm gains stay replicated and are synchronised DDP-style with asynchronous all-reduces.
- Each wrapped layer's `forward` is replaced (per instance — the module remains a `Linear`/`Embedding`) by a custom `autograd.Function` that takes the all-gathered weight, uses it and drops it; it deliberately does not save the weight for backward.
- *Scheduling.* The order in which the layers are used is recorded during each forward pass; the all-gather for call-site $i$ is launched asynchronously when call-site $i - 2$ has finished its forward, so at most one gathered weight is in flight while another is in use. The first pass (unknown order) gathers on demand.
- *Backward.* A linear layer all-gathers its weight again (prefetched one layer ahead, in reverse order), computes the input gradient and the full weight gradient, and launches an asynchronous reduce-scatter so that each rank receives the summed gradient of its own slice; embeddings scatter-add into a dense gradient and reduce-scatter it. `finish_gradient_synchronization()` waits for all collectives, divides by the world size, writes `.grad` in FP32 and discards any gathered weight that would be stale after the optimizer step.
- *Mixed precision.* With `compute_dtype`, shards are cast before they are communicated and gradients are communicated in the compute dtype (half the bytes), while master weights, `.grad` and optimizer updates stay in FP32.
- With the Gloo backend (used by the tests), which has no reduce-scatter, an all-reduce followed by slicing is used instead.

All four FSDP tests (correctness and gradient synchronisation, FP32 and FP16) passed in 5 consecutive runs on GPU and on CPU.

== Problem `fsdp_accounting`

*(a)* From the analysis above, a rank of a 2-GPU data-parallel run holds $4P$ (parameters, gradients, two moments) plus activations; optimizer-state sharding reduces this to $2P + 2P\/N$; FSDP shards all four, $4P\/N$. For xl ($P = 12.7$ GiB) and $N = 2$ that is 50.8 GiB → 38.1 GiB → *25.4 GiB*: FSDP saves 25.4 GiB (half of the non-activation memory) relative to plain DDP and a further 12.7 GiB relative to optimizer-state sharding, ignoring the buffers for the few gathered weights in flight. This is the difference between not fitting and fitting: the measured peak of the full xl model under FSDP is 29.0 GiB per GPU (25.4 GiB plus activations and transient full-size weights/gradients of the layers in flight), whereas neither DDP variant can run it at all.

*(b)* `cs336_systems/bench/fsdp_bench.py`, global batch 4, context 512, reference AdamW; BF16 runs use `compute_dtype=torch.bfloat16` under BF16 autocast. "Blocked on all-gather" is measured in separate instrumented steps that synchronise the GPU before every weight use and time how long the step then waits for the weight.

#figure(gen("fsdp"), caption: [FSDP on 2 GPUs: the full xl model and a 16-layer xl model, with and without scheduling the all-gathers ahead of use.])

#figure(image("generated/trace_fsdp.png", width: 100%), caption: [GPU kernel timeline of one FSDP training step on rank 0 (16-layer xl, BF16), from `torch.profiler`: NCCL all-gather/reduce-scatter kernels are active during the whole step.])

The full xl model trains under FSDP at 7.0 s per step (± 2.3 s; another job was active) with prefetching and 12.2 s without. *The communication does not finish in time for the forward pass on this machine:* in none of the 2,706 weight uses of the timed steps had the prefetched all-gather already completed when the layer needed it, and for the 16-layer model the step time is the same with and without prefetching (4.8 vs 4.7 s). The reason is the ratio of communication to compute. Per step each rank must receive half of every weight twice (forward and backward) and exchange every gradient once, roughly 10 GiB per rank per step in BF16 for full xl, over a ≈ 1 GB/s host-memory path, while the compute for 2 sequences is well under a second: the step is communication-bound by an order of magnitude and compute idles waiting for weights no matter how early the gathers are issued. What prefetching does achieve is visible in the instrumented runs: once a layer starts, the wait attributable to _starting the gather late_ drops from 7.8 s per step to 11 ms, i.e. the schedule itself is never the bottleneck. Communicating in BF16 instead of FP32 halves the step time (4.8 vs 9.3 s for 16 layers), a direct confirmation that bandwidth is the limit. On NVLink-class links (hundreds of GB/s per the calculations in @sec-parallelism, $N <= 1 + B W \/ C$) the same schedule would hide the gathers behind compute.
