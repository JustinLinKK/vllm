# Real Server Metrics For Serving-Scale Calibration

## Goal

This note is the handoff for real-server data collection needed to calibrate the
current ASTRA-sim serving model for:

- colocated serving
- prefill/decode-disaggregated serving

The target is not generic observability. The target is a dataset that lets us
fit and validate simulator outputs for:

- TTFT
- TPOT
- E2E latency
- request throughput
- token throughput
- SLO goodput
- prefill/decode/transfer stage costs
- batching and contention behavior

## What We Need From Every Run

Every benchmark run should save three artifacts:

1. `run_config.json`
   Exact serving configuration used by the real system.
2. `request_metrics.csv`
   One row per completed request.
3. `system_timeseries.csv`
   Time-series utilization and network counters sampled during the run.

If PD is enabled, also save:

4. `pd_stage_metrics.csv`
   Per-request or per-batch prefill, transfer, and decode stage timings.

The simulator now mirrors that split through:

- request-level service and queue-wait fields in `request_metrics.csv`
- optional `outputs.stage_metrics_output` for per-batch calibration rows
- enriched `outputs.event_trace_output` with request counts, queue depths, and
  decode base-latency flags

## Required Run Metadata

These fields should be recorded once per run because they are simulator inputs,
not fit knobs:

- model name and revision
- dense vs MoE model type
- number of layers, hidden size, attention heads, KV heads, expert count
- weight precision and KV-cache precision
- GPU SKU, GPU count, GPU memory size, GPU memory bandwidth
- node count
- intra-node interconnect type and bandwidth
- inter-node network type and bandwidth
- serving engine and exact commit or version
- runtime architecture: `colocated` or `pd_disaggregated`
- TP degree
- PP degree
- EP degree if used
- DP-attention degree if used
- replica count
- scheduler caps:
  `max_running_requests`, `max_decode_batch_size`,
  `max_prefill_batch_tokens`, `max_prefill_batch_size`
- request trace parameters:
  request rate, prompt length distribution, output length distribution,
  concurrency target, warmup duration, measurement duration
- for PD: prefill worker count, decode worker count, router settings, transfer
  path, and whether prefill and decode use different layouts

## Request-Level Metrics

These are the most important measurements because they map directly to simulator
outputs.

Record one row per request with:

- `request_id`
- `arrival_ts`
- `service_start_ts` if available
- `prefill_queue_enter_ts` if available
- `prefill_start_ts`
- `prefill_end_ts`
- `transfer_queue_enter_ts` if PD
- `transfer_start_ts` if PD
- `transfer_end_ts` if PD
- `decode_queue_enter_ts` if available
- `decode_start_ts`
- `first_token_ts`
- `finish_ts`
- `prompt_tokens`
- `output_tokens`
- `total_tokens`
- `success_or_failure`
- failure reason or timeout reason if any

From those timestamps we need to compute:

- `ttft_ns = first_token_ts - arrival_ts`
- `tpot_ns = (finish_ts - decode_start_ts) / output_tokens`
- `e2e_ns = finish_ts - arrival_ts`
- `queue_wait_ns = service_start_ts - arrival_ts` if available
- `prefill_queue_wait_ns = prefill_start_ts - prefill_queue_enter_ts` if available
- `decode_queue_wait_ns = decode_start_ts - decode_queue_enter_ts` if available
- `transfer_queue_wait_ns = transfer_start_ts - transfer_queue_enter_ts` if PD
- `prefill_duration_ns = prefill_end_ts - prefill_start_ts`
- `transfer_duration_ns = transfer_end_ts - transfer_start_ts` if PD
- `decode_duration_ns = finish_ts - decode_start_ts`

For each run, report summary stats for the above:

- mean
- p50
- p90
- p99

These summary fields should line up with the simulator’s current request-level
metrics and make direct closure testing possible.

## Throughput And Goodput Metrics

For each run, report:

- completed requests
- completed output tokens
- run makespan
- request throughput in requests per second
- output-token throughput in tokens per second
- TTFT SLO threshold used
- TPOT SLO threshold used
- E2E SLO threshold used
- good requests count
- goodput in requests per second
- SLO attainment fraction

Also record the fraction of requests failing each SLO separately:

- TTFT SLO miss fraction
- TPOT SLO miss fraction
- E2E SLO miss fraction

This matters because a simulator fit can have the right overall goodput but the
wrong failure mode.

## Batch-Shape Metrics

We need real batch behavior to fit `batch_efficiency` and decode interference.
At minimum, collect time-series or batch-level summaries for:

- active running requests over time
- prefill batch request count
- prefill batch token count
- decode batch request count
- decode generated tokens per step
- number of concurrent decode sequences
- chunked prefill chunk size if used
- scheduler admission rate over time
- request queue depth over time
- prefill queue depth over time
- decode queue depth over time

For each of those, save:

- mean
- p50
- p90
- p99
- max

If the engine can emit per-iteration scheduler traces, keep those too. They are
the cleanest way to fit contention and batching curves.

## GPU Compute And Memory Metrics

These metrics help separate compute-side mismatch from queueing mismatch.
Sample them at fixed cadence, ideally 100 ms to 1 s:

- SM or compute utilization per GPU
- HBM bandwidth utilization per GPU if available
- GPU memory used
- KV-cache memory used
- power draw if available
- achieved FLOPs if available from profiler traces
- kernel time breakdown if available:
  attention, FFN, expert kernels, all-reduce, all-gather, all-to-all, memcpy

For calibration, the most useful profiler-derived rollups are:

- prefill attention compute time per prompt token
- prefill FFN or expert compute time per prompt token
- decode attention compute time per output token
- decode FFN or expert compute time per output token

If only coarse profiling is possible, collect separate prefill-heavy and
decode-heavy runs so those coefficients can still be estimated.

## Communication Metrics

We need enough network data to fit `comm_scale` and the stage-specific
communication terms.

For colocated runs, collect:

- TP collective bytes and time per iteration if available
- PP activation bytes and time if PP is enabled
- EP dispatch or combine bytes and time if EP is enabled
- NCCL collective latency summaries
- per-link or per-NIC TX and RX throughput

For PD runs, also collect:

- prefill-to-decode transfer bytes per request
- prefill-to-decode transfer latency per request
- prefill-to-decode transfer bandwidth during steady state
- router queue depth
- router dispatch latency
- NIC TX and RX throughput on prefill nodes
- NIC TX and RX throughput on decode nodes

If direct transfer bytes are not exposed, save enough model metadata to infer
KV transfer size from prompt length and KV format.

## PD-Specific Stage Metrics

For PD calibration, stage separation matters more than aggregate latency. We
need measurements that isolate:

- prefill service time
- transfer service time
- decode service time
- overlap between prefill and decode
- idle time on decode workers waiting for transfer
- idle time on prefill workers blocked by scheduler or router backpressure

Useful PD-only counters include:

- prefill worker busy fraction
- decode worker busy fraction
- transfer link busy fraction
- number of in-flight handoffs
- handoff queue depth
- handoff failures or retries

## Minimal Experiment Matrix

The collection should cover both clean calibration runs and realistic
validation runs.

### A. Colocated Single-Request Baselines

Purpose: fit clean prefill and decode constants with minimal queueing.

- concurrency 1
- short prompt, short generation
- long prompt, short generation
- short prompt, long generation
- long prompt, long generation

### B. Colocated Concurrency Sweeps

Purpose: fit batching efficiency and decode interference.

- fixed model and layout
- request rate or concurrency sweep from light load to saturation
- same prompt/output distribution used in later validation

### C. PD Single-Request Baselines

Purpose: fit transfer overhead and stage isolation.

- concurrency 1
- same prompt/output sweep as colocated
- separate prefill and decode worker counts recorded

### D. PD Concurrency Sweeps

Purpose: fit overlap, backpressure, and handoff sensitivity.

- sweep request rate from light load to saturation
- sweep prefill worker count if possible
- sweep decode worker count if possible
- keep transfer path and network topology fixed per run

### E. Realistic Trace Validation

Purpose: final closure against simulator outputs.

- ShareGPT-like or production-like prompt/output mix
- colocated and PD runs on the same hardware cluster
- same SLO thresholds used by the simulator

## How The Metrics Map To Simulator Knobs

Use this mapping when turning measurements into fit parameters.

| Real measurement | Main simulator knob or output |
| --- | --- |
| single-request prefill duration vs prompt tokens | `cost_model.prefill.attention_compute_ns_per_token`, `cost_model.prefill.ffn_or_expert_compute_ns_per_token` |
| single-request decode duration vs output tokens | `cost_model.decode.attention_compute_ns_per_token`, `cost_model.decode.ffn_or_expert_compute_ns_per_token` |
| TP or PP slowdown relative to 1-GPU baseline | `comm_scale`, `cost_model.<stage>.tp_collective_ns_per_token`, `cost_model.<stage>.pp_activation_ns_per_token` |
| EP dispatch overhead | `cost_model.<stage>.ep_dispatch_ns_per_token` |
| DP-attention synchronization overhead | `cost_model.<stage>.dp_attention_sync_ns_per_token` |
| batch runtime vs batch size | `cost_model.prefill.batch_efficiency`, `cost_model.decode.batch_efficiency` |
| decode slowdown under sustained concurrency | `cost_model.decode.interference_factor` |
| PD transfer latency and bytes | `pd.transfer.latency_ns`, `pd.transfer.bandwidth_bytes_per_s`, `pd.transfer.efficiency`, `pd.transfer.bytes_per_prompt_token` |
| throughput and latency under realistic traces | TTFT, TPOT, E2E, throughput, goodput validation targets |

## Nice-To-Have Profiling Data

If the teammate can collect profiler traces without too much overhead, these are
high value:

- Nsight Systems timeline for one colocated run and one PD run
- NCCL trace or collective summaries
- per-kernel time grouped into attention, FFN, expert, communication, memcpy
- KV-cache allocation growth over time
- router timeline or handoff timeline in PD mode

These are not mandatory for every run, but they are very helpful when the fit is
directionally wrong and we need to decide whether the problem is compute,
communication, or scheduler behavior.

## Deliverable Checklist

The data collection is complete when we have:

- colocated runs with request-level latency metrics
- PD runs with request-level latency metrics
- run-level throughput and goodput for both modes
- stage-separated prefill, transfer, and decode timings for PD
- batching and queue-depth statistics
- GPU utilization and memory usage time series
- network and collective summaries
- exact run metadata for topology and scheduler settings

Without stage timing and queueing data, we can still fit surface TTFT and TPOT,
but we will not be able to calibrate the simulator in an interpretable way.
