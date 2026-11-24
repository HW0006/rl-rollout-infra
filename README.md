# rl-rollout-infra

A small, honest, working demo of the **infrastructure** behind code-generation
RL rollouts — not the RL algorithm itself, but the systems that would surround
one: a scheduler, an inference client, and a sandboxed execution environment.

This is a portfolio piece built to be read and discussed component by
component, not a production system. It is not "large-scale" — it's a
correctly-architected miniature of the real thing, built with no GPU
cluster and no budget beyond fractions of a cent in API calls.

## What it does

For each of a handful of coding problems (see `problems.json`), it:

1. **Schedules** `N` rollout jobs per problem onto a bounded worker pool
   (`scheduler.py`) — this gives backpressure (only `max_workers` jobs run
   at once no matter how many are queued), per-job timeouts, and retries
   with backoff so one stuck job can't stall the fleet.
2. Has each worker call Claude (`inference_client.py`) to generate a
   candidate solution — concurrency-limited, with exponential backoff on
   transient API errors. This is the "inference serving client" side of a
   rollout pipeline.
3. Runs each candidate in an **isolated Docker container**
   (`sandbox.py`) — no network, capped memory/CPU/process count, a
   wall-clock timeout, running as a non-root user — against the problem's
   test cases to compute a pass/fail reward. This is the same pattern used
   to compute reward signals for coding-RL environments.
4. **Aggregates** results (`aggregator.py`) into per-problem pass rates,
   best-of-N selection, and fleet-wide latency/throughput numbers.

## Architecture

```
main.py
  │
  ├─ problems.json ──► Job queue (scheduler.py)
  │                         │
  │                         ├─ worker 1 ─┐
  │                         ├─ worker 2 ─┤── inference_client.py ──► Claude API
  │                         ├─ ...       │          │
  │                         └─ worker N ─┘          ▼
  │                                        candidate code
  │                                              │
  │                                              ▼
  │                                        sandbox.py ──► Docker container
  │                                              │          (isolated, resource-limited)
  │                                              ▼
  │                                        pass/fail reward
  │                                              │
  └───────────────────────────────────► aggregator.py ──► report.json
```

## Setup

1. **Docker Desktop** must be installed and running (used for sandboxed
   execution — see https://www.docker.com/products/docker-desktop/).
2. Create a virtualenv and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Set your Anthropic API key, either as an environment variable or in a
   `.env` file in this directory:
   ```bash
   echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
   ```
   Get a key at https://console.anthropic.com/. Cost for a full run across
   all problems is on the order of a fraction of a cent (Haiku pricing).

## Run

```bash
python main.py
```

Prints a JSON report to stdout and writes it to `report.json`, including
each problem's best candidate solution, pass rate, and fleet-wide latency
and throughput.

## Honest scope notes

- This does **not** implement an RL training loop (no gradient updates, no
  policy weights) — it implements the infrastructure a training loop would
  sit on top of: rollout scheduling, concurrent inference, and sandboxed
  reward computation. That's deliberate — it's the infra, not the research.
- Isolation is Docker-level (network-disabled, resource-capped containers),
  not a hardened gVisor/Firecracker-style sandbox — appropriate for a demo,
  not for running truly adversarial code at scale.
- Scale here is "a handful of problems on one machine," not a distributed
  cluster. The architecture (queue + bounded workers + retries) is the same
  shape you'd scale out, but this doesn't demonstrate operating it at scale.

## Possible extensions

- Swap the in-process `asyncio.Queue` for a real broker (Redis/SQS) and run
  workers as separate processes/machines.
- Add real policy weight updates (e.g. a simple REINFORCE step) using the
  reward signal already being computed.
- Replace Docker with a faster-starting sandbox (e.g. a pre-warmed container
  pool) and measure the throughput difference.
