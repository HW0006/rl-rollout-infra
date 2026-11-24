"""Collects rollout results into per-problem and fleet-wide metrics.

This is the consumption side of a rollout pipeline: pass-rate / reward
summaries and best-of-N selection, plus the throughput and latency numbers
you'd actually look at to tell whether the fleet is healthy.
"""

import statistics
from collections import defaultdict


def aggregate(jobs: list):
    by_problem = defaultdict(list)
    for job in jobs:
        by_problem[job.problem_id].append(job)

    per_problem = {}
    for problem_id, problem_jobs in by_problem.items():
        best = max(problem_jobs, key=lambda j: j.reward)
        per_problem[problem_id] = {
            "n_samples": len(problem_jobs),
            "mean_reward": statistics.mean(j.reward for j in problem_jobs),
            "solved_best_of_n": best.reward == 1.0,
            "best_reward": best.reward,
            "best_candidate": best.candidate_code,
        }

    latencies = [j.latency for j in jobs]
    rewards = [j.reward for j in jobs]
    summary = {
        "total_jobs": len(jobs),
        "failed_jobs": sum(1 for j in jobs if j.status == "failed"),
        "mean_reward": statistics.mean(rewards) if rewards else 0.0,
        "mean_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_s": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0,
    }
    return per_problem, summary
