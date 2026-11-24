"""Entry point: loads coding problems, runs N rollouts per problem through
the scheduler -> inference client -> sandbox pipeline, and prints a report.
"""

import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv

from aggregator import aggregate
from inference_client import InferenceClient
from sandbox import run_in_sandbox
from scheduler import Job, Scheduler

N_SAMPLES = 4
MAX_WORKERS = 5


def load_problems(path: str = "problems.json") -> list:
    with open(path) as f:
        return json.load(f)


async def main() -> None:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY (env var or .env file) first.", file=sys.stderr)
        sys.exit(1)

    problems = load_problems()
    client = InferenceClient(api_key=api_key, max_concurrency=MAX_WORKERS)
    scheduler = Scheduler(client, run_in_sandbox, max_workers=MAX_WORKERS)

    for problem in problems:
        for i in range(N_SAMPLES):
            scheduler.enqueue(Job(
                problem_id=problem["id"],
                func_name=problem["func_name"],
                prompt=problem["prompt"],
                test_cases=problem["test_cases"],
                sample_index=i,
            ))

    start = time.monotonic()
    jobs = await scheduler.run()
    elapsed = time.monotonic() - start

    per_problem, summary = aggregate(jobs)
    summary["wall_clock_s"] = round(elapsed, 2)
    summary["throughput_jobs_per_s"] = round(summary["total_jobs"] / elapsed, 2) if elapsed else 0.0

    report = {"summary": summary, "per_problem": per_problem}
    print(json.dumps(report, indent=2))

    with open("report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
