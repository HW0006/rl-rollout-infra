"""Job queue + bounded worker pool: the "scheduler" piece.

A fixed-size worker pool pulls rollout jobs from a queue, which gives
natural backpressure (only max_workers jobs run concurrently regardless of
how many are enqueued). Each job gets its own timeout and retry budget so
one bad job (stuck API call, runaway container) can't stall the fleet.
"""

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class Job:
    problem_id: str
    func_name: str
    prompt: str
    test_cases: list
    sample_index: int
    status: str = "pending"
    reward: float = 0.0
    error: str | None = None
    latency: float = 0.0
    candidate_code: str = ""


class Scheduler:
    def __init__(self, inference_client, sandbox_run_fn, max_workers: int = 5,
                 job_timeout: float = 30.0, max_retries: int = 2):
        self.inference_client = inference_client
        self.sandbox_run_fn = sandbox_run_fn
        self.max_workers = max_workers
        self.job_timeout = job_timeout
        self.max_retries = max_retries
        self.queue: asyncio.Queue = asyncio.Queue()
        self.results: list[Job] = []

    def enqueue(self, job: Job) -> None:
        self.queue.put_nowait(job)

    async def _worker(self) -> None:
        while True:
            try:
                job = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await self._process(job)
            self.queue.task_done()

    async def _process(self, job: Job) -> None:
        start = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                job.candidate_code = await asyncio.wait_for(
                    self.inference_client.generate_candidate(job.prompt),
                    timeout=self.job_timeout,
                )
                loop = asyncio.get_running_loop()
                sandbox_result = await loop.run_in_executor(
                    None, self.sandbox_run_fn, job.candidate_code, job.func_name, job.test_cases,
                )
                job.reward = sandbox_result.reward
                job.error = sandbox_result.error
                job.status = "done"
                break
            except Exception as e:
                job.error = str(e)
                if attempt == self.max_retries:
                    job.status = "failed"
                else:
                    await asyncio.sleep(1.5 * (attempt + 1))
        job.latency = time.monotonic() - start
        self.results.append(job)

    async def run(self) -> list[Job]:
        workers = [asyncio.create_task(self._worker()) for _ in range(self.max_workers)]
        await asyncio.gather(*workers)
        return self.results
