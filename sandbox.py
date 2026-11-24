"""Isolated, resource-limited execution of untrusted candidate code.

This is the "sandboxed code execution" piece: it runs model-generated code
against test cases to compute a reward, the way a coding-RL environment
would, without trusting that code with network access, host files, or
unbounded CPU/memory/time.
"""

import json
import os
import tempfile

import docker

IMAGE = "python:3.11-slim"
TIMEOUT_SECONDS = 10
MEMORY_LIMIT = "128m"
NANO_CPUS = 500_000_000  # 0.5 CPU
PIDS_LIMIT = 64

HARNESS_TEMPLATE = """
import json

{candidate_code}

test_cases = json.loads({test_cases_json!r})
results = []
for args, expected in test_cases:
    try:
        actual = {func_name}(*args)
        results.append(actual == expected)
    except Exception:
        results.append(False)

print(json.dumps({{"passed": sum(results), "total": len(results)}}))
"""


class SandboxResult:
    def __init__(self, passed: int, total: int, error: str | None = None, timed_out: bool = False):
        self.passed = passed
        self.total = total
        self.error = error
        self.timed_out = timed_out

    @property
    def reward(self) -> float:
        return 0.0 if self.total == 0 else self.passed / self.total


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def run_in_sandbox(candidate_code: str, func_name: str, test_cases: list) -> SandboxResult:
    """Runs candidate_code + test_cases inside an isolated, network-disabled
    container with capped memory/CPU/pids and a wall-clock timeout."""
    client = _get_client()
    script = HARNESS_TEMPLATE.format(
        candidate_code=candidate_code,
        test_cases_json=json.dumps(test_cases),
        func_name=func_name,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chmod(tmpdir, 0o755)
        script_path = os.path.join(tmpdir, "run.py")
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o644)

        container = client.containers.run(
            IMAGE,
            command=["python", "/sandbox/run.py"],
            volumes={tmpdir: {"bind": "/sandbox", "mode": "ro"}},
            working_dir="/sandbox",
            network_disabled=True,
            mem_limit=MEMORY_LIMIT,
            nano_cpus=NANO_CPUS,
            pids_limit=PIDS_LIMIT,
            user="nobody",
            detach=True,
        )

        try:
            try:
                result = container.wait(timeout=TIMEOUT_SECONDS)
                exit_code = result.get("StatusCode", 1)
                logs = container.logs().decode("utf-8", errors="replace")
            except Exception:
                container.kill()
                return SandboxResult(0, len(test_cases), error="timeout", timed_out=True)
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

        if exit_code != 0:
            return SandboxResult(0, len(test_cases), error=logs.strip()[-500:])

        try:
            last_line = logs.strip().splitlines()[-1]
            parsed = json.loads(last_line)
            return SandboxResult(parsed["passed"], parsed["total"])
        except Exception as e:
            return SandboxResult(0, len(test_cases), error=f"parse error: {e}; logs={logs[:300]!r}")
