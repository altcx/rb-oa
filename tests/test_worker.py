"""Warm worker + anytime job queue."""

from __future__ import annotations

import asyncio
import time

import pytest

from services.solvers.runtime.worker import JobManager, JobMessage


@pytest.fixture
async def jm():
    m = JobManager()
    await m.start()
    yield m
    await m.stop()


async def test_echo_roundtrip(jm: JobManager):
    result = await jm.run("echo", {"hello": "world"}, timeout_s=30)
    assert result == {"hello": "world"}


async def test_unknown_kind_reports_error_without_killing_the_worker(jm: JobManager):
    with pytest.raises(RuntimeError) as exc:
        await jm.run("not_a_real_kind", {}, timeout_s=30)
    assert "unknown job kind" in str(exc.value)
    # the worker must survive a bad job: the next one still succeeds
    assert await jm.run("echo", {"n": 1}, timeout_s=30) == {"n": 1}


async def test_subscribers_see_messages(jm: JobManager):
    seen: list[JobMessage] = []
    unsub = jm.subscribe(lambda m: seen.append(m))
    await jm.run("echo", {"x": 2}, timeout_s=30)
    unsub()
    assert any(m.type == "result" for m in seen)


async def test_cancel_marks_job_done(jm: JobManager):
    job_id = await jm.submit("echo", {"x": 1})
    await jm.cancel(job_id)
    job = jm.jobs[job_id]
    assert job.cancelled and job.done.is_set()


async def test_submit_returns_immediately(jm: JobManager):
    """Submission must never block the event loop: the UI stays responsive."""
    t0 = time.perf_counter()
    await jm.submit("echo", {"x": 1})
    assert time.perf_counter() - t0 < 0.25


async def test_concurrent_jobs_all_complete(jm: JobManager):
    results = await asyncio.gather(*(jm.run("echo", {"i": i}, timeout_s=30) for i in range(5)))
    assert sorted(r["i"] for r in results) == [0, 1, 2, 3, 4]
