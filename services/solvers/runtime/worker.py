"""Warm solver process + anytime job queue.  Spec 6.3.

Two ideas, both about the user never seeing a spinner:

*Warm*.  A long-lived child process imports numpy / ortools / the solvers at
boot and preallocates its scratch arrays.  Nothing on the hot path pays import
or allocation cost.

*Anytime*.  Every solver emits best-so-far as it goes.  Progress messages flow
back over a queue and out to the WebSocket, so the first usable answer lands in
about 200 ms and improves in place.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

JobKind = str  # "optimize_factory" | "solve_builder" | "simulate_factory" | ...


@dataclass
class JobMessage:
    job_id: str
    type: str  # "progress" | "result" | "error" | "ready"
    data: Any = None
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------


def _warm_imports() -> None:
    """Pay every import cost once, before any job arrives."""
    import numpy  # noqa: F401

    try:
        import ortools.linear_solver.pywraplp  # noqa: F401
        import ortools.sat.python.cp_model  # noqa: F401
    except Exception:  # pragma: no cover - ortools optional at runtime
        pass
    try:
        from services.solvers.factory import bounds, optimize, sim  # noqa: F401
    except Exception:
        pass
    try:
        from services.solvers.builder import solve  # noqa: F401
    except Exception:
        pass


def _dispatch(kind: JobKind, payload: dict[str, Any], emit: Callable[[Any], None]) -> Any:
    """Run one job.  Imports are lazy so a half-built tree still serves the rest."""
    if kind == "optimize_factory":
        from services.core.rules.dsl import RuleFlags
        from services.solvers.factory.model import FactoryConfig, FactoryState
        from services.solvers.factory.optimize import optimize

        state = FactoryState.model_validate(payload["state"])
        flags = RuleFlags.model_validate(payload.get("flags") or {})
        seed = payload.get("seed_config")
        result = optimize(
            state,
            seconds=float(payload.get("seconds", 10.0)),
            flags=flags,
            seed_config=FactoryConfig.model_validate(seed) if seed else None,
            on_improve=lambda r: emit(_dump(r)),
        )
        return _dump(result)

    if kind == "simulate_factory":
        from services.core.rules.dsl import RuleFlags
        from services.solvers.factory.model import FactoryConfig, FactoryState
        from services.solvers.factory.sim import simulate

        return _dump(
            simulate(
                FactoryState.model_validate(payload["state"]),
                FactoryConfig.model_validate(payload["config"]),
                RuleFlags.model_validate(payload.get("flags") or {}),
            )
        )

    if kind == "factory_upper_bound":
        from services.core.rules.dsl import RuleFlags
        from services.solvers.factory.bounds import upper_bound
        from services.solvers.factory.model import FactoryState

        return _dump(
            upper_bound(
                FactoryState.model_validate(payload["state"]),
                RuleFlags.model_validate(payload.get("flags") or {}),
            )
        )

    if kind == "calibrate_factory":
        from services.solvers.factory.calibrate import calibrate
        from services.solvers.factory.model import FactoryConfig, FactoryState

        return _dump(
            calibrate(
                FactoryState.model_validate(payload["state"]),
                FactoryConfig.model_validate(payload["config"]),
                list(payload["observed"]),
            )
        )

    if kind == "solve_builder":
        from services.solvers.builder.model import BuilderPuzzle
        from services.solvers.builder.solve import solve_builder

        return _dump(
            solve_builder(
                BuilderPuzzle.model_validate(payload["puzzle"]),
                objective=payload.get("objective"),
                budget_s=float(payload.get("budget_s", 10.0)),
                max_builds=int(payload.get("max_builds", 200)),
            )
        )

    if kind == "solve_all_interpretations":
        from services.core.rules.resolve import solve_all_interpretations
        from services.solvers.builder.model import BuilderPuzzle

        puzzle = BuilderPuzzle.model_validate(payload["puzzle"])
        return _dump(
            solve_all_interpretations(puzzle.rules, puzzle, budget_s=float(payload.get("budget_s", 5.0)))
        )

    if kind == "echo":  # used by tests and by the warm-start health check
        return payload

    raise ValueError(f"unknown job kind {kind!r}")


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _child_main(cmd_q: Any, out_q: Any) -> None:  # pragma: no cover - runs in a child
    _warm_imports()
    out_q.put(JobMessage(job_id="", type="ready"))
    while True:
        item = cmd_q.get()
        if item is None:
            return
        job_id, kind, payload = item
        started = time.perf_counter()

        def emit(data: Any, _job_id: str = job_id, _started: float = started) -> None:
            out_q.put(
                JobMessage(
                    job_id=_job_id,
                    type="progress",
                    data=data,
                    elapsed_s=time.perf_counter() - _started,
                )
            )

        try:
            result = _dispatch(kind, payload, emit)
            out_q.put(
                JobMessage(
                    job_id=job_id,
                    type="result",
                    data=result,
                    elapsed_s=time.perf_counter() - started,
                )
            )
        except Exception as exc:
            out_q.put(
                JobMessage(
                    job_id=job_id,
                    type="error",
                    data={"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
                    elapsed_s=time.perf_counter() - started,
                )
            )


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    kind: JobKind
    submitted_at: float
    last_progress: Any = None
    result: Any = None
    error: str | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False


class JobManager:
    """Owns the warm process and fans job messages out to subscribers."""

    def __init__(self, *, method: str | None = None) -> None:
        # forkserver on POSIX: a fresh interpreter like spawn, but it does not
        # re-import the parent's __main__ (which under pytest or `python -c`
        # either costs seconds or deadlocks outright).
        if method is None:
            method = "forkserver" if os.name == "posix" else "spawn"
        try:
            self._ctx = mp.get_context(method)
        except ValueError:  # pragma: no cover - platform without that method
            self._ctx = mp.get_context()
        self._cmd_q: Any = None
        self._out_q: Any = None
        self._proc: Any = None
        self._pump: asyncio.Task | None = None
        self.jobs: dict[str, Job] = {}
        self._subscribers: list[Callable[[JobMessage], Any]] = []
        self._ready = asyncio.Event()
        self._inline = False  # set when the child could not be started
        self._reader: threading.Thread | None = None
        self._inbox: asyncio.Queue[JobMessage] = asyncio.Queue()

    # -- lifecycle ------------------------------------------------------

    async def start(self, timeout_s: float = 20.0) -> None:
        if self._proc is not None:
            return
        try:
            self._cmd_q = self._ctx.Queue()
            self._out_q = self._ctx.Queue()
            self._proc = self._ctx.Process(
                target=_child_main, args=(self._cmd_q, self._out_q), daemon=True
            )
            self._proc.start()
        except Exception:
            # No process isolation available (restricted sandbox, no /dev/shm).
            # Degrade to in-process execution rather than losing the feature.
            self._inline = True
            self._ready.set()
            return
        self._inbox = asyncio.Queue()
        loop = asyncio.get_running_loop()
        self._reader = threading.Thread(
            target=self._reader_loop, args=(loop, self._out_q), daemon=True, name="solver-reader"
        )
        self._reader.start()
        self._pump = asyncio.create_task(self._pump_loop())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout_s)
        except asyncio.TimeoutError:
            self._inline = True
            self._ready.set()

    def _reader_loop(self, loop: asyncio.AbstractEventLoop, out_q: Any) -> None:
        """Daemon thread: blocking queue reads must never hold up interpreter
        exit, so this thread is daemonised and woken by an explicit sentinel."""
        while True:
            try:
                msg = out_q.get()
            except (EOFError, OSError, ValueError):
                return
            if msg is None or getattr(msg, "type", None) == "__stop__":
                return
            try:
                loop.call_soon_threadsafe(self._inbox.put_nowait, msg)
            except RuntimeError:  # loop already closed
                return

    async def stop(self) -> None:
        if self._pump:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump
            self._pump = None
        if self._out_q is not None:
            with contextlib.suppress(Exception):
                self._out_q.put(JobMessage(job_id="", type="__stop__"))
        if self._reader is not None:
            self._reader.join(timeout=2.0)
            self._reader = None
        if self._proc is not None:
            try:
                self._cmd_q.put(None)
                self._proc.join(timeout=2.0)
            except Exception:
                pass
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
            self._proc = None
        for q in (self._cmd_q, self._out_q):
            with contextlib.suppress(Exception):
                if q is not None:
                    q.close()
                    q.join_thread()
        self._cmd_q = self._out_q = None

    async def restart(self) -> None:
        """Hard cancel: kill the worker mid-job and come back warm."""
        await self.stop()
        self._ready = asyncio.Event()
        await self.start()

    # -- subscription ---------------------------------------------------

    def subscribe(self, cb: Callable[[JobMessage], Any]) -> Callable[[], None]:
        self._subscribers.append(cb)

        def unsubscribe() -> None:
            if cb in self._subscribers:
                self._subscribers.remove(cb)

        return unsubscribe

    async def _emit(self, msg: JobMessage) -> None:
        for cb in list(self._subscribers):
            try:
                res = cb(msg)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass

    async def _pump_loop(self) -> None:
        while True:
            msg: JobMessage = await self._inbox.get()
            if msg.type == "__stop__":
                return
            if msg.type == "ready":
                self._ready.set()
                continue
            job = self.jobs.get(msg.job_id)
            if job is None or job.cancelled:
                continue
            if msg.type == "progress":
                job.last_progress = msg.data
            elif msg.type == "result":
                job.result = msg.data
                job.done.set()
            elif msg.type == "error":
                job.error = (msg.data or {}).get("error", "unknown error")
                job.done.set()
            await self._emit(msg)

    # -- submission -----------------------------------------------------

    async def submit(self, kind: JobKind, payload: dict[str, Any]) -> str:
        await self.start()
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, kind=kind, submitted_at=time.time())
        self.jobs[job_id] = job
        if self._inline:
            asyncio.create_task(self._run_inline(job, kind, payload))
        else:
            self._cmd_q.put((job_id, kind, payload))
        return job_id

    async def _run_inline(self, job: Job, kind: JobKind, payload: dict[str, Any]) -> None:
        """Fallback path: run in a thread so the event loop keeps serving."""
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        pending: queue.Queue = queue.Queue()

        def emit(data: Any) -> None:
            pending.put(data)

        async def drain() -> None:
            while True:
                await asyncio.sleep(0.05)
                while not pending.empty():
                    data = pending.get()
                    job.last_progress = data
                    await self._emit(
                        JobMessage(
                            job_id=job.id,
                            type="progress",
                            data=data,
                            elapsed_s=time.perf_counter() - started,
                        )
                    )

        drainer = asyncio.create_task(drain())
        try:
            result = await loop.run_in_executor(None, lambda: _dispatch(kind, payload, emit))
            job.result = result
            msg = JobMessage(job.id, "result", result, time.perf_counter() - started)
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            msg = JobMessage(
                job.id, "error", {"error": job.error}, time.perf_counter() - started
            )
        finally:
            drainer.cancel()
        job.done.set()
        await self._emit(msg)

    async def wait(self, job_id: str, timeout_s: float | None = None) -> Job:
        job = self.jobs[job_id]
        if timeout_s is None:
            await job.done.wait()
        else:
            try:
                await asyncio.wait_for(job.done.wait(), timeout_s)
            except asyncio.TimeoutError:
                pass
        return job

    async def run(self, kind: JobKind, payload: dict[str, Any], timeout_s: float = 60.0) -> Any:
        """Submit and await.  Raises RuntimeError on solver error."""
        job_id = await self.submit(kind, payload)
        job = await self.wait(job_id, timeout_s)
        if job.error:
            raise RuntimeError(job.error)
        if not job.done.is_set():
            raise TimeoutError(f"job {kind} exceeded {timeout_s}s")
        return job.result

    async def cancel(self, job_id: str, *, hard: bool = False) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        job.cancelled = True
        job.done.set()
        if hard:
            await self.restart()

    def best_so_far(self, job_id: str) -> Any:
        job = self.jobs.get(job_id)
        return None if job is None else (job.result or job.last_progress)

    def active(self) -> Iterable[Job]:
        return (j for j in self.jobs.values() if not j.done.is_set())


#: Process-wide manager, started by the FastAPI lifespan hook.
manager = JobManager()
