"""FastAPI app: REST + WebSocket, serving the built web UI.

Every long-running solve goes through the warm worker (``solvers/runtime``) and
streams best-so-far over the WebSocket, so the UI never shows a spinner where a
partial answer would do.

Modules that are optional at runtime (screen capture on a headless box, the
OpenRouter client without a key) are imported lazily and degrade to a clear
error instead of taking the server down.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from services.core.session import LeaderboardEntry, SessionState, sessions
from services.solvers.runtime.worker import JobMessage, manager

log = logging.getLogger("puzzle_copilot")

WEB_DIST = Path(__file__).resolve().parents[2] / "apps" / "web" / "dist"


# ---------------------------------------------------------------------------
# WebSocket hub
# ---------------------------------------------------------------------------


class Hub:
    def __init__(self) -> None:
        self._conns: dict[str, set[WebSocket]] = {}

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._conns.setdefault(session_id, set()).add(ws)

    def disconnect(self, session_id: str, ws: WebSocket) -> None:
        self._conns.get(session_id, set()).discard(ws)

    async def send(self, session_id: str, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._conns.get(session_id, ())):
            try:
                await ws.send_text(json.dumps(payload, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(session_id, ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        for sid in list(self._conns):
            await self.send(sid, payload)


hub = Hub()
#: job id -> session id, so worker progress reaches the right socket.
_job_owner: dict[str, str] = {}


async def _on_job_message(msg: JobMessage) -> None:
    session_id = _job_owner.get(msg.job_id)
    if session_id is None:
        return
    if msg.type == "error":
        await hub.send(session_id, {"type": "warning", "text": (msg.data or {}).get("error", "")})
        return
    data = msg.data or {}
    if isinstance(data, dict) and "best_value" in data:
        await hub.send(
            session_id,
            {
                "type": "optimizer_progress",
                "job_id": msg.job_id,
                "final": msg.type == "result",
                "best_value": data.get("best_value"),
                "bound": data.get("bound"),
                "iterations": data.get("iterations", 0),
                "elapsed_s": msg.elapsed_s,
                "actions": data.get("actions", []),
                "top_warning": data.get("top_warning"),
                "money_by_hour": data.get("money_by_hour"),
            },
        )
        session = sessions.get(session_id)
        if session and msg.type == "result":
            session.optimizer_best = data
            if data.get("best_value") is not None:
                session.record_score(
                    LeaderboardEntry(
                        label="optimizer",
                        score=float(data["best_value"]),
                        config_summary=_summarize_actions(data.get("actions", [])),
                        config=data.get("best"),
                        source="optimizer",
                    )
                )
            sessions.save(session)
    else:
        await hub.send(
            session_id,
            {"type": "job_result", "job_id": msg.job_id, "kind": msg.type, "data": data},
        )


def _summarize_actions(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "no change"
    head = ", ".join(f"{a.get('target')}:{a.get('setting')}={a.get('value')}" for a in actions[:3])
    return head + (f" (+{len(actions) - 3} more)" if len(actions) > 3 else "")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from services.core.settings.keys import install_redaction_filter

        install_redaction_filter()
    except Exception:  # module may not be present in a partial checkout
        log.debug("redaction filter unavailable")
    manager.subscribe(_on_job_message)
    await manager.start()
    yield
    await manager.stop()


app = FastAPI(title="Puzzle Copilot", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _err(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


# -- sessions ---------------------------------------------------------------


@app.get("/api/sessions")
def api_sessions() -> dict[str, Any]:
    return {
        "sessions": [
            {"id": s.id, "created_at": s.created_at, "puzzle_type": s.puzzle_type}
            for s in sessions.list()
        ]
    }


@app.post("/api/sessions")
def api_create_session(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    s = sessions.create(body.get("puzzle_type", "factory"))
    return {"id": s.id, "puzzle_type": s.puzzle_type, "created_at": s.created_at}


def _require(session_id: str) -> SessionState:
    s = sessions.get(session_id)
    if s is None:
        raise HTTPException(404, f"no session {session_id}")
    return s


# -- captures ---------------------------------------------------------------


def _capture_store():
    from services.core.capture.store import CaptureStore  # lazy: optional at runtime

    return CaptureStore()


@app.get("/api/sessions/{session_id}/captures")
def api_captures(session_id: str) -> dict[str, Any]:
    _require(session_id)
    try:
        store = _capture_store()
        items = [c.model_dump(mode="json") for c in store.list_captures(session_id)]
    except Exception as exc:
        return {"captures": [], "error": str(exc)}
    for c in items:
        c["thumb_url"] = f"/api/captures/{c['id']}/thumb"
        c["image_url"] = f"/api/captures/{c['id']}/image"
    items.sort(key=lambda c: c.get("created_at", 0), reverse=True)
    return {"captures": items}


@app.post("/api/captures/monitor")
async def api_capture_monitor(body: dict[str, Any] = Body(...)) -> Any:
    session = _require(body["session_id"])
    try:
        from services.core.capture.grab import grab_monitor

        store = _capture_store()
        cap = grab_monitor(int(body.get("monitor", 0)))
        meta = store.save(cap, session_id=session.id, puzzle_type=session.puzzle_type)
    except Exception as exc:
        return _err(503, f"capture unavailable: {exc}")
    payload = meta.model_dump(mode="json")
    payload["thumb_url"] = f"/api/captures/{meta.id}/thumb"
    await hub.send(session.id, {"type": "capture", "capture": payload})
    asyncio.create_task(_speculate(session, meta.id))
    return {"capture_id": meta.id}


@app.post("/api/captures/region")
async def api_capture_region(body: dict[str, Any] = Body(...)) -> Any:
    session = _require(body["session_id"])
    try:
        from services.core.capture.grab import grab_region

        store = _capture_store()
        cap = grab_region(int(body["x"]), int(body["y"]), int(body["w"]), int(body["h"]))
        meta = store.save(cap, session_id=session.id, puzzle_type=session.puzzle_type)
    except Exception as exc:
        return _err(503, f"capture unavailable: {exc}")
    payload = meta.model_dump(mode="json")
    payload["thumb_url"] = f"/api/captures/{meta.id}/thumb"
    await hub.send(session.id, {"type": "capture", "capture": payload})
    asyncio.create_task(_speculate(session, meta.id))
    return {"capture_id": meta.id}


@app.post("/api/captures/upload")
async def api_capture_upload(body: dict[str, Any] = Body(...)) -> Any:
    """Accept a base64 PNG.  Lets the tool be used on a machine where the
    screen-grab backend is unavailable, and lets tests drive the whole path."""
    session = _require(body["session_id"])
    raw = body["image_base64"].split(",", 1)[-1]
    try:
        store = _capture_store()
        meta = store.save_bytes(
            base64.b64decode(raw), session_id=session.id, puzzle_type=session.puzzle_type
        )
    except Exception as exc:
        return _err(503, f"capture store unavailable: {exc}")
    payload = meta.model_dump(mode="json")
    payload["thumb_url"] = f"/api/captures/{meta.id}/thumb"
    await hub.send(session.id, {"type": "capture", "capture": payload})
    return {"capture_id": meta.id}


@app.get("/api/captures/{capture_id}/image")
def api_capture_image(capture_id: str) -> Any:
    try:
        path = _capture_store().path_for(capture_id)
    except Exception as exc:
        return _err(404, str(exc))
    return FileResponse(path, media_type="image/png")


@app.get("/api/captures/{capture_id}/thumb")
def api_capture_thumb(capture_id: str) -> Any:
    try:
        data = _capture_store().thumbnail(capture_id)
    except Exception as exc:
        return _err(404, str(exc))
    return Response(content=data, media_type="image/png")


async def _speculate(session: SessionState, capture_id: str) -> None:
    """Start extraction the moment a capture lands (spec 6.2), before asked."""
    try:
        from services.core.extract.pipeline import speculate
    except Exception:
        return
    try:
        await speculate(session_id=session.id, capture_id=capture_id, puzzle_type=session.puzzle_type)
    except Exception as exc:
        log.debug("speculative extraction failed: %s", exc)


# -- state ------------------------------------------------------------------


@app.get("/api/sessions/{session_id}/state")
def api_state(session_id: str) -> dict[str, Any]:
    s = _require(session_id)
    return {
        "state": s.state or s.pending_state,
        "confirmed": s.confirmed_at is not None,
        "verdicts": [v.model_dump(mode="json") for v in s.verdicts],
        "unresolved": s.unresolved,
        "puzzle_type": s.puzzle_type,
    }


@app.post("/api/sessions/{session_id}/state/confirm")
async def api_confirm(session_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    s = _require(session_id)
    from services.core.extract.schemas import flatten, unflatten

    base = s.state or s.pending_state or {}
    flat = flatten(base)
    for path, value in (body.get("patch") or {}).items():
        flat[path] = value
    s.state = unflatten(flat)
    s.confirmed_at = time.time()
    accepted = set(body.get("patch") or {})
    for v in s.verdicts:
        if v.path in accepted:
            v.status = "manual"
    s.unresolved = [p for p in s.unresolved if p not in accepted]
    sessions.save(s)
    await hub.send(session_id, {"type": "state", **api_state(session_id)})
    # Speculative optimization: the instant state is confirmed, start solving.
    if s.puzzle_type == "factory" and s.state:
        with contextlib.suppress(Exception):
            await _start_optimize(s, seconds=10.0)
    return {"ok": True}


@app.post("/api/sessions/{session_id}/state")
async def api_set_state(session_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Hand-entered state.  A verified solver with manual data entry is a
    usable tool (spec 2.1), so this path is first-class, not a fallback."""
    s = _require(session_id)
    s.state = body.get("state")
    s.config = body.get("config", s.config)
    s.confirmed_at = time.time()
    s.verdicts = []
    s.unresolved = []
    sessions.save(s)
    await hub.send(session_id, {"type": "state", **api_state(session_id)})
    return {"ok": True}


# -- rules ------------------------------------------------------------------


@app.get("/api/sessions/{session_id}/rules")
def api_rules(session_id: str) -> dict[str, Any]:
    s = _require(session_id)
    unresolved = list(s.rule_unknowns)
    if not unresolved:
        with contextlib.suppress(Exception):
            unresolved = _default_unknowns(s)
    return {"rules": s.rules, "unresolved": unresolved, "calibrated": s.calibrated}


def _default_unknowns(s: SessionState) -> list[dict[str, Any]]:
    from services.core.rules.compile import experiment_for  # lazy
    from services.core.rules.dsl import (
        BUILDER_FLAG_OPTIONS,
        FACTORY_FLAG_OPTIONS,
        BuilderRules,
        FactoryRuleState,
    )

    out: list[dict[str, Any]] = []
    if s.puzzle_type == "builder":
        rules = BuilderRules.model_validate(s.rules or {})
        for flag in rules.unresolved():
            out.append(
                {
                    "flag": flag,
                    "options": list(BUILDER_FLAG_OPTIONS[flag]),
                    "experiment": experiment_for(flag, BUILDER_FLAG_OPTIONS[flag]).model_dump(
                        mode="json"
                    ),
                }
            )
    else:
        rs = FactoryRuleState.model_validate(s.rules or {})
        for flag in rs.unresolved():
            out.append(
                {
                    "flag": flag,
                    "options": list(FACTORY_FLAG_OPTIONS[flag]),
                    "experiment": experiment_for(flag, FACTORY_FLAG_OPTIONS[flag]).model_dump(
                        mode="json"
                    ),
                }
            )
    return out


# -- solving ----------------------------------------------------------------


async def _start_optimize(s: SessionState, seconds: float) -> str:
    job_id = await manager.submit(
        "optimize_factory",
        {"state": s.state, "seconds": seconds, "flags": (s.rules or {}).get("flags", {})},
    )
    _job_owner[job_id] = s.id
    return job_id


@app.post("/api/solve/optimize")
async def api_optimize(body: dict[str, Any] = Body(...)) -> Any:
    s = _require(body["session_id"])
    if s.state is None:
        return _err(400, "no confirmed state")
    if not s.calibrated and not body.get("override_calibration"):
        # Calibration is a gate, not a feature (spec 7.5).  The override is
        # explicit and the UI must shout about it.
        return _err(
            412,
            "calibration gate: no recorded run has matched the simulator yet. "
            "Run calibrate_factory, or pass override_calibration=true.",
        )
    return {"job_id": await _start_optimize(s, float(body.get("seconds", 10.0)))}


@app.post("/api/solve/bound")
async def api_bound(body: dict[str, Any] = Body(...)) -> Any:
    s = _require(body["session_id"])
    if s.state is None:
        return _err(400, "no confirmed state")
    try:
        result = await manager.run("factory_upper_bound", {"state": s.state}, timeout_s=20)
    except Exception as exc:
        return _err(500, str(exc))
    s.bound = result
    sessions.save(s)
    return result


@app.post("/api/solve/simulate")
async def api_simulate(body: dict[str, Any] = Body(...)) -> Any:
    s = _require(body["session_id"])
    try:
        return await manager.run(
            "simulate_factory",
            {"state": s.state, "config": body["config"], "flags": (s.rules or {}).get("flags", {})},
            timeout_s=30,
        )
    except Exception as exc:
        return _err(500, str(exc))


@app.post("/api/solve/calibrate")
async def api_calibrate(body: dict[str, Any] = Body(...)) -> Any:
    s = _require(body["session_id"])
    try:
        result = await manager.run(
            "calibrate_factory",
            {"state": s.state, "config": body["config"], "observed": body["observed"]},
            timeout_s=60,
        )
    except Exception as exc:
        return _err(500, str(exc))
    s.calibration = result
    s.calibrated = bool(result.get("matched"))
    s.observed_money_by_hour = list(body["observed"])
    if result.get("resolved_flags"):
        s.rules = {**(s.rules or {}), "flags": result["resolved_flags"]}
    sessions.save(s)
    await hub.send(s.id, {"type": "calibration", "result": result})
    return result


@app.get("/api/sessions/{session_id}/chart")
async def api_chart(session_id: str) -> Any:
    """The three money-over-time lines: current config, optimizer best, and the
    observed game run.  Three lines on one chart is the most informative view in
    the app (spec 11), so it gets a first-class endpoint rather than being
    scavenged out of tool results.

    A line the session does not have comes back as ``null`` — never as zeros,
    which would read as a real, terrible run.
    """
    s = _require(session_id)
    out: dict[str, Any] = {
        "current": None,
        "best": None,
        "observed": s.observed_money_by_hour,
        "bound": (s.bound or {}).get("ceiling"),
        "horizon_hours": (s.state or {}).get("horizon_hours"),
    }
    if s.state is None or s.puzzle_type != "factory":
        return out

    flags = (s.rules or {}).get("flags", {})
    config = s.config
    if config is None:
        with contextlib.suppress(Exception):
            from services.solvers.factory.model import FactoryConfig, FactoryState

            config = FactoryConfig.from_state(FactoryState.model_validate(s.state)).model_dump(
                mode="json"
            )
    for key, cfg in (("current", config), ("best", (s.optimizer_best or {}).get("best"))):
        if not cfg:
            continue
        try:
            sim = await manager.run(
                "simulate_factory", {"state": s.state, "config": cfg, "flags": flags}, timeout_s=30
            )
            out[key] = sim.get("money_by_hour")
        except Exception as exc:
            log.debug("chart line %s unavailable: %s", key, exc)
    return out


@app.post("/api/solve/builder")
async def api_solve_builder(body: dict[str, Any] = Body(...)) -> Any:
    s = _require(body["session_id"])
    if s.state is None:
        return _err(400, "no confirmed state")
    job_id = await manager.submit(
        "solve_builder",
        {
            "puzzle": s.state,
            "objective": body.get("objective"),
            "budget_s": float(body.get("budget_s", 10.0)),
        },
    )
    _job_owner[job_id] = s.id
    return {"job_id": job_id}


@app.post("/api/solve/interpretations")
async def api_interpretations(body: dict[str, Any] = Body(...)) -> Any:
    s = _require(body["session_id"])
    job_id = await manager.submit(
        "solve_all_interpretations",
        {"puzzle": s.state, "budget_s": float(body.get("budget_s", 5.0))},
    )
    _job_owner[job_id] = s.id
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str) -> Any:
    job = manager.jobs.get(job_id)
    if job is None:
        return _err(404, "no such job")
    return {
        "id": job.id,
        "kind": job.kind,
        "done": job.done.is_set(),
        "error": job.error,
        "result": job.result,
        "best_so_far": manager.best_so_far(job_id),
    }


@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel(job_id: str, body: dict[str, Any] = Body(default={})) -> Any:
    await manager.cancel(job_id, hard=bool(body.get("hard")))
    return {"ok": True}


# -- leaderboard ------------------------------------------------------------


@app.get("/api/sessions/{session_id}/leaderboard")
def api_leaderboard(session_id: str) -> dict[str, Any]:
    s = _require(session_id)
    return {"entries": [e.model_dump(mode="json") for e in sorted(s.leaderboard, key=lambda e: -e.score)]}


@app.post("/api/sessions/{session_id}/leaderboard")
async def api_leaderboard_add(session_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    s = _require(session_id)
    entry = s.record_score(
        LeaderboardEntry(
            label=body.get("label", "tested"),
            score=float(body["score"]),
            config_summary=body.get("config_summary", ""),
            config=body.get("config"),
            source=body.get("source", "manual"),
        )
    )
    sessions.save(s)
    await hub.send(session_id, {"type": "leaderboard", **api_leaderboard(session_id)})
    return {"ok": True, "entry": entry.model_dump(mode="json")}


# -- settings ---------------------------------------------------------------


@app.get("/api/settings/models")
async def api_models() -> Any:
    try:
        from services.core.wiring import catalog_and_roles

        return await catalog_and_roles()
    except Exception as exc:
        return _err(503, f"model catalog unavailable: {exc}")


@app.post("/api/settings/models")
async def api_set_models(body: dict[str, Any] = Body(...)) -> Any:
    try:
        from services.core.wiring import save_roles

        save_roles(body["roles"])
    except Exception as exc:
        return _err(503, str(exc))
    return {"ok": True}


@app.post("/api/settings/key")
async def api_set_key(body: dict[str, Any] = Body(...)) -> Any:
    try:
        from services.core.settings.keys import storage_backend, store_key, validate_key

        info = await validate_key(body["key"])
        if not info.valid:
            return {"valid": False, "error": info.error}
        store_key(body["key"])
        return {
            "valid": True,
            "label": info.label,
            "remaining_credit": info.remaining_credit,
            "backend": storage_backend(),
        }
    except Exception as exc:
        return _err(503, f"key validation unavailable: {exc}")


@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    return {
        "ok": True,
        "worker_ready": manager.jobs is not None,
        "web_dist": WEB_DIST.exists(),
        "active_jobs": len(list(manager.active())),
    }


# -- websocket --------------------------------------------------------------


@app.websocket("/ws/{session_id}")
async def ws_endpoint(ws: WebSocket, session_id: str) -> None:
    await hub.connect(session_id, ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = msg.get("type")
            if kind == "chat":
                asyncio.create_task(_run_agent(session_id, msg.get("text", "")))
            elif kind == "cancel":
                await manager.cancel(msg.get("job_id", ""), hard=bool(msg.get("hard")))
            elif kind == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        hub.disconnect(session_id, ws)
    except Exception:
        hub.disconnect(session_id, ws)


async def _run_agent(session_id: str, text: str) -> None:
    s = sessions.get(session_id)
    if s is None:
        return
    try:
        from services.core.wiring import build_agent_session, event_to_ws
    except Exception as exc:
        await hub.send(session_id, {"type": "warning", "text": f"agent unavailable: {exc}"})
        return
    s.chat.append({"role": "user", "text": text, "at": time.time()})
    final = ""
    try:
        agent = await build_agent_session(s)
        async for event in agent.run(text):
            if event.kind == "tool_call_finished" and event.result is not None:
                s.tool_results.append({"tool": event.tool, "result": event.result})
            if event.kind == "done":
                final = event.text
            await hub.send(session_id, event_to_ws(event))
    except Exception as exc:
        await hub.send(session_id, {"type": "warning", "text": f"agent error: {exc}"})
        return
    # Spec 13: the agent must never narrate a number it made up.
    try:
        from services.core.agent.guard import unsourced_numbers

        offenders = unsourced_numbers(final, s.tool_results, [text])
        if offenders:
            log.warning("unsourced numbers in agent output: %s", offenders)
            await hub.send(
                session_id,
                {
                    "type": "provenance_warning",
                    "numbers": offenders,
                    "text": "these numbers do not appear in any tool result from this conversation",
                },
            )
    except Exception:
        pass
    s.chat.append({"role": "assistant", "text": final, "at": time.time()})
    s.cost_spent = getattr(agent, "spent", s.cost_spent)
    sessions.save(s)


# -- static web app ---------------------------------------------------------

if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> Any:  # pragma: no cover - trivial
        candidate = WEB_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run("services.core.main:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
