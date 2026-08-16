"""Tool schemas + dispatch for the strategist loop.

Two constraints shape this file:

1.  **Everything is trimmed before it re-enters context.**  A 24-hour,
    12-machine simulation has ~288 per-machine log rows; an optimizer archive
    can hold hundreds of configs.  Feeding those back costs seconds of clock
    per turn and pushes the actual answer out of the window.  Every trim is
    announced inside the payload under ``_trimmed`` so the model knows it is
    looking at a slice and can ask for more.
2.  **No server.**  The registry takes a :class:`SessionStore` at construction,
    so the FastAPI layer supplies the real one and tests supply
    :class:`InMemorySessionStore`.

Solvers are imported *lazily inside the handlers*: they are written by other
agents and may not exist yet.  A missing solver is a structured error, never an
ImportError crash mid-run.
"""

from __future__ import annotations

import copy
import inspect
import json
import time
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from services.core.llm.protocol import ToolSpec

# Trim caps.  Chosen so a full result stays under a few kB of context.
MAX_LOG_ROWS = 40
MAX_ARCHIVE = 5
MAX_BUILDS = 10
MAX_WARNINGS = 10
MAX_ACTIONS = 25
MAX_INTERPRETATIONS = 12


# ---------------------------------------------------------------------------
# Session state interface
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionStore(Protocol):
    """What the tools need from the server.  FastAPI implements this."""

    def list_captures(self, session_id: str) -> list[dict[str, Any]]: ...

    def get_capture(self, session_id: str, capture_id: str) -> dict[str, Any] | None: ...

    def get_state(self, session_id: str) -> dict[str, Any] | None: ...

    def set_state(self, session_id: str, state: dict[str, Any]) -> None: ...

    def puzzle_type(self, session_id: str) -> str | None: ...

    def get_rules(self, session_id: str, puzzle: str) -> dict[str, Any] | None: ...

    def set_rules(self, session_id: str, puzzle: str, rules: dict[str, Any]) -> None: ...

    def get_config(self, session_id: str) -> dict[str, Any] | None: ...

    def set_config(self, session_id: str, config: dict[str, Any]) -> None: ...


class InMemorySessionStore:
    """Reference implementation; also what the tests run against."""

    def __init__(self) -> None:
        self.captures: dict[str, list[dict[str, Any]]] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.types: dict[str, str] = {}
        self.rules: dict[tuple[str, str], dict[str, Any]] = {}
        self.configs: dict[str, dict[str, Any]] = {}

    # -- captures -------------------------------------------------------
    def add_capture(self, session_id: str, capture: dict[str, Any]) -> None:
        self.captures.setdefault(session_id, []).append(capture)

    def list_captures(self, session_id: str) -> list[dict[str, Any]]:
        return list(self.captures.get(session_id, []))

    def get_capture(self, session_id: str, capture_id: str) -> dict[str, Any] | None:
        for c in self.captures.get(session_id, []):
            if c.get("id") == capture_id:
                return c
        return None

    # -- state ----------------------------------------------------------
    def get_state(self, session_id: str) -> dict[str, Any] | None:
        return self.states.get(session_id)

    def set_state(self, session_id: str, state: dict[str, Any]) -> None:
        self.states[session_id] = state

    def puzzle_type(self, session_id: str) -> str | None:
        return self.types.get(session_id)

    def set_puzzle_type(self, session_id: str, puzzle_type: str) -> None:
        self.types[session_id] = puzzle_type

    # -- rules / config --------------------------------------------------
    def get_rules(self, session_id: str, puzzle: str) -> dict[str, Any] | None:
        return self.rules.get((session_id, puzzle))

    def set_rules(self, session_id: str, puzzle: str, rules: dict[str, Any]) -> None:
        self.rules[(session_id, puzzle)] = rules

    def get_config(self, session_id: str) -> dict[str, Any] | None:
        return self.configs.get(session_id)

    def set_config(self, session_id: str, config: dict[str, Any]) -> None:
        self.configs[session_id] = config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def _trim(items: Any, cap: int, label: str, trimmed: list[str]) -> list[Any]:
    seq = list(items or [])
    if len(seq) > cap:
        trimmed.append(f"{label}: showing {cap} of {len(seq)}")
        return [_jsonable(x) for x in seq[:cap]]
    return [_jsonable(x) for x in seq]


def _err(tool: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "tool": tool, "error": message, **extra}


def _missing_solver(tool: str, module: str, exc: Exception) -> dict[str, Any]:
    return _err(
        tool,
        f"solver module '{module}' is not available ({exc}). "
        "It is being written by another component; do not report numbers you "
        "do not have.",
        unavailable=module,
    )


def _pointer(doc: Any, path: str) -> tuple[Any, str]:
    """Resolve an RFC-6901 pointer to (container, final_token)."""
    tokens = [t.replace("~1", "/").replace("~0", "~") for t in path.split("/")[1:]]
    if not tokens:
        raise KeyError("empty pointer")
    cursor = doc
    for tok in tokens[:-1]:
        cursor = cursor[int(tok)] if isinstance(cursor, list) else cursor[tok]
    return cursor, tokens[-1]


def apply_json_patch(doc: dict[str, Any], ops: list[dict[str, Any]]) -> dict[str, Any]:
    """Minimal RFC-6902: add / replace / remove.  Operates on a copy."""
    out = copy.deepcopy(doc)
    for op in ops:
        kind = op.get("op")
        container, token = _pointer(out, op.get("path", ""))
        if kind in {"add", "replace"}:
            if isinstance(container, list):
                if token == "-":
                    container.append(op.get("value"))
                elif kind == "add":
                    container.insert(int(token), op.get("value"))
                else:
                    container[int(token)] = op.get("value")
            else:
                container[token] = op.get("value")
        elif kind == "remove":
            if isinstance(container, list):
                del container[int(token)]
            else:
                container.pop(token, None)
        else:
            raise ValueError(f"unsupported json patch op: {kind!r}")
    return out


def _as_factory_state(raw: Any) -> Any:
    from services.solvers.factory.model import FactoryState

    if raw is None:
        raise ValueError("no factory state available")
    return raw if isinstance(raw, FactoryState) else FactoryState.model_validate(raw)


def _as_factory_config(raw: Any, state: Any) -> Any:
    from services.solvers.factory.model import FactoryConfig

    if raw is None:
        return FactoryConfig.from_state(state)
    return raw if isinstance(raw, FactoryConfig) else FactoryConfig.model_validate(raw)


def _as_builder_puzzle(raw: Any) -> Any:
    from services.solvers.builder.model import BuilderPuzzle

    if raw is None:
        raise ValueError("no builder puzzle available")
    return raw if isinstance(raw, BuilderPuzzle) else BuilderPuzzle.model_validate(raw)


def _as_flags(raw: Any) -> Any:
    from services.core.rules.dsl import RuleFlags

    if raw is None:
        return RuleFlags()
    return raw if isinstance(raw, RuleFlags) else RuleFlags.model_validate(raw)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def _obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


_SESSION = {"type": "string", "description": "Session id. Omit to use the active session."}
_STATE = {
    "type": "object",
    "description": "Optional explicit state. Omit to use the session's confirmed state.",
    "additionalProperties": True,
}
_CONFIG = {
    "type": "object",
    "description": "Optional explicit config. Omit to use the session's current config.",
    "additionalProperties": True,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_captures",
        "description": "List screenshots captured in this session, newest last.",
        "parameters": _obj({"session_id": _SESSION}),
    },
    {
        "name": "extract_state",
        "description": (
            "Run the vision extraction jury over the named captures and return "
            "the merged state plus any disputed fields. Ask the user only about "
            "disputed fields."
        ),
        "parameters": _obj(
            {
                "capture_ids": {"type": "array", "items": {"type": "string"}},
                "puzzle_type": {"type": "string", "enum": ["factory", "builder"]},
                "session_id": _SESSION,
            },
            ["capture_ids", "puzzle_type"],
        ),
    },
    {
        "name": "get_state",
        "description": "Return the session's current confirmed state.",
        "parameters": _obj({"session_id": _SESSION}),
    },
    {
        "name": "patch_state",
        "description": (
            "Apply an RFC-6902 JSON patch to the confirmed state. Use this to "
            "record a correction the user gave you."
        ),
        "parameters": _obj(
            {
                "json_patch": {
                    "type": "array",
                    "description": "Ops with op/path/value.",
                    "items": _obj(
                        {
                            "op": {"type": "string", "enum": ["add", "replace", "remove"]},
                            "path": {"type": "string"},
                            "value": {},
                        },
                        ["op", "path"],
                    ),
                },
                "session_id": _SESSION,
            },
            ["json_patch"],
        ),
    },
    {
        "name": "get_rules",
        "description": (
            "Return the rule spec for this puzzle plus the flags that are still "
            "unresolved. Never assert a rule that is not in the resolved set."
        ),
        "parameters": _obj(
            {
                "puzzle": {"type": "string", "enum": ["factory", "builder"]},
                "session_id": _SESSION,
            },
            ["puzzle"],
        ),
    },
    {
        "name": "propose_experiment",
        "description": (
            "Return the smallest in-game observation that resolves one "
            "unresolved rule flag, executable in under fifteen seconds."
        ),
        "parameters": _obj(
            {
                "puzzle": {"type": "string", "enum": ["factory", "builder"]},
                "flag": {"type": "string"},
            },
            ["puzzle", "flag"],
        ),
    },
    {
        "name": "factory_upper_bound",
        "description": (
            "LP relaxation: profit ceiling, per-hour profit, the bottleneck "
            "machine and its dual. Fast. Call this before the optimizer."
        ),
        "parameters": _obj({"state": _STATE, "session_id": _SESSION}),
    },
    {
        "name": "simulate_factory",
        "description": (
            "Deterministic hour-by-hour simulation of one config. Returns "
            "money_by_hour, warnings and a trimmed per-machine log."
        ),
        "parameters": _obj(
            {
                "state": _STATE,
                "config": _CONFIG,
                "flags": {"type": "object", "additionalProperties": True},
                "session_id": _SESSION,
            }
        ),
    },
    {
        "name": "optimize_factory",
        "description": (
            "Anytime search for the best config. Returns ranked configs, the "
            "diff from current, the bound and the top warning."
        ),
        "parameters": _obj(
            {
                "seconds": {"type": "number", "description": "Search budget in seconds."},
                "state": _STATE,
                "session_id": _SESSION,
            },
            ["seconds"],
        ),
    },
    {
        "name": "calibrate_factory",
        "description": (
            "Match a recorded per-hour money series against the simulator to "
            "pin unstated rule flags. Returns per-hour diff and resolved flags."
        ),
        "parameters": _obj(
            {
                "observed": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Money observed at the end of each hour.",
                },
                "state": _STATE,
                "config": _CONFIG,
                "session_id": _SESSION,
            },
            ["observed"],
        ),
    },
    {
        "name": "solve_builder",
        "description": (
            "Solve the builder puzzle exactly. Returns counts, minimal builds, "
            "free parts and the binding obstacle."
        ),
        "parameters": _obj(
            {
                "objective": {
                    "type": "string",
                    "enum": ["count_valid", "enumerate", "max_passed", "min_weight"],
                },
                "puzzle": {"type": "object", "additionalProperties": True},
                "budget_s": {"type": "number"},
                "session_id": _SESSION,
            }
        ),
    },
    {
        "name": "solve_all_interpretations",
        "description": (
            "Solve under every reading of the unresolved rules. Returns "
            "consensus actions, divergence grouped by the driving flag, the "
            "highest-leverage unknown and the value of information."
        ),
        "parameters": _obj(
            {
                "puzzle": {"type": "string", "enum": ["factory", "builder"]},
                "budget_s": {"type": "number"},
                "session_id": _SESSION,
            },
            ["puzzle"],
        ),
    },
    {
        "name": "compare_configs",
        "description": "Structured diff between two configs plus the simulated delta.",
        "parameters": _obj(
            {
                "a": {"type": "object", "additionalProperties": True},
                "b": {"type": "object", "additionalProperties": True},
                "state": _STATE,
                "session_id": _SESSION,
            },
            ["a", "b"],
        ),
    },
]

TOOL_NAMES: tuple[str, ...] = tuple(t["name"] for t in TOOL_SCHEMAS)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

Handler = Callable[..., Any] | Callable[..., Awaitable[Any]]


class ToolRegistry:
    """Schemas + dispatch, bound to one session's state accessor."""

    def __init__(self, store: SessionStore, session_id: str = "default") -> None:
        self.store = store
        self.session_id = session_id
        self.calls: list[dict[str, Any]] = []
        self._handlers: dict[str, Handler] = {
            "list_captures": self.list_captures,
            "extract_state": self.extract_state,
            "get_state": self.get_state,
            "patch_state": self.patch_state,
            "get_rules": self.get_rules,
            "propose_experiment": self.propose_experiment,
            "factory_upper_bound": self.factory_upper_bound,
            "simulate_factory": self.simulate_factory,
            "optimize_factory": self.optimize_factory,
            "calibrate_factory": self.calibrate_factory,
            "solve_builder": self.solve_builder,
            "solve_all_interpretations": self.solve_all_interpretations,
            "compare_configs": self.compare_configs,
        }

    # -- wire ------------------------------------------------------------

    def schemas(self) -> list[dict[str, Any]]:
        """The OpenAI-compatible ``tools`` array."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in TOOL_SCHEMAS
        ]

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name=t["name"], description=t["description"], parameters=t["parameters"])
            for t in TOOL_SCHEMAS
        ]

    @property
    def names(self) -> tuple[str, ...]:
        return TOOL_NAMES

    async def dispatch(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run one tool.  Always returns a JSON-serializable dict."""
        args = dict(arguments or {})
        started = time.perf_counter()
        handler = self._handlers.get(name)
        if handler is None:
            return _err(name, f"unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}")
        try:
            result = handler(**args)
            if inspect.isawaitable(result):
                result = await result
        except TypeError as exc:
            result = _err(name, f"bad arguments: {exc}")
        except Exception as exc:
            result = _err(name, f"{type(exc).__name__}: {exc}")
        if not isinstance(result, dict):
            result = {"ok": True, "result": _jsonable(result)}
        result.setdefault("ok", "error" not in result)
        result.setdefault("tool", name)
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        self.calls.append({"name": name, "arguments": args, "result": result})
        return result

    # -- session helpers -------------------------------------------------

    def _sid(self, session_id: str | None) -> str:
        return session_id or self.session_id

    def _state(self, state: Any, session_id: str | None) -> Any:
        return state if state is not None else self.store.get_state(self._sid(session_id))

    def _config(self, config: Any, session_id: str | None) -> Any:
        return config if config is not None else self.store.get_config(self._sid(session_id))

    # -- tools -----------------------------------------------------------

    def list_captures(self, session_id: str | None = None) -> dict[str, Any]:
        captures = self.store.list_captures(self._sid(session_id))
        trimmed: list[str] = []
        rows = [
            {k: v for k, v in c.items() if k not in {"data_url", "png", "bytes"}}
            for c in captures
        ]
        return {
            "captures": _trim(rows, 50, "captures", trimmed),
            "count": len(captures),
            "_trimmed": trimmed,
            "note": "image payloads are omitted; pass capture ids to extract_state",
        }

    async def extract_state(
        self,
        capture_ids: list[str],
        puzzle_type: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        sid = self._sid(session_id)
        try:
            from services.core.wiring import run_extraction  # type: ignore
        except ImportError as exc:
            return _err(
                "extract_state",
                f"extraction pipeline not available ({exc}). "
                "Ask the user to confirm the state manually, or use get_state.",
                unavailable="services.core.wiring",
            )
        missing = [cid for cid in capture_ids if self.store.get_capture(sid, cid) is None]
        if missing:
            return _err("extract_state", f"unknown capture ids: {', '.join(missing)}")
        try:
            result = await run_extraction(sid, list(capture_ids), puzzle_type)
        except Exception as exc:
            return _err("extract_state", f"{type(exc).__name__}: {exc}")

        # Disputed fields are the whole point of the payload: the model must ask
        # about those and nothing else.  The merged state stays out of context —
        # it is already in the session, and get_state will fetch it on demand.
        state = result.factory_state or result.builder_puzzle
        trimmed: list[str] = []
        return {
            "ok": True,
            "puzzle_type": result.puzzle_type,
            "disputed": _trim(result.disputed, 40, "disputed", trimmed),
            "unresolved": _trim(result.unresolved, 40, "unresolved", trimmed),
            "auto_confirmed_count": len(result.auto_confirmed),
            "state_available": state is not None,
            "elapsed_ms": round(result.elapsed_ms, 1),
            "used_delta": result.used_delta,
            "errors": result.errors,
            "trimmed": trimmed,
        }

    def get_state(self, session_id: str | None = None) -> dict[str, Any]:
        sid = self._sid(session_id)
        state = self.store.get_state(sid)
        return {
            "ok": state is not None,
            "state": _jsonable(state),
            "puzzle_type": self.store.puzzle_type(sid),
            "error": None if state is not None else "no confirmed state in this session yet",
        }

    def patch_state(
        self, json_patch: list[dict[str, Any]], session_id: str | None = None
    ) -> dict[str, Any]:
        sid = self._sid(session_id)
        state = self.store.get_state(sid)
        if state is None:
            return _err("patch_state", "no state to patch")
        updated = apply_json_patch(_jsonable(state), list(json_patch))
        self.store.set_state(sid, updated)
        return {"ok": True, "applied": len(json_patch), "state": updated}

    def get_rules(self, puzzle: str, session_id: str | None = None) -> dict[str, Any]:
        from services.core.rules.compile import factory_unknowns, unknowns_for
        from services.core.rules.dsl import BuilderRules, FactoryRuleState

        sid = self._sid(session_id)
        stored = self.store.get_rules(sid, puzzle) or {}
        if puzzle == "builder":
            rules = BuilderRules.model_validate(stored.get("rules", stored) or {})
            unknowns = unknowns_for(rules)
            spec = rules.model_dump()
        else:
            state = FactoryRuleState.model_validate(stored.get("flag_state", stored) or {})
            unknowns = factory_unknowns(state)
            spec = state.model_dump()
        return {
            "ok": True,
            "puzzle": puzzle,
            "rule_spec": _jsonable(spec),
            "unresolved_flags": [u.flag for u in unknowns],
            "unknowns": [
                {
                    "flag": u.flag,
                    "options": _jsonable(u.options),
                    "reason": u.reason,
                    "experiment": _jsonable(u.experiment),
                }
                for u in unknowns
            ],
        }

    def propose_experiment(self, puzzle: str, flag: str) -> dict[str, Any]:
        from services.core.rules.compile import ALL_FLAG_OPTIONS, experiment_for
        from services.core.rules.dsl import BUILDER_FLAG_OPTIONS, FACTORY_FLAG_OPTIONS

        table = BUILDER_FLAG_OPTIONS if puzzle == "builder" else FACTORY_FLAG_OPTIONS
        options = table.get(flag) or ALL_FLAG_OPTIONS.get(flag)
        if options is None:
            return _err(
                "propose_experiment",
                f"unknown flag {flag!r} for {puzzle}; known: {', '.join(sorted(table))}",
            )
        return {"ok": True, "experiment": _jsonable(experiment_for(flag, list(options)))}

    def factory_upper_bound(
        self, state: Any = None, session_id: str | None = None, flags: Any = None
    ) -> dict[str, Any]:
        try:
            from services.solvers.factory.bounds import upper_bound
        except ImportError as exc:
            return _missing_solver("factory_upper_bound", "services.solvers.factory.bounds", exc)
        st = _as_factory_state(self._state(state, session_id))
        bound = upper_bound(st, _as_flags(flags))
        trimmed: list[str] = []
        return {
            "ok": True,
            "ceiling": _jsonable(getattr(bound, "ceiling", None)),
            "per_hour_profit": _jsonable(getattr(bound, "per_hour_profit", None)),
            "bottleneck_machine": getattr(bound, "bottleneck_machine", None),
            "bottleneck_reason": getattr(bound, "bottleneck_reason", None),
            "duals": _jsonable(getattr(bound, "duals", None)),
            "throughput": _jsonable(getattr(bound, "throughput", None)),
            "notes": _trim(getattr(bound, "notes", None), 8, "notes", trimmed),
            "_trimmed": trimmed,
        }

    def simulate_factory(
        self,
        state: Any = None,
        config: Any = None,
        flags: Any = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            from services.solvers.factory.sim import simulate
        except ImportError as exc:
            return _missing_solver("simulate_factory", "services.solvers.factory.sim", exc)
        st = _as_factory_state(self._state(state, session_id))
        cfg = _as_factory_config(self._config(config, session_id), st)
        result = simulate(st, cfg, _as_flags(flags))
        trimmed: list[str] = []
        return {
            "ok": True,
            "money_by_hour": _jsonable(getattr(result, "money_by_hour", [])),
            "final_money": _jsonable(getattr(result, "final_money", None)),
            "total_revenue": _jsonable(getattr(result, "total_revenue", None)),
            "total_cost": _jsonable(getattr(result, "total_cost", None)),
            "warnings": _trim(getattr(result, "warnings", None), MAX_WARNINGS, "warnings", trimmed),
            "warning_count": len(getattr(result, "warnings", None) or []),
            "per_machine_log": _trim(
                getattr(result, "per_machine_log", None), MAX_LOG_ROWS, "per_machine_log", trimmed
            ),
            "ending_storage": _jsonable(getattr(result, "ending_storage", {})),
            "_trimmed": trimmed,
        }

    def optimize_factory(
        self,
        seconds: float,
        state: Any = None,
        session_id: str | None = None,
        flags: Any = None,
    ) -> dict[str, Any]:
        try:
            from services.solvers.factory.optimize import optimize
        except ImportError as exc:
            return _missing_solver("optimize_factory", "services.solvers.factory.optimize", exc)
        sid = self._sid(session_id)
        st = _as_factory_state(self._state(state, session_id))
        seed = self.store.get_config(sid)
        seed_cfg = _as_factory_config(seed, st) if seed is not None else None
        result = optimize(st, float(seconds), _as_flags(flags), seed_config=seed_cfg)
        trimmed: list[str] = []
        archive = getattr(result, "archive", None) or []
        return {
            "ok": True,
            "best_value": _jsonable(getattr(result, "best_value", None)),
            "baseline_value": _jsonable(getattr(result, "baseline_value", None)),
            "delta": _delta(
                getattr(result, "best_value", None), getattr(result, "baseline_value", None)
            ),
            "bound": _jsonable(getattr(result, "bound", None)),
            "actions": _trim(getattr(result, "actions", None), MAX_ACTIONS, "actions", trimmed),
            "ranked_configs": _trim(archive, MAX_ARCHIVE, "archive", trimmed),
            "archive_size": len(archive),
            "top_warning": _jsonable(getattr(result, "top_warning", None)),
            "endgame": _jsonable(getattr(result, "endgame", None)),
            "elapsed_s": _jsonable(getattr(result, "elapsed_s", None)),
            "iterations": _jsonable(getattr(result, "iterations", None)),
            "best": _jsonable(getattr(result, "best", None)),
            "_trimmed": trimmed,
        }

    def calibrate_factory(
        self,
        observed: list[float],
        state: Any = None,
        config: Any = None,
        session_id: str | None = None,
        flags: Any = None,
    ) -> dict[str, Any]:
        try:
            from services.solvers.factory.calibrate import calibrate
        except ImportError as exc:
            return _missing_solver("calibrate_factory", "services.solvers.factory.calibrate", exc)
        st = _as_factory_state(self._state(state, session_id))
        cfg = _as_factory_config(self._config(config, session_id), st)
        result = calibrate(st, cfg, [float(x) for x in observed], flags)
        trimmed: list[str] = []
        return {
            "ok": True,
            "matched": _jsonable(getattr(result, "matched", None)),
            "per_hour": _jsonable(getattr(result, "per_hour", None)),
            "resolved_flags": _jsonable(getattr(result, "resolved_flags", None)),
            "candidates": _trim(
                getattr(result, "candidates", None), MAX_ARCHIVE, "candidates", trimmed
            ),
            "discriminating_hours": _jsonable(getattr(result, "discriminating_hours", None)),
            "message": getattr(result, "message", ""),
            "_trimmed": trimmed,
        }

    def solve_builder(
        self,
        objective: str | None = None,
        puzzle: Any = None,
        budget_s: float = 5.0,
        session_id: str | None = None,
        max_builds: int = MAX_BUILDS,
    ) -> dict[str, Any]:
        try:
            from services.solvers.builder.solve import solve_builder as _solve
        except ImportError as exc:
            return _missing_solver("solve_builder", "services.solvers.builder.solve", exc)
        pz = _as_builder_puzzle(self._state(puzzle, session_id))
        report = _solve(
            puzzle=pz, objective=objective, budget_s=float(budget_s), max_builds=int(max_builds)
        )
        trimmed: list[str] = []
        return {
            "ok": True,
            "total_valid": _jsonable(getattr(report, "total_valid", None)),
            "binding_obstacle": getattr(report, "binding_obstacle", None),
            "binding_obstacle_eliminated": _jsonable(
                getattr(report, "binding_obstacle_eliminated", None)
            ),
            "obstacle_elimination": _jsonable(getattr(report, "obstacle_elimination", {})),
            "minimal_builds": _trim(
                getattr(report, "minimal_builds", None), MAX_BUILDS, "minimal_builds", trimmed
            ),
            "builds": _trim(getattr(report, "builds", None), MAX_BUILDS, "builds", trimmed),
            "free_parts": _jsonable(getattr(report, "free_parts", [])),
            "free_multiplier": _jsonable(getattr(report, "free_multiplier", 1)),
            "weight_slack": _jsonable(getattr(report, "weight_slack", None)),
            "best_build": _jsonable(getattr(report, "best_build", None)),
            "exact": _jsonable(getattr(report, "exact", None)),
            "method": getattr(report, "method", ""),
            "warnings": _trim(getattr(report, "warnings", None), MAX_WARNINGS, "warnings", trimmed),
            "_trimmed": trimmed,
        }

    def solve_all_interpretations(
        self, puzzle: str, budget_s: float = 5.0, session_id: str | None = None
    ) -> dict[str, Any]:
        from services.core.rules import resolve as R

        sid = self._sid(session_id)
        raw_state = self.store.get_state(sid)
        if puzzle == "builder":
            pz = _as_builder_puzzle(raw_state)
            result = R.solve_all_interpretations(pz.rules, pz, float(budget_s))
        else:
            st = _as_factory_state(raw_state)
            cfg = _as_factory_config(self.store.get_config(sid), st)
            stored = self.store.get_rules(sid, "factory") or {}
            from services.core.rules.dsl import FactoryRuleState

            flag_state = FactoryRuleState.model_validate(stored.get("flag_state", stored) or {})
            result = R.solve_all_interpretations_factory(st, cfg, flag_state, float(budget_s))
        trimmed: list[str] = []
        return {
            "ok": True,
            "consensus_actions": _trim(
                result.consensus_actions, MAX_ACTIONS, "consensus_actions", trimmed
            ),
            "divergent_actions": _jsonable(result.divergent_actions),
            "highest_leverage_unknown": _jsonable(result.highest_leverage_unknown),
            "value_of_information": result.value_of_information,
            "combinations_tried": result.combinations_tried,
            "truncated": result.truncated,
            "notes": result.notes,
            "interpretations": _trim(
                result.interpretations, MAX_INTERPRETATIONS, "interpretations", trimmed
            ),
            "_trimmed": trimmed,
        }

    def compare_configs(
        self,
        a: Any,
        b: Any,
        state: Any = None,
        session_id: str | None = None,
        flags: Any = None,
    ) -> dict[str, Any]:
        try:
            from services.solvers.factory.optimize import config_diff
        except ImportError as exc:
            return _missing_solver("compare_configs", "services.solvers.factory.optimize", exc)
        st = _as_factory_state(self._state(state, session_id))
        ca, cb = _as_factory_config(a, st), _as_factory_config(b, st)
        diff = config_diff(ca, cb)
        payload: dict[str, Any] = {"ok": True, "diff": _jsonable(diff), "changes": len(diff or [])}
        try:
            from services.solvers.factory.sim import simulate
        except ImportError as exc:
            payload["simulated_delta"] = None
            payload["note"] = f"simulator unavailable ({exc}); diff only"
            return payload
        f = _as_flags(flags)
        ra, rb = simulate(st, ca, f), simulate(st, cb, f)
        payload["a_final_money"] = _jsonable(getattr(ra, "final_money", None))
        payload["b_final_money"] = _jsonable(getattr(rb, "final_money", None))
        payload["simulated_delta"] = _delta(
            getattr(rb, "final_money", None), getattr(ra, "final_money", None)
        )
        payload["a_warning_count"] = len(getattr(ra, "warnings", None) or [])
        payload["b_warning_count"] = len(getattr(rb, "warnings", None) or [])
        return payload


def _delta(new: Any, old: Any) -> float | None:
    try:
        return float(new) - float(old)
    except (TypeError, ValueError):
        return None


def tools_json() -> str:
    """The tools array as a JSON string (handy for debugging a request body)."""
    return json.dumps(
        [
            {"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}}
            for t in TOOL_SCHEMAS
        ],
        indent=2,
    )
