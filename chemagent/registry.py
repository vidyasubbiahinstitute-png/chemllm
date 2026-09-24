"""Tool registry: the single contract between the LLM and the cheminformatics layer.

Every capability the agent can use is registered here with a JSON Schema.
The same registry emits (a) Anthropic tool schemas for the model and
(b) a dispatch table for deterministic execution.  Nothing the agent can do
exists outside this registry -- that is what makes the workflow auditable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

_REGISTRY: dict[str, "Tool"] = {}


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]
    category: str = "general"
    phase: int = 1
    writes_files: bool = False

    def anthropic_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def tool(name: str, description: str, schema: dict, category: str = "general",
         phase: int = 1, writes_files: bool = False):
    """Decorator registering a python function as an LLM-callable tool."""
    def deco(fn):
        _REGISTRY[name] = Tool(name=name, description=description,
                               input_schema=schema, fn=fn, category=category,
                               phase=phase, writes_files=writes_files)
        return fn
    return deco


def get(name: str) -> Tool:
    if name not in _REGISTRY:
        raise KeyError(f"unknown tool {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_tools(max_phase: int = 2) -> list[Tool]:
    return [t for t in _REGISTRY.values() if t.phase <= max_phase]


def anthropic_tools(max_phase: int = 2) -> list[dict]:
    return [t.anthropic_schema() for t in all_tools(max_phase)]


@dataclass
class ToolCall:
    """One audited tool invocation -- the unit of the provenance log."""
    name: str
    arguments: dict
    ok: bool
    result: Any = None
    error: str | None = None
    seconds: float = 0.0
    files: list[str] = field(default_factory=list)

    def to_model(self, max_chars: int = 6000) -> str:
        """Serialise the result for return to the model."""
        if not self.ok:
            return json.dumps({"ok": False, "error": self.error})
        s = json.dumps({"ok": True, "result": self.result}, default=str)
        if len(s) > max_chars:
            s = s[:max_chars] + ' ...[truncated; full result kept in the run log]'
        return s


def dispatch(name: str, arguments: dict) -> ToolCall:
    """Execute a tool with full error capture. Errors are returned, never raised:
    the agent must be able to see a failure and recover from it."""
    t0 = time.time()
    dropped: list[str] = []
    try:
        t = get(name)
        allowed = set(t.input_schema.get("properties", {}))
        if allowed:
            dropped = [k for k in arguments if k not in allowed]
            arguments = {k: v for k, v in arguments.items() if k in allowed}
        res = t.fn(**arguments)
    except Exception as exc:  # noqa: BLE001 - deliberate: surface to the model
        return ToolCall(name=name, arguments=arguments, ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                        seconds=time.time() - t0)
    files: list[str] = []
    if isinstance(res, dict):
        for k in ("file", "files", "png", "svg", "path", "paths"):
            v = res.get(k)
            if isinstance(v, str):
                files.append(v)
            elif isinstance(v, list):
                files.extend([x for x in v if isinstance(x, str)])
    if dropped and isinstance(res, dict):
        res = {**res, "ignored_arguments": dropped}
    return ToolCall(name=name, arguments=arguments, ok=True, result=res,
                    seconds=time.time() - t0, files=sorted(set(files)))


def registry_table() -> list[dict]:
    """Machine-readable description of the whole tool surface (for docs/tests)."""
    rows = []
    for t in _REGISTRY.values():
        props = t.input_schema.get("properties", {})
        req = set(t.input_schema.get("required", []))
        rows.append({
            "tool": t.name,
            "phase": t.phase,
            "category": t.category,
            "inputs": ", ".join(f"{k}{'' if k in req else '?'}" for k in props),
            "writes_files": t.writes_files,
            "description": t.description.strip().split("\n")[0],
        })
    return sorted(rows, key=lambda r: (r["phase"], r["category"], r["tool"]))
