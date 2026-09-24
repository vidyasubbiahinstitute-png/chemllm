"""The orchestration loop: natural language in, tool-grounded chemistry out.

The model never computes chemistry itself. It plans, calls registered tools, reads the
returned values and writes the explanation. Everything numeric or structural in the
final answer has to have come through a tool call, and the run log records which one.
"""
from __future__ import annotations

import json
import os
import time

from .registry import ToolCall, anthropic_tools, dispatch

OUTDIR = os.environ.get("CHEMAGENT_OUTDIR", "outputs")

SYSTEM = """You are a chemistry agent for surfactant, carbohydrate-polymer and biopolymer work
in a home-care and personal-care R&D setting. You serve formulation scientists and chemists.

HOW YOU WORK
1. Resolve before you reason. Call resolve_input on every chemical the user names or draws,
   before anything else. Work from the SMILES it returns, never from a structure you recall.
2. Compute, never estimate. Molecular weight, logP, HLB, charge, CMC, DP, DS, linkage
   positions and similarity come from tools. If no tool returns a number, say the number is
   not available and name what would be needed to get it. Never fill a gap from memory.
3. Show the change. Any modification, reaction or derivatisation answer must include a
   before/after panel from draw_before_after and a compare_structures result.
4. Surface ambiguity. If a tool reports more than one reactive site, an unresolved name, an
   extended (non-canonical) parameter or a polymer assumption, put it in the answer. Do not
   quietly choose for the user on a question of chemical intent.
5. Export what is useful. When a structure is a deliverable, write a file with
   export_structure (.mol/.sdf for modelling, .pdb for 3D viewers) and name the file.
6. Polymers are specifications, not molecules. For carbohydrate polymers, state the repeat
   unit, linkage, DP of the exemplar you built, DS and every assumption the tool returned.
   Never present an oligomer exemplar as if it were the polymer.

HOW YOU ANSWER
Write for a chemist who is skim-reading. Lead with the answer. Then: what was built or
changed and why it matters chemically; the key numbers in a short table; the files produced;
then assumptions and caveats. No preamble, no restating the question.
"""


def run_agent(prompt: str, max_steps: int = 14, phase: int = 2,
              model: str | None = None, verbose: bool = True,
              log_name: str | None = None, host=None) -> dict:
    """Run one natural-language request to completion.

    `host` is the platform LLM client (anything exposing .llm and .reasoning_model).
    If omitted it is taken from the calling kernel's globals.
    """
    if host is None:
        import inspect
        frame = inspect.currentframe()
        while frame is not None and host is None:
            host = frame.f_globals.get("host")
            frame = frame.f_back
    if host is None:
        raise RuntimeError("no LLM client found: pass host=<client exposing .llm()>")
    model = model or host.reasoning_model()

    tools = anthropic_tools(max_phase=phase)
    if not tools:
        raise RuntimeError("the tool registry is empty -- import chemagent so the tool modules "
                           "register themselves before calling run_agent")
    messages = [{"role": "user", "content": prompt}]
    calls: list[ToolCall] = []
    t0 = time.time()
    final_text = ""
    steps = 0

    for steps in range(1, max_steps + 1):
        r = host.llm(messages=messages, tools=tools, system=SYSTEM, model=model,
                     max_tokens=4096, thinking={"type": "disabled"})
        if "error" in r:
            raise RuntimeError(f"LLM call failed: {r['error']}")
        content = r.get("content") or [{"type": "text", "text": r.get("text", "")}]
        uses = [b for b in content if b.get("type") == "tool_use"]
        for i, b in enumerate(uses):          # the client strips ids; re-mint them so the
            b.setdefault("id", f"call_{steps}_{i}")   # tool_result blocks can reference them
        messages.append({"role": "assistant", "content": content})

        if not uses:
            final_text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            break

        results = []
        for b in uses:
            call = dispatch(b["name"], b.get("input") or {})
            calls.append(call)
            if verbose:
                flag = "ok " if call.ok else "ERR"
                print(f"  [{flag}] {call.name}({json.dumps(call.arguments)[:110]}) "
                      f"{call.seconds:.2f}s")
                if not call.ok:
                    print(f"        -> {call.error}")
            results.append({"type": "tool_result", "tool_use_id": b["id"],
                            "content": call.to_model(),
                            **({"is_error": True} if not call.ok else {})})
        messages.append({"role": "user", "content": results})

    files = sorted({f for c in calls for f in c.files})
    run = {
        "prompt": prompt,
        "answer": final_text,
        "steps": steps,
        "seconds": round(time.time() - t0, 1),
        "n_tool_calls": len(calls),
        "n_failed_tool_calls": sum(1 for c in calls if not c.ok),
        "tools_used": [c.name for c in calls],
        "files": files,
        "model": model,
        "tool_log": [{"tool": c.name, "arguments": c.arguments, "ok": c.ok,
                      "error": c.error, "seconds": round(c.seconds, 3),
                      "result": c.result} for c in calls],
    }
    if log_name:
        os.makedirs(OUTDIR, exist_ok=True)
        path = os.path.join(OUTDIR, log_name)
        with open(path, "w") as fh:
            json.dump(run, fh, indent=2, default=str)
        run["log_file"] = path
    return run
