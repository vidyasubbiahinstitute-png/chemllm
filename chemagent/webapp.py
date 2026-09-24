"""HTML front end for the chemistry agent PoC.

    python -m chemagent.webapp                 # http://127.0.0.1:8000
    python -m chemagent.webapp --port 9000 --host 0.0.0.0

One self-contained HTML page (`chemagent/static/index.html` — no build step, no framework,
no CDN) over a small JSON API. The chemistry runs server-side because it is RDKit; the page
holds no chemical knowledge of its own, only layout.

The same discipline as the rest of the PoC applies at this layer: every endpoint dispatches
through `chemagent.registry`, returns the tool's own values unaltered, and passes the tools'
`caveats` / `assumptions` / `note` / `warning` fields through to the page so a withheld number
stays visible.

Embedding the page in an existing site: serve this app behind your own path and iframe `/`,
or call the JSON endpoints in §API directly from your own markup — every panel is one POST
with a JSON body and a JSON reply.
"""
from __future__ import annotations

import argparse
import base64
import os
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chemagent  # noqa: F401  -- import side-effects populate the tool registry
from chemagent import registry
from chemagent.compare import compare_series, compare_structures
from chemagent.descriptors import functional_groups, molecular_descriptors, surfactant_profile
from chemagent.export import export_dataset, export_structure
from chemagent.polymer import (MONOMERS, NAMED_POLYMERS, SUBSTITUENTS, analyse_polysaccharide,
                               build_polysaccharide)
from chemagent.registry import registry_table
from chemagent.render import draw_before_after, draw_structure
from chemagent.resolve import LEXICON, resolve_input
from chemagent.transform import HEAD_TEMPLATES, REACTIONS, apply_reaction, build_surfactant

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
OUTDIR = os.path.abspath(os.environ.get("CHEMAGENT_OUTDIR", "outputs"))
os.makedirs(OUTDIR, exist_ok=True)

app = FastAPI(title="Chemistry agent PoC", docs_url="/api/docs", openapi_url="/api/openapi.json")


# ------------------------------------------------------------------ shared helpers
def _uid(stem: str, ext: str) -> str:
    return f"{stem}_{uuid.uuid4().hex[:8]}.{ext}"


def _data_uri(path: str) -> str | None:
    """Inline a rendered PNG so the page needs no second request and no file route."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")


def _rows(d: dict, keys: list[str] | None = None, skip: tuple = ()) -> list[list[str]]:
    out = []
    for k, v in d.items():
        if k in skip or (keys and k not in keys):
            continue
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out.append([f"{k}.{k2}", "" if v2 is None else str(v2)])
        elif isinstance(v, list):
            out.append([k, ", ".join(str(x) for x in v) if v else "-"])
        else:
            out.append([k, "" if v is None else str(v)])
    return out


def _caveats(*results: dict) -> list[str]:
    out: list[str] = []
    for r in results:
        for key in ("caveats", "assumptions", "note", "notes", "warning", "ambiguity_note",
                    "ignored_arguments"):
            v = r.get(key)
            if not v:
                continue
            for it in (v if isinstance(v, list) else [v]):
                if str(it) not in out:
                    out.append(str(it))
    return out


def _downloads(paths: list[str]) -> list[dict]:
    return [{"name": os.path.basename(p), "url": "/files/" + os.path.basename(p)}
            for p in paths if p and os.path.exists(p)]


def _error(exc: Exception) -> JSONResponse:
    """Tool failures are data, not stack traces -- the page renders them in place."""
    return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=200)


# ------------------------------------------------------------------ request bodies
class CharacteriseIn(BaseModel):
    query: str
    use_network: bool = True
    want_3d: bool = False


class ModifyIn(BaseModel):
    start: str
    reaction: str
    co_reactant: str = ""
    repeat: int = 1
    counterion: str = "Na"


class BuildIn(BaseModel):
    tail_carbons: int = 12
    head_group: str = "sulfate"
    n_eo: int = 0
    counterion: str = ""
    branching: str = "linear"
    unsaturation: bool = False


class SeriesIn(BaseModel):
    head_group: str = "sulfate"
    chains: str = "10 12 14 16"
    n_eo: int = 0
    counterion: str = ""


class PolymerIn(BaseModel):
    monomer: str = "b_glcp"
    linkage: int = 4
    dp: int = 6
    target_dp: int = 0
    branch_monomer: str = ""
    branch_linkage: int = 6
    branch_every: int = 0
    substituent: str = ""
    ds: float = 0.0
    positions: str = ""


class AnalyseIn(BaseModel):
    smiles: str


class AgentIn(BaseModel):
    message: str
    phase2: bool = True


# ------------------------------------------------------------------ metadata
@app.get("/api/meta")
def meta() -> dict[str, Any]:
    """Everything the page needs to populate its controls. The page hardcodes no chemistry."""
    return {
        "head_groups": sorted(HEAD_TEMPLATES),
        "reactions": [{"key": k, "description": v["description"], "reagents": v["reagents"],
                       "product_class": v["product_class"],
                       "two_component": "{co}" in v["smarts"] or v["smarts"].count(">>") and
                                        v["smarts"].split(">>")[0].count(".") > 0}
                      for k, v in sorted(REACTIONS.items())],
        "monomers": [{"key": k, "name": v["name"], "short": v["short"]}
                     for k, v in sorted(MONOMERS.items())],
        "substituents": [{"key": k, "name": v.get("name", k),
                          "typical_ds": v.get("typical_ds", "")}
                         for k, v in sorted(SUBSTITUENTS.items())],
        "named_polymers": [{"key": k, "name": v["name"], "summary": v["summary"],
                            "build": v.get("build"), "notes": v.get("structural_notes", []),
                            "relevance": v.get("formulation_relevance", "")}
                           for k, v in sorted(NAMED_POLYMERS.items())],
        "lexicon_examples": sorted({v["name"] for v in LEXICON.values()})[:60],
        "registry": registry_table(),
        "counts": {"tools": len(registry.all_tools()), "lexicon": len(LEXICON),
                   "reactions": len(REACTIONS), "head_groups": len(HEAD_TEMPLATES),
                   "monomers": len(MONOMERS), "named_polymers": len(NAMED_POLYMERS),
                   "substituents": len(SUBSTITUENTS)},
        "agent_available": _get_llm_client() is not None,
    }


# ------------------------------------------------------------------ W1 characterise
@app.post("/api/characterise")
def api_characterise(body: CharacteriseIn):
    try:
        res = resolve_input(query=body.query.strip(), use_network=body.use_network)
        if not res.get("resolved"):
            return {"ok": False, "unresolved": True, "reason": res["reason"],
                    "suggestions": res.get("suggestions") or [], "next_step": res["next_step"]}
        smi = res["smiles"]
        png = os.path.join(OUTDIR, _uid("structure", "png"))
        draw_structure(smiles=[smi], legends=[res.get("preferred_name") or smi],
                       filename=os.path.basename(png))
        desc = molecular_descriptors(smiles=smi)
        prof = surfactant_profile(smiles=smi)
        fgs = functional_groups(smiles=smi)
        files = [export_structure(smiles=smi, fmt=f, filename=_uid("structure", f))["file"]
                 for f in (["mol", "sdf", "pdb"] if body.want_3d else ["mol", "sdf"])]
        return {"ok": True, "image": _data_uri(png),
                "title": res.get("preferred_name") or "Resolved structure",
                "summary": [["resolved via", res["source"] + (f" (CID {res['pubchem_cid']})"
                                                             if res.get("pubchem_cid") else "")],
                            ["SMILES", smi], ["InChIKey", res["inchikey"]],
                            ["lexicon class", res.get("class") or "not in the curated lexicon"],
                            ["charge class", prof["charge_class"]],
                            ["head group", prof["head_group"]],
                            ["tail carbons", prof["hydrophobic_tail_carbons"]],
                            ["ethoxylate units", prof["ethoxylate_units"]]],
                "descriptors": _rows(desc, skip=("smiles",)) +
                               [["functional groups", ", ".join(f"{k} x{v}" for k, v in
                                                                fgs["groups"].items()) or "-"]],
                "profile": _rows(prof, keys=["hlb_davies", "hlb_griffin", "counterions"]) +
                           _rows({"packing": prof["packing"], "cmc": prof["cmc_estimate"]}),
                "caveats": _caveats(prof, res), "files": _downloads(files)}
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ------------------------------------------------------------------ W2 modify
@app.post("/api/modify")
def api_modify(body: ModifyIn):
    try:
        res = resolve_input(query=body.start.strip())
        if not res.get("resolved"):
            return {"ok": False, "unresolved": True, "reason": res["reason"],
                    "suggestions": res.get("suggestions") or [], "next_step": res["next_step"]}
        co = None
        if body.co_reactant.strip():
            cres = resolve_input(query=body.co_reactant.strip())
            if not cres.get("resolved"):
                return {"ok": False, "unresolved": True,
                        "reason": "co-reactant: " + cres["reason"],
                        "suggestions": cres.get("suggestions") or [],
                        "next_step": cres["next_step"]}
            co = cres["smiles"]
        rxn = apply_reaction(smiles=res["smiles"], reaction=body.reaction,
                             co_reactant_smiles=co, repeat=int(body.repeat),
                             counterion=body.counterion or "none")
        png = os.path.join(OUTDIR, _uid("before_after", "png"))
        panel = draw_before_after(smiles_before=rxn["smiles_before"],
                                  smiles_after=rxn["smiles_after"],
                                  label_before=res.get("preferred_name") or "starting material",
                                  label_after=rxn["product_class"] or "product",
                                  filename=os.path.basename(png))
        cmp_ = compare_structures(smiles_a=rxn["smiles_before"], smiles_b=rxn["smiles_after"])
        return {"ok": True, "image": _data_uri(png),
                "title": body.reaction.replace("_", " "),
                "description": rxn["description"],
                "summary": [["reagents implied", rxn["reagents"]],
                            ["product class", rxn["product_class"]],
                            ["before", f"{rxn['smiles_before']}  (MW {rxn['mw_before']})"],
                            ["after", f"{rxn['smiles_after']}  (MW {rxn['mw_after']}, "
                                      f"{rxn['formula_after']})"],
                            ["reactive sites matched", rxn["distinct_products_first_step"]]],
                "ambiguous": bool(rxn["regiochemical_ambiguity"]),
                "comparison": [["Tanimoto (ECFP4)", cmp_["tanimoto_ecfp4"]],
                               ["Tanimoto (MACCS)", cmp_["tanimoto_maccs"]],
                               ["MCS atoms", cmp_["mcs_atoms"]],
                               ["conserved fraction", panel["conserved_fraction"]],
                               ["groups gained", ", ".join(f"{k} x{v}" for k, v in
                                                           cmp_["functional_groups_gained"].items()) or "-"],
                               ["groups lost", ", ".join(f"{k} x{v}" for k, v in
                                                         cmp_["functional_groups_lost"].items()) or "-"]] +
                              [[f"delta {k}", v] for k, v in
                               cmp_["descriptor_delta_b_minus_a"].items()],
                "caveats": _caveats(rxn)}
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ------------------------------------------------------------------ W3 build
@app.post("/api/build")
def api_build(body: BuildIn):
    try:
        spec = build_surfactant(tail_carbons=int(body.tail_carbons), head_group=body.head_group,
                                n_eo=int(body.n_eo), counterion=body.counterion or None,
                                unsaturation=1 if body.unsaturation else 0,
                                branching=body.branching)
        smi = spec["smiles"]
        label = (f"C{body.tail_carbons} {body.head_group.replace('_', ' ')}"
                 + (f", {body.n_eo} EO" if body.n_eo else ""))
        png = os.path.join(OUTDIR, _uid("built", "png"))
        draw_structure(smiles=[smi], legends=[label], filename=os.path.basename(png))
        prof = surfactant_profile(smiles=smi)
        desc = molecular_descriptors(smiles=smi)
        files = [export_structure(smiles=smi, fmt=f, filename=_uid("built", f))["file"]
                 for f in ("mol", "sdf", "pdb")]
        return {"ok": True, "image": _data_uri(png), "title": label,
                "summary": [["SMILES", smi], ["formula", spec["formula"]], ["MW", spec["mw"]],
                            ["declared class", spec["surfactant_class"]],
                            ["computed charge class", prof["charge_class"]],
                            ["counterion", spec["counterion"]]],
                "properties": _rows({"clogp_crippen": desc["clogp_crippen"],
                                     "tpsa": desc["tpsa"],
                                     "rotatable_bonds": desc["rotatable_bonds"],
                                     "hlb_davies": prof["hlb_davies"],
                                     "hlb_griffin": prof["hlb_griffin"],
                                     "tail_carbons": prof["hydrophobic_tail_carbons"],
                                     "ethoxylate_units": prof["ethoxylate_units"]}) +
                              _rows({"packing": prof["packing"], "cmc": prof["cmc_estimate"]}),
                "caveats": _caveats(spec, prof), "files": _downloads(files)}
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ------------------------------------------------------------------ W4 series
SERIES_COLUMNS = ["name", "formula", "mw", "charge_class", "tail_carbons", "eo_units", "clogp",
                  "tpsa", "hlb_davies", "hlb_griffin", "cpp", "predicted_aggregate",
                  "cmc_mM_estimate", "cmc_basis"]


@app.post("/api/series")
def api_series(body: SeriesIn):
    try:
        ns = [int(x) for x in body.chains.replace(",", " ").split()]
        if not ns:
            raise ValueError("give at least one chain length, for example '10 12 14 16'")
        recs, smis, legs = [], [], []
        for n in ns:
            smi = build_surfactant(tail_carbons=n, head_group=body.head_group,
                                   n_eo=int(body.n_eo),
                                   counterion=body.counterion or None)["smiles"]
            name = f"C{n} {body.head_group.replace('_', ' ')}" + (f" {body.n_eo}EO" if body.n_eo else "")
            recs.append({"name": name, "smiles": smi}); smis.append(smi); legs.append(name)
        rows = compare_series(records=recs)["rows"]
        png = os.path.join(OUTDIR, _uid("series", "png"))
        draw_structure(smiles=smis, legends=legs, filename=os.path.basename(png))
        sdf = export_dataset(records=recs, filename=_uid("series", "sdf"))["file"]
        return {"ok": True, "image": _data_uri(png), "columns": SERIES_COLUMNS,
                "rows": [[("" if r.get(c) is None else r.get(c)) for c in SERIES_COLUMNS]
                         for r in rows],
                "caveats": ["Rows whose cmc_basis is not_applicable have no published "
                            "homologous-series correlation for that head group; the CMC column is "
                            "deliberately empty rather than estimated."],
                "files": _downloads([sdf])}
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ------------------------------------------------------------------ W5/W6 polymer
@app.post("/api/polymer")
def api_polymer(body: PolymerIn):
    try:
        kw: dict[str, Any] = dict(monomer=body.monomer, linkage=int(body.linkage), dp=int(body.dp))
        if int(body.target_dp) > 0:
            kw["target_dp"] = int(body.target_dp)
        if body.branch_monomer and int(body.branch_every) > 0:
            kw.update(branch_monomer=body.branch_monomer,
                      branch_linkage=int(body.branch_linkage),
                      branch_every=int(body.branch_every))
        if body.substituent:
            kw.update(substituent=body.substituent, ds=float(body.ds))
            if body.positions.strip():
                kw["substituent_positions"] = [int(x) for x in
                                               body.positions.replace(",", " ").split()]
        built = build_polysaccharide(**kw)
        smi = built["exemplar_smiles"]
        png = os.path.join(OUTDIR, _uid("polymer", "png"))
        draw_structure(smiles=[smi], legends=[f"{built['spec']['linkage_label']} DP {body.dp}"],
                       filename=os.path.basename(png))
        ana = analyse_polysaccharide(smiles=smi)
        sp, pq = built["spec"], built["polymer_quantities"]
        files = [export_structure(smiles=smi, fmt=f, name=f"{body.monomer}_1_{body.linkage}_DP{body.dp}",
                                  filename=_uid("polymer", f))["file"] for f in ("mol", "sdf")]
        return {"ok": True, "image": _data_uri(png),
                "title": f"{sp['monomer_name']} {sp['main_chain_linkage']} exemplar at DP {body.dp}",
                "summary": [["repeat unit", sp["linkage_label"]],
                            ["anomeric configuration", sp["anomeric_configuration"]],
                            ["branch", sp["branch"] or "none"],
                            ["substituent", sp["substituent"] or "none"],
                            ["target DS", sp["target_ds"] if sp["substituent"] else "-"],
                            ["BigSMILES repeat unit", built["bigsmiles_repeat_unit"]],
                            ["exemplar formula", built["exemplar_formula"]],
                            ["exemplar MW", built["exemplar_mw"]]],
                "polymer_quantities": _rows(pq),
                "read_back": [["composition", ", ".join(f"{k} x{v}" for k, v in
                                                        ana["residue_composition"].items())],
                              ["linkages", ", ".join(f"{l['donor_residue']}{l['label']}"
                                                     f"{l['acceptor_residue']}"
                                                     for l in ana["glycosidic_linkages"])],
                              ["reducing end residue", ana["reducing_end_residues"] or "none found"],
                              ["net charge", ana["net_charge"]],
                              ["branch points", ana["branch_points"] or "none"]],
                "caveats": _caveats(built, ana), "files": _downloads(files)}
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@app.post("/api/analyse")
def api_analyse(body: AnalyseIn):
    try:
        ana = analyse_polysaccharide(smiles=body.smiles.strip())
        return {"ok": True,
                "title": f"{ana['n_sugar_rings']} sugar ring(s), "
                         f"{ana['n_glycosidic_bonds']} glycosidic bond(s)",
                "summary": [["composition", ", ".join(f"{k} x{v}" for k, v in
                                                      ana["residue_composition"].items()) or "-"],
                            ["formula", ana["formula"]], ["MW", ana["mw"]],
                            ["net charge", ana["net_charge"]],
                            ["reducing end", ana["reducing_end_residues"] or
                             "none found (fully substituted or cyclic)"],
                            ["branch points", ana["branch_points"] or "none"]],
                "residue_columns": ["residue", "identity", "anomeric", "ring form",
                                    "anomeric CIP", "reducing end", "substituents"],
                "residues": [[r["residue"], r["identity"]["monomer_name"],
                              r["identity"]["anomeric_configuration"], r["ring_form"],
                              r["anomeric_cip"],
                              "yes" if r["free_anomeric_oh_reducing_end"] else "no",
                              ", ".join(f"O{s['position']}: {s['substituent_smiles']}"
                                        for s in r["substituents"]) or "-"]
                             for r in ana["residues"]],
                "linkage_columns": ["donor residue", "donor position", "acceptor residue",
                                    "acceptor position", "linkage"],
                "linkages": [[l["donor_residue"], l["donor_position"], l["acceptor_residue"],
                              l["acceptor_position"], l["label"]]
                             for l in ana["glycosidic_linkages"]],
                "caveats": _caveats(ana)}
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ------------------------------------------------------------------ agent
def _get_llm_client():
    """Return an object exposing .llm(...) and .reasoning_model(), or None."""
    import builtins
    h = getattr(builtins, "host", None)
    if h is not None and hasattr(h, "llm"):
        return h
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    class _Client:
        """Adapter giving the anthropic SDK the surface chemagent.agent expects."""
        def __init__(self):
            self._c = anthropic.Anthropic(api_key=key)
            self._model = os.environ.get("CHEMAGENT_MODEL", "claude-sonnet-4-5")

        def reasoning_model(self):
            return self._model

        def llm(self, messages=None, tools=None, system=None, model=None, max_tokens=4096,
                thinking=None, **_):
            r = self._c.messages.create(model=model or self._model, max_tokens=max_tokens,
                                        system=system or "", tools=tools or [],
                                        messages=messages or [])
            content = [b.model_dump() for b in r.content]
            return {"text": "".join(b.get("text", "") for b in content
                                    if b.get("type") == "text"),
                    "content": content, "model": r.model, "stop_reason": r.stop_reason,
                    "usage": r.usage.model_dump()}
    return _Client()


AGENT_UNAVAILABLE = ("The agent needs an LLM client. Set ANTHROPIC_API_KEY and install the "
                     "`anthropic` package, then restart the server. Every other panel works "
                     "without it.")


@app.post("/api/agent")
def api_agent(body: AgentIn):
    client = _get_llm_client()
    if client is None:
        return {"ok": False, "error": AGENT_UNAVAILABLE}
    from chemagent.agent import run_agent
    try:
        run = run_agent(body.message.strip(), phase=2 if body.phase2 else 1, host=client,
                        verbose=False, log_name=_uid("web_run", "json"))
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    return {"ok": True, "answer": run["answer"] or "(no answer returned)",
            "steps": run["steps"], "seconds": run["seconds"],
            "trace": [{"tool": c["tool"], "ok": c["ok"], "error": c["error"],
                       "arguments": c["arguments"]} for c in run["tool_log"]],
            "files": _downloads(run["files"])}


# ------------------------------------------------------------------ files and page
@app.get("/files/{name}")
def get_file(name: str):
    """Serve a generated structure file. Basename only -- no traversal out of OUTDIR."""
    safe = os.path.basename(name)
    path = os.path.join(OUTDIR, safe)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, filename=safe, media_type="application/octet-stream")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chemistry agent PoC — HTML front end")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    import uvicorn
    uvicorn.run("chemagent.webapp:app" if args.reload else app, host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
