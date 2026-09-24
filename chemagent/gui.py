"""Browser GUI for the chemistry agent PoC.

Every panel is also a named HTTP endpoint, so the same GUI can be driven from a script:
`Client("http://127.0.0.1:7860").predict(..., api_name="/characterise")`.

    python -m chemagent.gui                    # http://127.0.0.1:7860
    python -m chemagent.gui --share            # temporary public link
    python -m chemagent.gui --port 8080

The GUI is a thin skin over the same registered tools the agent calls, so a panel and
an agent request that do the same thing give byte-identical results. Nothing in this
module computes chemistry; it only arranges inputs, dispatches through
`chemagent.registry`, and lays the returned values out.

The "Ask the agent" tab needs an LLM client. It works out of the box on a host that
injects `host`, and otherwise uses the `anthropic` package with ANTHROPIC_API_KEY.
Every other tab is fully deterministic and needs neither.
"""
from __future__ import annotations

import argparse
import inspect
import os
import uuid

import gradio as gr

import chemagent  # noqa: F401  -- import side-effects populate the tool registry
from chemagent import registry
from chemagent.compare import compare_series, compare_structures
from chemagent.descriptors import functional_groups, molecular_descriptors, surfactant_profile
from chemagent.export import export_structure
from chemagent.polymer import (MONOMERS, NAMED_POLYMERS, SUBSTITUENTS, analyse_polysaccharide,
                               build_polysaccharide)
from chemagent.registry import registry_table
from chemagent.render import draw_before_after, draw_structure
from chemagent.resolve import LEXICON, resolve_input
from chemagent.transform import HEAD_TEMPLATES, REACTIONS, apply_reaction, build_surfactant

OUTDIR = os.environ.get("CHEMAGENT_OUTDIR", "outputs")
os.makedirs(OUTDIR, exist_ok=True)

DESIGN_RULE = (
    "The model plans; it never computes. Every number below is produced by a registered "
    "cheminformatics tool and is reproducible from the same call. Where a correlation is "
    "outside its applicability domain the tool says so instead of estimating."
)

# ------------------------------------------------------------------ small helpers
def _uid(stem: str, ext: str) -> str:
    return f"{stem}_{uuid.uuid4().hex[:8]}.{ext}"


def _kv(d: dict, keys: list[str] | None = None, skip: tuple = ()) -> list[list[str]]:
    """Flatten a tool result into (property, value) rows for a Dataframe."""
    rows = []
    for k, v in d.items():
        if k in skip or (keys and k not in keys):
            continue
        if isinstance(v, dict):
            for k2, v2 in v.items():
                rows.append([f"{k}.{k2}", "" if v2 is None else str(v2)])
        elif isinstance(v, list):
            rows.append([k, ", ".join(str(x) for x in v) if v else "-"])
        else:
            rows.append([k, "" if v is None else str(v)])
    return rows


def _caveats(*results: dict) -> str:
    """Collect every caveat / assumption / note / warning a tool returned."""
    out = []
    for r in results:
        for key in ("caveats", "assumptions", "note", "notes", "warning", "ambiguity_note",
                    "ignored_arguments"):
            v = r.get(key)
            if not v:
                continue
            items = v if isinstance(v, list) else [v]
            for it in items:
                if str(it) not in out:
                    out.append(str(it))
    if not out:
        return ""
    return "**Assumptions and caveats returned by the tools**\n\n" + "\n".join(f"- {c}" for c in out)


def _chatbot_kwargs() -> dict:
    """Gradio 5 needs type='messages' to opt into the OpenAI-style history this module
    produces; Gradio 6 made that the only format and removed the argument."""
    import inspect
    return ({"type": "messages"}
            if "type" in inspect.signature(gr.Chatbot.__init__).parameters else {})


def _fail(exc: Exception) -> str:
    return f"**Could not complete this request**\n\n`{type(exc).__name__}: {exc}`"


LEX_CHOICES = sorted({v["name"] for v in LEXICON.values()})
HEAD_CHOICES = sorted(HEAD_TEMPLATES)
RXN_CHOICES = sorted(REACTIONS)
MONO_CHOICES = sorted(MONOMERS)
SUB_CHOICES = ["(none)"] + sorted(SUBSTITUENTS)
POLY_CHOICES = sorted(NAMED_POLYMERS)
COUNTERIONS = ["(class default)", "Na", "K", "Li", "NH4", "Cl", "Br", "none"]


# ------------------------------------------------------------------ tab 1: characterise
def do_characterise(query: str, use_network: bool, want_3d: bool):
    blank = (None, "", [], [], "", None)
    if not (query or "").strip():
        return (None, "Enter a name, SMILES or InChI.", [], [], "", None)
    try:
        res = resolve_input(query=query.strip(), use_network=use_network)
        if not res.get("resolved"):
            sugg = ", ".join(res.get("suggestions") or []) or "none"
            return (None,
                    f"**Not resolved.** {res['reason']}\n\nNear matches: {sugg}\n\n"
                    f"Next step: {res['next_step']}", [], [], "", None)
        smi = res["smiles"]
        png = os.path.join(OUTDIR, _uid("structure", "png"))
        draw_structure(smiles=[smi], legends=[res.get("preferred_name") or smi],
                       filename=os.path.basename(png))
        desc = molecular_descriptors(smiles=smi)
        prof = surfactant_profile(smiles=smi)
        fgs = functional_groups(smiles=smi)
        head = (f"### {res.get('preferred_name') or 'Resolved structure'}\n\n"
                f"- **Resolved via** `{res['source']}`"
                + (f" (CID {res['pubchem_cid']})" if res.get("pubchem_cid") else "") + "\n"
                f"- **SMILES** `{smi}`\n"
                f"- **InChIKey** `{res['inchikey']}`\n"
                f"- **Class** {res.get('class') or 'not classified in the lexicon'}\n"
                f"- **Charge class** {prof['charge_class']} · **head group** {prof['head_group']} · "
                f"**tail** C{prof['hydrophobic_tail_carbons']}"
                + (f" · **{prof['ethoxylate_units']} EO**" if prof["ethoxylate_units"] else ""))
        prop_rows = _kv(desc, skip=("smiles",)) + [["functional groups", ", ".join(
            f"{k} x{v}" for k, v in fgs["groups"].items()) or "-"]]
        surf_rows = _kv(prof, keys=["hlb_davies", "hlb_griffin", "counterions"]) \
            + _kv({"packing": prof["packing"], "cmc_estimate": prof["cmc_estimate"]})
        files = [export_structure(smiles=smi, fmt=f, filename=_uid("structure", f))["file"]
                 for f in (["mol", "sdf", "pdb"] if want_3d else ["mol", "sdf"])]
        return (png, head, prop_rows, surf_rows, _caveats(prof, res), files)
    except Exception as exc:  # noqa: BLE001
        return (None, _fail(exc), [], [], "", None)


# ------------------------------------------------------------------ tab 2: modify
def do_modify(start: str, reaction: str, co_reactant: str, repeat: int, counterion: str):
    if not (start or "").strip():
        return (None, "Enter a starting material.", [], "")
    try:
        res = resolve_input(query=start.strip())
        if not res.get("resolved"):
            return (None, f"**Not resolved.** {res['reason']}", [], "")
        co = None
        if (co_reactant or "").strip():
            cres = resolve_input(query=co_reactant.strip())
            if not cres.get("resolved"):
                return (None, f"**Co-reactant not resolved.** {cres['reason']}", [], "")
            co = cres["smiles"]
        rxn = apply_reaction(smiles=res["smiles"], reaction=reaction, co_reactant_smiles=co,
                             repeat=int(repeat),
                             counterion="none" if counterion.startswith("(") else counterion)
        png = os.path.join(OUTDIR, _uid("before_after", "png"))
        panel = draw_before_after(smiles_before=rxn["smiles_before"],
                                 smiles_after=rxn["smiles_after"],
                                 label_before=res.get("preferred_name") or "starting material",
                                 label_after=rxn["product_class"] or "product",
                                 filename=os.path.basename(png))
        cmp_ = compare_structures(smiles_a=rxn["smiles_before"], smiles_b=rxn["smiles_after"])
        md = (f"### {reaction.replace('_', ' ')}\n\n"
              f"{rxn['description']}\n\n"
              f"- **Reagents implied** {rxn['reagents']}\n"
              f"- **Product class** {rxn['product_class']}\n"
              f"- **Before** `{rxn['smiles_before']}` (MW {rxn['mw_before']})\n"
              f"- **After** `{rxn['smiles_after']}` (MW {rxn['mw_after']}, {rxn['formula_after']})\n"
              f"- **Reactive sites matched** {rxn['distinct_products_first_step']}"
              + ("  \n\n> **Regiochemical ambiguity:** more than one site matched and the largest "
                 "product was kept. Confirm the intended site before using this structure."
                 if rxn["regiochemical_ambiguity"] else ""))
        rows = ([["Tanimoto (ECFP4)", cmp_["tanimoto_ecfp4"]],
                 ["Tanimoto (MACCS)", cmp_["tanimoto_maccs"]],
                 ["MCS atoms", cmp_["mcs_atoms"]],
                 ["conserved fraction of starting material", panel["conserved_fraction"]],
                 ["groups gained", ", ".join(f"{k} x{v}" for k, v in cmp_["functional_groups_gained"].items()) or "-"],
                 ["groups lost", ", ".join(f"{k} x{v}" for k, v in cmp_["functional_groups_lost"].items()) or "-"]]
                + [[f"delta {k}", v] for k, v in cmp_["descriptor_delta_b_minus_a"].items()])
        return (png, md, [[str(a), str(b)] for a, b in rows], _caveats(rxn))
    except Exception as exc:  # noqa: BLE001
        return (None, _fail(exc), [], "")


# ------------------------------------------------------------------ tab 3: build
def do_build(tail_c: int, head: str, n_eo: int, counterion: str, branching: str, unsat: bool):
    try:
        spec = build_surfactant(tail_carbons=int(tail_c), head_group=head, n_eo=int(n_eo),
                                counterion=None if counterion.startswith("(") else counterion,
                                unsaturation=1 if unsat else 0, branching=branching)
        smi = spec["smiles"]
        png = os.path.join(OUTDIR, _uid("built", "png"))
        label = (f"C{tail_c} {head.replace('_', ' ')}" + (f", {n_eo} EO" if n_eo else ""))
        draw_structure(smiles=[smi], legends=[label], filename=os.path.basename(png))
        prof = surfactant_profile(smiles=smi)
        desc = molecular_descriptors(smiles=smi)
        md = (f"### {label}\n\n- **SMILES** `{smi}`\n"
              f"- **Formula** {spec['formula']} · **MW** {spec['mw']}\n"
              f"- **Class** {spec['surfactant_class']} (computed: {prof['charge_class']}) · "
              f"**counterion** {spec['counterion']}")
        rows = _kv({"clogp_crippen": desc["clogp_crippen"], "tpsa": desc["tpsa"],
                    "rotatable_bonds": desc["rotatable_bonds"],
                    "hlb_davies": prof["hlb_davies"], "hlb_griffin": prof["hlb_griffin"],
                    "tail_carbons": prof["hydrophobic_tail_carbons"],
                    "ethoxylate_units": prof["ethoxylate_units"]}) \
            + _kv({"packing": prof["packing"], "cmc_estimate": prof["cmc_estimate"]})
        files = [export_structure(smiles=smi, fmt=f, filename=_uid("built", f))["file"]
                 for f in ("mol", "sdf", "pdb")]
        return png, md, rows, _caveats(spec, prof), files
    except Exception as exc:  # noqa: BLE001
        return None, _fail(exc), [], "", None


# ------------------------------------------------------------------ tab 4: series
def do_series(head: str, chains: str, n_eo: int, counterion: str):
    try:
        ns = [int(x) for x in chains.replace(",", " ").split()]
        if not ns:
            raise ValueError("give at least one chain length, e.g. '10 12 14 16'")
        recs, smis, legs = [], [], []
        for n in ns:
            smi = build_surfactant(tail_carbons=n, head_group=head, n_eo=int(n_eo),
                                   counterion=None if counterion.startswith("(") else counterion
                                   )["smiles"]
            name = f"C{n} {head.replace('_', ' ')}" + (f" {n_eo}EO" if n_eo else "")
            recs.append({"name": name, "smiles": smi}); smis.append(smi); legs.append(name)
        tab = compare_series(records=recs)["rows"]
        png = os.path.join(OUTDIR, _uid("series", "png"))
        draw_structure(smiles=smis, legends=legs, filename=os.path.basename(png))
        from chemagent.export import export_dataset
        sdf = export_dataset(records=recs, filename=_uid("series", "sdf"))["file"]
        cols = ["name", "formula", "mw", "charge_class", "tail_carbons", "eo_units", "clogp",
                "tpsa", "hlb_davies", "hlb_griffin", "cpp", "predicted_aggregate",
                "cmc_mM_estimate", "cmc_basis"]
        rows = [[("" if r.get(c) is None else r.get(c)) for c in cols] for r in tab]
        note = ("Rows whose `cmc_basis` is `not_applicable` have no published homologous-series "
                "correlation for that head group; the CMC column is deliberately empty rather "
                "than estimated.")
        return png, gr.Dataframe(value=rows, headers=cols), note, [sdf]
    except Exception as exc:  # noqa: BLE001
        return None, gr.Dataframe(value=[], headers=["name"]), _fail(exc), None


# ------------------------------------------------------------------ tab 5: polymer build
def on_named_polymer(name: str):
    """Fill the builder controls from a curated named-polymer entry."""
    if not name:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), "")
    v = NAMED_POLYMERS[name]
    b = v.get("build") or {}
    notes = (f"### {v['name']}\n\n{v['summary']}\n\n"
             + "\n".join(f"- {n}" for n in v.get("structural_notes", []))
             + f"\n\n**Formulation relevance** — {v.get('formulation_relevance', 'n/a')}")
    if not b:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(),
                notes + "\n\n> This polymer has **no build parameters** in the curated library "
                        "(a declared Phase 2 gap). The controls were left unchanged.")
    return (gr.update(value=b.get("monomer", "b_glcp")),
            gr.update(value=b.get("linkage", 4)),
            gr.update(value=b.get("branch_monomer") or "(none)"),
            gr.update(value=b.get("branch_linkage") or 6),
            gr.update(value=b.get("branch_every") or 0),
            gr.update(value=b.get("substituent") or "(none)"),
            gr.update(value=b.get("ds") or 0.0),
            notes)


def do_polymer(monomer: str, linkage: int, dp: int, target_dp: int, branch_mono: str,
               branch_link: int, branch_every: int, substituent: str, ds: float, positions: str):
    try:
        pos = [int(x) for x in positions.replace(",", " ").split()] if positions.strip() else None
        kw = dict(monomer=monomer, linkage=int(linkage), dp=int(dp))
        if int(target_dp) > 0:
            kw["target_dp"] = int(target_dp)
        if branch_mono and not branch_mono.startswith("(") and int(branch_every) > 0:
            kw.update(branch_monomer=branch_mono, branch_linkage=int(branch_link),
                      branch_every=int(branch_every))
        if substituent and not substituent.startswith("("):
            kw.update(substituent=substituent, ds=float(ds))
            if pos:
                kw["substituent_positions"] = pos
        built = build_polysaccharide(**kw)
        smi = built["exemplar_smiles"]
        png = os.path.join(OUTDIR, _uid("polymer", "png"))
        draw_structure(smiles=[smi], legends=[f"{built['spec']['linkage_label']} DP {dp}"],
                       filename=os.path.basename(png))
        ana = analyse_polysaccharide(smiles=smi)
        sp, pq = built["spec"], built["polymer_quantities"]
        md = (f"### {sp['monomer_name']} {sp['main_chain_linkage']} exemplar at DP {dp}\n\n"
              f"- **Repeat unit** {sp['linkage_label']} · **anomeric** {sp['anomeric_configuration']}\n"
              f"- **Branch** {sp['branch'] or 'none'}\n"
              f"- **Substituent** {sp['substituent'] or 'none'}"
              + (f" at target DS {sp['target_ds']}, realised DS "
                 f"{pq['realised_ds_in_exemplar']} in this exemplar" if sp['substituent'] else "")
              + f"\n- **BigSMILES repeat unit** `{built['bigsmiles_repeat_unit']}`\n"
              f"- **Exemplar** {built['exemplar_formula']}, MW {built['exemplar_mw']}\n\n"
              f"**Composition read back from the built structure** — "
              + ", ".join(f"{k} x{v}" for k, v in ana["residue_composition"].items())
              + " · linkages "
              + ", ".join(f"{l['donor_residue']}{l['label']}{l['acceptor_residue']}"
                          for l in ana["glycosidic_linkages"])
              + f" · reducing end at residue {ana['reducing_end_residues'] or 'none found'}")
        rows = _kv(pq) + [["exemplar net charge", ana["net_charge"]],
                          ["sugar rings found", ana["n_sugar_rings"]],
                          ["branch points", ana["branch_points"] or "-"]]
        files = [export_structure(smiles=smi, fmt=f, name=f"{monomer}_1_{linkage}_DP{dp}",
                                 filename=_uid("polymer", f))["file"] for f in ("mol", "sdf")]
        return png, md, [[str(a), str(b)] for a, b in rows], _caveats(built, ana), files
    except Exception as exc:  # noqa: BLE001
        return None, _fail(exc), [], "", None


# ------------------------------------------------------------------ tab 6: polymer analyse
def do_analyse(smiles: str):
    if not (smiles or "").strip():
        return "Paste a carbohydrate SMILES.", [], [], ""
    try:
        ana = analyse_polysaccharide(smiles=smiles.strip())
        md = (f"### {ana['n_sugar_rings']} sugar ring(s), {ana['n_glycosidic_bonds']} glycosidic bond(s)\n\n"
              f"- **Composition** " + (", ".join(f"{k} x{v}" for k, v in ana["residue_composition"].items()) or "-")
              + f"\n- **Formula** {ana['formula']} · **MW** {ana['mw']} · **net charge** {ana['net_charge']}\n"
              f"- **Reducing end** residue {ana['reducing_end_residues'] or 'none found (fully substituted or cyclic)'}\n"
              f"- **Branch points** {ana['branch_points'] or 'none'}")
        res_rows = [[r["residue"], r["identity"]["monomer_name"],
                     r["identity"]["anomeric_configuration"], r["ring_form"],
                     r["anomeric_cip"], "yes" if r["free_anomeric_oh_reducing_end"] else "no",
                     ", ".join(f"O{s['position']}: {s['substituent_smiles']}"
                               for s in r["substituents"]) or "-"]
                    for r in ana["residues"]]
        link_rows = [[l["donor_residue"], l["donor_position"], l["acceptor_residue"],
                      l["acceptor_position"], l["label"]] for l in ana["glycosidic_linkages"]]
        return md, res_rows, link_rows, _caveats(ana)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc), [], [], ""


# ------------------------------------------------------------------ tab 7: agent
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
        """Adapter giving the anthropic SDK the same surface chemagent.agent expects."""
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
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            return {"text": text, "content": content, "model": r.model,
                    "stop_reason": r.stop_reason, "usage": r.usage.model_dump()}
    return _Client()


AGENT_UNAVAILABLE = (
    "The agent tab needs an LLM client. Set `ANTHROPIC_API_KEY` in the environment and "
    "`pip install anthropic`, then restart the GUI. Every other tab works without it."
)


def do_agent(message: str, history: list, phase2: bool):
    history = list(history or [])
    if not (message or "").strip():
        return history, "", None
    client = _get_llm_client()
    if client is None:
        return (history + [{"role": "user", "content": message},
                           {"role": "assistant", "content": AGENT_UNAVAILABLE}], "", None)
    from chemagent.agent import run_agent
    try:
        run = run_agent(message.strip(), phase=2 if phase2 else 1, host=client, verbose=False,
                        log_name=_uid("gui_run", "json"))
    except Exception as exc:  # noqa: BLE001
        return history + [{"role": "user", "content": message},
                          {"role": "assistant", "content": _fail(exc)}], "", None
    trace = "\n".join(
        f"{i}. `{c['tool']}` {'ok' if c['ok'] else 'FAILED: ' + str(c['error'])}"
        for i, c in enumerate(run["tool_log"], 1)) or "no tool calls"
    answer = (run["answer"] or "*(no answer returned)*") + \
        f"\n\n<details><summary>{run['n_tool_calls']} tool calls in " \
        f"{run['steps']} turns ({run['seconds']} s)</summary>\n\n{trace}\n\n</details>"
    files = [f for f in run["files"] if os.path.exists(f)] or None
    return (history + [{"role": "user", "content": message},
                       {"role": "assistant", "content": answer}], "", files)


# ------------------------------------------------------------------ layout
def build_ui() -> gr.Blocks:
    # Gradio 6 moved `theme` from the Blocks constructor to launch()
    blocks_kw = {"title": "Chemistry agent PoC"}
    if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kw["theme"] = gr.themes.Soft()
    with gr.Blocks(**blocks_kw) as demo:
        gr.Markdown("# Chemistry agent — surfactants and carbohydrate polymers")
        gr.Markdown(f"*{DESIGN_RULE}*")

        with gr.Tab("Characterise"):
            gr.Markdown("Resolve a name, SMILES or InChI, then render and profile it. "
                        "**W1** in the workflow specification.")
            with gr.Row():
                with gr.Column(scale=1):
                    c_in = gr.Textbox(label="Name, SMILES or InChI", value="sodium lauryl sulfate",
                                      placeholder="SDS · CTAB · cocamidopropyl betaine · CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]")
                    gr.Examples([["SDS"], ["cocamidopropyl betaine"], ["CTAB"], ["decyl glucoside"],
                                 ["CCCCCCCCCCCCOCCOCCOS(=O)(=O)[O-].[Na+]"]], inputs=c_in)
                    c_net = gr.Checkbox(label="Fall back to PubChem for unknown names", value=True)
                    c_3d = gr.Checkbox(label="Also export a 3D .pdb (slower)", value=False)
                    c_go = gr.Button("Characterise", variant="primary")
                    c_files = gr.File(label="Structure files", file_count="multiple")
                with gr.Column(scale=2):
                    c_img = gr.Image(label="Structure", type="filepath", height=300)
                    c_md = gr.Markdown()
                    with gr.Row():
                        c_props = gr.Dataframe(headers=["property", "value"], label="Descriptors",
                                               wrap=True)
                        c_surf = gr.Dataframe(headers=["property", "value"],
                                              label="Surfactant profile", wrap=True)
                    c_cav = gr.Markdown()
            c_go.click(do_characterise, [c_in, c_net, c_3d],
                       [c_img, c_md, c_props, c_surf, c_cav, c_files],
                       api_name="characterise")

        with gr.Tab("Modify / react"):
            gr.Markdown("Apply a curated transformation and see what changed. **W2**.")
            with gr.Row():
                with gr.Column(scale=1):
                    m_in = gr.Textbox(label="Starting material", value="lauric acid")
                    m_rxn = gr.Dropdown(RXN_CHOICES, label="Transformation", value="amidation")
                    m_co = gr.Textbox(label="Co-reactant (two-component reactions only)",
                                      value="DMAPA",
                                      info="amidation and esterification need one")
                    m_rep = gr.Number(label="Apply n times in series", value=1, precision=0,
                                      minimum=1, maximum=12)
                    m_ci = gr.Dropdown(COUNTERIONS, label="Counterion on the product", value="Na")
                    m_go = gr.Button("Apply transformation", variant="primary")
                    m_rxn_md = gr.Markdown()
                with gr.Column(scale=2):
                    m_img = gr.Image(label="Before / after — changed atoms highlighted",
                                     type="filepath", height=340)
                    m_tab = gr.Dataframe(headers=["property", "value"],
                                         label="Similarity and property change", wrap=True)
                    m_cav = gr.Markdown()
            m_go.click(do_modify, [m_in, m_rxn, m_co, m_rep, m_ci],
                       [m_img, m_rxn_md, m_tab, m_cav], api_name="modify")

        with gr.Tab("Build from a spec"):
            gr.Markdown("Describe the surfactant you want instead of drawing it. **W3**.")
            with gr.Row():
                with gr.Column(scale=1):
                    b_n = gr.Slider(6, 22, value=12, step=1,
                                    label="Carbons in the hydrophobe (includes the acyl carbon)")
                    b_head = gr.Dropdown(HEAD_CHOICES, label="Head group", value="sulfate")
                    b_eo = gr.Slider(0, 20, value=0, step=1, label="EO units (ether sulfate / ethoxylate)")
                    b_ci = gr.Dropdown(COUNTERIONS, label="Counterion", value="(class default)")
                    b_br = gr.Dropdown(["linear", "iso", "2-ethylhexyl-like"], label="Branching",
                                       value="linear")
                    b_un = gr.Checkbox(label="One cis double bond (needs >= 12 carbons)", value=False)
                    b_go = gr.Button("Build", variant="primary")
                    b_files = gr.File(label="Structure files", file_count="multiple")
                with gr.Column(scale=2):
                    b_img = gr.Image(label="Structure", type="filepath", height=280)
                    b_md = gr.Markdown()
                    b_tab = gr.Dataframe(headers=["property", "value"], label="Properties", wrap=True)
                    b_cav = gr.Markdown()
            b_go.click(do_build, [b_n, b_head, b_eo, b_ci, b_br, b_un],
                       [b_img, b_md, b_tab, b_cav, b_files], api_name="build_surfactant")

        with gr.Tab("Compare a series"):
            gr.Markdown("Build a homologous series and tabulate the property trend. **W4**.")
            with gr.Row():
                s_head = gr.Dropdown(HEAD_CHOICES, label="Head group", value="sulfate")
                s_chains = gr.Textbox(label="Chain lengths", value="10 12 14 16")
                s_eo = gr.Slider(0, 12, value=0, step=1, label="EO units")
                s_ci = gr.Dropdown(COUNTERIONS, label="Counterion", value="(class default)")
                s_go = gr.Button("Build series", variant="primary")
            s_tab = gr.Dataframe(label="Series properties", wrap=True)
            s_note = gr.Markdown()
            s_img = gr.Image(label="Series", type="filepath", height=300)
            s_files = gr.File(label="SDF for downstream modelling", file_count="multiple")
            s_go.click(do_series, [s_head, s_chains, s_eo, s_ci], [s_img, s_tab, s_note, s_files],
                       api_name="series")

        with gr.Tab("Polymer builder (Phase 2)"):
            gr.Markdown("Build a carbohydrate-polymer exemplar from a repeat-unit spec, or load "
                        "the curated parameters for a named polymer. **W5 / W6**.")
            with gr.Row():
                with gr.Column(scale=1):
                    p_named = gr.Dropdown(POLY_CHOICES, label="Load a named polymer",
                                          value=None, info="fills the controls below")
                    p_mono = gr.Dropdown(MONO_CHOICES, label="Monomer", value="b_glcp")
                    p_link = gr.Dropdown([2, 3, 4, 6], label="Main-chain linkage (1->n)", value=4)
                    p_dp = gr.Slider(2, 20, value=6, step=1, label="Exemplar DP to build")
                    p_tdp = gr.Number(label="Real polymer DP for Mn (0 = same as exemplar)",
                                      value=0, precision=0)
                    with gr.Accordion("Branching", open=False):
                        p_bmono = gr.Dropdown(["(none)"] + MONO_CHOICES, label="Branch monomer",
                                              value="(none)")
                        p_blink = gr.Dropdown([2, 3, 4, 6], label="Branch position", value=6)
                        p_bevery = gr.Number(label="Branch every n residues (0 = none)", value=0,
                                             precision=0)
                    with gr.Accordion("Derivatisation", open=False):
                        p_sub = gr.Dropdown(SUB_CHOICES, label="Substituent", value="(none)")
                        p_ds = gr.Number(label="Target degree of substitution", value=0.0)
                        p_pos = gr.Textbox(label="Positions available (blank = all free)",
                                           value="", placeholder="2 3 6")
                    p_go = gr.Button("Build exemplar", variant="primary")
                    p_files = gr.File(label="Structure files", file_count="multiple")
                with gr.Column(scale=2):
                    p_img = gr.Image(label="Exemplar", type="filepath", height=340)
                    p_md = gr.Markdown()
                    p_tab = gr.Dataframe(headers=["quantity", "value"],
                                         label="Polymer-level quantities", wrap=True)
                    p_cav = gr.Markdown()
            p_named.change(on_named_polymer, [p_named],
                           [p_mono, p_link, p_bmono, p_blink, p_bevery, p_sub, p_ds, p_md])
            p_go.click(do_polymer,
                       [p_mono, p_link, p_dp, p_tdp, p_bmono, p_blink, p_bevery, p_sub, p_ds, p_pos],
                       [p_img, p_md, p_tab, p_cav, p_files], api_name="build_polysaccharide")

        with gr.Tab("Polymer analyser (Phase 2)"):
            gr.Markdown("Interpret a carbohydrate structure you already have: residue identity, "
                        "anomeric configuration, every glycosidic linkage, substituents and the "
                        "reducing end.")
            a_in = gr.Textbox(label="Carbohydrate SMILES", lines=3)
            a_go = gr.Button("Analyse", variant="primary")
            a_md = gr.Markdown()
            a_res = gr.Dataframe(headers=["residue", "identity", "anomeric", "ring form",
                                          "anomeric CIP", "reducing end", "substituents"],
                                 label="Residues", wrap=True)
            a_link = gr.Dataframe(headers=["donor residue", "donor position", "acceptor residue",
                                           "acceptor position", "linkage"],
                                  label="Glycosidic linkages", wrap=True)
            a_cav = gr.Markdown()
            a_go.click(do_analyse, [a_in], [a_md, a_res, a_link, a_cav], api_name="analyse")

        with gr.Tab("Ask the agent"):
            gr.Markdown("Natural language over the same tools. The tool calls behind each answer "
                        "are listed under the reply.")
            g_chat = gr.Chatbot(label="Conversation", height=420, **_chatbot_kwargs())
            with gr.Row():
                g_in = gr.Textbox(label="Request", scale=4,
                                  placeholder="Swap the sulfate on SDS for an amine oxide and tell me what changes")
                g_p2 = gr.Checkbox(label="Phase 2 tools", value=True, scale=1)
                g_go = gr.Button("Send", variant="primary", scale=1)
            g_files = gr.File(label="Files produced", file_count="multiple")
            if _get_llm_client() is None:
                gr.Markdown(f"> {AGENT_UNAVAILABLE}")
            g_go.click(do_agent, [g_in, g_chat, g_p2], [g_chat, g_in, g_files], api_name="ask")
            g_in.submit(do_agent, [g_in, g_chat, g_p2], [g_chat, g_in, g_files])

        with gr.Tab("Tool registry"):
            gr.Markdown(f"Every capability in the system. {len(registry.all_tools())} tools; "
                        "the GUI panels and the agent both dispatch through this table.")
            gr.Dataframe(value=[[r["tool"], r["phase"], r["category"], r["inputs"],
                                 "yes" if r["writes_files"] else "no", r["description"]]
                                for r in registry_table()],
                         headers=["tool", "phase", "category", "arguments", "writes files",
                                  "description"], wrap=True)
            gr.Markdown("**Curated data layer** — "
                        f"{len(LEXICON)} lexicon entries · {len(REACTIONS)} reactions · "
                        f"{len(HEAD_TEMPLATES)} head groups · {len(MONOMERS)} monomer templates · "
                        f"{len(NAMED_POLYMERS)} named polymers · {len(SUBSTITUENTS)} derivatisations")
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="Chemistry agent PoC GUI")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    args = ap.parse_args()
    import inspect
    kw = dict(server_name=args.host, server_port=args.port, share=args.share,
              allowed_paths=[OUTDIR])
    if "show_api" in inspect.signature(gr.Blocks.launch).parameters:   # removed in Gradio 6
        kw["show_api"] = False
    build_ui().launch(**kw)


if __name__ == "__main__":
    main()
