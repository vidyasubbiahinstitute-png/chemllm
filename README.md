# chemagent — chemistry LLM agent PoC

Reference implementation for the workflow specified in WORKFLOW.md: surfactant and
carbohydrate-polymer reasoning over a deterministic RDKit tool layer.

Full run guide, environment variables and troubleshooting: RUNNING.md

## Install
    conda create -n chem -c conda-forge python=3.12 rdkit pandas matplotlib pillow requests py3dmol
    pip install fastapi uvicorn   # HTML front end
    pip install gradio            # Gradio front end

## Browser front ends
Eight panels either way: characterise, modify/react, build from a spec, compare a series,
polymer builder, polymer analyser, ask the agent, tool registry.

HTML app — self-contained page + JSON API, no build step, no CDN; this is the one to embed:
    pip install fastapi uvicorn
    python -m chemagent.webapp              # http://127.0.0.1:8000

  POST /api/characterise /api/modify /api/build /api/series /api/polymer /api/analyse /api/agent
  GET  /api/meta (control vocabulary)  /files/{name} (generated structures)  /api/docs

Gradio app — same panels, no front-end code to maintain:
    pip install gradio
    python -m chemagent.gui                 # http://127.0.0.1:7860
    python -m chemagent.gui --share         # temporary public link

## Use the tool layer directly (no LLM)
    import chemagent
    from chemagent.transform import build_surfactant
    from chemagent.descriptors import surfactant_profile
    s = build_surfactant(tail_carbons=14, head_group="amine_oxide")["smiles"]
    surfactant_profile(s)

## Use the agent loop
    from chemagent.agent import run_agent
    run = run_agent("Build a C14 amine oxide and compare it to SDS", host=host)
    run["answer"]; run["files"]; run["tool_log"]

`host` is any client exposing `.llm(messages=, tools=, system=, model=, max_tokens=)` and
`.reasoning_model()`. Outputs are written to $CHEMAGENT_OUTDIR (default ./outputs).
Set CHEMAGENT_OFFLINE=1 to disable the PubChem fallback in the resolution cascade.

## Modules
  chemagent/__init__.py
  chemagent/agent.py
  chemagent/gui.py
  chemagent/webapp.py
  chemagent/static/index.html
  chemagent/compare.py
  chemagent/descriptors.py
  chemagent/export.py
  chemagent/polymer.py
  chemagent/registry.py
  chemagent/render.py
  chemagent/resolve.py
  chemagent/transform.py

## Curated data (2 files, reviewed JSON)
  chemagent/data/lexicon.json    37 surfactants and feedstocks with synonyms and provenance notes
  chemagent/data/monomers.json   11 InChIKey-validated monomer templates, 20 named polymers,
                                 10 derivatisation chemistries

## Validation
outputs/validation_report.csv — 42 checks, all passing: monomer templates and spec-built
surfactants against PubChem InChIKeys, assembled disaccharides against reference disaccharides
(full stereochemistry layer), and the builder carbon-count contract across all 21 head groups.
