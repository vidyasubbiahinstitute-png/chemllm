# Running the chemistry agent

**What in this guide has been executed.** From a clean extraction of `chemagent_poc.tar.gz` into
an empty directory, verbatim: `tar xzf … && cd chemagent-poc`, `pip install -e ".[web]"`, both
install-check commands in §6 (25 tools; SDS InChIKey `DBMJMQXJHONAFJ-UHFFFAOYSA-M`), `chemagent-web`
serving the page, the §3 curl example returning Cetyltrimethylammonium bromide / cationic, output
files appearing in `./outputs`, and `chemagent-gui` serving the Gradio page — all from a working
directory outside the source tree, on Linux with Python 3.12.

**What has not been executed, and is written from the package metadata rather than from a run:**
the conda and `venv` creation lines in §2 (the environment used already had RDKit installed), the
agent tab against a real `ANTHROPIC_API_KEY` (§4), the `CHEMAGENT_OUTDIR` and `CHEMAGENT_OFFLINE`
overrides (§5), the Windows activation paths, and the fixes in the troubleshooting table (§7).
Treat those as documentation, not as verified procedure.

## 1. What you need

Python 3.10 or newer and RDKit. Nothing else is required for the deterministic panels — no
database, no service, no API key. RDKit ships as a binary wheel, so a plain `pip` install works;
conda is the safer route on Windows and on older Linux.

## 2. Install

Unpack the bundle, then pick one route.

**conda (recommended)**

    tar xzf chemagent_poc.tar.gz && cd chemagent-poc
    conda create -n chem -c conda-forge python=3.12 rdkit pandas pillow
    conda activate chem
    pip install -e ".[web]"

**pip only**

    tar xzf chemagent_poc.tar.gz && cd chemagent-poc
    python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
    pip install -e ".[web]"

The extras decide which front end you get:

| Extra | Pulls in | Gives you |
|---|---|---|
| *(none)* | rdkit, pandas, pillow | the Python tool layer |
| `[web]` | fastapi, uvicorn | the HTML app |
| `[gradio]` | gradio | the Gradio app |
| `[agent]` | anthropic | the natural-language agent |
| `[viewer]` | py3Dmol | the interactive 3D viewer tool |
| `[all]` | all of the above | everything |

`pip install -e .` puts two commands on your PATH and makes the package importable from any
directory, so you do not have to work inside the source tree.

## 3. Run it

**HTML app** — the one to use for a demo or to embed in an existing site:

    chemagent-web                          # http://127.0.0.1:8000
    chemagent-web --port 9000              # different port
    chemagent-web --host 0.0.0.0           # reachable from other machines on the network
    python -m chemagent.webapp             # identical, if you did not install the package

Open the printed URL. Eight tabs: Characterise, Modify / react, Build from a spec, Compare a
series, Polymer builder, Polymer analyser, Ask the agent, Tool registry.

**Gradio app** — same panels, no front-end code to maintain:

    chemagent-gui                          # http://127.0.0.1:7860
    chemagent-gui --share                  # temporary public link for a remote demo

**Neither** — the tool layer is ordinary Python:

    from chemagent.transform import build_surfactant
    from chemagent.descriptors import surfactant_profile
    smi = build_surfactant(tail_carbons=14, head_group="amine_oxide")["smiles"]
    surfactant_profile(smi)["hlb_davies"]

**As a service** — every panel is also an HTTP endpoint, so nothing needs a browser:

    curl -X POST http://127.0.0.1:8000/api/characterise \
         -H 'Content-Type: application/json' -d '{"query": "CTAB"}'

`GET /api/docs` lists the schema of every endpoint. The Gradio app exposes the same panels as
named routes: `Client(url).predict(..., api_name="/characterise")`.

## 4. Turning on the agent tab

The seven deterministic panels need no model. The agent tab does:

    pip install -e ".[web,agent]"
    export ANTHROPIC_API_KEY=sk-ant-...        # Windows: set ANTHROPIC_API_KEY=...
    chemagent-web

Without a key the tab renders a notice saying so and every other tab keeps working. To pin a
different model, set `CHEMAGENT_MODEL`.

## 5. Where files go

Rendered PNGs and exported `.mol` / `.sdf` / `.pdb` files are written to `./outputs` **relative
to the directory you launch from**, and served back through `/files/<name>`. Point that somewhere
else with `CHEMAGENT_OUTDIR`:

    CHEMAGENT_OUTDIR=/data/chemagent chemagent-web

| Variable | Default | Effect |
|---|---|---|
| `CHEMAGENT_OUTDIR` | `./outputs` | where rendered images and structure files land |
| `CHEMAGENT_OFFLINE` | unset | set to `1` to disable the PubChem fallback in name resolution |
| `ANTHROPIC_API_KEY` | unset | enables the agent tab |
| `CHEMAGENT_MODEL` | `claude-sonnet-4-5` | model used by the agent loop |

## 6. Check the install

    python -c "import chemagent; print(len(chemagent.all_tools()), 'tools')"      # 25 tools
    python -c "from chemagent.resolve import resolve_input; print(resolve_input('SDS')['inchikey'])"

Name resolution works offline for the 37 curated lexicon entries; anything outside it needs
network access to PubChem, or `CHEMAGENT_OFFLINE=1` to skip the attempt.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: chemagent` | running from outside the source tree without installing | `pip install -e .` from the unpacked directory |
| `ModuleNotFoundError: fastapi` / `gradio` | front-end extra not installed | `pip install -e ".[web]"` or `".[gradio]"` |
| `[Errno 98] Address already in use` | something already on the port | `chemagent-web --port 9000` |
| Agent tab says it is unavailable | no `ANTHROPIC_API_KEY`, or `anthropic` not installed | §4 |
| A name will not resolve | not in the lexicon and PubChem unreachable | give a SMILES instead, or add the material to `chemagent/data/lexicon.json` |
| `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed` | you are behind a SOCKS proxy; httpx needs the extra | `pip install "httpx[socks]"` |
| A panel shows an error message instead of results | a tool refused the request | that is the designed behaviour — the message names the reason, e.g. a position that carries a branch cannot also be substituted |
| A CMC or HLB cell is empty | the correlation does not apply to that structure | also designed — the tool declines rather than estimating; the caveats block says why |

## 8. Ports and exposure

Both front ends bind `127.0.0.1` by default, so they are reachable only from the machine they run
on. `--host 0.0.0.0` exposes the app to your network; there is no authentication layer, so put it
behind your own reverse proxy or SSO before doing that on anything shared. `gradio --share`
creates a temporary public tunnel — convenient for a remote demo, not for routine use.

## 9. Serving it as a web page

### On your own machine

    chemagent-web

then open **http://127.0.0.1:8000** in any browser. That is the whole thing — the app is already
a web page; nothing is generated as a static file you open with `file://`, because the chemistry
runs server-side in RDKit.

### The server is on another machine

Forward the port over SSH and browse to it locally, which needs no firewall change and no
exposure:

    ssh -L 8000:127.0.0.1:8000 you@server      # keep this open
    # then open http://127.0.0.1:8000 in your own browser

### Other people on your network should reach it

    chemagent-web --host 0.0.0.0 --port 8000

then `http://<server-ip>:8000`. There is no login, so only do this on a trusted network — see §8.

### Behind your own domain or reverse proxy

The app is a normal ASGI application, so any reverse proxy works. Mapping it to the domain root
is the simplest arrangement:

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }

To serve it under a sub-path instead — `https://tools.example.com/chem/` — the page resolves its
own requests relative to the page it was loaded from, so the front end follows the sub-path
without edits. The proxy must strip the prefix before forwarding:

    location /chem/ {
        proxy_pass http://127.0.0.1:8000/;      # the trailing slash strips /chem
        proxy_set_header Host $host;
    }

*Verified: relative-path resolution serving at the domain root. The sub-path proxy configuration
above is documentation, not a tested deployment.*

### Inside a page you already have

Two ways, depending on how much of the look you want to keep.

**Embed the whole app** — one line in your existing page:

    <iframe src="https://tools.example.com/chem/" style="width:100%;height:900px;border:0"></iframe>

**Use only the chemistry** — ignore `index.html` and call the JSON endpoints from your own
markup. Every panel is one POST with a JSON body and a JSON reply; rendered structures come back
as base64 data URIs you can drop straight into an `<img>`:

    const r = await fetch("/api/characterise", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query: "SDS"})
    });
    const d = await r.json();
    document.querySelector("#structure").src = d.image;      // data:image/png;base64,...
    // d.summary, d.descriptors, d.profile are [label, value] rows; d.caveats is a list of strings
    // d.files is [{name, url}] pointing at /files/<name>

Endpoints: `/api/characterise`, `/api/modify`, `/api/build`, `/api/series`, `/api/polymer`,
`/api/analyse`, `/api/agent`, plus `/api/meta` for the dropdown vocabulary and `/files/{name}`
for generated structure files. The full schema is browsable at `/api/docs` while the server runs.

If you render your own UI, carry the `caveats` list through to the screen. It is where a withheld
CMC, a regiochemical-ambiguity flag and the commercial-material distribution notes arrive, and
dropping it turns a careful answer into a confident-looking one.
