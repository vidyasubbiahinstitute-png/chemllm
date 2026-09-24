# Chemistry LLM Agent — end-to-end workflow specification

**Scope:** structure rendering, chemical reasoning and polymer manipulation for surfactants,
carbohydrate polymers and biopolymers.
**Status:** this document specifies the workflow *and* describes a working reference
implementation. Every capability described in §5–§7 is implemented and was exercised end to end;
§10 reports what was verified and §8.5 lists what is not yet covered.

---

## 1. Purpose and how this maps to the brief

The brief asks for a chemistry-aware LLM workflow that lets a non-specialist interact with
surfactant and carbohydrate-polymer structures in natural language, with structure rendering,
format conversion, property extraction, reaction reasoning, chemical modification and
polymer-specific handling. This document turns that into an executable specification: an agent
loop, a tool contract, a data layer, an output contract, guardrails, and an evaluation plan.

| Brief requirement | Where it is met |
|---|---|
| Interpret prompts about surfactants | §4 input resolution, §6 W1–W4 |
| Convert inputs to standard representations | `resolve_input`, `convert_format` (§5) |
| Render 2D / 3D structures | `draw_structure`, `draw_svg`, `view_3d`, `export_structure` |
| Retrieve molecular information and descriptors | `molecular_descriptors`, `functional_groups`, `surfactant_profile` |
| Reaction and modification capability | §6 W2, 18-reaction curated library, graph edits, spec builder |
| Show original and modified structures side by side | `draw_before_after` (MCS-highlighted), mandated by the system prompt |
| Short explanation of the chemical change | §7 output contract, clause 4 |
| Head-group / tail-group modification | `build_surfactant` (21 head groups), `modify_tail_length` |
| Similarity and functional-group information | `compare_structures`, `compare_series` |
| Repeat units, glycosidic linkages, chain length, branching, substitution, derivatisation | §8, `build_polysaccharide`, `analyse_polysaccharide` |
| Polymer interpretation understandable to a third party | §8.2 three-layer representation + assumption register |
| Downloadable .pdb, .mol, .sdf | `export_structure`, `export_dataset` |
| Scriptable ChemDraw-like alternative in an LLM pipeline | The whole tool layer is callable directly from Python without the LLM (§12.2) |

---

## 2. The governing design decision

**The model plans; it never computes.** Molecular weight, logP, HLB, charge, CMC, degree of
polymerisation, degree of substitution, linkage positions and similarity are all produced by
deterministic code. The LLM chooses which tool to call, reads what comes back, and writes the
explanation.

This is the difference between a chemistry agent and a chemistry-flavoured chatbot, and it is
what makes the output auditable: every number in an answer traces to a recorded tool call with
recorded arguments. It also means the failure mode is "the agent says it cannot compute that"
rather than "the agent produces a plausible wrong number", which is the failure mode that would
disqualify the tool from formulation work.

---

## 3. Architecture

![Architecture of the chemistry agent]({{artifact:art_e6a916f4-4950-4ddb-9628-75bc94118991}})

### 3.1 Layers

| Layer | Responsibility | Implementation |
|---|---|---|
| Input | Accept whatever form the chemist has | Free text, name, SMILES, InChI, mol block, or a structured spec |
| Orchestration | Plan, call tools, read results, compose the answer | `chemagent/agent.py`, one LLM loop with the tool schemas |
| Tool layer | All chemistry | `chemagent/{resolve,descriptors,transform,render,export,compare,polymer}.py` over RDKit |
| Data layer | All chemical facts | Curated JSON: ingredient lexicon, reaction library, head-group templates, monomer templates, named polymers, derivatisations; plus PubChem for names outside the lexicon |
| Audit | Provenance for every answer | `chemagent/registry.py` captures tool, arguments, result, duration and success for each call; written to a JSON run log |
| Output | Reviewable by a chemist, consumable by a model | PNG/SVG, before/after panel, .mol/.sdf/.pdb/.xyz, CSV/SDF tables, explanation, assumptions |

### 3.2 The tool contract

Every tool is registered with a name, a plain-language description written *for the model*, and a
JSON Schema. The same registry emits the model-facing schemas and the dispatch table, so there is
exactly one definition of what the system can do. Nothing the agent can do exists outside the
registry.

Three conventions make the registry safe to hand to a model:

- **Errors are returned, not raised.** A failed call comes back as `{"ok": false, "error": ...}`
  so the agent can read the failure and correct itself. In the seven demonstration runs, two calls
  failed — one on an unsupported keyword, one on a valence clash — and both were corrected on
  the following turn.
- **Unknown arguments are dropped, not fatal.** A model that invents an extra keyword gets its
  call executed with the valid arguments and an `ignored_arguments` list in the result.
- **Tools refuse rather than guess.** Where a correlation has an applicability domain, the tool
  checks it and returns `status: "not_applicable"` with the reason. See §9.

### 3.3 The agent loop

```
user request
  → model plans and emits tool calls
  → registry dispatches, captures result + provenance
  → results returned to the model
  → repeat until the model answers (cap: 14–18 turns)
  → answer + files + run log
```

Observed cost in the demonstration set: 3–8 model turns and 6–17 tool calls per request,
27–68 seconds wall clock.

---

## 4. Input handling

### 4.1 Resolution cascade

`resolve_input` is the mandatory first call. It tries, in order, and reports which one succeeded:

| Order | Source | Why it is in this position |
|---|---|---|
| 1 | Curated internal lexicon (37 entries, with synonyms) | Reviewed, offline, deterministic; house names and surfactant class names resolve here |
| 2 | Direct structure parse (SMILES / InChI / mol block) | Unambiguous when the user supplies a structure |
| 3 | PubChem PUG-REST name lookup | Coverage for anything outside the lexicon; the CID is reported |
| 4 | Structured build from a specification | For "a C14 amine oxide" there is no name to look up |
| 5 | Escalate | Return `resolved: false` with near-miss suggestions and ask the user |

Every resolution returns canonical SMILES, InChI, InChIKey, the salt-free parent, the neutral
parent, the fragment count, and the **source**. A chemist reviewing an answer can see whether the
structure came from the reviewed lexicon or from a public database.

### 4.2 The five input types in the brief

| Input type | Handling |
|---|---|
| Natural-language description | The model translates it into a tool call; if it implies a spec it goes to `build_surfactant` or `build_polysaccharide`, not to a recalled structure |
| Chemical / common / trade / class names | Lexicon first (synonyms, INCI-style names, abbreviations such as SDS, LAS, CAPB, CTAB, APG), then PubChem |
| SMILES and other structure formats | Direct parse, then standardisation and salt stripping |
| Surfactant-specific inputs (head group, tail length, charge type, HLB target, modification) | `build_surfactant` takes tail carbons, head group, EO number, counterion, unsaturation and branching; `list_head_groups` enumerates the 21 available heads |
| Carbohydrate polymer inputs (repeat unit, linkage, DP, branching, substitution) | `build_polysaccharide` takes monomer, linkage position, DP, branch monomer/position/frequency, substituent and target DS; `list_named_polymers` maps a polymer name onto those parameters |

### 4.3 Ambiguity is surfaced, never resolved silently

Commercial materials are distributions, and names are loose. Where the input under-determines the
structure the tool says so:

- A name that resolves to a distribution (SLES, CAPB, APG, coconut fatty acid) carries a `note`
  recording that the lexicon entry is a single congener of a mixture.
- A reaction that matches more than one site returns `regiochemical_ambiguity: true` and the
  number of distinct products. Sulfating glucose, for example, reports five reactive hydroxyls.
- An unresolved name returns suggestions and the instruction to ask the user.

---

## 5. Tool layer

25 tools in 7 categories; 20 available in Phase 1, 5 more for Phase 2 polymers. The machine-readable
registry (name, phase, category, arguments, whether it writes files, description) is exported as a
CSV deliverable.

| Category | Tools | What they cover |
|---|---|---|
| Input / format | `resolve_input`, `convert_format`, `list_lexicon` | Resolution cascade; SMILES / InChI / InChIKey / molblock / V3000 |
| Properties | `molecular_descriptors`, `functional_groups`, `surfactant_profile` | Mass, formula, charge, cLogP, TPSA, HBD/HBA, rotatable bonds, sp3 fraction, stereocentres; 24-pattern functional-group census; charge class, head-group type, tail length, EO count, Davies and Griffin HLB, Tanford packing parameter, Klevens CMC |
| Modification | `apply_reaction`, `apply_custom_reaction`, `build_surfactant`, `modify_tail_length`, `list_reactions`, `list_head_groups` | 18 curated reaction SMARTS; arbitrary SMARTS with an unreviewed flag; spec-driven construction; graph edits to chain length |
| Render | `draw_structure`, `draw_before_after`, `draw_svg`, `view_3d` | 2D depiction and grids; MCS-highlighted before/after panels; SVG; interactive 3D HTML |
| Export | `export_structure`, `export_dataset` | .mol, .sdf, .pdb, .xyz, .smi with ETKDGv3 embedding and MMFF94/UFF optimisation; multi-structure SDF with properties |
| Compare | `compare_structures`, `compare_series` | ECFP4 and MACCS Tanimoto, MCS, functional groups gained/lost, descriptor deltas; series tabulation |
| Polymer (Phase 2) | `build_polysaccharide`, `analyse_polysaccharide`, `list_monomers`, `list_named_polymers`, `list_substituents` | Repeat-unit assembly, linkage and branching, derivatisation at a target DS, structural interpretation of a supplied carbohydrate |

---

## 6. Workflows

Each workflow below is a named pattern the agent follows. They compose: a real request usually
runs two or three of them.

### W1 — Characterise

**Trigger.** "What is X?", "give me the properties of X", "render X".

**Steps.** `resolve_input` → `draw_structure` / `draw_svg` → `molecular_descriptors` →
`functional_groups` → `surfactant_profile` → `export_structure` if a file is wanted.

**Output.** Depiction, a property table, the charge class and head-group assignment, and the
caveats attached to every semi-empirical number.

**Failure modes.** Name not resolvable → escalate to the user. Correlation outside its domain →
`not_applicable` with the reason, never a substituted estimate.

**Acceptance.** Every reported number appears in a recorded tool result; the source of the
structure is stated.

### W2 — Modify or react

**Trigger.** "Sulfate this", "make the betaine", "what happens if I hydrolyse it".

**Steps.** `resolve_input` → `list_reactions` if the transformation name is uncertain →
`apply_reaction` (repeat for multi-step routes) → `compare_structures` → `draw_before_after` →
`export_structure`.

**Output.** Original and modified structures side by side with the changed atoms highlighted, the
reagents implied by each step, the product class, similarity and the functional groups gained and
lost, and a plain-language explanation of what changed and why it matters.

**Worked example — the cocamidopropyl betaine route, built from lauric acid in two steps:**

![Lauric acid to cocamidopropyl betaine: the full route]({{artifact:art_fa76ba02-f2ca-45f0-86b2-9aa1bea8d000}})

![Before and after: lauric acid versus cocamidopropyl betaine, changed atoms highlighted]({{artifact:art_c7f8fbef-8a87-423e-93ba-b2e524dd56cd}})

The agent resolved lauric acid and DMAPA, applied `amidation` then `betainisation`, and reported
the intermediate amide as well as the final zwitterion — the actual manufacturing route, not a
one-step hand-wave.

**Failure modes.** No reactive site → the tool names the functional group that is missing.
Multiple sites → ambiguity flag, and the agent must put it in the answer.

### W3 — Build from a specification

**Trigger.** "Give me a C14 amine oxide", "a 3-EO ether sulfate", "swap the head group".

**Steps.** `list_head_groups` → `build_surfactant` → W1 to characterise the result.

A head-group swap is this workflow applied twice: read the tail length off the original with
`surfactant_profile`, then rebuild with the new head. Below, SDS → lauramine oxide at constant
C12 tail — the tail is conserved and the whole head is replaced, which the MCS highlighting makes
obvious at a glance:

![SDS to lauramine oxide, constant C12 tail]({{artifact:art_bb41414b-836f-44b6-9651-f60cd7c9e905}})

**Acceptance.** The realised carbon count matches the requested `tail_carbons` for every head
group (verified: 21/21, §10).

### W4 — Compare a series

**Trigger.** "How does chain length affect …", "compare these four".

**Steps.** `build_surfactant` per member → `compare_series` → `draw_structure` grid →
`export_dataset`.

![Sodium alkyl sulfate homologous series, C10 to C16]({{artifact:art_6976f1bc-96d2-4e20-8339-f9febcff9ffd}})

The property layer over the same series and over head groups at constant tail:

![Chain length sets CMC; head group sets hydrophilicity]({{artifact:art_47849a43-a66f-4820-9c8c-07b9e3f30051}})

**Output.** A table with formula, MW, charge class, head group, tail carbons, EO count, cLogP,
TPSA, Davies HLB, Griffin HLB, packing parameter, predicted aggregate morphology and CMC estimate
per member, plus an SDF for downstream modelling.

### W5 — Build a polysaccharide (Phase 2)

**Trigger.** "Build cellulose at DP 6", "what is guar structurally".

**Steps.** `list_named_polymers` if the user gave a name → `build_polysaccharide` →
`analyse_polysaccharide` to verify the built structure → `draw_structure` → `export_structure`.

![Cellulose and amylose at DP 6: the same monomer, different anomeric configuration]({{artifact:art_b7cb7ce7-2005-4ab7-849f-522524e6b26d}})

**Output.** The exemplar SMILES at the built DP, the repeat-unit mass, Mn at the real target DP, a
BigSMILES-style repeat-unit string, the linkage list confirmed from the assembled structure, the
reducing end, and the assumption register.

**Design point.** The analyser is run on the agent's own output. Linkages and anomeric
configuration in the answer are read back off the built molecule, not asserted from the spec.

### W6 — Derivatise and interpret a polysaccharide (Phase 2)

**Trigger.** "CMC at DS 0.7", "cationic guar", "sulfated galactan".

**Steps.** `list_substituents` → `build_polysaccharide` with `substituent`, `ds` and
`substituent_positions` → `analyse_polysaccharide` → `draw_before_after` against the
unsubstituted parent → `compare_structures`.

![Cellulose versus carboxymethyl cellulose at DS 0.7]({{artifact:art_3277b0b9-1fed-45a8-aec3-40f8f94a1964}})

![Guar versus cationic guar]({{artifact:art_fe6b7dc1-82da-4b66-ae27-afd7ab3ce3db}})

**Output.** Realised DS in the exemplar (which differs from the target when DP is small), charge
per repeat unit, Mn including substituent mass, the positions actually substituted, and the
substitution-distribution caveat.

**Constraint enforced.** A position carrying a branch cannot also carry a substituent; the tool
refuses with the list of positions that are free.

### W7 — Export and hand off

Any workflow can terminate in files: `.mol` / `.sdf` for modelling, `.pdb` / `.xyz` for 3D
viewers, multi-record SDF with properties for a screening set, CSV for a report. 3D formats run
ETKDGv3 embedding then MMFF94 (UFF fallback), and the result reports which force field was used
and whether it converged. Counterions are dropped for 3D formats, and this is stated.

---

## 7. Output contract

Every answer carries, in this order:

1. The answer itself — the structure, the property, the recommendation.
2. What was built or changed, and why it matters chemically.
3. The numbers, in a short table, each traceable to a tool call.
4. For any modification: the before/after panel, and one paragraph naming the part of the molecule
   that changed and the chemical consequence.
5. The files produced, by name.
6. Assumptions and caveats, including every `note` and `assumption` the tools returned.

Clause 6 is not decoration. A commercial surfactant is a distribution, an HLB is a design
heuristic, a polymer exemplar is one chain out of a distribution, and a head-group area is a
literature-typical value. An answer that hides those is more dangerous than no answer.

---

## 8. Phase 2: carbohydrate polymers and biopolymers

### 8.1 Why polymers need their own treatment

A polysaccharide is not one molecule. The brief's Phase 2 requirements — repeat-unit
representation, glycosidic linkages, chain length, branching, substitution patterns,
derivatisation, non-standard inputs — are all consequences of that. A single SMILES cannot carry
DP distribution, DS distribution or substitution pattern, so the workflow must be explicit about
which layer any given number comes from.

### 8.2 The three-layer representation

| Layer | What it is | What it is used for |
|---|---|---|
| **SPEC** | Monomer, linkage, anomeric configuration, DP, branching, substituent, DS | The parametric description a formulator actually works with; the thing that gets stored and compared |
| **EXEMPLAR** | One fully specified oligomer SMILES at a stated DP | Everything RDKit computes: depiction, formula, linkage verification, 3D export |
| **POLYMER** | Repeat-unit mass, Mn at the real DP, realised DS, charge per repeat unit, BigSMILES repeat unit | The numbers that describe the material rather than the exemplar |

An answer that reports an exemplar quantity as if it were a polymer quantity is a defect. The
tool returns the two separately, and the output contract requires the DP of the exemplar to be
stated.

### 8.3 Assembly method

Residues are prepared from validated monomer templates with dummy attachment points and joined
with RDKit's `molzip`. The anomeric oxygen is *replaced* in place rather than deleted, which
preserves the bond ordering at C1 and therefore its stereochemistry; the acceptor hydroxyl gains
a dummy. Stereochemistry is carried through from the templates rather than re-derived, and the
result was checked against reference disaccharides (§10).

Conventions:

- The reducing end carries the anomeric configuration of the monomer template. Building from
  `b_glcp` gives β at the reducing end, which is why the DP 2 β-1,4-glucan matches β-cellobiose
  and the α-1,4 one matches α-maltose.
- Substituents are placed deterministically across the available positions to hit the target DS as
  closely as the DP allows; the **realised** DS is always reported alongside the target.
- Branches are placed at a fixed period; the branch position is removed from the substitutable
  set.

### 8.4 Interpreting a carbohydrate the tool did not build

`analyse_polysaccharide` numbers each pyranose/furanose ring independently: the anomeric carbon is
the ring carbon bonded to both the ring oxygen and an exocyclic O/N, and numbering proceeds around
the ring away from the ring oxygen, with the exocyclic carbon on the last ring carbon as C6. From
that it reports every glycosidic linkage as donor→acceptor positions, the reducing end, branch
points, substituents per position, and the residue identity.

Residue identity — and therefore α/β — comes from chirality-aware matching against the curated
monomer templates, with hydroxyl oxygens relaxed so a free OH, a glycosidic bond and a substituent
all match, and with a completeness check so N-acetylglucosamine is not read as glucosamine. A ring
that matches no template is reported as *not matched to a curated template* with no α/β
assignment, rather than guessed. The raw CIP descriptor is also reported, with the warning that it
is not a stable α/β indicator: it flips when the anomeric OH is glycosylated, which is visible in
the cellulose exemplar where the five internal residues read S and the reducing end reads R while
all six are β.

### 8.5 Declared gaps

These are known and deliberately left open rather than approximated silently. Each is reported by
the relevant `list_named_polymers` entry when the user asks about that polymer.

| Gap | Affects | What it needs |
|---|---|---|
| Alternating / block sequence builder | Hyaluronan, pullulan, mixed-linkage β-glucan, alginate M/G blocks, chitosan DDA | A sequence specification instead of a single repeat unit |
| α-L-guluronate template | Alginate gelation (egg-box) | One curated, validated monomer template |
| 3,6-anhydrogalactose template | κ- and ι-carrageenan | One curated template plus its sulfate positions |
| Furanose templates and 2-linked anomeric chemistry | Inulin, fructans, arabinofuranose | Template curation |
| Trisaccharide side chains with pyruvate/acetate | Xanthan | Branch subtree specification |
| Degree of esterification as distinct from DS | Pectin | A second substitution parameter on the uronic acid |
| Reactivity-weighted substitution | All cellulose ethers | A substitution model rather than deterministic placement |
| Chain-length and DS distributions | All polymers | Distribution objects rather than point values |

---

## 9. Guardrails

| Guardrail | Mechanism | Enforced where |
|---|---|---|
| Resolve before reasoning | `resolve_input` mandated as the first call | System prompt; every demonstration run complied |
| No invented numbers | The model has no calculator; all quantities come from tools | Architecture |
| Correlations stay in their domain | Klevens CMC returns `not_applicable` with a reason outside its series or chain-length range; Griffin HLB refuses on ionics; the packing parameter refuses without a ≥6-carbon tail | `surfactant_profile` |
| Extended parameters are flagged | Davies group numbers outside the original table are marked `in_original_davies_table: false` and the profile sets `hlb_davies_uses_extended_group_numbers` | `surfactant_profile` |
| Unreviewed chemistry is labelled | `apply_custom_reaction` sets `unreviewed_transformation: true` and a warning that a chemist must verify it | `transform.py` |
| Ambiguity reaches the user | Multi-site reactions, distribution notes and unresolved names are returned as structured fields, and the output contract requires them in the answer | Registry + system prompt |
| Polymer exemplars are not polymers | Exemplar and polymer quantities are returned in separate blocks with an assumptions list | `build_polysaccharide` |
| Every answer is auditable | Tool, arguments, result, duration and success captured per call and written to a JSON run log | `registry.py`, `agent.py` |

Observed behaviour in the demonstration set: S2 and S4 both asked for the full property set of a
structure whose head group has no published Klevens constants (a betaine and an amine oxide). Both
answers reported HLB, packing parameter and the rest, and reported **no CMC**, naming the missing
correlation instead. A separate smoke run on a 3-EO ether sulfate behaved the same way, since the
Klevens constants are fitted to the non-ethoxylated series. That is the guardrail doing its job on
questions where a plausible number was available to invent.

---

## 10. Validation and evaluation

### 10.1 What has been verified

42 automated checks, all passing, exported as a CSV deliverable:

| Check | n | Method | Result |
|---|---|---|---|
| Monomer templates | 11 | InChIKey compared against the PubChem record for the named reference compound | 11/11 exact match |
| Glycosidic assembly | 6 | Build the DP 2 oligomer and compare its InChIKey to the reference disaccharide: β-cellobiose, α-maltose, β-laminaribiose, β-gentiobiose, β-sophorose, α-isomaltose | 6/6 exact match, including full stereochemistry |
| Builder carbon-count contract | 21 | Build C12 with every head group and count the realised chain carbons | 21/21 give 12 |
| Spec-built identity | 4 | Build oleic acid, SDS, CTAB and lauramine oxide from specifications and compare InChIKeys to PubChem | 4/4 exact match |

The disaccharide check is the load-bearing one for Phase 2: matching β-cellobiose and α-maltose on
the full InChIKey, including the stereochemistry layer, demonstrates that the assembly method
preserves anomeric and ring stereochemistry rather than merely producing the right connectivity.

Separately, all 18 curated reactions were fired against appropriate substrates and all produced
the expected product class, and all 37 lexicon entries parse and standardise.

### 10.2 Test tiers for ongoing development

- **Unit** — every tool against fixed inputs with expected outputs; every reaction against a
  positive and a negative substrate.
- **Chemistry acceptance** — InChIKey identity against public references for anything the tool
  builds that has a name, as in §10.1. This is the tier that catches stereochemistry bugs.
- **Contract** — every tool's declared schema matches its signature; every error path returns a
  structured error rather than raising.
- **Agent-level** — a scenario suite (the seven in the transcripts, extended) asserting the tools
  called, the files produced, the absence of unrecovered failures, and the presence of the
  required caveats in the answer.
- **Anti-hallucination** — a set of questions whose correct answer is a refusal (CMC of an
  ethoxylated sulfate, HLB of an unnamed structure, linkage of an unresolvable input). The
  measured quantity is the refusal rate, and it should be 100 %.
- **Human review** — a chemist scores a sample of answers on the rubric in §10.3.

### 10.3 Chemist acceptance rubric

Score each sampled answer 0–2 on: structure correctness; property correctness; correct handling of
what the tool could *not* do; clarity of the before/after; completeness of the assumptions; and
usefulness of the explanation. A release candidate should score ≥ 10/12 on 90 % of sampled
answers, with structure correctness never below 2.

### 10.4 Metrics to track

Resolution rate by input type; refusal correctness (refused when it should, answered when it
could); tool-call failure rate and recovery rate; turns and wall clock per request; the fraction
of answers whose numeric claims all trace to a recorded call (target: 1.0); and chemist rubric
score.

### 10.5 Demonstration set

Seven scenarios spanning both phases: characterise, multi-step route, homologous series,
head-group swap, polymer build, polymer derivatisation, named polymer plus modification.
Aggregate: **70 tool calls, 2 recovered failures, 0 unrecovered failures.** The two were an
unsupported keyword argument (S2) and a valence error from substituting a position that
already carried a branch (S7); both are now prevented by the registry's argument filter and
the builder's branch-position guard respectively. Full prompts, call traces and answers are
in the transcripts deliverable.

---

## 11. Data governance

- **The data layer is the only source of chemical facts.** Lexicon, reactions, head groups,
  monomers, named polymers and derivatisations live in reviewable JSON, version-controlled
  alongside the code. Adding a house ingredient is a data change, not a code change.
- **Provenance is reported, not implied.** Each resolution states whether it came from the curated
  lexicon, a direct parse, or PubChem — with the CID in the last case.
- **Approximations are annotated in the data, not in the prose.** Lexicon entries for
  distributions (SLES, CAPB, LAS isomers, Span 20, APG) carry a `note` that travels with every
  answer that uses them.
- **Public data.** PubChem is public-domain; usage is a name-to-structure lookup with the CID
  recorded. Any internal formulation data added later should stay behind the same tool interface
  so access control sits in one place.
- **Nothing proprietary is sent to the model beyond what the user typed** — the tool layer runs
  locally and returns values, so structures resolved from the internal lexicon never leave the
  environment except as the SMILES the agent reasons about.

---

## 12. Deployment and integration

### 12.1 Shape

The reference implementation is a Python package with no service dependencies beyond RDKit and an
LLM endpoint. `run_agent(prompt)` returns the answer, the file list and the full run log. That
makes five deployment shapes available without redesign: a notebook function, the HTML app and
the Gradio app in §12.2, a chat interface over the same loop, and a batch script that runs a list
of prompts.

### 12.2 Browser front ends

Two, sharing one tool layer. Neither holds chemical knowledge of its own.

**HTML app** — one self-contained page over a JSON API, no build step, no framework, no CDN:

    pip install fastapi uvicorn
    python -m chemagent.webapp                    # http://127.0.0.1:8000
    python -m chemagent.webapp --host 0.0.0.0 --port 9000

`chemagent/static/index.html` is plain HTML, CSS and vanilla JavaScript; `chemagent/webapp.py`
is a FastAPI app exposing one POST endpoint per workflow (`/api/characterise`, `/api/modify`,
`/api/build`, `/api/series`, `/api/polymer`, `/api/analyse`, `/api/agent`), plus `/api/meta` for
the control vocabulary and `/files/{name}` for generated structure files. Rendered structures come
back inline as base64 data URIs, so a panel is one request and one reply. This is the form to use
for embedding: serve the app behind your own path and iframe `/`, replace `index.html` with your
own markup against the same endpoints, or call the endpoints from a script — they are ordinary
JSON, documented at `/api/docs`.

**Gradio app** — the same eight panels with no front-end code to maintain, for demos and internal use:

    pip install gradio
    python -m chemagent.gui            # http://127.0.0.1:7860
    python -m chemagent.gui --share    # temporary public link

Panels in both: **Characterise** (W1), **Modify / react** (W2), **Build from a spec** (W3),
**Compare a series** (W4), **Polymer builder** (W5/W6, with a named-polymer dropdown that fills
the controls from the curated library), **Polymer analyser**, **Ask the agent**, **Tool registry**.

Three properties make them skins rather than second implementations:

- **They dispatch through the same registered tools.** A panel, an HTTP call and an agent request
  that do the same thing return identical values; there is no UI-only chemistry path to keep in
  sync. Verified: `/api/characterise` on SDS returns the same Davies HLB (40.48) and CMC
  (7.586 mM) as `surfactant_profile` called directly, and CMC at DP 4 returns the same realised
  DS (0.75) and Mn (164,552 at DP 800) through both front ends.
- **They carry the guardrails to the screen.** Each panel renders the `caveats`, `assumptions`,
  `note` and `warning` fields the tools returned, so a `not_applicable` CMC or a
  regiochemical-ambiguity flag is visible rather than silently absent. Tool errors render as
  messages in place — "position 6 carries the branch, so it cannot also be substituted" — not as
  stack traces, and an unresolvable input renders the resolver's own suggestions and next step.
- **Every panel is reachable without a browser.** The HTML app's endpoints are plain JSON; the
  Gradio panels are named API routes (`Client(url).predict(..., api_name="/characterise")`).

The agent panel needs an LLM client (`ANTHROPIC_API_KEY`, or a host that injects one) and reports
that it is unavailable when there is none; every other panel is deterministic and works offline
apart from the optional PubChem fallback. `/files/{name}` serves basenames only and 404s on
traversal attempts.

### 12.3 The tool layer is useful without the LLM

Every tool is an ordinary Python function with a JSON Schema. A modelling script can call
`build_surfactant`, `surfactant_profile` and `export_dataset` directly to generate a screening
set, with no model in the loop. This is the "scriptable ChemDraw alternative" the brief asks for,
and it is also how the tools are unit-tested.

### 12.4 Integration points

Descriptor and structure output (SDF with properties) is the natural handoff to QSPR or
coarse-grained simulation workflows; the exemplar `.pdb` is the handoff to a 3D viewer or to
molecular dynamics setup. Both are already emitted.

---

## 13. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Plausible-but-wrong numbers | A formulator acts on an invented CMC | Model cannot compute; correlations check their domain and refuse; every number traces to a call |
| Single structure taken for a commercial material | Wrong conclusion about a distribution-dominated property | Distribution notes in the lexicon, assumption register on every polymer build, output contract clause 6 |
| Stereochemistry silently wrong in assembled polymers | Everything downstream is wrong and it is invisible in a 2D picture | InChIKey validation against reference disaccharides in CI; templates validated against PubChem |
| Curated library drifts out of date or contains errors | Systematic error across many answers | Data layer is reviewed JSON under version control; the validation suite re-runs against public references |
| Over-trust in HLB / CPP heuristics | Formulation decisions made on a design heuristic | Caveats returned by the tool and mandated in the answer; both HLB scales reported when applicable so disagreement is visible |
| Scope creep into property prediction | A PoC becomes an unvalidated prediction engine | Property *estimation* is restricted to named, cited correlations with declared domains; anything else is out of scope until it has a validation set |
| Model drift on prompt changes | Guardrail compliance regresses silently | The anti-hallucination tier in §10.2 runs on every prompt change |

---

## 14. Phasing

**Phase 1 — surfactant agent.** Delivered in the reference implementation: resolution cascade,
descriptors and surfactant profiling, 18-reaction library, spec-driven builder with 21 head
groups, tail-length edits, before/after rendering, comparison, and all export formats.
*Exit criteria:* the four Phase 1 scenarios run clean; builder carbon-count and spec-built
identity checks pass; chemist rubric ≥ 10/12.

**Phase 2 — carbohydrate polymers.** Delivered: three-layer representation, 11 validated monomer
templates, linkage-directed assembly with branching, 10 derivatisation chemistries at target DS,
20 named polymers mapped to build parameters, structural analysis with residue typing.
*Exit criteria:* disaccharide InChIKey checks pass; a formulator can get from a polymer name to a
verified exemplar with its assumptions in one request.

**Phase 2b — the declared gaps.** The sequence builder in §8.5 is the highest-value next
increment: it unlocks hyaluronan, alginate M/G, chitosan DDA and mixed-linkage glucans in one
change, and it is a specification change rather than new chemistry.

**Phase 3 — candidates, not committed.** Structure–property models trained on internal
formulation data behind the same tool interface; a reaction-feasibility check on the curated
library; batch mode over ingredient lists; and a UI over the same loop.

---

## Appendix A — curated reaction library

| Reaction | Product class |
|---|---|
| `sulfation_of_alcohol` | anionic surfactant (sulfate) |
| `ethoxylation`, `propoxylation` | nonionic / ether intermediate |
| `sulfonation_of_alkylbenzene` | anionic surfactant (sulfonate, LAS) |
| `amidation`, `esterification` | amide / ester intermediates |
| `ester_hydrolysis`, `amide_hydrolysis` | acid + alcohol / acid + amine |
| `quaternisation_methyl`, `quaternisation_benzyl` | cationic surfactant (quat) |
| `betainisation` | amphoteric surfactant (betaine) |
| `amine_oxidation` | amphoteric / pH-responsive surfactant |
| `alcohol_to_carboxylic_acid` | fatty acid |
| `carboxymethylation_of_hydroxyl` | anionic ether-carboxylate (CMC route) |
| `hydroxypropylation_of_hydroxyl` | nonionic ether (HPMC route) |
| `acetylation_of_hydroxyl` | ester |
| `sulfation_of_sugar_hydroxyl` | polyanionic polysaccharide |
| `alkene_hydrogenation` | saturated chain |

## Appendix B — head groups available to the builder

anionic: sulfate, ether sulfate, sulfonate, benzenesulfonate, carboxylate, phosphate,
sarcosinate, isethionate, taurate · cationic: trimethylammonium, benzyldimethylammonium ·
amphoteric: amine oxide, betaine, amidopropyl betaine · nonionic: ethoxylate, glucoside,
monoglyceride, amide MEA · feedstock: alcohol, carboxylic acid, tertiary amine

## Appendix C — monomer templates (all InChIKey-validated)

β-/α-D-glucopyranose, β-/α-D-galactopyranose, β-D-mannopyranose, β-D-xylopyranose,
N-acetyl-β-D-glucosamine, β-D-glucosamine, β-D-glucopyranuronic acid,
α-D-galactopyranuronic acid, β-D-mannopyranuronic acid

## Appendix D — named polymers recognised

cellulose, CMC, HEC, HPMC, amylose, amylopectin, starch, OSA starch, chitin, chitosan, dextran,
pullulan, alginate, pectin, hyaluronan, carrageenan, guar, cereal β-glucan, xanthan, inulin

## Appendix E — derivatisation chemistries

carboxymethyl, hydroxyethyl, hydroxypropyl, methyl, acetyl, sulfate, phosphate,
cationic (CHPTAC), octenylsuccinate, succinate — each with its reagent chemistry, typical DS
range and known effect on the polymer.

## Appendix F — deliverables

| File | Contents |
|---|---|
| [chemagent_poc.tar.gz]({{artifact:349d8cef-4c76-4a01-8c91-f0c63018685a}}) | The reference implementation: 10 modules, 2 curated data files, README and the exported tables |
| [demo_transcripts.md]({{artifact:a4e16490-8082-405e-92c0-d21617b0ec43}}) | Seven end-to-end runs: prompts, tool traces, answers |
| [tool_registry.csv]({{artifact:1197142d-096b-4af6-a185-1e9aa357b154}}) | The full tool surface: name, phase, category, arguments, description |
| [validation_report.csv]({{artifact:217753d6-cfc7-4873-b1f8-486b2316dce1}}) | The 42 validation checks with computed and reference InChIKeys |
| [surfactant_series_table.csv]({{artifact:e4e7220a-0eb4-43d9-aafa-c8bd422af8fe}}) | 12 surfactants across chain length and head group with the full property set |
| [fig_architecture.png]({{artifact:1f41e0f8-9c43-417e-84e5-9d23e73f943b}}) | The architecture diagram (§3) |
| [fig_surfactant_property_layer.png]({{artifact:4ca627a0-3e50-4f60-956d-ce78b70de046}}) | Chain-length and head-group property trends (§6 W4) |
