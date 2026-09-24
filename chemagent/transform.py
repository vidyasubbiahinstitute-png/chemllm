"""Chemical modification: a curated reaction library, structure-directed edits, and a
spec-driven surfactant builder.

Three complementary routes, because chemists ask for modifications in three ways:
  * "sulfate this alcohol"          -> apply_reaction     (curated reaction SMARTS)
  * "make the tail two carbons longer" -> modify_tail_length (graph edit)
  * "give me a C14 amine oxide"     -> build_surfactant   (head group + tail spec)

Every route returns the starting SMILES, the product SMILES, the transformation applied
and any ambiguity (multiple reactive sites) so the agent can flag it rather than
silently picking one.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from .registry import tool

# --------------------------------------------------------------- reaction library
REACTIONS: dict[str, dict] = {
    "sulfation_of_alcohol": {
        "smarts": "[C;X4;!$(C=[O,N,S]):1][OX2;H1:2]>>[C:1][O:2][S](=O)(=O)[O-]",
        "description": "Sulfation of a primary/secondary alcohol to the alkyl sulfate ester (anion).",
        "reagents": "SO3 / chlorosulfonic acid, then neutralisation",
        "product_class": "anionic surfactant (sulfate)",
        "note": "Sulfate esters hydrolyse under acid; sulfonates do not.",
    },
    "ethoxylation": {
        "smarts": "[C;X4:1][OX2;H1:2]>>[C:1][O:2]CCO",
        "description": "Insert one oxyethylene (EO) unit at a hydroxyl; repeat for higher EO numbers.",
        "reagents": "ethylene oxide, base catalysed",
        "product_class": "nonionic / ether intermediate",
        "note": "Industrial ethoxylation gives a Poisson-like EO distribution, not a discrete adduct.",
    },
    "propoxylation": {
        "smarts": "[C;X4:1][OX2;H1:2]>>[C:1][O:2]CC(C)O",
        "description": "Insert one oxypropylene (PO) unit at a hydroxyl.",
        "reagents": "propylene oxide, base catalysed",
        "product_class": "nonionic / ether intermediate",
    },
    "sulfonation_of_alkylbenzene": {
        "smarts": "[CX4:1][c:2]1[cH:3][cH:4][cH:5][cH:6][cH:7]1>>[C:1][c:2]1[cH:3][cH:4][c:5]([S](=O)(=O)[O-])[cH:6][cH:7]1",
        "description": "Para-sulfonation of a mono-alkylbenzene to the linear alkylbenzene sulfonate (LAS).",
        "reagents": "SO3 / oleum, then neutralisation",
        "product_class": "anionic surfactant (sulfonate)",
    },
    "amidation": {
        "smarts": "[CX3:1](=[OX1:2])[OX2H1].[NX3;H2,H1;!$(N[C,S]=[O,S,N]):3]>>[C:1](=[O:2])[N:3]",
        "description": "Condense a carboxylic acid with a primary/secondary amine to the amide.",
        "reagents": "thermal condensation or via the methyl ester",
        "product_class": "amide intermediate (e.g. the CAPB route)",
        "two_reactants": True,
    },
    "esterification": {
        "smarts": "[CX3:1](=[OX1:2])[OX2H1].[OX2H1:3][CX4:4]>>[C:1](=[O:2])[O:3][C:4]",
        "description": "Fischer esterification of a carboxylic acid with an alcohol.",
        "reagents": "acid catalysis, water removal",
        "product_class": "ester",
        "two_reactants": True,
    },
    "ester_hydrolysis": {
        "smarts": "[CX3:1](=[OX1:2])[OX2:3][CX4:4]>>[C:1](=[O:2])[OH].[OH:3][C:4]",
        "description": "Hydrolysis (saponification) of an ester to the acid and the alcohol.",
        "reagents": "NaOH / H2O",
        "product_class": "acid + alcohol",
    },
    "amide_hydrolysis": {
        "smarts": "[CX3:1](=[OX1:2])[NX3:3]>>[C:1](=[O:2])[OH].[N:3]",
        "description": "Hydrolysis of an amide to the acid and the amine.",
        "reagents": "strong acid or base, heat",
        "product_class": "acid + amine",
    },
    "quaternisation_methyl": {
        "smarts": "[NX3;H0;!$(N[C,S]=[O,S,N]);!$([N+]):1]>>[N+:1]C",
        "description": "Quaternise a tertiary amine with a methylating agent to the quaternary ammonium cation.",
        "reagents": "methyl chloride / dimethyl sulfate",
        "product_class": "cationic surfactant (quat)",
    },
    "quaternisation_benzyl": {
        "smarts": "[NX3;H0;!$(N[C,S]=[O,S,N]);!$([N+]):1]>>[N+:1]Cc1ccccc1",
        "description": "Benzyl quaternisation of a tertiary amine (benzalkonium-type cationic).",
        "reagents": "benzyl chloride",
        "product_class": "cationic surfactant (quat)",
    },
    "betainisation": {
        "smarts": "[NX3;H0;!$(N[C,S]=[O,S,N]);!$([N+]):1]>>[N+:1]CC(=O)[O-]",
        "description": "Carboxymethylate a tertiary amine with sodium chloroacetate to the betaine zwitterion.",
        "reagents": "sodium chloroacetate, aqueous, ~70 C",
        "product_class": "amphoteric surfactant (betaine)",
    },
    "amine_oxidation": {
        "smarts": "[NX3;H0;!$(N[C,S]=[O,S,N]);!$([N+]):1]>>[N+:1][O-]",
        "description": "Oxidise a tertiary amine to the amine N-oxide.",
        "reagents": "hydrogen peroxide, aqueous",
        "product_class": "amphoteric / pH-responsive surfactant",
    },
    "alcohol_to_carboxylic_acid": {
        "smarts": "[CX4;H2:1][OX2;H1]>>[CX3:1](=O)[OH]",
        "description": "Oxidise a primary alcohol to the carboxylic acid.",
        "reagents": "TEMPO/oxidant or catalytic aerobic oxidation",
        "product_class": "fatty acid",
    },
    "carboxymethylation_of_hydroxyl": {
        "smarts": "[#6:1][OX2;H1:2]>>[#6:1][O:2]CC(=O)[O-]",
        "description": "Williamson carboxymethylation of a hydroxyl (the CMC / carboxymethyl route).",
        "reagents": "NaOH then sodium monochloroacetate",
        "product_class": "anionic ether-carboxylate",
    },
    "hydroxypropylation_of_hydroxyl": {
        "smarts": "[#6:1][OX2;H1:2]>>[#6:1][O:2]CC(C)O",
        "description": "Hydroxypropylation of a hydroxyl (HPMC / hydroxypropyl starch route).",
        "reagents": "propylene oxide, alkaline",
        "product_class": "nonionic ether",
    },
    "acetylation_of_hydroxyl": {
        "smarts": "[#6:1][OX2;H1:2]>>[#6:1][O:2]C(C)=O",
        "description": "Acetylation of a hydroxyl.",
        "reagents": "acetic anhydride",
        "product_class": "ester",
    },
    "sulfation_of_sugar_hydroxyl": {
        "smarts": "[#6:1][OX2;H1:2]>>[#6:1][O:2][S](=O)(=O)[O-]",
        "description": "Sulfate half-ester on a carbohydrate hydroxyl (carrageenan/heparin-like motif).",
        "reagents": "SO3-pyridine / chlorosulfonic acid",
        "product_class": "polyanionic polysaccharide",
    },
    "alkene_hydrogenation": {
        "smarts": "[CX3:1]=[CX3:2]>>[CX4:1][CX4:2]",
        "description": "Hydrogenate a C=C double bond (hardening an unsaturated tail).",
        "reagents": "H2, Ni/Pd",
        "product_class": "saturated chain",
    },
}

COUNTERION = {"Na": "[Na+]", "K": "[K+]", "Li": "[Li+]", "NH4": "[NH4+]",
              "Cl": "[Cl-]", "Br": "[Br-]", "Ca": "[Ca+2]", "Mg": "[Mg+2]", "none": ""}


def _canon(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else smi


@tool(
    name="list_reactions",
    category="transform",
    phase=1,
    description="List the curated surfactant/polymer reaction library: each entry gives the "
                "transformation name, a plain-language description, the reagents implied and the "
                "product class. Call this before apply_reaction if unsure which name to use.",
    schema={"type": "object", "properties": {}, "required": []},
)
def list_reactions() -> dict:
    return {"count": len(REACTIONS),
            "reactions": [{"name": k, **{kk: vv for kk, vv in v.items() if kk != "smarts"}}
                          for k, v in REACTIONS.items()]}


@tool(
    name="apply_reaction",
    category="transform",
    phase=1,
    description=(
        "Apply a named transformation from the curated reaction library to a structure. "
        "Returns every distinct product, the number of reactive sites matched, and the reagents "
        "implied. If more than one site matched, the ambiguity is reported -- surface that to the "
        "user rather than silently choosing. Use co_reactant_smiles for two-component reactions "
        "(amidation, esterification)."
    ),
    schema={
        "type": "object",
        "properties": {
            "smiles": {"type": "string", "description": "Starting material."},
            "reaction": {"type": "string", "description": "Name from list_reactions."},
            "co_reactant_smiles": {"type": "string",
                                   "description": "Second reactant for two-component reactions."},
            "repeat": {"type": "integer",
                       "description": "Apply the transformation n times in series (e.g. 3 EO units)."},
            "counterion": {"type": "string",
                           "description": "Add a counterion to the product to balance charge: Na, K, Cl, Br, none."},
        },
        "required": ["smiles", "reaction"],
    },
)
def apply_reaction(smiles: str, reaction: str, co_reactant_smiles: str | None = None,
                   repeat: int = 1, counterion: str = "none") -> dict:
    if reaction not in REACTIONS:
        raise ValueError(f"unknown reaction {reaction!r}; call list_reactions for the library")
    meta = REACTIONS[reaction]
    rxn = AllChem.ReactionFromSmarts(meta["smarts"])
    current = Chem.MolFromSmiles(smiles)
    if current is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")

    sites_first = 0
    for step in range(max(1, int(repeat))):
        reactants = (current,)
        if meta.get("two_reactants"):
            if not co_reactant_smiles:
                raise ValueError(f"{reaction} needs co_reactant_smiles")
            co = Chem.MolFromSmiles(co_reactant_smiles)
            if co is None:
                raise ValueError(f"unparsable co-reactant SMILES: {co_reactant_smiles!r}")
            reactants = (current, co)
        products = rxn.RunReactants(reactants)
        if not products:
            raise ValueError(
                f"{reaction} found no reactive site in {Chem.MolToSmiles(current)!r} "
                f"(step {step + 1}); check the substrate has the required functional group")
        uniq = {}
        for pset in products:
            for p in pset:
                try:
                    Chem.SanitizeMol(p)
                except Exception:
                    continue
                uniq[Chem.MolToSmiles(p)] = p
        if step == 0:
            sites_first = len(uniq)
        main = max(uniq.values(), key=lambda m: m.GetNumHeavyAtoms())
        by_products = [s for s in uniq if s != Chem.MolToSmiles(main)]
        current = main

    product_smiles = Chem.MolToSmiles(current)
    if counterion and counterion != "none":
        q = Chem.GetFormalCharge(current)
        ion = COUNTERION.get(counterion)
        if ion is None:
            raise ValueError(f"unknown counterion {counterion!r}")
        ion_q = Chem.GetFormalCharge(Chem.MolFromSmiles(ion))
        if q != 0 and ion_q != 0 and (q < 0) != (ion_q < 0):
            n_ion = abs(q) // abs(ion_q)
            product_smiles = _canon(product_smiles + "." + ".".join([ion] * max(1, n_ion)))

    return {
        "reaction": reaction,
        "description": meta["description"],
        "reagents": meta["reagents"],
        "product_class": meta.get("product_class"),
        "note": meta.get("note"),
        "smiles_before": _canon(smiles),
        "smiles_after": product_smiles,
        "repeats_applied": max(1, int(repeat)),
        "distinct_products_first_step": sites_first,
        "regiochemical_ambiguity": sites_first > 1,
        "ambiguity_note": ("more than one reactive site matched; the largest product was kept. "
                           "Confirm the intended site with the user."
                           if sites_first > 1 else None),
        "byproducts": by_products,
        "mw_before": round(Descriptors.MolWt(Chem.MolFromSmiles(smiles)), 2),
        "mw_after": round(Descriptors.MolWt(Chem.MolFromSmiles(product_smiles)), 2),
        "formula_after": rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(product_smiles)),
    }


# --------------------------------------------------------------- graph edits
def _tail_path(mol: Chem.Mol) -> list[int]:
    from .descriptors import longest_aliphatic_chain
    return longest_aliphatic_chain(mol)["atoms"]


@tool(
    name="modify_tail_length",
    category="transform",
    phase=1,
    description=(
        "Lengthen or shorten the hydrophobic tail of a surfactant by a number of CH2 units, "
        "keeping the head group untouched. Positive delta extends, negative shortens. "
        "Use this for chain-length series (C10 -> C12 -> C14) without rebuilding the molecule."
    ),
    schema={"type": "object",
            "properties": {"smiles": {"type": "string"},
                           "delta_carbons": {"type": "integer"}},
            "required": ["smiles", "delta_carbons"]},
)
def modify_tail_length(smiles: str, delta_carbons: int) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    frags = Chem.GetMolFrags(mol, asMols=True)
    main = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    others = [Chem.MolToSmiles(f) for f in frags if f is not main]

    path = _tail_path(main)
    if not path:
        raise ValueError("no aliphatic chain found to modify")

    # terminal CH3 end of the chain: the endpoint carrying no heteroatom neighbour
    def is_free_end(i):
        a = main.GetAtomWithIdx(i)
        return (a.GetTotalNumHs() >= 2
                and all(nb.GetSymbol() == "C" for nb in a.GetNeighbors()))
    ends = [p for p in (path[0], path[-1]) if is_free_end(p)]
    if not ends:
        raise ValueError("could not identify a free methyl terminus on the tail")
    end = ends[0]

    rw = Chem.RWMol(main)
    n_before = sum(1 for a in main.GetAtoms() if a.GetSymbol() == "C")
    if delta_carbons > 0:
        prev = end
        for _ in range(delta_carbons):
            new = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(prev, new, Chem.BondType.SINGLE)
            prev = new
    elif delta_carbons < 0:
        for _ in range(-delta_carbons):
            a = rw.GetAtomWithIdx(end)
            nbrs = [nb.GetIdx() for nb in a.GetNeighbors()]
            if a.GetSymbol() != "C" or len(nbrs) != 1:
                raise ValueError("cannot shorten further without cutting into the head group")
            nxt = nbrs[0]
            rw.RemoveAtom(end)
            end = nxt if nxt < end else nxt - 1
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    new_smiles = ".".join([Chem.MolToSmiles(out)] + others)
    return {"smiles_before": _canon(smiles), "smiles_after": _canon(new_smiles),
            "delta_carbons": delta_carbons,
            "tail_carbons_before": len(path),
            "tail_carbons_after": len(_tail_path(out)),
            "total_carbons_before": n_before,
            "mw_before": round(Descriptors.MolWt(main), 2),
            "mw_after": round(Descriptors.MolWt(out), 2)}


# --------------------------------------------------------------- spec-driven builder
HEAD_TEMPLATES: dict[str, dict] = {
    "sulfate":              {"t": "{T}OS(=O)(=O)[O-]", "ci": "Na", "class": "anionic"},
    "ether_sulfate":        {"t": "{T}{EO}OS(=O)(=O)[O-]", "ci": "Na", "class": "anionic", "needs_eo": True},
    "sulfonate":            {"t": "{T}S(=O)(=O)[O-]", "ci": "Na", "class": "anionic"},
    "benzenesulfonate":     {"t": "{T}c1ccc(cc1)S(=O)(=O)[O-]", "ci": "Na", "class": "anionic"},
    "carboxylate":          {"t": "{T-1}C(=O)[O-]", "ci": "Na", "class": "anionic"},
    "carboxylic_acid":      {"t": "{T-1}C(=O)O", "ci": "none", "class": "feedstock"},
    "phosphate":            {"t": "{T}OP(=O)([O-])[O-]", "ci": "Na", "class": "anionic"},
    "sarcosinate":          {"t": "{T-1}C(=O)N(C)CC(=O)[O-]", "ci": "Na", "class": "anionic"},
    "isethionate":          {"t": "{T-1}C(=O)OCCS(=O)(=O)[O-]", "ci": "Na", "class": "anionic"},
    "taurate":              {"t": "{T-1}C(=O)N(C)CCS(=O)(=O)[O-]", "ci": "Na", "class": "anionic"},
    "alcohol":              {"t": "{T}O", "ci": "none", "class": "feedstock"},
    "ethoxylate":           {"t": "{T}{EO}O", "ci": "none", "class": "nonionic", "needs_eo": True},
    "glucoside":            {"t": "{T}O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O", "ci": "none", "class": "nonionic"},
    "monoglyceride":        {"t": "{T-1}C(=O)OCC(O)CO", "ci": "none", "class": "nonionic"},
    "amide_mea":            {"t": "{T-1}C(=O)NCCO", "ci": "none", "class": "nonionic"},
    "trimethylammonium":    {"t": "{T}[N+](C)(C)C", "ci": "Br", "class": "cationic"},
    "benzyldimethylammonium": {"t": "{T}[N+](C)(C)Cc1ccccc1", "ci": "Cl", "class": "cationic"},
    "amine_oxide":          {"t": "{T}[N+](C)(C)[O-]", "ci": "none", "class": "amphoteric"},
    "betaine":              {"t": "{T}[N+](C)(C)CC(=O)[O-]", "ci": "none", "class": "amphoteric"},
    "amidopropyl_betaine":  {"t": "{T-1}C(=O)NCCC[N+](C)(C)CC(=O)[O-]", "ci": "none", "class": "amphoteric"},
    "tertiary_amine":       {"t": "{T}N(C)C", "ci": "none", "class": "feedstock"},
}


def _tail_smiles(n: int, unsaturation: int = 0, branching: str = "linear") -> str:
    if n < 2:
        raise ValueError("tail must have at least 2 carbons")
    if branching == "iso" and n >= 4:
        return "CC(C)" + "C" * (n - 3)
    if branching == "2-ethylhexyl-like" and n >= 7:
        return "CCCC" + "C(CC)" + "C" * (n - 7)
    if unsaturation >= 1:
        if n < 12:
            raise ValueError("a cis-9 double bond needs a hydrophobe of at least 12 carbons "
                             f"(requested {n}); use unsaturation=0")
        # single cis double bond at the 9,10 position, counting from the head end
        return "CCCCCCCC/C=C\\" + "C" * (n - 10)
    return "C" * n


@tool(
    name="build_surfactant",
    category="transform",
    phase=1,
    description=(
        "Build a surfactant from a specification instead of a structure: tail carbon count, head "
        "group, EO number, counterion, unsaturation and branching. Use this when the user describes "
        "what they want ('a C14 amine oxide', 'a 3-EO ether sulfate') rather than supplying a "
        "structure, and for head-group swaps (read the tail length off the original, then rebuild). "
        "Call list_head_groups to see the available head groups."
    ),
    schema={
        "type": "object",
        "properties": {
            "tail_carbons": {"type": "integer",
                             "description": "Carbons in the hydrophobe, including the acyl carbon for amide/ester heads."},
            "head_group": {"type": "string"},
            "n_eo": {"type": "integer", "description": "Oxyethylene units (ether_sulfate, ethoxylate)."},
            "counterion": {"type": "string", "description": "Na, K, Cl, Br, NH4, none. Defaults to the class-typical ion."},
            "unsaturation": {"type": "integer", "description": "0 = saturated, 1 = one cis double bond."},
            "branching": {"type": "string", "enum": ["linear", "iso", "2-ethylhexyl-like"]},
        },
        "required": ["tail_carbons", "head_group"],
    },
)
def build_surfactant(tail_carbons: int, head_group: str, n_eo: int = 0,
                     counterion: str | None = None, unsaturation: int = 0,
                     branching: str = "linear") -> dict:
    if head_group not in HEAD_TEMPLATES:
        raise ValueError(f"unknown head group {head_group!r}; "
                         f"available: {sorted(HEAD_TEMPLATES)}")
    spec = HEAD_TEMPLATES[head_group]
    if spec.get("needs_eo") and n_eo < 1:
        raise ValueError(f"{head_group} requires n_eo >= 1")
    smi = spec["t"]
    if "{T-1}" in smi:
        # the head template supplies the acyl carbon, so the alkyl part carries one fewer
        smi = smi.replace("{T-1}", _tail_smiles(tail_carbons - 1, unsaturation, branching))
    if "{T}" in smi:
        smi = smi.replace("{T}", _tail_smiles(tail_carbons, unsaturation, branching))
    smi = smi.replace("{EO}", "OCC" * max(0, n_eo))
    ci = counterion or spec["ci"]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"constructed an invalid SMILES: {smi!r}")
    q = Chem.GetFormalCharge(mol)
    if q != 0 and ci not in (None, "none"):
        ion = COUNTERION[ci]
        ion_q = Chem.GetFormalCharge(Chem.MolFromSmiles(ion))
        n_ion = max(1, abs(q) // max(1, abs(ion_q)))
        smi = smi + "." + ".".join([ion] * n_ion)
    final = _canon(smi)
    m = Chem.MolFromSmiles(final)
    return {"smiles": final,
            "head_group": head_group,
            "surfactant_class": spec["class"],
            "tail_carbons": tail_carbons,
            "n_eo": n_eo,
            "counterion": ci,
            "branching": branching,
            "unsaturation": unsaturation,
            "formula": rdMolDescriptors.CalcMolFormula(m),
            "mw": round(Descriptors.MolWt(m), 2),
            "assumptions": ["Single discrete structure; a real material of this description "
                            "would be a chain-length (and, if ethoxylated, EO) distribution."]}


@tool(
    name="list_head_groups",
    category="transform",
    phase=1,
    description="List the head groups available to build_surfactant, with their surfactant class.",
    schema={"type": "object", "properties": {}, "required": []},
)
def list_head_groups() -> dict:
    return {"head_groups": [{"name": k, "class": v["class"],
                             "requires_n_eo": bool(v.get("needs_eo"))}
                            for k, v in HEAD_TEMPLATES.items()]}


@tool(
    name="apply_custom_reaction",
    category="transform",
    phase=1,
    description=(
        "Apply an arbitrary reaction SMARTS that is not in the curated library. Use sparingly: "
        "the curated library is reviewed, this is not. The SMARTS is validated before use and the "
        "result is returned with an explicit 'unreviewed_transformation' flag that must be passed "
        "on to the user."
    ),
    schema={"type": "object",
            "properties": {"smiles": {"type": "string"},
                           "reaction_smarts": {"type": "string"},
                           "co_reactant_smiles": {"type": "string"}},
            "required": ["smiles", "reaction_smarts"]},
)
def apply_custom_reaction(smiles: str, reaction_smarts: str,
                          co_reactant_smiles: str | None = None) -> dict:
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    if rxn is None:
        raise ValueError("invalid reaction SMARTS")
    rxn.Initialize()
    mol = Chem.MolFromSmiles(smiles)
    reactants = (mol,) if rxn.GetNumReactantTemplates() == 1 else (
        mol, Chem.MolFromSmiles(co_reactant_smiles or ""))
    if any(r is None for r in reactants):
        raise ValueError("a reactant SMILES did not parse")
    out = {}
    for pset in rxn.RunReactants(reactants):
        for p in pset:
            try:
                Chem.SanitizeMol(p)
                out[Chem.MolToSmiles(p)] = True
            except Exception:
                continue
    if not out:
        raise ValueError("the SMARTS matched no reactive site")
    return {"smiles_before": _canon(smiles), "products": sorted(out),
            "n_products": len(out), "unreviewed_transformation": True,
            "warning": "This transformation is not from the curated library; a chemist must verify it."}
