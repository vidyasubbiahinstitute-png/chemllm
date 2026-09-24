"""Descriptors, functional-group census and surfactant-specific profiling.

Design rule for this layer: every number is either (a) computed directly from the
graph, or (b) produced by a *named, cited* empirical correlation whose applicability
domain is checked first. When a correlation does not apply, the tool says so instead
of returning a number. The agent is not permitted to supply the number itself.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem import rdmolops

from .registry import tool

# ---------------------------------------------------------------- functional groups
FG_SMARTS: dict[str, str] = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "carboxylate": "[CX3](=O)[O-]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "amide": "[CX3](=O)[NX3]",
    "primary_amine": "[NX3;H2;!$(N[C,S]=[O,S,N])]",
    "secondary_amine": "[NX3;H1;!$(N[C,S]=[O,S,N])]",
    "tertiary_amine": "[NX3;H0;!$(N[C,S]=[O,S,N]);!$([N+])]",
    "quaternary_ammonium": "[NX4+]",
    "amine_oxide": "[NX4+][O-]",
    "sulfate_ester": "[OX2][SX4](=O)(=O)[OX1H0-,OX2H1]",
    "sulfonate": "[#6][SX4](=O)(=O)[OX1H0-,OX2H1]",
    "sulfonamide": "[SX4](=O)(=O)[NX3]",
    "phosphate_ester": "[OX2]P(=O)([OX2,OX1-])[OX2,OX1-]",
    "primary_alcohol": "[CX4H2;!$(C[!#6;!#1])][OX2H]",
    "secondary_alcohol": "[CX4H1][OX2H]",
    "hydroxyl": "[OX2H][#6]",
    "ether": "[OX2]([#6])[#6]",
    "ethoxylate_unit": "[OX2][CH2][CH2][OX2]",
    "aromatic_ring": "c1ccccc1",
    "alkene": "[CX3]=[CX3]",
    "nitrile": "[NX1]#[CX2]",
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",
    "glycosidic_acetal": "[CX4H1]([OX2][#6])([OX2][#6])",
    "anomeric_hemiacetal": "[CX4H1]([OX2H])([OX2][CX4])",
    "acetal_ring_oxygen": "[OX2r5,OX2r6]",
    "betaine_zwitterion": "[NX4+]CC(=O)[O-]",
}


def mol_from_smiles(smiles: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    return m


def split_ions(mol: Chem.Mol) -> tuple[Chem.Mol, list[str]]:
    """Separate the surface-active organic ion from simple counterions."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    organics = [f for f in frags
                if f.GetNumHeavyAtoms() > 2 and any(a.GetSymbol() == "C" for a in f.GetAtoms())]
    counter = [Chem.MolToSmiles(f) for f in frags if f not in organics]
    main = max(organics, key=lambda f: f.GetNumHeavyAtoms()) if organics else mol
    return main, counter


def longest_aliphatic_chain(mol: Chem.Mol) -> dict:
    """Longest acyclic carbon path = the hydrophobic tail, for the usual surfactant case."""
    idx = [a.GetIdx() for a in mol.GetAtoms()
           if a.GetSymbol() == "C" and not a.IsInRing() and not a.GetIsAromatic()]
    if not idx:
        return {"n_carbons": 0, "atoms": []}
    sub = set(idx)
    adj = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
               if n.GetIdx() in sub] for i in idx}

    def bfs(start):
        seen = {start: [start]}
        queue = [start]
        far, path = start, [start]
        while queue:
            cur = queue.pop(0)
            for nb in adj[cur]:
                if nb not in seen:
                    seen[nb] = seen[cur] + [nb]
                    queue.append(nb)
                    if len(seen[nb]) > len(path):
                        far, path = nb, seen[nb]
        return far, path

    a, _ = bfs(idx[0])
    _, path = bfs(a)
    return {"n_carbons": len(path), "atoms": path}


# ---------------------------------------------------------------- basic descriptors
@tool(
    name="molecular_descriptors",
    category="properties",
    phase=1,
    description=(
        "Compute graph-derived molecular descriptors for a SMILES: formula, average and "
        "monoisotopic mass, net formal charge, counterions, cLogP (Crippen), TPSA, H-bond "
        "donors/acceptors, rotatable bonds, ring counts, heavy atoms, fraction sp3. "
        "All values are computed, none are estimated."
    ),
    schema={"type": "object",
            "properties": {"smiles": {"type": "string"}},
            "required": ["smiles"]},
)
def molecular_descriptors(smiles: str) -> dict:
    mol = mol_from_smiles(smiles)
    main, counter = split_ions(mol)
    return {
        "smiles": Chem.MolToSmiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw_average": round(Descriptors.MolWt(mol), 3),
        "mw_monoisotopic": round(Descriptors.ExactMolWt(mol), 4),
        "mw_active_ion": round(Descriptors.MolWt(main), 3),
        "net_formal_charge": rdmolops.GetFormalCharge(mol),
        "active_ion_charge": rdmolops.GetFormalCharge(main),
        "counterions": counter,
        "clogp_crippen": round(Crippen.MolLogP(main), 3),
        "molar_refractivity": round(Crippen.MolMR(main), 2),
        "tpsa": round(rdMolDescriptors.CalcTPSA(main), 2),
        "hbd": rdMolDescriptors.CalcNumHBD(main),
        "hba": rdMolDescriptors.CalcNumHBA(main),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(main),
        "rings": rdMolDescriptors.CalcNumRings(main),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(main),
        "heavy_atoms": main.GetNumHeavyAtoms(),
        "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(main), 3),
        "n_stereocentres": len(Chem.FindMolChiralCenters(main, includeUnassigned=True,
                                                         useLegacyImplementation=False)),
    }


@tool(
    name="functional_groups",
    category="properties",
    phase=1,
    description="Count the functional groups present in a molecule using a curated SMARTS "
                "catalogue (acids, esters, amides, amines, quats, sulfates, sulfonates, "
                "ethoxylate units, glycosidic acetals, and more).",
    schema={"type": "object",
            "properties": {"smiles": {"type": "string"}},
            "required": ["smiles"]},
)
def functional_groups(smiles: str) -> dict:
    mol = mol_from_smiles(smiles)
    found = {}
    for name, sma in FG_SMARTS.items():
        patt = Chem.MolFromSmarts(sma)
        if patt is None:
            continue
        n = len(mol.GetSubstructMatches(patt, uniquify=True))
        if n:
            found[name] = n
    return {"smiles": Chem.MolToSmiles(mol), "groups": found,
            "n_group_types": len(found)}


# ---------------------------------------------------------------- surfactant profile
DAVIES_GROUPS = [
    # (label, SMARTS, group number, in_original_davies_table)
    ("sulfate_sodium_salt", "[OX2][SX4](=O)(=O)[O-]", 38.7, True),
    ("carboxylate_sodium_salt", "[CX3](=O)[O-]", 19.1, True),
    ("sulfonate_sodium_salt", "[#6][SX4](=O)(=O)[O-]", 11.0, False),
    ("tertiary_amine", "[NX3;H0;!$(N[C,S]=[O,S,N]);!$([N+])]", 9.4, True),
    ("quaternary_ammonium", "[NX4+;!$([NX4+][O-])]", 9.4, False),
    ("amine_oxide", "[NX4+][O-]", 9.4, False),
    ("ester_free", "[CX3](=O)[OX2H0][#6]", 2.4, True),
    ("carboxylic_acid", "[CX3](=O)[OX2H1]", 2.1, True),
    ("hydroxyl_free", "[OX2H][#6]", 1.9, True),
    ("amide", "[CX3](=O)[NX3]", 2.7, False),
]

HEAD_AREAS_A2 = {  # literature-typical head-group areas at the micelle surface
    "sulfate": 62.0, "sulfonate": 55.0, "carboxylate": 40.0,
    "trimethylammonium": 60.0, "betaine": 65.0, "amine_oxide": 50.0,
    "glucoside": 45.0, "ethoxylate": 50.0, "unknown": 55.0,
}

KLEVENS = {  # log10(CMC / mol L-1) = A - B * n_carbons(tail)
    "alkyl_sulfate_na":       {"A": 1.42, "B": 0.295, "T": "25 C", "n_range": [8, 18]},
    "alkyl_carboxylate_na":   {"A": 1.85, "B": 0.300, "T": "20 C", "n_range": [8, 18]},
    "alkyl_sulfonate_na":     {"A": 1.59, "B": 0.294, "T": "40 C", "n_range": [8, 18]},
    "alkyltrimethylammonium_br": {"A": 1.77, "B": 0.292, "T": "25 C", "n_range": [8, 18]},
}


def classify_head(mol: Chem.Mol) -> str:
    checks = [("sulfate", "[OX2][SX4](=O)(=O)[OX1H0-,OX2H1]"),
              ("sulfonate", "[#6][SX4](=O)(=O)[OX1H0-,OX2H1]"),
              ("betaine", "[NX4+]CC(=O)[O-]"),
              ("amine_oxide", "[NX4+][O-]"),
              ("trimethylammonium", "[NX4+](C)(C)C"),
              ("carboxylate", "[CX3](=O)[O-]"),
              ("glucoside", "[CX4H1]([OX2][#6])([OX2][#6])"),
              ("ethoxylate", "[OX2][CH2][CH2][OX2]")]
    for name, sma in checks:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(sma)):
            return name
    return "unknown"


def charge_class(main: Chem.Mol) -> str:
    q = rdmolops.GetFormalCharge(main)
    pos = any(a.GetFormalCharge() > 0 for a in main.GetAtoms())
    neg = any(a.GetFormalCharge() < 0 for a in main.GetAtoms())
    if pos and neg and q == 0:
        return "zwitterionic/amphoteric"
    if q < 0:
        return "anionic"
    if q > 0:
        return "cationic"
    return "nonionic"


@tool(
    name="surfactant_profile",
    category="properties",
    phase=1,
    description=(
        "Surfactant-specific interpretation of a structure: charge class, head-group type, "
        "hydrophobic tail length, ethoxylate (EO) count, Griffin HLB (nonionics), Davies HLB with "
        "the full group accounting, Tanford tail volume/length and the critical packing parameter "
        "with the predicted aggregate morphology, and a Klevens CMC estimate where a validated "
        "homologous-series correlation exists. Correlations that do not apply return "
        "'not_applicable' with the reason -- do not substitute your own estimate."
    ),
    schema={"type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "head_area_A2": {"type": "number",
                                 "description": "Override the literature-typical head-group area used for the packing parameter."},
            },
            "required": ["smiles"]},
)
def surfactant_profile(smiles: str, head_area_A2: float | None = None) -> dict:
    mol = mol_from_smiles(smiles)
    main, counter = split_ions(mol)
    head = classify_head(main)
    cls = charge_class(main)
    chain = longest_aliphatic_chain(main)
    n_tail = chain["n_carbons"]
    n_eo = len(main.GetSubstructMatches(Chem.MolFromSmarts("[OX2][CH2][CH2][OX2]")))

    # --- Davies HLB with transparent accounting
    contribs, extended = [], False
    for label, sma, value, canonical in DAVIES_GROUPS:
        n = len(main.GetSubstructMatches(Chem.MolFromSmarts(sma), uniquify=True))
        if n:
            contribs.append({"group": label, "count": n, "group_number": value,
                             "in_original_davies_table": canonical})
            extended = extended or not canonical
    n_eo_units = n_eo
    if n_eo_units:
        contribs.append({"group": "oxyethylene_(CH2CH2O)", "count": n_eo_units,
                         "group_number": 0.33, "in_original_davies_table": True})
    lipophilic_c = sum(1 for a in main.GetAtoms()
                       if a.GetSymbol() == "C" and not a.GetIsAromatic()
                       and not any(nb.GetSymbol() in ("O", "N", "S") for nb in a.GetNeighbors()))
    davies = 7.0 + sum(c["count"] * c["group_number"] for c in contribs) - 0.475 * lipophilic_c

    # --- Griffin HLB: only defined for nonionic surfactants
    if cls == "nonionic":
        hydrophile_mass = 0.0
        for a in main.GetAtoms():
            if a.GetSymbol() in ("O", "N"):
                hydrophile_mass += a.GetMass() + a.GetTotalNumHs() * 1.008
            elif a.GetSymbol() == "C" and any(nb.GetSymbol() == "O" for nb in a.GetNeighbors()):
                hydrophile_mass += a.GetMass() + a.GetTotalNumHs() * 1.008
        griffin = round(20.0 * hydrophile_mass / Descriptors.MolWt(main), 2)
        griffin_note = "Griffin HLB = 20 x M(hydrophile)/M(total); hydrophile = O/N atoms plus O-bearing carbons."
    else:
        griffin, griffin_note = None, "Griffin HLB is defined for nonionic surfactants only."

    # --- Tanford geometry + critical packing parameter
    if n_tail >= 6:
        n_t = n_tail - (1 if head in ("carboxylate",) else 0)  # acyl carbon is part of the head
        v = 27.4 + 26.9 * n_t          # A^3, Tanford
        lc = 1.5 + 1.265 * n_t         # A,   Tanford
        a0 = head_area_A2 if head_area_A2 else HEAD_AREAS_A2.get(head, 55.0)
        cpp = v / (a0 * lc)
        morph = ("spherical micelle" if cpp < 1 / 3 else
                 "cylindrical/rod micelle" if cpp < 0.5 else
                 "flexible bilayer / vesicle" if cpp < 1.0 else
                 "planar bilayer" if cpp <= 1.05 else "inverted / reverse phase")
        packing = {"tail_carbons_used": n_t, "tail_volume_A3": round(v, 1),
                   "tail_length_A": round(lc, 2), "head_area_A2": a0,
                   "critical_packing_parameter": round(cpp, 3),
                   "predicted_aggregate": morph,
                   "basis": "Tanford volume/length relations; head-group area is a literature-typical value, not measured."}
    else:
        packing = {"status": "not_applicable",
                   "reason": f"no hydrophobic tail of >=6 carbons found (longest chain = {n_tail} C)"}

    # --- CMC, only for series with published Klevens constants
    series = None
    if head == "sulfate" and cls == "anionic" and n_eo == 0:
        series = "alkyl_sulfate_na"
    elif head == "sulfonate" and cls == "anionic":
        series = "alkyl_sulfonate_na"
    elif head == "carboxylate" and cls == "anionic":
        series = "alkyl_carboxylate_na"
    elif head == "trimethylammonium" and cls == "cationic":
        series = "alkyltrimethylammonium_br"
    if series and KLEVENS[series]["n_range"][0] <= n_tail <= KLEVENS[series]["n_range"][1]:
        k = KLEVENS[series]
        log_cmc = k["A"] - k["B"] * n_tail
        cmc = {"series": series, "log10_cmc_M": round(log_cmc, 3),
               "cmc_mM": round(10 ** log_cmc * 1000, 3), "reference_temperature": k["T"],
               "equation": f"log10(CMC/M) = {k['A']} - {k['B']} x n_C",
               "basis": "Klevens-type homologous-series correlation; pure surfactant in water, no added electrolyte."}
    else:
        why = "no validated Klevens correlation is held for this surfactant class"
        if n_eo:
            why = ("Klevens constants apply to the non-ethoxylated homologous series; this "
                   f"structure carries {n_eo} EO unit(s), which shifts the CMC")
        elif n_tail and series is None:
            why = f"no Klevens constants are held for a {head} head group on a {cls} surfactant"
        elif series:
            why = (f"chain length {n_tail} C is outside the fitted range "
                   f"{KLEVENS[series]['n_range']} for {series}")
        cmc = {"status": "not_applicable",
               "reason": f"{why} (head={head}, class={cls}, n_tail={n_tail}, n_eo={n_eo}). "
                         "Measure it, or supply a class-specific correlation."}

    return {
        "smiles": Chem.MolToSmiles(mol),
        "charge_class": cls,
        "head_group": head,
        "counterions": counter,
        "hydrophobic_tail_carbons": n_tail,
        "ethoxylate_units": n_eo,
        "hlb_griffin": griffin,
        "hlb_griffin_note": griffin_note,
        "hlb_davies": round(davies, 2),
        "hlb_davies_accounting": contribs,
        "hlb_davies_lipophilic_carbons": lipophilic_c,
        "hlb_davies_uses_extended_group_numbers": extended,
        "packing": packing,
        "cmc_estimate": cmc,
        "caveats": [
            "HLB and CPP are semi-empirical design heuristics, not measurements.",
            "Davies group numbers outside the original table are flagged as extended.",
            "Commercial materials are distributions; these values describe the single structure supplied.",
        ],
    }
