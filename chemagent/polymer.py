"""Phase 2: carbohydrate polymers and biopolymers.

Representation strategy
-----------------------
A polysaccharide is not one molecule, so the PoC carries three linked layers and is
explicit about which one any answer comes from:

  1. SPEC      - the parametric description a formulator actually works with
                 (monomer, linkage, anomeric configuration, DP, branching, DS).
  2. EXEMPLAR  - a single, fully specified oligomer SMILES built from the spec at a
                 stated DP. Everything RDKit computes is computed on this.
  3. POLYMER   - repeat-unit level quantities derived analytically from the spec
                 (repeat mass, Mn at the target DP, DS, charge per repeat unit),
                 plus a BigSMILES-style repeat-unit string.

Assembly uses dummy-atom templates and RDKit's molzip, which forms the glycosidic bond
by substituting matched attachment points -- stereochemistry at the anomeric centre and
at the ring carbons is carried through unchanged from the validated monomer templates.
"""
from __future__ import annotations

import json
import os

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from .registry import tool

_HERE = os.path.dirname(__file__)
with open(os.path.join(_HERE, "data", "monomers.json")) as fh:
    _MDATA = json.load(fh)
MONOMERS: dict = _MDATA["monomers"]
NAMED_POLYMERS: dict = _MDATA["named_polymers"]
SUBSTITUENTS: dict = _MDATA["substituents"]


# ------------------------------------------------------------------ template surgery
def _template(key: str) -> Chem.Mol:
    if key not in MONOMERS:
        raise ValueError(f"unknown monomer {key!r}; available: {sorted(MONOMERS)}")
    m = Chem.MolFromSmiles(MONOMERS[key]["smiles"])
    if m is None:
        raise ValueError(f"monomer template for {key!r} is not parsable")
    return m


def _find_mapped(mol: Chem.Mol, mapno: int) -> int:
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() == mapno:
            return a.GetIdx()
    raise ValueError(f"template has no position {mapno}")


def _attach(rw: Chem.RWMol, o_idx: int, frag_smiles: str) -> None:
    """Replace the H on a hydroxyl oxygen with a fragment (etherification/esterification)."""
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None:
        raise ValueError(f"bad substituent SMILES: {frag_smiles!r}")
    offset = rw.GetNumAtoms()
    rw.InsertMol(frag)
    rw.AddBond(o_idx, offset, Chem.BondType.SINGLE)
    o = rw.GetAtomWithIdx(o_idx)
    o.SetNumExplicitHs(0)
    o.SetNoImplicit(True)


def _residue(key: str, donor_tag: int | None, acceptor: dict[int, int] | None,
             substituents: dict[int, str] | None) -> Chem.Mol:
    """One sugar residue prepared for molzip.

    donor_tag   -- attach a dummy in place of the anomeric OH oxygen (this residue
                   donates its C1 to a glycosidic bond).
    acceptor    -- {position: tag}; attach a dummy to that hydroxyl oxygen.
    substituents-- {position: fragment SMILES} for derivatisation.
    """
    rw = Chem.RWMol(_template(key))
    if substituents:
        for pos, frag in substituents.items():
            _attach(rw, _find_mapped(rw, pos), frag)
    if acceptor:
        for pos, tag in acceptor.items():
            o_idx = _find_mapped(rw, pos)
            d = rw.AddAtom(Chem.Atom(0))
            rw.GetAtomWithIdx(d).SetAtomMapNum(tag)
            rw.AddBond(o_idx, d, Chem.BondType.SINGLE)
            o = rw.GetAtomWithIdx(o_idx)
            o.SetNumExplicitHs(0)
            o.SetNoImplicit(True)
    if donor_tag is not None:
        o_idx = _find_mapped(rw, 1)
        a = rw.GetAtomWithIdx(o_idx)
        # in-place replacement keeps the bond ordering at C1, preserving its chiral tag
        a.SetAtomicNum(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetAtomMapNum(tag_ := donor_tag)
        _ = tag_
    mol = rw.GetMol()
    for a in mol.GetAtoms():                      # strip positional labels; keep dummy tags
        if a.GetAtomicNum() != 0:
            a.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    return mol


def _zip(residues: list[Chem.Mol]) -> Chem.Mol:
    combined = residues[0]
    for r in residues[1:]:
        combined = Chem.CombineMols(combined, r)
    params = Chem.rdmolops.MolzipParams()
    params.label = Chem.rdmolops.MolzipLabel.AtomMapNumber
    out = Chem.molzip(combined, params)
    Chem.SanitizeMol(out)
    return out


# ------------------------------------------------------------------ assembly
def _plan_substitution(dp: int, positions: list[int], ds: float) -> dict[int, list[int]]:
    """Deterministically spread substituents so the realised DS is as close as possible
    to the target. Returns {residue_index: [positions]}."""
    total = int(round(ds * dp))
    plan: dict[int, list[int]] = {}
    if total <= 0 or not positions:
        return plan
    slots = [(i, p) for i in range(dp) for p in positions]
    if total > len(slots):
        raise ValueError(f"DS {ds} needs {total} substitutions but only {len(slots)} "
                         f"hydroxyls at positions {positions} across DP {dp}")
    step = len(slots) / total
    for k in range(total):
        i, p = slots[int(k * step)]
        plan.setdefault(i, []).append(p)
    return plan


@tool(
    name="build_polysaccharide",
    category="polymer",
    phase=2,
    writes_files=False,
    description=(
        "Build an explicit oligosaccharide exemplar from a repeat-unit specification: monomer, "
        "glycosidic linkage position, anomeric configuration, degree of polymerisation, optional "
        "regular branching, and optional derivatisation at a target degree of substitution. "
        "Returns the exemplar SMILES plus polymer-level quantities (repeat mass, Mn, realised DS, "
        "charge per repeat unit) and an explicit assumptions list. Call list_monomers or "
        "list_named_polymers first if the user gave a polymer name rather than a spec."
    ),
    schema={
        "type": "object",
        "properties": {
            "monomer": {"type": "string", "description": "Key from list_monomers, e.g. b_glcp, a_glcp, b_galp, glcnac."},
            "linkage": {"type": "integer", "description": "Acceptor position of the main-chain glycosidic bond: 2, 3, 4 or 6."},
            "dp": {"type": "integer", "description": "Degree of polymerisation of the exemplar to build (keep <= 12 for speed)."},
            "target_dp": {"type": "integer", "description": "The real polymer DP to report Mn for, if different from the exemplar."},
            "branch_monomer": {"type": "string"},
            "branch_linkage": {"type": "integer", "description": "Position on the backbone carrying the branch (commonly 6)."},
            "branch_every": {"type": "integer", "description": "Attach a branch every n backbone residues."},
            "substituent": {"type": "string", "description": "Key from list_substituents, e.g. carboxymethyl, hydroxypropyl, acetyl, sulfate."},
            "ds": {"type": "number", "description": "Target degree of substitution (substituents per repeat unit)."},
            "substituent_positions": {"type": "array", "items": {"type": "integer"},
                                      "description": "Hydroxyls available for substitution, e.g. [6] or [2,3,6]."},
        },
        "required": ["monomer", "linkage", "dp"],
    },
)
def build_polysaccharide(monomer: str, linkage: int, dp: int, target_dp: int | None = None,
                         branch_monomer: str | None = None, branch_linkage: int | None = None,
                         branch_every: int | None = None, substituent: str | None = None,
                         ds: float = 0.0, substituent_positions: list[int] | None = None) -> dict:
    if dp < 1:
        raise ValueError("dp must be >= 1")
    if dp > 30:
        raise ValueError("exemplar DP is capped at 30; report polymer quantities with target_dp instead")
    mono = MONOMERS[monomer] if monomer in MONOMERS else None
    if mono is None:
        raise ValueError(f"unknown monomer {monomer!r}; available: {sorted(MONOMERS)}")
    if linkage not in mono["positions"]:
        raise ValueError(f"{monomer} has no free hydroxyl at position {linkage}; "
                         f"available: {mono['positions']}")

    # substitution plan (never on the positions consumed by the glycosidic bonds)
    free_positions = [p for p in mono["positions"] if p not in (1, linkage)]
    sub_pos = substituent_positions or free_positions
    bad = [p for p in sub_pos if p not in free_positions]
    if bad:
        raise ValueError(f"positions {bad} are not free for substitution "
                         f"(free: {free_positions}; position {linkage} carries the main chain)")
    if branch_monomer and branch_every:
        bl = branch_linkage or 6
        if substituent_positions and bl in sub_pos:
            raise ValueError(
                f"position {bl} carries the branch, so it cannot also be substituted; "
                f"choose substituent_positions from {[p for p in free_positions if p != bl]}")
        sub_pos = [p for p in sub_pos if p != bl]
    frag = None
    if substituent:
        if substituent not in SUBSTITUENTS:
            raise ValueError(f"unknown substituent {substituent!r}; available: {sorted(SUBSTITUENTS)}")
        frag = SUBSTITUENTS[substituent]["smiles"]
    plan = _plan_substitution(dp, sub_pos, ds) if frag else {}

    # branch plan
    branch_sites: list[int] = []
    if branch_monomer and branch_every:
        blink = branch_linkage or 6
        if blink not in free_positions:
            raise ValueError(f"branch position {blink} is not free (free: {free_positions})")
        branch_sites = [i for i in range(dp) if (i + 1) % branch_every == 0 and i != dp - 1]

    # junction tags: residue i donates with tag i+1, residue i+1 accepts with tag i+1
    residues: list[Chem.Mol] = []
    tagmap: dict[int, str] = {}
    for i in range(dp):
        donor = (i + 1) if i < dp - 1 else None
        acceptor = {linkage: i} if i > 0 else {}
        subs = {p: frag for p in plan.get(i, [])} if frag else {}
        if i in branch_sites:
            btag = 1000 + i
            acceptor[branch_linkage or 6] = btag
            residues.append(_residue(branch_monomer, btag, None, None))
            tagmap[btag] = f"branch {branch_monomer} (1->{branch_linkage or 6}) on residue {i+1}"
        residues.append(_residue(monomer, donor, acceptor or None, subs or None))

    mol = _zip(residues)
    smiles = Chem.MolToSmiles(mol)

    # ---- polymer-level quantities from the spec, not from the exemplar
    unit = Chem.MolFromSmiles(MONOMERS[monomer]["smiles"].replace(":1", "").replace(":2", "")
                              .replace(":3", "").replace(":4", "").replace(":6", ""))
    monomer_mass = Descriptors.MolWt(unit)
    repeat_mass = monomer_mass - 18.015                      # condensation loses one water
    sub_mass = 0.0
    realised_ds = 0.0
    if frag:
        fm = Chem.MolFromSmiles(frag)
        sub_mass = Descriptors.MolWt(fm) - 1.008             # replaces one hydroxyl H
        realised_ds = sum(len(v) for v in plan.values()) / dp
    tdp = target_dp or dp
    mn = repeat_mass * tdp + 18.015 + sub_mass * realised_ds * tdp

    charge_per_unit = Chem.GetFormalCharge(Chem.MolFromSmiles(frag)) * realised_ds if frag else 0.0
    linkage_label = f"{MONOMERS[monomer]['short']}-(1->{linkage})"

    return {
        "exemplar_smiles": smiles,
        "exemplar_dp": dp,
        "exemplar_formula": rdMolDescriptors.CalcMolFormula(mol),
        "exemplar_mw": round(Descriptors.MolWt(mol), 2),
        "exemplar_heavy_atoms": mol.GetNumHeavyAtoms(),
        "spec": {"monomer": monomer, "monomer_name": MONOMERS[monomer]["name"],
                 "anomeric_configuration": MONOMERS[monomer]["anomeric"],
                 "main_chain_linkage": f"(1->{linkage})",
                 "linkage_label": linkage_label,
                 "branch": (f"{branch_monomer} (1->{branch_linkage or 6}) every {branch_every} units"
                            if branch_sites else None),
                 "n_branches_in_exemplar": len(branch_sites),
                 "substituent": substituent, "target_ds": ds,
                 "substituent_positions": sub_pos if frag else None},
        "polymer_quantities": {
            "repeat_unit_mass_unsubstituted": round(repeat_mass, 3),
            "substituent_added_mass": round(sub_mass, 3) if frag else 0.0,
            "realised_ds_in_exemplar": round(realised_ds, 3),
            "target_dp_for_Mn": tdp,
            "Mn_estimate": round(mn, 1),
            "formal_charge_per_repeat_unit": round(charge_per_unit, 3),
        },
        "bigsmiles_repeat_unit": bigsmiles_repeat(monomer, linkage),
        "assumptions": [
            f"Exemplar built at DP {dp}; a real sample is a distribution of chain lengths.",
            "Substituents are placed deterministically to hit the target DS; in a real "
            "derivatised polysaccharide the distribution along and around the chain is "
            "statistical and reactivity-weighted (O6 > O2 > O3 for cellulose ethers).",
            "Mn is computed from the repeat-unit mass and the target DP, not measured.",
            "Anomeric configuration and ring form are fixed by the monomer template.",
        ],
        "junctions": tagmap,
    }


def bigsmiles_repeat(monomer: str, linkage: int) -> str:
    """A BigSMILES-style stochastic object for the repeat unit.

    [>] marks the anomeric (donor) attachment point, [<] the acceptor hydroxyl that the
    next residue's C1 bonds to. End groups are left implicit.
    """
    raw = MONOMERS[monomer]["smiles"]
    body = raw.replace("[OH:1]", "[>]")
    # the acceptor oxygen is retained in the chain; the descriptor sits on its outer side
    acc = f"[OH:{linkage}]"
    body = body.replace(acc, "[<]O" if raw.startswith(acc) else "O[<]")
    for p in (2, 3, 4, 6):
        body = body.replace(f"[OH:{p}]", "O")
    return "{[]" + body + "[]}"


# residue-identity queries: the validated monomer templates with every oxygen relaxed to
# [OX2] so a free OH and an O-glycosyl/O-substituted oxygen both match, chirality retained
def _residue_queries() -> dict[str, tuple[Chem.Mol, set[int]]]:
    out = {}
    for key, v in MONOMERS.items():
        m = Chem.MolFromSmiles(v["smiles"])
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        relaxed = set()
        rw = Chem.RWMol(m)
        for a in list(rw.GetAtoms()):
            # only hydroxyl oxygens are relaxed; carbonyl oxygens must stay exact
            if (a.GetAtomicNum() == 8 and a.GetDegree() == 1 and not a.IsInRing()
                    and a.GetTotalNumHs() == 1
                    and a.GetBonds()[0].GetBondType() == Chem.BondType.SINGLE):
                rw.ReplaceAtom(a.GetIdx(), Chem.AtomFromSmarts("[OX2]"))
                relaxed.add(a.GetIdx())
        q = rw.GetMol()
        Chem.SanitizeMol(q, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        out[key] = (q, relaxed)
    return out


_RESIDUE_QUERIES = None


def identify_residues(mol: Chem.Mol, rings: list[tuple[int, ...]]) -> dict[int, dict]:
    """Assign each sugar ring to a curated monomer template (and therefore to an
    unambiguous alpha/beta assignment) by chirality-aware substructure matching."""
    global _RESIDUE_QUERIES
    if _RESIDUE_QUERIES is None:
        _RESIDUE_QUERIES = _residue_queries()
    assignment: dict[int, dict] = {}
    # most specific (largest) template first, so N-acetylglucosamine is not read as glucosamine
    for key, (q, relaxed) in sorted(_RESIDUE_QUERIES.items(),
                                    key=lambda kv: -kv[1][0].GetNumHeavyAtoms()):
        for match in mol.GetSubstructMatches(q, useChirality=True, uniquify=True):
            mset = set(match)
            # a residue is only this monomer if nothing hangs off the matched atoms except
            # through the relaxed hydroxyl oxygens (which carry linkages and substituents)
            ok = True
            for qi, mi in enumerate(match):
                if qi in relaxed:
                    continue
                at = mol.GetAtomWithIdx(mi)
                if any(nb.GetIdx() not in mset for nb in at.GetNeighbors()):
                    ok = False
                    break
            if not ok:
                continue
            for ri, ring in enumerate(rings):
                if set(ring) <= mset and ri not in assignment:
                    assignment[ri] = {"monomer_key": key,
                                      "monomer_name": MONOMERS[key]["name"],
                                      "short": MONOMERS[key]["short"],
                                      "anomeric_configuration": MONOMERS[key]["anomeric"]}
    return assignment


# ------------------------------------------------------------------ analysis
def _ring_numbering(mol: Chem.Mol, ring: tuple[int, ...]) -> dict[int, int] | None:
    """Map atom index -> carbohydrate position number for one pyranose/furanose ring."""
    ring_set = set(ring)
    oxys = [i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "O"]
    if len(oxys) != 1:
        return None
    ring_o = oxys[0]
    carbons = [i for i in ring if i != ring_o]
    # anomeric carbon: ring carbon bonded to the ring O and to an exocyclic O or N
    anomeric = None
    for c in mol.GetAtomWithIdx(ring_o).GetNeighbors():
        if c.GetIdx() in ring_set and any(
                nb.GetSymbol() in ("O", "N") and nb.GetIdx() not in ring_set
                for nb in c.GetNeighbors()):
            anomeric = c.GetIdx()
            break
    if anomeric is None:
        return None
    numbering = {anomeric: 1}
    prev, cur, n = ring_o, anomeric, 1
    while True:
        nxt = [nb.GetIdx() for nb in mol.GetAtomWithIdx(cur).GetNeighbors()
               if nb.GetIdx() in ring_set and nb.GetIdx() != prev]
        if not nxt or nxt[0] == ring_o:
            break
        prev, cur = cur, nxt[0]
        n += 1
        numbering[cur] = n
    # the exocyclic carbon on the last ring carbon is C(n+1), e.g. C6 of a hexopyranose
    last = max(numbering, key=lambda k: numbering[k])
    for nb in mol.GetAtomWithIdx(last).GetNeighbors():
        if nb.GetSymbol() == "C" and nb.GetIdx() not in ring_set:
            numbering[nb.GetIdx()] = numbering[last] + 1
            break
    return numbering


@tool(
    name="analyse_polysaccharide",
    category="polymer",
    phase=2,
    description=(
        "Analyse a carbohydrate structure supplied as SMILES: number of sugar rings (DP of the "
        "exemplar), ring sizes, every glycosidic linkage with its donor->acceptor position "
        "(e.g. 1->4), the reducing end, substituent census per residue, net charge, and the "
        "carbohydrate-relevant descriptors. Use this to interpret a structure the user supplied, "
        "or to verify one the agent built."
    ),
    schema={"type": "object", "properties": {"smiles": {"type": "string"}},
            "required": ["smiles"]},
)
def analyse_polysaccharide(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    from rdkit.Chem import rdCIPLabeler
    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except Exception:
        pass

    ri = mol.GetRingInfo()
    sugar_rings = []
    for ring in ri.AtomRings():
        if len(ring) not in (5, 6):
            continue
        n_o = sum(1 for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "O")
        n_c = sum(1 for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "C")
        if n_o == 1 and n_c == len(ring) - 1 and all(
                not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            numbering = _ring_numbering(mol, ring)
            if numbering:
                sugar_rings.append({"ring": ring, "size": len(ring), "numbering": numbering})

    ident = identify_residues(mol, [r["ring"] for r in sugar_rings])
    residues = []
    for k, r in enumerate(sugar_rings):
        anom = next(i for i, n in r["numbering"].items() if n == 1)
        a = mol.GetAtomWithIdx(anom)
        exo = [nb for nb in a.GetNeighbors()
               if nb.GetSymbol() in ("O", "N") and nb.GetIdx() not in set(r["ring"])]
        free_anomeric_oh = any(o.GetSymbol() == "O" and o.GetTotalNumHs() == 1 for o in exo)
        subs = []
        for idx, pos in r["numbering"].items():
            at = mol.GetAtomWithIdx(idx)
            for nb in at.GetNeighbors():
                if nb.GetSymbol() == "O" and nb.GetIdx() not in set(r["ring"]):
                    heavy = [x for x in nb.GetNeighbors() if x.GetIdx() != idx]
                    if not heavy:
                        continue
                    other = heavy[0]
                    in_other_ring = any(other.GetIdx() in set(s["ring"]) for s in sugar_rings)
                    if not in_other_ring and other.GetSymbol() != "H":
                        env = Chem.MolFragmentToSmiles(
                            mol, atomsToUse=_collect_substituent(mol, nb.GetIdx(), idx,
                                                                 {i for s in sugar_rings for i in s["ring"]}))
                        subs.append({"position": pos, "substituent_smiles": env})
        residues.append({
            "residue": k + 1,
            "ring_size": r["size"],
            "ring_form": "pyranose" if r["size"] == 6 else "furanose",
            "identity": ident.get(k, {"monomer_key": None,
                                      "monomer_name": "not matched to a curated template",
                                      "anomeric_configuration": "unassigned"}),
            "anomeric_cip": a.GetPropsAsDict().get("_CIPCode", "unassigned"),
            "free_anomeric_oh_reducing_end": free_anomeric_oh,
            "substituents": subs,
        })

    # glycosidic bonds: exocyclic O bridging an anomeric carbon of one ring to a carbon of another
    linkages = []
    ring_of = {}
    for k, r in enumerate(sugar_rings):
        for i in r["ring"]:
            ring_of[i] = k
        for i, n in r["numbering"].items():
            ring_of.setdefault(i, k)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "O" or atom.GetDegree() != 2:
            continue
        nbrs = atom.GetNeighbors()
        if any(nb.GetIdx() in [i for r in sugar_rings for i in r["ring"]] and
               atom.GetIdx() in [i for r in sugar_rings for i in r["ring"]] for nb in nbrs):
            continue
        tagged = []
        for nb in nbrs:
            k = ring_of.get(nb.GetIdx())
            if k is None:
                tagged.append(None)
            else:
                tagged.append((k, sugar_rings[k]["numbering"].get(nb.GetIdx())))
        if all(t is not None for t in tagged) and tagged[0][0] != tagged[1][0]:
            (k1, p1), (k2, p2) = tagged
            donor, acceptor = ((k1, p1), (k2, p2)) if p1 == 1 else ((k2, p2), (k1, p1))
            linkages.append({"donor_residue": donor[0] + 1, "donor_position": donor[1],
                             "acceptor_residue": acceptor[0] + 1, "acceptor_position": acceptor[1],
                             "label": f"({donor[1]}->{acceptor[1]})"})

    reducing = [r["residue"] for r in residues if r["free_anomeric_oh_reducing_end"]]
    return {
        "smiles": Chem.MolToSmiles(mol),
        "n_sugar_rings": len(sugar_rings),
        "ring_forms": [r["ring_form"] for r in residues],
        "residues": residues,
        "glycosidic_linkages": linkages,
        "n_glycosidic_bonds": len(linkages),
        "branch_points": [r for r in
                          {l["acceptor_residue"] for l in linkages}
                          if sum(1 for l in linkages if l["acceptor_residue"] == r) > 1],
        "reducing_end_residues": reducing,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": round(Descriptors.MolWt(mol), 2),
        "net_charge": Chem.GetFormalCharge(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "residue_composition": _composition(residues),
        "notes": ["Linkage positions are derived from ring numbering (anomeric carbon = C1, "
                  "numbered away from the ring oxygen); this is reliable for pyranose and "
                  "furanose rings but not for open-chain or fused-ring carbohydrates.",
                  "alpha/beta comes from matching each residue against the curated monomer "
                  "templates with chirality; residues reported as 'not matched to a curated "
                  "template' have no alpha/beta assignment and need curation before use.",
                  "The raw CIP descriptor at the anomeric carbon is also given, but it is not a "
                  "stable alpha/beta indicator: it changes when the anomeric OH is glycosylated."],
    }


def _composition(residues: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in residues:
        label = r["identity"].get("short") or r["identity"]["monomer_name"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def _collect_substituent(mol, start, exclude, ring_atoms, limit=25):
    seen, stack = {start}, [start]
    while stack and len(seen) < limit:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            i = nb.GetIdx()
            if i in seen or i == exclude or i in ring_atoms:
                continue
            seen.add(i)
            stack.append(i)
    return sorted(seen)


@tool(
    name="list_monomers",
    category="polymer",
    phase=2,
    description="List the curated carbohydrate monomer templates (key, name, ring form, anomeric "
                "configuration and the hydroxyl positions available for linkage or substitution).",
    schema={"type": "object", "properties": {}, "required": []},
)
def list_monomers() -> dict:
    return {"monomers": [{"key": k, "name": v["name"], "short": v["short"],
                          "anomeric": v["anomeric"], "ring": v["ring"],
                          "free_positions": v["positions"]}
                         for k, v in MONOMERS.items()]}


@tool(
    name="list_substituents",
    category="polymer",
    phase=2,
    description="List the derivatisation chemistries available for polysaccharides "
                "(carboxymethyl, hydroxypropyl, hydroxyethyl, methyl, acetyl, sulfate, "
                "phosphate, cationic CHPTAC, octenylsuccinate).",
    schema={"type": "object", "properties": {}, "required": []},
)
def list_substituents() -> dict:
    return {"substituents": [{"key": k, **v} for k, v in SUBSTITUENTS.items()]}


@tool(
    name="list_named_polymers",
    category="polymer",
    phase=2,
    description=(
        "Resolve a polysaccharide by name (cellulose, amylose, amylopectin, chitin, chitosan, "
        "CMC, HEC, HPMC, dextran, pullulan, alginate, pectin, xanthan, inulin, hyaluronan, "
        "carrageenan, guar) into the build_polysaccharide parameters that describe it, together "
        "with the curated structural notes for that polymer. Call this whenever the user names a "
        "polymer instead of specifying a repeat unit."
    ),
    schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": []},
)
def list_named_polymers(name: str | None = None) -> dict:
    if name is None:
        return {"polymers": [{"key": k, "name": v["name"], "summary": v["summary"]}
                             for k, v in NAMED_POLYMERS.items()]}
    key = name.lower().replace(" ", "_").replace("-", "_")
    alias = {k2: k for k, v in NAMED_POLYMERS.items()
             for k2 in [k, *[a.lower().replace(" ", "_").replace("-", "_")
                             for a in v.get("synonyms", [])]]}
    if key not in alias:
        return {"resolved": False, "input": name,
                "available": sorted(NAMED_POLYMERS),
                "next_step": "ask the user for the repeat unit and linkage, or use build_polysaccharide directly"}
    v = NAMED_POLYMERS[alias[key]]
    return {"resolved": True, "key": alias[key], **v}
