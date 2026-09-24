"""Comparison: similarity, maximum common substructure, functional-group and property deltas."""
from __future__ import annotations

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFMCS
from rdkit.Chem import rdFingerprintGenerator

from .descriptors import FG_SMARTS, molecular_descriptors, surfactant_profile
from .registry import tool

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _mol(smi: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError(f"unparsable SMILES: {smi!r}")
    return m


@tool(
    name="compare_structures",
    category="compare",
    phase=1,
    description=(
        "Compare two structures: ECFP4 and MACCS Tanimoto similarity, the maximum common "
        "substructure, which functional groups were gained or lost, and the change in key "
        "descriptors. Use this after any modification so the answer quantifies what changed."
    ),
    schema={"type": "object",
            "properties": {"smiles_a": {"type": "string"}, "smiles_b": {"type": "string"},
                           "label_a": {"type": "string"}, "label_b": {"type": "string"}},
            "required": ["smiles_a", "smiles_b"]},
)
def compare_structures(smiles_a: str, smiles_b: str,
                       label_a: str = "A", label_b: str = "B") -> dict:
    a, b = _mol(smiles_a), _mol(smiles_b)
    fa, fb = _MORGAN.GetFingerprint(a), _MORGAN.GetFingerprint(b)
    tanimoto = DataStructs.TanimotoSimilarity(fa, fb)
    maccs = DataStructs.TanimotoSimilarity(MACCSkeys.GenMACCSKeys(a),
                                           MACCSkeys.GenMACCSKeys(b))
    mcs = rdFMCS.FindMCS([a, b], timeout=20, ringMatchesRingOnly=True)

    def fgs(m):
        out = {}
        for n, s in FG_SMARTS.items():
            k = len(m.GetSubstructMatches(Chem.MolFromSmarts(s), uniquify=True))
            if k:
                out[n] = k
        return out

    ga, gb = fgs(a), fgs(b)
    gained = {k: gb[k] - ga.get(k, 0) for k in gb if gb[k] > ga.get(k, 0)}
    lost = {k: ga[k] - gb.get(k, 0) for k in ga if ga[k] > gb.get(k, 0)}

    da, db = molecular_descriptors(smiles_a), molecular_descriptors(smiles_b)
    keys = ["mw_average", "clogp_crippen", "tpsa", "hbd", "hba",
            "rotatable_bonds", "net_formal_charge", "heavy_atoms"]
    delta = {k: round(db[k] - da[k], 3) for k in keys}

    return {
        "label_a": label_a, "label_b": label_b,
        "tanimoto_ecfp4": round(tanimoto, 4),
        "tanimoto_maccs": round(maccs, 4),
        "mcs_smarts": mcs.smartsString,
        "mcs_atoms": mcs.numAtoms,
        "mcs_bonds": mcs.numBonds,
        "mcs_fraction_of_a": round(mcs.numAtoms / a.GetNumAtoms(), 3),
        "mcs_fraction_of_b": round(mcs.numAtoms / b.GetNumAtoms(), 3),
        "functional_groups_gained": gained,
        "functional_groups_lost": lost,
        "descriptor_delta_b_minus_a": delta,
        "descriptors_a": {k: da[k] for k in keys},
        "descriptors_b": {k: db[k] for k in keys},
    }


@tool(
    name="compare_series",
    category="compare",
    phase=1,
    description="Tabulate descriptors and surfactant properties across a set of structures "
                "(e.g. a chain-length or head-group series) so they can be ranked or plotted.",
    schema={"type": "object",
            "properties": {
                "records": {"type": "array",
                            "items": {"type": "object",
                                      "properties": {"smiles": {"type": "string"},
                                                     "name": {"type": "string"}},
                                      "required": ["smiles"]}}},
            "required": ["records"]},
)
def compare_series(records: list[dict]) -> dict:
    rows = []
    for i, rec in enumerate(records):
        smi = rec["smiles"]
        d = molecular_descriptors(smi)
        p = surfactant_profile(smi)
        cmc = p["cmc_estimate"]
        rows.append({
            "name": rec.get("name", f"entry_{i+1}"),
            "smiles": d["smiles"],
            "formula": d["formula"],
            "mw": d["mw_average"],
            "charge_class": p["charge_class"],
            "head_group": p["head_group"],
            "tail_carbons": p["hydrophobic_tail_carbons"],
            "eo_units": p["ethoxylate_units"],
            "clogp": d["clogp_crippen"],
            "tpsa": d["tpsa"],
            "hlb_davies": p["hlb_davies"],
            "hlb_griffin": p["hlb_griffin"],
            "cpp": p["packing"].get("critical_packing_parameter"),
            "predicted_aggregate": p["packing"].get("predicted_aggregate"),
            "cmc_mM_estimate": cmc.get("cmc_mM"),
            "cmc_basis": cmc.get("series", cmc.get("status")),
        })
    return {"n": len(rows), "rows": rows}
