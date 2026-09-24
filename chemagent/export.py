"""3D embedding and downloadable structure files (.mol, .sdf, .pdb, .xyz) + a 3D viewer."""
from __future__ import annotations

import os

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from .registry import tool

OUTDIR = os.environ.get("CHEMAGENT_OUTDIR", "outputs")
os.makedirs(OUTDIR, exist_ok=True)


def embed3d(smiles: str, add_hs: bool = True, optimise: bool = True,
            seed: int = 0xC0FFEE) -> tuple[Chem.Mol, dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:  # drop counterions: they have no meaningful 3D relationship
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    mol = Chem.AddHs(mol) if add_hs else mol
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError("3D embedding failed even with random coordinates")
    info = {"forcefield": None, "converged": None}
    if optimise:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            res = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
            info = {"forcefield": "MMFF94", "converged": res == 0}
        else:
            res = AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
            info = {"forcefield": "UFF", "converged": res == 0}
    return mol, info


@tool(
    name="export_structure",
    category="export",
    phase=1,
    writes_files=True,
    description=(
        "Write a downloadable structure file. Formats: mol (2D or 3D V2000), sdf, pdb, xyz, smi. "
        "3D formats trigger ETKDGv3 embedding followed by MMFF94 (or UFF) optimisation; the force "
        "field used and whether it converged are reported. Counterions are dropped for 3D formats."
    ),
    schema={
        "type": "object",
        "properties": {
            "smiles": {"type": "string"},
            "fmt": {"type": "string", "enum": ["mol", "sdf", "pdb", "xyz", "smi"]},
            "filename": {"type": "string"},
            "name": {"type": "string", "description": "Molecule title written into the file."},
            "three_d": {"type": "boolean", "description": "Generate 3D coordinates (default: true for pdb/xyz)."},
        },
        "required": ["smiles", "fmt"],
    },
)
def export_structure(smiles: str, fmt: str, filename: str | None = None,
                     name: str = "", three_d: bool | None = None) -> dict:
    fmt = fmt.lower().lstrip(".")
    if three_d is None:
        three_d = fmt in ("pdb", "xyz")
    filename = filename or f"{(name or 'structure').replace(' ', '_')}.{fmt}"
    path = os.path.join(OUTDIR, filename)
    info: dict = {}

    if fmt == "smi":
        mol = Chem.MolFromSmiles(smiles)
        with open(path, "w") as fh:
            fh.write(f"{Chem.MolToSmiles(mol)}\t{name or 'structure'}\n")
    elif three_d:
        mol, info = embed3d(smiles)
        mol.SetProp("_Name", name or "structure")
        if fmt == "pdb":
            Chem.MolToPDBFile(mol, path)
        elif fmt == "xyz":
            Chem.MolToXYZFile(mol, path)
        elif fmt == "sdf":
            w = Chem.SDWriter(path); w.write(mol); w.close()
        else:
            Chem.MolToMolFile(mol, path)
    else:
        mol = Chem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol)
        mol.SetProp("_Name", name or "structure")
        if fmt == "sdf":
            w = Chem.SDWriter(path); w.write(mol); w.close()
        elif fmt == "mol":
            Chem.MolToMolFile(mol, path)
        else:
            raise ValueError(f"{fmt} requires 3D coordinates; set three_d=true")

    return {"file": path, "format": fmt, "three_d": three_d,
            "n_atoms": mol.GetNumAtoms(),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "mw": round(Descriptors.MolWt(mol), 2), **info}


@tool(
    name="export_dataset",
    category="export",
    phase=1,
    writes_files=True,
    description="Write a multi-structure SDF with per-molecule properties attached "
                "(name, computed MW, formula and any extra key/value pairs supplied). "
                "Use this to hand a set of candidates to downstream modelling.",
    schema={
        "type": "object",
        "properties": {
            "records": {"type": "array",
                        "items": {"type": "object",
                                  "properties": {"smiles": {"type": "string"},
                                                 "name": {"type": "string"},
                                                 "properties": {"type": "object"}},
                                  "required": ["smiles"]}},
            "filename": {"type": "string"},
        },
        "required": ["records"],
    },
)
def export_dataset(records: list[dict], filename: str = "dataset.sdf") -> dict:
    path = os.path.join(OUTDIR, filename)
    w = Chem.SDWriter(path)
    n = 0
    for rec in records:
        mol = Chem.MolFromSmiles(rec["smiles"])
        if mol is None:
            continue
        AllChem.Compute2DCoords(mol)
        mol.SetProp("_Name", rec.get("name", f"mol_{n+1}"))
        mol.SetProp("SMILES", Chem.MolToSmiles(mol))
        mol.SetProp("MW", f"{Descriptors.MolWt(mol):.2f}")
        mol.SetProp("Formula", rdMolDescriptors.CalcMolFormula(mol))
        for k, v in (rec.get("properties") or {}).items():
            mol.SetProp(str(k), str(v))
        w.write(mol)
        n += 1
    w.close()
    return {"file": path, "n_written": n}


@tool(
    name="view_3d",
    category="render",
    phase=1,
    writes_files=True,
    description="Generate a self-contained interactive 3D viewer (HTML) for a structure, "
                "plus the underlying PDB file.",
    schema={"type": "object",
            "properties": {"smiles": {"type": "string"}, "filename": {"type": "string"},
                           "style": {"type": "string", "enum": ["stick", "sphere", "line"]}},
            "required": ["smiles"]},
)
def view_3d(smiles: str, filename: str = "view3d.html", style: str = "stick") -> dict:
    import py3Dmol
    mol, info = embed3d(smiles)
    pdb = Chem.MolToPDBBlock(mol)
    v = py3Dmol.view(width=720, height=520)
    v.addModel(pdb, "pdb")
    v.setStyle({style: {"radius": 0.15} if style == "stick" else {}})
    v.setBackgroundColor("white")
    v.zoomTo()
    html = v._make_html()
    path = os.path.join(OUTDIR, filename)
    with open(path, "w") as fh:
        fh.write(html)
    pdb_path = os.path.join(OUTDIR, filename.rsplit(".", 1)[0] + ".pdb")
    with open(pdb_path, "w") as fh:
        fh.write(pdb)
    return {"files": [path, pdb_path], "n_atoms": mol.GetNumAtoms(), **info}
