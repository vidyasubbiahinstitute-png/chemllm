"""2D depiction: single structures, grids, and before/after modification panels.

The before/after panel is the workhorse output of the modification workflow: it
highlights exactly the atoms that are not shared between the two structures, so a
chemist can see the change without reading the SMILES.
"""
from __future__ import annotations

import os

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

from .registry import tool

OUTDIR = os.environ.get("CHEMAGENT_OUTDIR", "outputs")
os.makedirs(OUTDIR, exist_ok=True)

HIGHLIGHT = (0.95, 0.55, 0.15)   # changed atoms
KEEP = (0.55, 0.78, 0.95)        # conserved scaffold


def _prep(smiles: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    AllChem.Compute2DCoords(m)
    return m


def _draw(mols, legends, highlights=None, colors=None, per_mol=(520, 420),
          n_cols=2, title=None) -> str:
    n = len(mols)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    w, h = per_mol[0] * n_cols, per_mol[1] * n_rows
    d = rdMolDraw2D.MolDraw2DCairo(w, h, per_mol[0], per_mol[1])
    opts = d.drawOptions()
    opts.legendFontSize = 20
    opts.addStereoAnnotation = True
    opts.bondLineWidth = 2
    d.DrawMolecules(list(mols), legends=list(legends),
                    highlightAtoms=highlights, highlightAtomColors=colors)
    d.FinishDrawing()
    return d.GetDrawingText()


@tool(
    name="draw_structure",
    category="render",
    phase=1,
    writes_files=True,
    description="Render a 2D depiction of one or more structures to a PNG file. "
                "Pass several SMILES to get a labelled grid (e.g. a homologous series).",
    schema={
        "type": "object",
        "properties": {
            "smiles": {"type": "array", "items": {"type": "string"},
                       "description": "One or more SMILES."},
            "legends": {"type": "array", "items": {"type": "string"},
                        "description": "Optional label under each structure."},
            "filename": {"type": "string", "description": "Output PNG name (no path)."},
        },
        "required": ["smiles"],
    },
)
def draw_structure(smiles: list[str], legends: list[str] | None = None,
                   filename: str = "structure.png") -> dict:
    if isinstance(smiles, str):
        smiles = [smiles]
    mols = [_prep(s) for s in smiles]
    legends = legends or [Chem.MolToSmiles(m) for m in mols]
    legends = [str(x) for x in legends][: len(mols)]
    n = len(mols)
    n_cols = n if n <= 3 else (2 if n == 4 else 3)
    png = _draw(mols, legends, n_cols=n_cols)
    path = os.path.join(OUTDIR, filename)
    with open(path, "wb") as fh:
        fh.write(png)
    return {"file": path, "n_structures": len(mols), "legends": legends}


@tool(
    name="draw_before_after",
    category="render",
    phase=1,
    writes_files=True,
    description=(
        "Render a side-by-side before/after panel for a chemical modification. Atoms that are "
        "not part of the maximum common substructure are highlighted in orange in both panels, "
        "so the change is visually explicit. Use this for every modification or reaction answer."
    ),
    schema={
        "type": "object",
        "properties": {
            "smiles_before": {"type": "string"},
            "smiles_after": {"type": "string"},
            "label_before": {"type": "string"},
            "label_after": {"type": "string"},
            "filename": {"type": "string"},
        },
        "required": ["smiles_before", "smiles_after"],
    },
)
def draw_before_after(smiles_before: str, smiles_after: str,
                      label_before: str = "before", label_after: str = "after",
                      filename: str = "before_after.png") -> dict:
    a, b = _prep(smiles_before), _prep(smiles_after)
    mcs = rdFMCS.FindMCS([a, b], timeout=20,
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                         ringMatchesRingOnly=True, completeRingsOnly=False)
    core = Chem.MolFromSmarts(mcs.smartsString) if mcs.numAtoms else None
    hits_a = set(a.GetSubstructMatch(core)) if core is not None else set()
    hits_b = set(b.GetSubstructMatch(core)) if core is not None else set()
    diff_a = [i for i in range(a.GetNumAtoms()) if i not in hits_a]
    diff_b = [i for i in range(b.GetNumAtoms()) if i not in hits_b]
    highlights = [diff_a, diff_b]
    colors = [{i: HIGHLIGHT for i in diff_a}, {i: HIGHLIGHT for i in diff_b}]
    png = _draw([a, b], [label_before, label_after], highlights=highlights,
                colors=colors, n_cols=2, per_mol=(560, 440))
    path = os.path.join(OUTDIR, filename)
    with open(path, "wb") as fh:
        fh.write(png)
    return {"file": path,
            "mcs_smarts": mcs.smartsString,
            "mcs_atoms": mcs.numAtoms,
            "atoms_changed_before": len(diff_a),
            "atoms_changed_after": len(diff_b),
            "conserved_fraction": round(mcs.numAtoms / max(a.GetNumAtoms(), 1), 3)}


@tool(
    name="draw_svg",
    category="render",
    phase=1,
    writes_files=True,
    description="Render a single structure as a scalable SVG (for reports and slides).",
    schema={"type": "object",
            "properties": {"smiles": {"type": "string"}, "filename": {"type": "string"},
                           "legend": {"type": "string"}},
            "required": ["smiles"]},
)
def draw_svg(smiles: str, filename: str = "structure.svg", legend: str = "") -> dict:
    m = _prep(smiles)
    d = rdMolDraw2D.MolDraw2DSVG(600, 460)
    d.drawOptions().addStereoAnnotation = True
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, legend=legend)
    d.FinishDrawing()
    path = os.path.join(OUTDIR, filename)
    with open(path, "w") as fh:
        fh.write(d.GetDrawingText())
    return {"file": path}
