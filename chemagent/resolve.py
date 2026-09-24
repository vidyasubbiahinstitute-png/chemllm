"""Input resolution: free text / name / SMILES / spec  ->  a standardised molecule.

Resolution cascade (first hit wins, provenance always reported):
  1. direct parse   -- SMILES / InChI / mol block
  2. curated lexicon -- internal surfactant + ingredient dictionary (offline, auditable)
  3. public registry -- PubChem PUG-REST name lookup (network; optional)
  4. structured spec -- build_surfactant() from head group + tail length
  5. escalate        -- return `unresolved` so the agent asks the user, never guesses
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

from .registry import tool

RDLogger.DisableLog("rdApp.*")

_HERE = os.path.dirname(__file__)
with open(os.path.join(_HERE, "data", "lexicon.json")) as fh:
    LEXICON: dict = json.load(fh)

# name -> key, including synonyms, normalised
_ALIAS: dict[str, str] = {}
for _k, _v in LEXICON.items():
    for _n in [_k, _v["name"], *_v.get("synonyms", [])]:
        _ALIAS[re.sub(r"[^a-z0-9]", "", _n.lower())] = _k

ALLOW_NETWORK = os.environ.get("CHEMAGENT_OFFLINE", "0") != "1"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse_any(text: str) -> Chem.Mol | None:
    """Try SMILES, then InChI, then mol block."""
    text = text.strip()
    if not text:
        return None
    m = Chem.MolFromSmiles(text)
    if m is not None:
        return m
    if text.lower().startswith("inchi="):
        m = inchi.MolFromInchi(text)
        if m is not None:
            return m
    if "\n" in text and ("V2000" in text or "V3000" in text):
        m = Chem.MolFromMolBlock(text)
        if m is not None:
            return m
    return None


@lru_cache(maxsize=512)
def _pubchem_name_to_smiles(name: str) -> tuple[str | None, str | None]:
    """(smiles, cid) from PubChem PUG-REST; (None, None) when unavailable."""
    if not ALLOW_NETWORK:
        return None, None
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    url = base + urllib.parse.quote(name) + "/property/IsomericSMILES,CanonicalSMILES/JSON"
    try:
        with urllib.request.urlopen(url, timeout=12) as resp:
            data = json.loads(resp.read().decode())
        rec = data["PropertyTable"]["Properties"][0]
        smi = rec.get("IsomericSMILES") or rec.get("CanonicalSMILES")
        return smi, str(rec.get("CID"))
    except Exception:
        return None, None


def standardise(mol: Chem.Mol, keep_salt: bool = True) -> dict:
    """Clean up a molecule and report both the full (salt) form and the parent ion."""
    mol = rdMolStandardize.Cleanup(mol)
    parent = rdMolStandardize.FragmentParent(mol)
    uncharger = rdMolStandardize.Uncharger()
    out = {
        "smiles": Chem.MolToSmiles(mol if keep_salt else parent),
        "parent_smiles": Chem.MolToSmiles(parent),
        "neutral_parent_smiles": Chem.MolToSmiles(uncharger.uncharge(parent)),
        "inchi": inchi.MolToInchi(mol) or "",
        "inchikey": inchi.MolToInchiKey(mol) or "",
        "n_fragments": len(Chem.GetMolFrags(mol)),
    }
    return out


@tool(
    name="resolve_input",
    category="io",
    phase=1,
    description=(
        "Resolve any chemical input -- SMILES, InChI, chemical name, common/trade name or "
        "surfactant class name -- into a standardised structure. Always call this first. "
        "Returns canonical SMILES, InChI, InChIKey, the salt-free parent, and the resolution "
        "source so the provenance of the structure is explicit. If the input cannot be resolved "
        "it returns resolved=false with suggestions; in that case ASK the user, do not invent a structure."
    ),
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "SMILES, InChI, or a chemical/common/trade name."},
            "use_network": {"type": "boolean",
                            "description": "Allow a PubChem lookup if the name is not in the curated lexicon (default true)."},
        },
        "required": ["query"],
    },
)
def resolve_input(query: str, use_network: bool = True) -> dict:
    q = query.strip()
    key = _norm(q)

    # 1. curated lexicon first for names (authoritative, offline, reviewed)
    if key in _ALIAS:
        entry = LEXICON[_ALIAS[key]]
        mol = Chem.MolFromSmiles(entry["smiles"])
        std = standardise(mol)
        return {"resolved": True, "source": "curated_lexicon", "lexicon_key": _ALIAS[key],
                "input": q, "preferred_name": entry["name"], "class": entry.get("class"),
                "note": entry.get("note"), **std}

    # 2. direct structure parse
    mol = parse_any(q)
    if mol is not None:
        std = standardise(mol)
        match = next((k for k, v in LEXICON.items()
                      if Chem.MolToSmiles(Chem.MolFromSmiles(v["smiles"])) == std["smiles"]), None)
        return {"resolved": True, "source": "direct_parse", "input": q,
                "preferred_name": LEXICON[match]["name"] if match else None,
                "class": LEXICON[match].get("class") if match else None, **std}

    # 3. public registry
    if use_network:
        smi, cid = _pubchem_name_to_smiles(q)
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                return {"resolved": True, "source": "pubchem", "pubchem_cid": cid,
                        "input": q, "preferred_name": q, **standardise(mol)}

    # 4. escalate with near-misses rather than guessing
    cands = [LEXICON[k]["name"] for k in LEXICON
             if key[:6] and key[:6] in _norm(LEXICON[k]["name"])][:8]
    return {"resolved": False, "input": q,
            "reason": "not a parsable structure and not found in the curated lexicon or PubChem",
            "suggestions": cands,
            "next_step": "ask the user for a SMILES, or use build_surfactant with an explicit head group and tail length"}


@tool(
    name="list_lexicon",
    category="io",
    phase=1,
    description="List the curated internal surfactant/ingredient lexicon, optionally filtered by class "
                "(anionic, cationic, nonionic, amphoteric, feedstock, polymer_precursor).",
    schema={"type": "object",
            "properties": {"surfactant_class": {"type": "string"}},
            "required": []},
)
def list_lexicon(surfactant_class: str | None = None) -> dict:
    rows = [{"key": k, "name": v["name"], "class": v.get("class"), "smiles": v["smiles"]}
            for k, v in LEXICON.items()
            if surfactant_class is None or v.get("class") == surfactant_class]
    return {"count": len(rows), "entries": rows}


@tool(
    name="convert_format",
    category="io",
    phase=1,
    description="Convert a resolved structure between machine-readable formats: "
                "smiles, canonical_smiles, inchi, inchikey, molblock, molfile_v3000.",
    schema={
        "type": "object",
        "properties": {
            "smiles": {"type": "string"},
            "to": {"type": "string",
                   "enum": ["smiles", "canonical_smiles", "inchi", "inchikey",
                            "molblock", "molfile_v3000"]},
        },
        "required": ["smiles", "to"],
    },
)
def convert_format(smiles: str, to: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    if to in ("smiles", "canonical_smiles"):
        val = Chem.MolToSmiles(mol)
    elif to == "inchi":
        val = inchi.MolToInchi(mol)
    elif to == "inchikey":
        val = inchi.MolToInchiKey(mol)
    elif to == "molblock":
        val = Chem.MolToMolBlock(Chem.AddHs(mol))
    elif to == "molfile_v3000":
        val = Chem.MolToV3KMolBlock(mol)
    else:
        raise ValueError(f"unsupported target format {to!r}")
    return {"format": to, "value": val}
