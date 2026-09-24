"""Chemistry LLM agent PoC: surfactant and carbohydrate-polymer reasoning over a
deterministic cheminformatics tool layer.

Layers
------
registry    tool contract + audited dispatch
resolve     input resolution (name / SMILES / InChI / spec) and format conversion
descriptors molecular descriptors, functional groups, surfactant profiling
render      2D depiction, before/after modification panels
export      3D embedding and downloadable .mol/.sdf/.pdb/.xyz files
transform   curated reaction library, graph edits, spec-driven surfactant builder
compare     similarity, MCS, functional-group and property deltas
polymer     Phase 2: carbohydrate repeat units, linkages, branching, derivatisation
agent       the LLM orchestration loop over the registry
"""
from . import registry  # noqa: F401
from . import resolve  # noqa: F401
from . import descriptors  # noqa: F401
from . import render  # noqa: F401
from . import export  # noqa: F401
from . import transform  # noqa: F401
from . import compare  # noqa: F401
from . import polymer  # noqa: F401

from .registry import (anthropic_tools, dispatch, all_tools,  # noqa: F401
                       registry_table, get)

__version__ = "0.1.0"
