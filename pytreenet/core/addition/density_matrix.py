"""
Implements the density matrix addition for TTNs.
"""
from __future__ import annotations

import numpy as np

from ..ttn import TreeTensorNetwork
from ...contractions.state_state_contraction import build_full_subtree_cache

def density_matrix_addition(ttns: list[TreeTensorNetwork]
                            ) -> TreeTensorNetwork:
    