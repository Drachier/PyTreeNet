"""
This module impements an interface to truncate a tree tensor network state.
"""
from __future__ import annotations
from enum import Enum

class TruncationMethod(Enum):
    """
    The available truncation methods.
    """
    RECURSIVE = "recursive"
    SVD = "svd"
    VARIATIONAL = "variational"
