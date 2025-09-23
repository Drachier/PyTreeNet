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

    def randomisable(self) -> bool:
        """
        Checks if the truncation method has a randomised version.

        Returns:
            bool: True if the truncation method has a randomised version,
                False otherwise.
        """
        return self in {TruncationMethod.RECURSIVE, TruncationMethod.SVD}
