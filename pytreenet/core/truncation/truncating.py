"""
This module impements an interface to truncate a tree tensor network state.
"""
from __future__ import annotations
from enum import Enum
from typing import Callable, TYPE_CHECKING

from .svd_truncation import svd_truncation
from .recursive_truncation import recursive_truncation
from .variational import single_site_fitting

if TYPE_CHECKING:
    from ...ttns.ttns import TTNS

class TruncationMethod(Enum):
    """
    The available truncation methods.
    """
    RECURSIVE = "recursive"
    SVD = "svd"
    VARIATIONAL = "variational"
    NONE = "none"

    def randomisable(self) -> bool:
        """
        Checks if the truncation method has a randomised version.

        Returns:
            bool: True if the truncation method has a randomised version,
                False otherwise.
        """
        return self in {TruncationMethod.RECURSIVE, TruncationMethod.SVD}

    def get_function(self) -> Callable:
        """
        Get the truncation function corresponding to the truncation method.

        Returns:
            Callable: The truncation function.
        """
        if self == TruncationMethod.RECURSIVE:
            return recursive_truncation
        if self == TruncationMethod.SVD:
            return svd_truncation
        if self == TruncationMethod.VARIATIONAL:
            return single_site_fitting
        if self == TruncationMethod.NONE:
            raise ValueError("Truncation method 'NONE' does not have a "
                             "corresponding function.")
        raise ValueError(f"Unknown truncation method: {self}")

def truncate_ttns(ttns: TTNS,
                  method: TruncationMethod,
                  *args,
                  **kwargs
                  ) -> TTNS:
    """
    Truncate a tree tensor network state using the specified method.

    Args:
        ttns (TTNS): The tree tensor network state to be truncated.
        method (TruncationMethod): The method to use for the truncation.
        *args: Additional arguments to pass to the truncation function.
        **kwargs: Additional keyword arguments to pass to the truncation
            function.

    Returns:
        TTNS: The truncated tree tensor network state.
    """
    if method == TruncationMethod.NONE:
        return ttns
    truncation_function = method.get_function()
    return truncation_function(ttns, *args, **kwargs)
