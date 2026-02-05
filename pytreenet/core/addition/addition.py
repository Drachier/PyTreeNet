"""
Module to use from the outside to perform addition operations.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from enum import Enum

from .density_matrix import density_matrix_addition
from .direct import (direct_addition,
                     direct_addition_and_truncation)

if TYPE_CHECKING:
    from ..ttn import TreeTensorNetwork

class AdditionMethod(Enum):
    """
    Enumeration of addition methods.
    """
    DIRECT = "direct"
    DIRECT_TRUNCATE = "direct_truncate"
    DENSITY_MATRIX = "density_matrix"

    def get_function(self) -> Callable:
        """
        Gets the addition function corresponding to the addition method.

        Returns:
            Callable: The addition function.
        """
        if self == AdditionMethod.DIRECT:
            return direct_addition
        elif self == AdditionMethod.DIRECT_TRUNCATE:
            return direct_addition_and_truncation
        elif self == AdditionMethod.DENSITY_MATRIX:
            return density_matrix_addition
        else:
            raise ValueError(f"Unknown addition method: {self}")

def add_two_ttns(ttn1: TreeTensorNetwork,
                 ttn2: TreeTensorNetwork,
                 method: AdditionMethod,
                 *args,
                 **kwargs) -> TreeTensorNetwork:
    """
    Adds two TTNS using the specified addition method.

    Args:
        ttn1 (TreeTensorNetwork): The first TTNS.
        ttn2 (TreeTensorNetwork): The second TTNS.
        method (AdditionMethod): The addition method to use.
        *args: Additional positional arguments for the addition function.
        **kwargs: Additional keyword arguments for the addition function.

    Returns:
        TreeTensorNetwork: The resulting TTNS after addition.
    """
    add_func = method.get_function()
    return add_func([ttn1, ttn2], *args, **kwargs)

def add_ttns(ttns: list[TreeTensorNetwork],
              method: AdditionMethod,
              *args,
              **kwargs) -> TreeTensorNetwork:
    """
    Adds multiple TTNS using the specified addition method.

    Args:
        ttns (list[TreeTensorNetwork]): The list of TTNS to add.
        method (AdditionMethod): The addition method to use.
        *args: Additional positional arguments for the addition function.
        **kwargs: Additional keyword arguments for the addition function.

    Returns:
        TreeTensorNetwork: The resulting TTNS after addition.
    """
    add_func = method.get_function()
    result_ttn = add_func(ttns, *args, **kwargs)
    return result_ttn
