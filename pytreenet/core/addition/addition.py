"""
Module to use from the outside to perform addition operations.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from enum import Enum

from .direct import (direct_addition,
                     direct_addition_and_truncation)
from ...ttns.ttns_ttno.dm_approach import (dm_addition,
                                           dm_linear_combination)
from ...ttns.ttns_ttno.half_dm_approach import (half_dm_addition,
                                                half_dm_linear_combination)
from ...ttns.ttns_ttno.src import (src_addition,
                                   src_linear_combination)

if TYPE_CHECKING:
    from ..ttn import TreeTensorNetwork

class AdditionMethod(Enum):
    """
    Enumeration of addition methods.
    """
    DIRECT = "direct"
    DIRECT_TRUNCATE = "direct_truncate"
    DENSITY_MATRIX = "density_matrix"
    HALF_DENSITY_MATRIX = "half_density_matrix"
    SRC = "src"

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
            return dm_addition
        elif self == AdditionMethod.HALF_DENSITY_MATRIX:
            return half_dm_addition
        elif self == AdditionMethod.SRC:
            return src_addition
        else:
            raise ValueError(f"Unknown addition method: {self}")

    def lin_comb_possible(self) -> bool:
        """
        Checks if the addition method can be used for linear combinations.

        Returns:
            bool: True if the addition method can be used for linear combinations, False otherwise.
        """
        return self in {AdditionMethod.DENSITY_MATRIX,
                        AdditionMethod.HALF_DENSITY_MATRIX,
                        AdditionMethod.SRC}

    def lin_comb_function(self) -> Callable:
        """
        Gets the linear combination function corresponding to the addition method.

        Returns:
            Callable: The linear combination function.

        Raises:
            ValueError: If the addition method cannot be used for linear combinations.
        """
        if self == AdditionMethod.DENSITY_MATRIX:
            return dm_linear_combination
        elif self == AdditionMethod.HALF_DENSITY_MATRIX:
            return half_dm_linear_combination
        elif self == AdditionMethod.SRC:
            return src_linear_combination
        else:
            raise ValueError(f"Addition method {self} cannot be used for linear combinations!")

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
