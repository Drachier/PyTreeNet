"""
This module provides commonly used operators as numpy arrays.
"""
from __future__ import annotations
from typing import Union, List

import numpy as np

def pauli_matrices(asarray: bool=True) -> Union[List, np.ndarray]:
    """
    Returns the three Pauli matrices X, Y, and Z in Z-basis as ndarray, if asarray is True
    otherwise it returns them as lists.
    """
    X = [[0,1],
         [1,0]]
    Y = [[0,-1j],
         [1j,0]]
    Z = [[1,0],
         [0,-1]]
    if asarray:
        X = np.asarray(X, dtype="complex")
        Y = np.asarray(Y, dtype="complex")
        Z = np.asarray(Z, dtype="complex")

    return X, Y, Z