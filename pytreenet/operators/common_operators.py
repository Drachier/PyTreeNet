"""
This module provides commonly used operators as numpy arrays.
"""
from __future__ import annotations
from typing import Union, List, Tuple

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

def bosonic_operators(dimension: int = 2) -> Tuple[np.ndarray]:
    """
    Supplies the common bosonic operators. The creation and anihilation operators
    don't have the numerically correct entries, but only 1s as entries.

    Args:
        dimension (int, optional): The dimension of the bosonics space to be considers. This determines
        the size of all the operators.. Defaults to 2.

    Returns:
        Tuple[np.ndarray]:
            * creation_op: Bosonic creation operator, i.e. a matrix with the subdiagonal entries
             equal to 1 and all others 0.
            * annihilation_op: Bosonic anihilation operator, i.e. a matrix with the superdiagonal
             entries equal to 1 and all other 0.
            * number_op: The bosonic number operator, i.e. a diagonal matrix with increasing
              integers on the diagonal from 0 to dimension-1.
    """
    creation_op = np.eye(dimension, k=-1)
    annihilation_op = np.conj(creation_op.T)

    number_vector = np.asarray(range(0,dimension))
    number_op = np.diag(number_vector)
    return (creation_op, annihilation_op, number_op)
