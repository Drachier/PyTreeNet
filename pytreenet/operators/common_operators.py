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

def swap_gate(dimension: int = 2) -> np.ndarray:
    """
    A SWAP gate acts on two systems with the same pysical dimension and swaps
    their states.

    Args:
        dimension (int, optional): Physical dimension of the two sites,
         which has to be the same for both. Defaults to 2.

    Returns:
        np.ndarray: A SWAP-gate for two `dimension`-dimensional systems.
    """
    swap = np.zeros((dimension**2, dimension**2), dtype=complex)

    for i in range(dimension**2):
        for j in range(dimension**2):

            # Basically find the indices in base dimension
            output_sys1 = int(i / dimension)
            output_sys2 = int(i % dimension)

            input_sys1 = int(j / dimension)
            input_sys2 = int(j % dimension)

            if (output_sys1 == input_sys2) and (input_sys1 == output_sys2):
                swap[i,j] = 1

    return swap
