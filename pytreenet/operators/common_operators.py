"""
This module provides commonly used operators as numpy arrays.

The operators provided are
* Pauli matrices X, Y, and Z
* Bosonic creation, annihilation, and number operator
* SWAP gates of arbitrary dimension
"""
from __future__ import annotations
from typing import Tuple

import numpy as np

from ..util.ttn_exceptions import positivity_check

def pauli_matrices() -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Returns the three Pauli matrices X, Y, and Z in Z-basis as ndarray.
    """
    X = [[0,1],
         [1,0]]
    Y = [[0,-1j],
         [1j,0]]
    Z = [[1,0],
         [0,-1]]
    X = np.asarray(X, dtype="complex")
    Y = np.asarray(Y, dtype="complex")
    Z = np.asarray(Z, dtype="complex")
    return (X, Y, Z)

def bosonic_operators(dimension: int = 2) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Supplies the common bosonic operators.

    The common operators are the creation, annihilation, and number operator.

    Args:
        dimension (int, optional): The dimension of the bosonic space to be
            considered. This determines the size of all the operators. Defaults
            to 2.

    Returns:
        Tuple[np.ndarray,np.ndarray,np.ndarray]:
            * creation_op: Bosonic creation operator.
            * annihilation_op: Bosonic anihilation operator.
            * number_op: The bosonic number operator, i.e. a diagonal matrix with increasing
              integers on the diagonal from 0 to dimension-1.
    """
    positivity_check(dimension, "dimension")
    sqrt_number_vec = np.asarray([np.sqrt(i)
                                  for i in range(1, dimension)],
                                  dtype=complex)

    creation_op = np.diag(sqrt_number_vec, k=-1)
    annihilation_op = creation_op.T
    number_op = creation_op @ annihilation_op
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
    positivity_check(dimension, "dimension")
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

def projector(dimension: int, index: int) -> np.ndarray:
    """
    A projector on a specific index of a system with a given dimension.

    Args:
        dimension (int): The dimension of the system.
        index (int): The index of the state to project to.
    
    Returns:
        np.ndarray: The projector on the index of the system.
    """
    matrix = np.zeros((dimension, dimension), dtype=complex)
    matrix[index, index] = 1
    return matrix

def ket_i(value: int, dimension: int) -> np.ndarray:
    """
    Generates the ith computational basis state for a system of a given dimension.

    Args:
        value (int): The index of the state.
        dimension (int): The dimension of the system.

    Returns:
        np.ndarray: The ith computational basis state.

    """
    matrix = np.zeros((dimension), dtype=complex)
    matrix[value] = 1
    return matrix

def superposition(rel_phase: float = 0) -> np.ndarray:
    """
    Generates a superposition state with equal weights.

    Args:
        rel_phase (float, optional): The relative phase of the superposition.
            Defaults to 0.

    Returns:
        np.ndarray: The superposition state.

    """
    return np.array([1, np.exp(1j*rel_phase)], dtype=complex) / np.sqrt(2)
