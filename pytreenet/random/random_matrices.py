"""
This module contains functions to generate random matrices.
"""
import numpy as np
from scipy.stats import unitary_group

from ..util.ttn_exceptions import positivity_check
from ..util.crandn import (crandn,
                           # Keep historical dependency
                           crandn_like,
                           RandomDistribution)

def random_matrix(size: int = 2) -> np.ndarray:
    """
    Creates a random matrix of given size.

    Args:
        size (int, optional): Size of the matrix. Defaults to 2.

    Returns:
        np.ndarray: The random matrix.
    """
    positivity_check(size, "size")
    return crandn((size,size))

def random_hermitian_matrix(size: int = 2) -> np.ndarray:
    """
    Creates a random hermitian matrix H^\\dagger = H

    Args:
        size (int, optional): Size of the matrix. Defaults to 2.

    Returns:
        np.ndarray: The hermitian matrix.
    """
    positivity_check(size, "size")
    rmatrix = random_matrix(size)
    return 0.5 * (rmatrix + rmatrix.T.conj())

def random_unitary_matrix(size: int = 2,
                          seed: int | None = None
                          ) -> np.ndarray:
    """
    Creates a random unitary matrix U^\\dagger U = I

    Args:
        size (int, optional): Size of the matrix. Defaults to 2.

    Returns:
        np.ndarray: The unitary matrix.
    """
    return unitary_group.rvs(size, random_state=seed)

def haar_random_state(size: int = 2,
                      seed: int | None = None
                      ) -> np.ndarray:
    """
    Generates a random normalized state vector using the Haar measure.

    Args:
        size (int, optional): Size of the state vector. Defaults to 2.
        seed (int, optional): Seed for the random number generator.
            Defaults to 42.

    Returns:
        np.ndarray: The random normalized state vector.
    """
    unit = random_unitary_matrix(size, seed=seed)
    state = unit[:,0]
    state /= np.linalg.norm(state)
    return state
