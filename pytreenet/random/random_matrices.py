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

def random_nearly_dependent_diagonals(size: int = 4,
                                      condition: float = 1e16,
                                      seed: int | None = None
                                      ) -> list[np.ndarray]:
    """
    Creates diagonal matrices that are dependent exactly but not numerically.

    The first `size` matrices span the diagonal matrices of the given size,
    with their singular values spaced geometrically from one down to one over
    `condition`. The last one is a further random diagonal matrix, so it is a
    combination of the others exactly, a diagonal matrix of this size having
    only `size` free entries, while its coordinates on them are of the order of
    `condition` and rebuilding it from them in double precision loses it.

    Args:
        size (int, optional): Size of the matrices, and the number of them
            that span the space. Defaults to 4.
        condition (float, optional): The ratio of the largest to the smallest
            singular value of the spanning set. Defaults to 1e16, at which the
            set is rank deficient in double precision.
        seed (int, optional): Seed of the random number generator. Defaults to
            None.

    Returns:
        list[np.ndarray]: The size + 1 diagonal matrices.
    """
    positivity_check(size, "size")
    positivity_check(condition, "condition")
    diagonals = crandn((size + 1, size), seed=seed)
    left, _, right = np.linalg.svd(diagonals[:size])
    values = np.geomspace(1.0, 1.0 / condition, size)
    diagonals[:size] = left @ np.diag(values) @ right
    return [np.diag(diagonal) for diagonal in diagonals]

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
