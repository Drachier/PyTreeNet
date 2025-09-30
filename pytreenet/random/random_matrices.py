"""
This module contains functions to generate random matrices.
"""
from typing import Tuple, Union
from enum import Enum

import numpy as np
import numpy.random as npr
from scipy.linalg import expm
from scipy.stats import unitary_group

from ..util.ttn_exceptions import positivity_check

class RandomDistribution(Enum):
    """
    Enumeration of supported random distributions.
    """
    NORMAL = "normal"
    UNIFORM = "uniform"

    def generating_function(self, rng: npr.Generator):
        """
        Returns the random number generating function corresponding to the
        distribution.

        Args:
            rng (npr.Generator): The random number generator.

        Returns:
            Callable: The random number generating function.
        """
        if self == RandomDistribution.NORMAL:
            return rng.normal
        elif self == RandomDistribution.UNIFORM:
            return rng.uniform
        else:
            raise ValueError(f"Unknown distribution: {self}!")

def crandn(size: Union[Tuple[int,...],int],
           *args,
           seed=None,
           distribution: RandomDistribution = RandomDistribution.NORMAL,
           **kwargs
           ) -> np.ndarray:
    """
    Draw random samples from the standard complex normal (Gaussian)
      distribution.

    Args:
        size (Tuple[int,...]): The size/shape of the output array.
        *args: Additional dimensions to be added to the size.
        seed (int, optional): Seed for the random number generator.
            Defaults to None.
        distribution (RandomDistribution, optional): The distribution to
            sample from. Defaults to RandomDistribution.NORMAL.
        **kwargs: Additional keyword arguments for the random number
            generation.

    Returns:
        np.ndarray: The array of random complex numbers.
    """
    rng = npr.default_rng(seed)
    gen_func = distribution.generating_function(rng)
    if isinstance(size, int) and len(args) > 0:
        size = tuple([size] + list(args))
    elif isinstance(size,int):
        size = (size,)
    # 1/sqrt(2) is a normalization factor
    return (gen_func(size=size, **kwargs) +
            1j*gen_func(size=size, **kwargs)) / np.sqrt(2)

def crandn_like(array: np.ndarray) -> np.ndarray:
    """
    Draw random samples from the standard complex normal (Gaussian)
      distribution with the same shape as the input array.

    Args:
        array (np.ndarray): The input array.

    Returns:
        np.ndarray: The array of random complex numbers.
    """
    return crandn(array.shape)

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
                          seed: int = 42
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
                     seed: int = 42
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
