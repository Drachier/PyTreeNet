"""
This module contains functions to generate random matrices.
"""
from typing import Tuple, Union

import numpy as np
from scipy.linalg import expm

from ..util.ttn_exceptions import positivity_check

def crandn(size: Union[Tuple[int,...],int], *args) -> np.ndarray:
    """
    Draw random samples from the standard complex normal (Gaussian)
      distribution.

    Args:
        size (Tuple[int,...]): The size/shape of the output array.

    Returns:
        np.ndarray: The array of random complex numbers.
    """
    if isinstance(size, int) and len(args) > 0:
        size = [size] + list(args)
    elif isinstance(size,int):
        size = (size,)
    # 1/sqrt(2) is a normalization factor
    return (np.random.standard_normal(size)
       + 1j*np.random.standard_normal(size)) / np.sqrt(2)

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

def random_unitary_matrix(size: int = 2) -> np.ndarray:
    """
    Creates a random unitary matrix U^\\dagger U = I

    Args:
        size (int, optional): Size of the matrix. Defaults to 2.

    Returns:
        np.ndarray: The unitary matrix.
    """
    positivity_check(size, "size")
    rherm = random_hermitian_matrix(size)
    return expm(1j * rherm)
