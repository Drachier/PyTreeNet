"""
Some useful tools
"""
from copy import deepcopy, copy
from collections import Counter

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, eigsh
from scipy.sparse.linalg import expm as expm_sparse
from scipy.sparse import csr_matrix

def crandn(size):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    # 1/sqrt(2) is a normalization factor
    return (np.random.standard_normal(size)
       + 1j*np.random.standard_normal(size)) / np.sqrt(2)

def copy_object(obj, deep=True):
    """
    Returns a normal copy of obj, if deep=False and a deepcopy if deep=True.
    """
    if deep:
        new_obj = deepcopy(obj)
    else:
        new_obj = copy(obj)

    return new_obj

def sort_dictionary(dictionary):
    """
    Adapted from https://www.geeksforgeeks.org/python-sort-a-dictionary/ .
    """
    return {key: val for key, val in sorted(dictionary.items(), key = lambda ele: ele[1], reverse = False)}

def compare_lists_by_value(list1, list2):
    if len(list1) != len(list2):
        return False
    if Counter(list1) == Counter(list2):
        return True
    return False

def fast_exp_action(exponent: np.ndarray,
                    vector: np.ndarray,
                    mode: str = "fastest") -> np.ndarray:
    """
    Result = exp( exponent) @ vector

    Args:
        exponent (np.ndarray): The exponent in matrix form.
        vector (np.ndarray): The input vector in vector form.
        mode (str, optional): The mode to use. Defaults to "fastest".

    Raises:
        NotImplementedError: If an unimplemented mode is used.

    Returns:
        np.ndarray: The result of the exponentiation and multiplication.
    """
    if mode == "fastest":
        mode = "chebyshev"
    if mode == "expm":
        return expm(exponent) @ vector
    if mode == "eigsh":
        if exponent.shape[0] < 4:
            return expm(exponent) @ vector
        k = min(exponent.shape[0]-2, 8)
        w, v, = eigsh(exponent, k=k)
        return v @ np.diag(np.exp(w)) @ np.linalg.pinv(v) @ vector
    if mode == "chebyshev":
        return expm_multiply(exponent, vector,
                             traceA=np.trace(exponent))
    if mode == "sparse":
        exponent = csr_matrix(exponent)
        vector = csr_matrix(exponent).transpose()
        exponent_ = expm_sparse(exponent)
        result = exponent_.dot(vector)
        return result.toarray()
    if mode == "none":
        return vector
    errstr = mode + " is not a possible mode for exponent action!"
    raise NotImplementedError(errstr)
