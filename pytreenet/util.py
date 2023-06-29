"""
Some useful tools
"""
import numpy as np

from copy import deepcopy, copy
from collections import Counter

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

def random_hermitian_matrix(size=2):
    matrix = crandn((size,size))
    return matrix + matrix.T

def build_swap_gate(dimension=2):
    """
    A SWAP gate acts on two systems with the same pysical dimension and swappes
    their states.

    Parameters
    ----------
    dimension : int, optional
        Physical dimension of the two sites, which has to be the same for both.
        The default is 2.

    Returns
    -------
    swap_gate: ndarry
        A SWAP-gate for two `dimension`-dimensional systems.

    """

    swap_gate = np.zeros((dimension**2, dimension**2), dtype=complex)

    for i in range(dimension**2):
        for j in range(dimension**2):

            # Basically find the indices in base dimension
            output_sys1 = int(i / dimension)
            output_sys2 = int(i % dimension)

            input_sys1 = int(j / dimension)
            input_sys2 = int(j % dimension)

            if (output_sys1 == input_sys2) and (input_sys1 == output_sys2):
                swap_gate[i,j] = 1

    return swap_gate

def compare_lists_by_value(list1, list2):
    if Counter(list1) == Counter(list2):
        return True
    else:
        return False