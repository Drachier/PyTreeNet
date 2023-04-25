"""
Some useful tools
"""
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from copy import deepcopy

def crandn(size):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    # 1/sqrt(2) is a normalization factor
    return (np.random.standard_normal(size)
       + 1j*np.random.standard_normal(size)) / np.sqrt(2)

def pauli_matrices(asarray=True):
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
        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

    return X, Y, Z

def copy_object(obj, deep=True):
    """
    Returns a normal copy of obj, if deep=False and a deepcopy if deep=True.
    """
    if deep:
        new_obj = deepcopy(obj)
    else:
        new_obj = obj

    return new_obj

def sort_dictionary(dictionary):
    """
    Adapted from https://www.geeksforgeeks.org/python-sort-a-dictionary/ .
    """
    return {key: val for key, val in sorted(dictionary.items(), key = lambda ele: ele[1], reverse = False)}

def random_hermitian_matrix(size=2):
    matrix = crandn((size,size))
    return matrix + matrix.T

def create_bosonic_operators(dimension=2):
    """
    Supplies the common bosonic operators. The creation and anihilation operators
    don't have the numerically correct entries, but only 1s as entries,

    Parameters
    ----------
    dimension : int, optional
        The dimension of the bosonics space to be considers. This determines
        the size of all the operators. The default is 2.

    Returns
    -------
    creation_op : ndarray
        Bosonic creation operator, i.e. a matrix with the subdiagonal entries
        equal to 1 and all others 0.
    annihilation_op : ndarray
        Bosonic anihilation operator, i.e. a matrix with the superdiagonal
        entries equal to 1 and all other 0.
    number_op : ndarray
        The bosonic number operator, i.e. a diagonal matrix with increasing
        integers on the diagonal from 0 to dimension-1.

    """


    creation_op = np.eye(dimension, k=-1)
    annihilation_op = np.conj(creation_op.T)

    number_vector = np.asarray(range(0,dimension))
    number_op = np.diag(number_vector)

    return creation_op, annihilation_op, number_op

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

def fast_exp_action(exponent, vector, mode="fastest"):
    """
    result = exp( exponent ) * vector
    
    Parameters
    ----------
    
    exponent : ndarray wit shape (n, n)
    vector : ndarray with shape (n,)
    mode : {"fastest", "expm", "eigsh", "chebyshec", "none"}
    
    Returns
    -------
    result : ndarray with shape (n,)
    
    """
    if mode == "fastest":
        mode = "chebyshev"
    
    if mode == "expm":
        return expm(exponent) @ vector
    elif mode == "eigsh":
        if exponent.shape[0] < 4:
            return expm(exponent) @ vector
        else:
            k = min(exponent.shape[0]-2, 8)
            w, v, = eigsh(exponent, k=k)
            return v @ np.diag(np.exp(w)) @ np.linalg.pinv(v) @ vector
    elif mode == "chebyshev":
        return expm_multiply(exponent, vector, traceA=np.trace(exponent))
    elif mode == "none":
        return vector
    else:
        raise NotImplementedError

def zero_state(shape):
    state = np.zeros(shape)
    state[0, 0, 0] = 1
    return state