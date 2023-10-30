"""
Some useful tools
"""
import numpy as np
from scipy.linalg import expm as expm
from scipy.sparse.linalg import expm_multiply, eigsh
from scipy.sparse.linalg import expm as expm_sparse
from scipy.sparse import csr_matrix

from copy import deepcopy
from tqdm import tqdm

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
        X = np.asarray(X, dtype=complex)
        Y = np.asarray(Y, dtype=complex)
        Z = np.asarray(Z, dtype=complex)

    return X, Y, Z

def zero():
    return np.array([1, 0], dtype=complex)

def one():
    return np.array([0, 1], dtype=complex)

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
    Supplies the common bosonic operators.

    Args:
        dimension (int, optional): The dimension of the bosonics space to be considers.
        This determines the size of all the operators. Defaults to 2.

    Returns:
        Tuple[np.ndarray]:
            * creation_op: Bosonic creation operator.
            * annihilation_op: Bosonic anihilation operator.
            * number_op: The bosonic number operator, i.e. a diagonal matrix with increasing
              integers on the diagonal from 0 to dimension-1.
    """
    if dimension < 1:
        errstr = "The dimension must be positive!"
        raise ValueError(errstr)
    sqrt_number_vec = np.asarray([np.sqrt(i)
                                  for i in range(1, dimension)])

    creation_op = np.diag(sqrt_number_vec, k=-1)
    annihilation_op = creation_op.T
    number_op = creation_op @ annihilation_op
    return (creation_op, annihilation_op, number_op)

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
    elif mode == "sparse":
        exponent = csr_matrix(exponent)
        vector = csr_matrix(exponent).transpose()
        exponent_ = expm_sparse(exponent)
        result = exponent_.dot(vector)
        return result.toarray()
    elif mode == "none":
        return vector
    else:
        raise NotImplementedError

def zero_state(shape):
    state = np.zeros(shape)
    state[0, 0, 0] = 1
    return state

def state_vector_time_evolution(state, hamiltonian, final_time, time_step_size, operators):
    num_time_steps = int(np.ceil(final_time / time_step_size))
    results = np.zeros((len(operators) + 1, num_time_steps + 1), dtype=complex)

    for time_step in tqdm(range(num_time_steps + 1)):
        if time_step != 0:
            state = fast_exp_action(-1j*time_step_size*hamiltonian, state)
        
        conj_state = np.conjugate(state.T)
        for i, operator in enumerate(operators):
            results[i,time_step] = conj_state @ operator @ state
            
        results[-1,time_step] = time_step * time_step_size

    print(state)
    return results

def multikron(*args):
    op = args
    if type(op[0]) == str:
        assert [s for s in op[0] if s not in ["0", "1"]] == []
        op = [zero() if s=="0" else one() for s in op[0]]
    operators = list(reversed(op))
    result = operators[0]
    for i, o in enumerate(operators):
        if i>0:
            result = np.kron(o, result)
    return result
