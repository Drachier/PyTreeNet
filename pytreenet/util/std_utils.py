"""
Some useful tools that do not fit into any other category.
"""
from typing import Tuple, Any, Dict, List
from copy import deepcopy, copy
from collections import Counter

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from numpy.linalg import norm
from scipy.sparse.linalg import norm as spnorm
from scipy.linalg import expm
from scipy.sparse import issparse 


def copy_object(obj: Any, deep=True) -> Any:
    """
    Copy an object.

    Shortens the commonly used if-else statement for copying objects to
    distinguish between deep and shallow copies.

    Args:
        obj (Any): The object to copy.
        deep (bool, optional): Whether to perform a deep copy. Defaults to
          True.
    
    Returns:
        Any: The copied object.
    """
    if deep:
        new_obj = deepcopy(obj)
    else:
        new_obj = copy(obj)
    return new_obj

def sort_dictionary(dictionary: Dict) -> Dict:
    """
    Sort a dictionary by its values.

    Args:
        dictionary (Dict): The dictionary to sort.

    Returns:
        Dict: The sorted dictionary.
    """
    return dict(sorted(dictionary.items(),key = lambda ele: ele[1], reverse = False))

def compare_lists_by_value(list1: List, list2: List) -> bool:
    """
    Compare, if two lists have the same elements.

    Args:
        list1 (List): The first list.
        list2 (List): The second list.
    
    Returns:
        bool: Whether the two lists have the same elements.
    """
    if len(list1) != len(list2):
        return False
    if Counter(list1) == Counter(list2):
        return True
    return False

def permute_tuple(tup: Tuple, permutation: List[int]) -> Tuple:
    """
    Permute the elements of a tuple.

    Args:
        tup (Tuple): The original tuple.
        permutation (List[int]): The permutation of the elements.
    
    Returns:
        Tuple: The permuted tuple.
    """
    assert len(tup) == len(permutation)
    return tuple(tup[i] for i in permutation)


def fast_exp_action(t, exponent: np.ndarray,
                    vector: np.ndarray,
                    parameters) -> np.ndarray:

    if exponent.shape[0] > parameters["size_threshold"] and parameters.get("mode") == "Krylov":
         method = "Krylov"
    elif exponent.shape[0] > parameters["size_threshold"] and parameters.get("mode") == "Taylor":
         method = "Taylor"                         
    else:
        method = "low_trunc"

    if method == "low_trunc":
        # SciPy approach
        exponent = t*exponent
        return expm_multiply(exponent, vector)
    
    elif method == "Krylov":
        # Expokit-like approach
        return expmv(t, 
                     exponent, 
                     vector, 
                     tol=parameters["Krylov_tol"], 
                     krylov_dim=parameters["krylov_dim"])
        
    elif method == "Taylor":
        # Taylor series expansion
        exponent = t*exponent
        result = taylor_expm_action_numpy(exponent, 
                                          vector, 
                                          Taylor_num_terms=parameters["Taylor_num_terms"])
        return result


def taylor_expm_action_numpy(A: np.ndarray,
                             v: np.ndarray,
                             Taylor_num_terms: int = 3) -> np.ndarray:
    """
    Approximates exp(A) @ v using the Taylor series expansion.

    Args:
        A (np.ndarray): Square matrix of shape (n, n). Can be real or complex.
        v (np.ndarray): Vector of shape (n,). Can be real or complex.
        Taylor_num_terms (int, optional): Number of terms in the Taylor series. Defaults to 20.

    Returns:
        np.ndarray: Approximated result of exp(A) @ v.
    """
    #A_sparse = np.where(np.abs(A) < 1e-8, 0, A)
    #A_sparse = csr_matrix(A_sparse) 
    #sparsity = 1.0 - (A_sparse.nnz / A.size)
    #if sparsity > 0.7 :s
    #    A = A_sparse

    result = v.copy()
    term = v.copy()
    for n in range(1, Taylor_num_terms):

        term = A @ term / n
        result = result + term  
        
        #print(np.linalg.norm(term))
        #if np.linalg.norm(term) < 1e-4:
        #    break                
    return result

def expmv(t, A, v, tol=1e-7, krylov_dim=30):
    """
        Evaluates exp(t * A) @ v efficiently using Krylov subspace projection
        techniques and matrix-vector product operations.

    This function implements the expv function of the Expokit library
    (https://www.maths.uq.edu.au/expokit). It is in particular suitable for
    large sparse matrices.

    Args:
    t (float): real or complex time-like parameter
    A (array or sparse): an numpy.array or scipy.sparse square matrix
    v (array): a vector with size compatible with A
    tol (real): the required tolerance (default 1e-7)
    krylov_dim (int): dimension of the Krylov subspace
                      (typically a number between 15 and 50, default 30)

    Returns:
    The array w(t) = exp(t * A) @ v.
    """
    assert A.shape[1] == A.shape[0], "A must be square"
    assert A.shape[1] == v.shape[0], "v and A must have compatible shapes"

    n = A.shape[0]
    m = min(krylov_dim, n)

    anorm = spnorm(A, ord=np.inf) if issparse(A) else norm(A, ord=np.inf)
    out_type = np.result_type(type(t), A.dtype, v.dtype)

    # safety factors
    gamma = 0.9
    delta = 1.2

    btol = 1e-7     # tolerance for "happy-breakdown"
    maxiter = 10    # max number of time-step refinements

    rndoff = anorm*np.spacing(1)

    # estimate first time-step and round to two significant digits
    beta = norm(v)
    r = 1/m
    fact = (((m+1)/np.exp(1.0)) ** (m+1))*np.sqrt(2.0*np.pi*(m+1))
    tau = (1.0/anorm) * ((fact*tol)/(4.0*beta*anorm)) ** r

    outvec = np.zeros(v.shape, dtype=out_type)

    # storage for Krylov subspace vectors
    v_m = np.zeros((m + 1, len(v)), dtype=out_type)
    # for i in range(1, m + 2):
    #     vm.append(np.empty_like(outvec))
    h_m = np.zeros((m+2, m+2), dtype=outvec.dtype)

    t_f = np.abs(t)

    # For some reason numpy.sign has a different definition than Julia or MATLAB
    if isinstance(t, complex):
        tsgn = t / np.abs(t)
    else:
        tsgn = np.sign(t)

    t_k = 0. * t_f
    w = np.array(v, dtype=out_type)
    p = np.empty_like(w)

    m_x = m
    while t_k < t_f:
        tau = min(t_f - t_k, tau)
        # Arnoldi procedure
        v_m[0] = w / beta
        m_x = m

        for j in range(m):
            p = A.dot(v_m[j])

            h_m[:j+1, j] = v_m[:j+1, :].conj() @ p
            tmp = h_m[:j+1, j][:, np.newaxis] * v_m[:j+1]
            p -= np.sum(tmp, axis=0)

            s = norm(p)
            if s < btol: # happy-breakdown
                tau = t_f - t_k
                err_loc = btol

                f_m = expm(tsgn * tau * h_m[:j+1, :j+1])

                tmp = beta * f_m[:j+1, 0][:, np.newaxis] * v_m[:j+1, :]
                w = np.sum(tmp, axis=0)

                m_x = j
                break

            h_m[j+1, j] = s
            v_m[j+1] = p / s

        h_m[m + 1, m] = 1.

        if m_x == m:
            avnorm = norm(A @ v_m[m])

        # propagate using adaptive step size
        i = 1
        while i < maxiter and m_x == m:
            f_m = expm(tsgn * tau * h_m)

            err1 = abs(beta * f_m[m, 0])
            err2 = abs(beta * f_m[m+1, 0] * avnorm)

            if err1 > 10*err2:	# err1 >> err2
                err_loc = err2
                r = 1/m
            elif err1 > err2:
                err_loc = (err1*err2)/(err1-err2)
                r = 1/m
            else:
                err_loc = err1
                r = 1/(m-1)

            # is time step sufficient?
            if err_loc <= delta * tau * (tau*tol/err_loc) ** r:
                tmp = beta * f_m[:m+1, 0][:, np.newaxis] * v_m[:m+1, :]
                w = np.sum(tmp, axis=0)
                break

            # estimate new time-step
            tau = gamma * tau * (tau * tol / err_loc) ** r
            i += 1

        if i == maxiter:
            raise(RuntimeError("Number of iteration exceeded maxiter. "
                               "Requested tolerance might be too high."))

        beta = norm(w)
        t_k += tau
        tau = gamma * tau * (tau * tol / err_loc) ** r # estimate new time-step
        err_loc = max(err_loc, rndoff)

        h_m.fill(0.)

    return w














# very efficient out-side the algorithm but not working out inside 
# TODO : figure out why.

def lanczos_hermitian(A, v, tol=1e-10, max_iter=None, ortho_threshold=1e-6, min_iter=10):
    """
    Lanczos iteration with dynamic stopping for Hermitian matrices.
    Args:
        A: Hermitian matrix (NumPy array of shape (N, N)).
        v: Initial vector (NumPy array of shape (N,)).
        tol: Convergence tolerance for residual norm.
        max_iter: Maximum Lanczos iterations (default is min(200, 0.3 * N)).
        ortho_threshold: Orthogonality threshold for dynamic stopping.
        min_iter: Minimum number of iterations before stopping.
    Returns:
        T: Tridiagonal matrix (NumPy array of shape (k, k), where k <= max_iter).
        Q: Orthonormal basis (NumPy array of shape (N, k), where k <= max_iter).
    """
    N = A.shape[0]
    if max_iter is None:
        max_iter = min(200, int(0.3 * N))

    alpha, beta = [], []
    Q = np.zeros((N, max_iter), dtype=A.dtype)

    # Normalize initial vector
    q = v / np.linalg.norm(v)
    q_prev = np.zeros_like(q)
    beta_prev = 0.0

    for k in range(max_iter):
        Q[:, k] = q
        z = A @ q
        alpha_k = np.vdot(q, z)
        alpha.append(alpha_k)

        # Compute new Lanczos vector
        z -= alpha_k * q + beta_prev * q_prev
        beta_k = np.linalg.norm(z)

        # Check convergence after minimum iterations
        if k >= min_iter:
            # Combine residual norm and orthogonality error checks
            ortho_error = np.max(np.abs(np.dot(Q[:, :k+1].T.conj(), z)))
            if beta_k < tol or ortho_error < ortho_threshold:
                break

        beta.append(beta_k)

        # Reorthogonalize to maintain orthogonality
        for j in range(k + 1):
            z -= np.vdot(Q[:, j], z) * Q[:, j]

        beta_prev = beta_k
        q_prev = q
        q = z / (beta_k + 1e-12)

    # Construct tridiagonal matrix T
    T = np.zeros((k+1, k+1), dtype=A.dtype)
    for i in range(k+1):
        T[i, i] = alpha[i]
        if i > 0:
            T[i, i-1] = T[i-1, i] = beta[i-1]

    return T, Q[:, :k+1]

def lanczos_matrix_exp_hermitian(A, v, tol=1e-10, max_iter=None, ortho_threshold =1e-6, min_iter=10):
    """
    Compute exp(A)v using dynamic Lanczos iteration for Hermitian A.
    Args:
        A: Hermitian matrix (NumPy array of shape (N, N)).
        v: Initial vector (NumPy array of shape (N,)).
        tol: Convergence tolerance for residual norm.
        max_iter: Maximum Lanczos iterations (default is min(200, 0.3 * N)).
        ortho_threshold: Orthogonality threshold for dynamic stopping.
        min_iter: Minimum number of iterations before stopping.
    Returns:
        Approximation of exp(A)v (NumPy array of shape (N,)).
    """
    T, Q = lanczos_hermitian(A, v, tol, max_iter, ortho_threshold, min_iter)

    # Compute exp(T) in the Krylov subspace
    exp_T = expm(T)

    # Compute exp(A)v = Q @ exp(T) @ Q^H @ v
    approx = Q @ (exp_T @ (Q.conj().T @ v))
    return approx
