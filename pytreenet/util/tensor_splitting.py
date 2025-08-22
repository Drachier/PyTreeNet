"""
Functions that decompose a tensor.

The functions in this module are used to decompose a tensor into two or more
tensors. The QR decomposition and the singular value decomposition are
the most common decompositions used in tensor networks and implemented.
While the QR-Decomposition is faster, the SVD allows for a truncation of the
bond dimension, by discarding sufficiently small singular values.
"""
from typing import Tuple, Union, Optional
from enum import Enum
from warnings import warn
from dataclasses import dataclass
import numpy as np

from .tensor_util import tensor_matricization
from .ttn_exceptions import positivity_check
from ..gpu_config import GPUConfig

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class SplitMode(Enum):
    """
    Different modes on how to deal with the dimensions of a decomposed tensor.

    Decompositions are not unique, but differ in how to deal with the dimension
    of the new created bond. In principle one can increase the dimension as
    much as is desired, by adding zero vectors to the resulting tensors.

    FULL: The resulting tensor has the keeps the inout dimension of the
        original tensor.
    REDUCED: The resulting tensor has the minimum dimension between the
        input and output tensor. This means there are no zero vectors added.
        This is the default mode of numpy.
    KEEP: One of the resulting tensors has the same shape as the input tensor,
        if only one leg is split of from the rest.
    """
    FULL = "full"
    REDUCED = "reduced"
    KEEP = "keep"
    def numpy_qr_mode(self) -> str:
        """
        Returns the string required in the numpy QR decomposition.
        """
        if self is SplitMode.FULL:
            return "complete"
        return "reduced"

    def numpy_svd_mode(self) -> bool:
        """
        Returns the numpy SVD mode required.

        A numpy SVD has to be told, if it should return the full or reduced
        matrices by using a boolean.
        """
        return self is not SplitMode.REDUCED

def _cupy_tensor_qr_decomposition(tensor: np.ndarray,
                                 q_legs: Tuple[int, ...],
                                 r_legs: Tuple[int, ...],
                                 mode: SplitMode = SplitMode.REDUCED) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated QR decomposition using CuPy.
    
    This provides significant speedup over CPU-based QR decomposition
    and is compatible with PyTreeNet's interface.
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is required for GPU acceleration.")
    
    # Move tensor to GPU
    tensor_gpu = cp.asarray(tensor)
    
    # Determine the shape for matricization
    q_shape = tuple(tensor.shape[i] for i in q_legs)
    r_shape = tuple(tensor.shape[i] for i in r_legs)
    
    q_size = np.prod(q_shape)
    r_size = np.prod(r_shape)
    
    # Create permutation to group q_legs and r_legs together
    perm = list(q_legs) + list(r_legs)
    tensor_permuted = tensor_gpu.transpose(perm)
    
    # Reshape to matrix form
    matrix = tensor_permuted.reshape(q_size, r_size)
    
    # Perform QR decomposition on GPU
    if mode == SplitMode.REDUCED:
        q_matrix, r_matrix = cp.linalg.qr(matrix, mode='reduced')
    elif mode == SplitMode.FULL:
        q_matrix, r_matrix = cp.linalg.qr(matrix, mode='complete')
    else:  # SplitMode.KEEP
        q_matrix, r_matrix = cp.linalg.qr(matrix, mode='reduced')
    
    # Reshape back to tensor form
    bond_dim = q_matrix.shape[1]
    q_tensor_shape = q_shape + (bond_dim,)
    r_tensor_shape = (bond_dim,) + r_shape
    
    q_tensor = q_matrix.reshape(q_tensor_shape)
    r_tensor = r_matrix.reshape(r_tensor_shape)
    
    # KEEP mode padding if needed
    if mode == SplitMode.KEEP:
        orig_bond_dim = r_size
        if bond_dim < orig_bond_dim:
            diff = orig_bond_dim - bond_dim
            # Pad Q tensor
            padding_q = [(0, 0)] * (q_tensor.ndim - 1) + [(0, diff)]
            q_tensor = cp.pad(q_tensor, padding_q)
            # Pad R tensor  
            padding_r = [(0, diff)] + [(0, 0)] * (r_tensor.ndim - 1)
            r_tensor = cp.pad(r_tensor, padding_r)
    
    # Convert back to CPU NumPy arrays
    return cp.asnumpy(q_tensor), cp.asnumpy(r_tensor)

def tensor_qr_decomposition(tensor: np.ndarray,
                            q_legs: Tuple[int,...],
                            r_legs: Tuple[int,...],
                            mode: SplitMode = SplitMode.REDUCED) -> Tuple[np.ndarray,np.ndarray]:
    """
    Computes the QR decomposition of a tensor with respect to the given legs.
    Automatically uses GPU acceleration if enabled in GPUConfig.

    Args:
        tensor (np.ndarray): Tensor on which the QR-decomp is to applied.
        q_legs (Tuple[int]): Legs of tensor that should be associated to the
            Q tensor after QR-decomposition.
        r_legs (Tuple[int]): Legs of tensor that should be associated to the
            R tensor after QR-decomposition.
        mode (SplitMode, optional): Reduced returns a QR deocomposition with
            minimum dimension between Q and R. Full returns the decomposition
            with dimension between Q and R as the output dimension of Q. Keep
            causes Q to have the same shape as the input tensor. Defaults to
            SplitMode.REDUCED.

    Returns:
        Tuple[np.ndarray,np.ndarray]: (Q, R)::

                  |2                             |1
                __|_      r_legs = (1, )       __|_        ____
               |    |     q_legs = (0,2)      |    |      | R  |
            ___|    |___  -------------->  ___| Q  |______|____|____
            0  |____|  1                   0  |____| 2   0        1

    """
    config = GPUConfig()

    # Check if GPU acceleration should be used
    if (config.use_gpu and 
        config.gpu_available and 
        tensor.size >= config.gpu_threshold):
        try:
            if config.verbose:
                print(f"Using GPU QR for tensor size: {tensor.size:,}")
            return _cupy_tensor_qr_decomposition(tensor, q_legs, r_legs, mode)
        except Exception as e:
            if config.verbose:
                print(f"GPU QR failed ({e}), falling back to CPU")

    # CPU implementation
    correctly_order = q_legs + r_legs == list(range(len(q_legs) + len(r_legs)))
    matrix = tensor_matricization(tensor, q_legs, r_legs,
                                  correctly_ordered=correctly_order)
    q, r = np.linalg.qr(matrix, mode=mode.numpy_qr_mode())
    shape = tensor.shape
    q_shape = _determine_tensor_shape(shape, q, q_legs, output=True)
    r_shape = _determine_tensor_shape(shape, r, r_legs, output=False)
    q = q.reshape(q_shape)
    r = r.reshape(r_shape)
    if mode is SplitMode.KEEP:
        orig_bond_dim = np.prod(r.shape[1:])
        diff = orig_bond_dim - q.shape[-1]
        padding_q = [(0,0)] * (q.ndim-1)
        padding_q.append((0,diff))
        q = np.pad(q,padding_q)
        padding_r = [(0,diff)]
        padding_r.extend([(0,0)]*(r.ndim-1))
        r = np.pad(r,padding_r)
    return q, r

def cupy_svd(matrix: np.ndarray, full_matrices: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform SVD using CuPy for GPU acceleration.
    Args:
        matrix (np.ndarray): Matrix to decompose using SVD
        full_matrices (bool): Whether to compute full or reduced SVD
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (U, S, Vh) as numpy arrays
    """
    if cp is None:
        raise RuntimeError("CuPy not available for GPU SVD")

    # Transfer to GPU
    gpu_matrix = cp.asarray(matrix)

    # Perform SVD on GPU
    u_gpu, s_gpu, vh_gpu = cp.linalg.svd(gpu_matrix, full_matrices=full_matrices)

    # Transfer back to CPU
    u = cp.asnumpy(u_gpu)
    s = cp.asnumpy(s_gpu)
    vh = cp.asnumpy(vh_gpu)

    return u, s, vh

def cupy_randomized_svd(matrix: np.ndarray, k: int, n_iter: int = 1,
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU implementation of randomized SVD using CuPy.
    """
    # Move input to GPU
    matrix_gpu = cp.asarray(matrix)
    m, n = matrix_gpu.shape
    max_rank = min(m, n)

    if k > max_rank:
        raise ValueError(f"Target rank k={k} exceeds matrix dimensions {matrix.shape}")


    ell = int(k)

    # Random number generator
    if random_state is not None:
        cp.random.seed(random_state)

    # Random Gaussian test matrix
    if cp.iscomplexobj(matrix_gpu):
        Omega = (cp.random.randn(n, ell) + 1j * cp.random.randn(n, ell)).astype(matrix_gpu.dtype)
    else:
        Omega = cp.random.randn(n, ell, dtype=matrix_gpu.dtype)

    # Initial range finding
    Y0 = matrix_gpu @ Omega
    Q, _ = cp.linalg.qr(Y0, mode='reduced')

    # Power iterations
    for _ in range(n_iter):
        Y_hat = matrix_gpu.conj().T @ Q
        Q_hat, _ = cp.linalg.qr(Y_hat, mode='reduced')
        Y = matrix_gpu @ Q_hat
        Q, _ = cp.linalg.qr(Y, mode='reduced')

    # B = Q† A
    B = Q.conj().T @ matrix_gpu

    # SVD of the smaller matrix
    U_tilde, s, Vh = cp.linalg.svd(B, full_matrices=False)

    # U = Q Ũ
    U = Q @ U_tilde

    # Truncate to rank-k
    U_k = U[:, :k]
    s_k = s[:k]
    Vh_k = Vh[:k, :]

    # Move results back to CPU
    return cp.asnumpy(U_k), cp.asnumpy(s_k), cp.asnumpy(Vh_k)

def randomized_svd(matrix, k, n_iter=1, random_state=None):
    """
    Rank-k approximation of matrix using randomized SVD with power iterations.

    Args:
        matrix : array_like, shape (m, n)
            Input matrix (real or complex)
        k : int
            Target rank for the approximation (must be ≤ min(m, n))
        n_iter : int, default=4
            Number of power iterations (≥ 0)
        random_state : int, optional
            Seed for random number generation

    Returns:
        U_k : ndarray, shape (m, k)
            Left singular vectors
        s_k : ndarray, shape (k,)
            Singular values (sorted in descending order)
        Vh_k : ndarray, shape (k, n)
            Conjugate transpose of right singular vectors

    """
    m, n = matrix.shape
    max_rank = min(m, n)
    
    if k > max_rank:
        raise ValueError(
            f"Target rank k={k} exceeds maximum possible rank {max_rank} "
            f"for matrix shape {matrix.shape}")

    # NO OVERSAMPLING: ℓ = k exactly
    # This guarantees ℓ ≤ min(m, n) whenever k ≤ min(m, n)
    ell = k

    # Set up random number generator
    rng = np.random.RandomState(random_state)

    # Generate random matrix Ω ∈ ℝ^(n×k) or ℂ^(n×k)
    if np.iscomplexobj(matrix):
        # Complex case: Gaussian entries with real and imaginary parts
        Omega = (rng.randn(n, ell) + 1j * rng.randn(n, ell)).astype(matrix.dtype)
    else:
        # Real case: Standard Gaussian entries
        Omega = rng.randn(n, ell).astype(matrix.dtype)

    # ALGORITHM : Randomized SVD with q power iterations
    
    # Step 1-3: Initial range finding
    Y0 = matrix @ Omega                    # Y₀ = AΩ
    Q, _ = np.linalg.qr(Y0, mode='reduced') # [Q₀, R₀] = qr(Y₀)

    # Steps 4-9: Power iterations (if n_iter > 0)
    for iteration in range(n_iter):
        # Step 5: Ỹⱼ ← A† Qⱼ₋₁
        Y_hat = matrix.conj().T @ Q
        
        # Step 6: [Q̃ⱼ, R̃ⱼ] ← qr(Ỹⱼ)  
        Q_hat, _ = np.linalg.qr(Y_hat, mode='reduced')
        
        # Step 7: Yⱼ ← AQ̃ⱼ
        Y = matrix @ Q_hat
        
        # Step 8: [Qⱼ, Rⱼ] ← qr(Yⱼ)
        Q, _ = np.linalg.qr(Y, mode='reduced')

    # Steps 10-14: Final factorization
    # Step 11: B ← Q† A
    B = Q.conj().T @ matrix
    
    # Step 12: [Ũ, Σ, Ṽ†] ← svd(B)
    U_tilde, s, Vh = np.linalg.svd(B, full_matrices=False)
    
    # Step 13: U ← QŨ  
    U = Q @ U_tilde
    
    # Step 14: Return rank-k approximation
    # Since ℓ = k, we get exactly k singular values (no truncation needed!)
    U_k = U[:, :k]
    s_k = s[:k] 
    Vh_k = Vh[:k, :]

    return U_k, s_k, Vh_k

def tensor_svd(tensor: np.ndarray,
               u_legs: Tuple[int,...],
               v_legs: Tuple[int,...],
               mode: SplitMode = SplitMode.REDUCED,
               randomized_cutoff: int = float("inf")) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Perform a singular value decomposition on a tensor.

    Args:
        tensor (np.ndarray): Tensor on which the svd is to be performed.
        u_legs (Tuple[int]): Legs of tensor that are to be associated to U
            after the SVD.
        v_legs (Tuple[int]): Legs of tensor that are to be associated to V
            after the SVD.
        mode (SplitMode, optional): Determines if the full or reduced matrices
            u and vh obtained by the SVD are returned. The default is
            SplitMode.REDUCED.
        randomized_cutoff (int): If the minimum dimension of the
            matricized tensor is larger than this value, a randomized SVD is
            used. Defaults to float("inf").

    Returns:
        Tuple[np.ndarray,np.ndarray,np.ndarray]: (U, S, V), where S is the
            vector of singular values and U and V are tensors equivalent to an
            isometry::

                      |2                             |1
                    __|_      v_legs = (1, )       __|_        ____        ____
                   |    |     q_legs = (0,2)      |    |      |  S |      |    |
                ___|    |___  -------------->  ___|  U |______|____|______| Vh |____1 
                0  |____|  1                   0  |____| 2   0       1  0 |____|

    """
    correctly_ordered = u_legs + v_legs == list(range(len(u_legs) + len(v_legs)))
    full_matrices = mode.numpy_svd_mode()
    matrix = tensor_matricization(tensor, u_legs, v_legs,
                                  correctly_ordered=correctly_ordered)
    m, n = matrix.shape
    min_dim = min(m, n)

    flag = False
    config = GPUConfig()
    if min_dim >= randomized_cutoff:
        k = randomized_cutoff
        n_iter = 2
        if config.use_gpu:
            if config.verbose:
                print(f"Using GPU randomized SVD for tensor size: {matrix.size}")
            u, s, vh = cupy_randomized_svd(matrix, k, n_iter)
        else:
            u, s, vh = randomized_svd(matrix, k, n_iter)
        flag = True

    elif config.should_use_gpu(matrix.size) and not flag:
        if config.verbose:
            print(f"Using GPU SVD for tensor size: {matrix.size}")
        u, s, vh = cupy_svd(matrix, full_matrices=full_matrices)
    else:
        u, s, vh = np.linalg.svd(matrix, full_matrices=full_matrices)

    shape = tensor.shape
    u_shape = _determine_tensor_shape(shape, u, u_legs, output=True)
    vh_shape = _determine_tensor_shape(shape, vh, v_legs, output=False)
    u = u.reshape(u_shape)
    vh = vh.reshape(vh_shape)
    return u, s, vh

def _determine_tensor_shape(old_shape: Tuple[int,...],
                            matrix: np.ndarray,
                            legs: Tuple[int,...],
                            output: bool = True) -> Tuple[int,...]:
    """
    Determines the shape of a tensor after a decomposition.

    Determines the new shape a matrix is to be reshaped to after a decomposition
    of a tensor with some original shape, to again obtain a tensor.
    Works only if all legs to be reshaped are combined in the input or output
    leg of the matrix.
    
    Args:
        old_shape (Tuple[int]): Shape of the original tensor.
        matrix (np.ndarray): Matrix to be reshaped.
        legs (Tuple[int]): Which legs of the original tensor are associated to
            the matrix.
        output (bool, optional): If the legs of the original tensor are
            associated to the input or output of matrix. Defaults to True.

    Returns:
        Tuple[int]: New shape to which matrix is to be reshaped.
    """
    leg_shape = [old_shape[i] for i in legs]
    if output:
        matrix_dimension = [matrix.shape[1]]
        leg_shape.extend(matrix_dimension)
        new_shape = leg_shape
    else:
        matrix_dimension = [matrix.shape[0]]
        matrix_dimension.extend(leg_shape)
        new_shape = matrix_dimension

    return tuple(new_shape)

@dataclass
class SVDParameters:
    """
    Holds all the parameters required for a truncated singular value
    decomposition.

    Attributes:
        max_bond_dim (int, optional): The maximum bond dimension allowed
            between nodes. Defaults to 100.
        max_product_dim (int, optional): 
            - The maximum product of bond dimensions
              of all neighbours of a node. 
            - Used to apply greedy optimization to
              distribute the bond dimensions, such that the product of bond
              dimensions is bellow this value. 
            - Defaults to infinity, which means
            that no greedy optimization is applied.
        randomized_cutoff (int, optional): 
            - If minimum dimension of the matricized 
              tensor is larger than this value, a randomized SVD is used.
            - Defaults to infinity, which means that the regular SVD is used.
        rel_tol (float, optional): singular values s for which
            (s / largest singular value) < rel_tol are truncated. Defaults to
            1e-15.
        total_tol (float, optional): singular values s for which s < total_tol
            are truncated. Defaults to 1e-15.
        renorm (bool, optional): If True, the truncated singular value vector
            is scaled to have the same norm as the original vector. Defaults to
            False.
        sum_trunc (bool, optional): If True, all singular values with index
            larger than K are truncated, where K is the smallest index such
            .. math::
                \sum_{i=K}^{r} (s_i / ||s|| )^2 < \text{total_tol}^2
            where r is the number of singular values. Defaults to False.
    """
    max_bond_dim: int = float("inf")
    max_product_dim: int = float("inf")
    randomized_cutoff: int = float("inf")
    rel_tol: float = 1e-15
    total_tol: float = 1e-15
    renorm: bool = False
    sum_trunc: bool = False
    sum_renorm: bool = False

    def __post_init__(self):
        """
        Check the validity of the parameters.
        """
        self.check_truncation_parameters()

    def check_truncation_parameters(self):
        """
        Checks if the truncation parameters are valid.

        The maximum bond dimension has to be a positive integer or infinity.
        The relative tolerance has to be positive or -infinity.
        The total tolerance has to be positive or -infinity.
        """
        max_bond_dim = self.max_bond_dim
        if (not isinstance(max_bond_dim,int)) and (max_bond_dim != float("inf")):
            raise TypeError(f"'max_bond_dim' has to be int not {type(max_bond_dim)}!")
        positivity_check(max_bond_dim, "max_bond_dim")
        rel_tol = self.rel_tol
        if (rel_tol < 0) and (rel_tol != float("-inf")):
            raise ValueError("'rel_tol' has to be positive or -inf.")
        total_tol = self.total_tol
        if (total_tol < 0) and (total_tol != float("-inf")):
            raise ValueError("'total_tol' has to be positive or -inf.")
        if self.sum_trunc:
            assert self.total_tol != float("-inf"), "If 'sum_trunc' is True, 'total_tol' has to be positive or 0!"

def renormalise_singular_values(s: np.ndarray,
                                new_s: np.ndarray) -> np.ndarray:
    """
    Renormalises a truncated singular value vector.

    The vector is scaled to have the same norm as the original vector.

    Args:
        s (np.ndarray): The original vector of singular values.
        new_s (np.ndarray): The truncated vector of singular values.

    Returns:
        np.ndarray: The renormalised vector new_s.
    """
    norm_old = np.sum(s)
    norm_new = np.sum(new_s)
    new_s = new_s * norm_old / norm_new
    return new_s

def _sum_truncation_index(s: np.ndarray,
                          total_tol: float,
                          norming: bool = True) -> int:
    """
    Determines the index for truncation of the singular values.

    The index is determined by the condition

    .. math::
        \sum_{i=K}^{r} (s_i / ||s|| )^2 < \text{total_tol}^2

    where r is the number of singular values.

    Args:
        s (np.ndarray): Vector of singular values sorted in descending order.
        total_tol (float): Tolarance for the sum of the squared singular
            values.
        norming (bool, optional): If True, the sum of the squared singular
            values is normalised by the squared norm of the vector. Defaults to
            True.
    
    Returns:
        int: The index K for truncation.
    
    """
    normsq = np.linalg.norm(s)**2
    if len(s) == 0:
        raise ValueError("No singular values to truncate!")
    if norming:
        normsq = np.linalg.norm(s) ** 2
    else:
        normsq = 1.0
    for k in range(len(s)):
        tail_sum = np.sum(s[k:] ** 2) / normsq
        if tail_sum < total_tol ** 2:
            return k
    return len(s)

def sum_truncation(s: np.ndarray,
                   total_tol: float,
                   norming: bool = True
                   ) -> np.ndarray:
    """
    Truncates the singular values of a tensor given as a vector by according to

    .. math::
        \sum_{i=K}^{r} (s_i / ||s|| )^2 < \text{total_tol}^2

    where r is the number of singular values.

    Args:
        s (np.ndarray): Vector of singular values sorted in descending order.
        total_tol (float): Tolarance for the sum of the squared singular
            values.
        norming (bool, optional): If True, the truncated singular value vector
            is scaled to have the same norm as the original vector. Defaults to
            True.

    Returns:
        np.ndarray: The truncated vector of singular values.

    """
    trunc_index = _sum_truncation_index(s, total_tol, norming=norming)
    return s[:trunc_index]

def value_truncation(s: np.ndarray,
                     total_tol: float,
                     rel_tol: float) -> np.ndarray:
    """
    Truncates a vector of by removing all singular values that are smaller than

    .. math::
        \max(\text{rel_tol} \cdot \text{max}(s), \text{total_tol}).

    Args:
        s (np.ndarray): Vector of singular values sorted in descending order.
        total_tol (float): Absolute value tolerance for the singular values.
        rel_tol (float): Tolarance for the relative size of the singular values.

    Returns:
        np.ndarray: The truncated vector of singular values.

    """
    max_singular_value = s[0]
    min_singular_value_cutoff = max(rel_tol * max_singular_value,
                                    total_tol)
    s_temp = s[s > min_singular_value_cutoff]
    return s_temp

def truncate_singular_values(s: np.ndarray,
                             svd_params: SVDParameters) -> Tuple[np.ndarray,np.ndarray]:
    """
    Truncates the singular values of a tensor given as a vector.

    Args:
        s (np.ndarray): Vector of singular values sorted in descending order.
        svd_params (SVDParameters): Parameters for the truncation of the singular
            values.

    Returns:
        Tuple[np.ndarray,np.ndarray]: (new_s, s_trunc), where is is the
            shortened vector of singular values and s_trunc is a vector of the
            truncated singular values.
    """
    if len(s) == 0:
        raise ValueError("No singular values to truncate!")
    if svd_params.sum_trunc:
        s_temp = sum_truncation(s, svd_params.total_tol, svd_params.sum_renorm)
    else:
        s_temp = value_truncation(s, svd_params.total_tol, svd_params.rel_tol)
    max_bond_dim = svd_params.max_bond_dim
    if len(s_temp) > max_bond_dim:
        new_s = s_temp[:max_bond_dim]
        s_trunc = s[max_bond_dim:]
    elif len(s_temp) == 0:
        warn("All singular values were truncated. Returning only the largest singular value.")
        s_trunc = s[1:]
        new_s = np.array([s[0]])
    else:
        new_s = s_temp
        s_trunc = s[len(s_temp):]
    if svd_params.renorm:
        new_s = renormalise_singular_values(s, new_s)
    return new_s, s_trunc

def truncated_tensor_svd(tensor: np.ndarray,
                         u_legs: Tuple[int,...],
                         v_legs: Tuple[int,...],
                         svd_params: SVDParameters) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Performs a singular value decomposition of a tensor including truncation.

    This means some of the singular values are discarded and the resulting
    tensors are truncated to the new bond dimension.

    Args:
        tensor (np.ndarray): Tensor on which the svd is to be performed.
        u_legs (Tuple[int]): Legs of tensor that are to be associated to U
            after the SVD.
        v_legs (Tuple[int]): Legs of tensor that are to be associated to V
            after the SVD.
        svd_params (SVDParameters): Parameters for the truncation of the
            singular values.
    
    Returns:
        Tuple[np.ndarray,np.ndarray,np.ndarray]: (U, S, V), where U and V are
         the truncated tensors and S are the singular values::

                  |2                             |1
                __|_      v_legs = (1, )       __|_        ____        ____
               |    |     q_legs = (0,2)      |    |      |  S |      |    |
            ___|    |___  -------------->  ___|  U |______|____|______| Vh |____1 
            0  |____|  1                   0  |____| 2   0       1  0 |____|

    """
    u, s, vh = tensor_svd(tensor, 
                          u_legs, 
                          v_legs, 
                          mode = SplitMode.REDUCED,
                          randomized_cutoff = svd_params.randomized_cutoff)
    s, _ = truncate_singular_values(s, svd_params)
    new_bond_dim = len(s)
    u = u[..., :new_bond_dim]
    vh = vh[:new_bond_dim, ...]
    return u, np.asarray(s), vh

class ContractionMode(Enum):
    """
    Which tensor the singular values are contracted into.

    VCONTR: Contracts the singular values with the V tensor.
    UCONTR: Contracts the singular values with the U tensor.
    EQUAL: Contracts the squareroot of the singular values with each tensor.
    """
    UCONTR = "ucontr"
    VCONTR = "vcontr"
    EQUAL = "equal"

def contr_truncated_svd_splitting(tensor: np.ndarray,
                                  u_legs: Tuple[int,...],
                                  v_legs: Tuple[int,...],
                                  contr_mode: ContractionMode = ContractionMode.VCONTR,
                                  svd_params: SVDParameters = SVDParameters()) -> Tuple[np.ndarray,np.ndarray]:
    """
    Performs a truncated singular value decomposition, but the singular values
     are contracted with the V tensor.

    Args:
        tensor (np.ndarray): Tensor on which the svd is to be performed.
        u_legs (Tuple[int]): Legs of tensor that are to be associated to U
            after the SVD.
        v_legs (Tuple[int]): Legs of tensor that are to be associated to V
            after the SVD.
        contr_mode (ContractionMode): Determines how the singular values are
            contracted into the other tensors.
        svd_params (SVDParameters): Parameters for the truncation of the
    """
    u, s, vh = truncated_tensor_svd(tensor, u_legs, v_legs, svd_params)
    if contr_mode == ContractionMode.VCONTR:
        svh = np.tensordot(np.diag(s), vh, axes=(1, 0))
        return u, svh
    if contr_mode == ContractionMode.UCONTR:
        us = np.tensordot(u,np.diag(s), axes=(-1, 0))
        return us, vh
    if contr_mode == ContractionMode.EQUAL:
        s_root = np.diag(np.sqrt(s))
        svh = np.tensordot(s_root, vh, axes=(1, 0))
        us = np.tensordot(u,s_root,axes=(-1, 0))
        return us, svh
    raise ValueError("Invalid contraction mode!")

def idiots_splitting(tensor: np.ndarray,
                     a_legs: Tuple[int,...],
                     b_legs: Tuple[int,...],
                     a_tensor: Union[np.ndarray,None] = None,
                     b_tensor: Union[np.ndarray,None] = None,
                     strict_checks: bool = True) -> Tuple[np.ndarray,np.ndarray]:
    """
    An idiots splitting of a tensor by two given compatible tensors.

    Performs checks if the given tensors are compatible with the given legs.

    Args:
        tensor (np.ndarray): Tensor to be split.
        a_legs (Tuple[int]): Legs of tensor that are to be associated to A.
        b_legs (Tuple[int]): Legs of tensor that are to be associated to B.
        a_tensor (Union[np.ndarray,None], optional): Given tensor A. Leg
            connecting to B is the last leg. Defaults to None.
        b_tensor (Union[np.ndarray,None], optional): Given tensor B. Leg
            connecting to A is the first leg. Defaults to None.
        strict_checks (bool, optional): If True, raises exceptions for 
            incompatible tensors. If False, only prints warnings. Defaults to True.

    Returns:
        Tuple[np.ndarray,np.ndarray]: (A, B), the two split tensors.

    """
    if a_tensor is None or b_tensor is None:
        raise ValueError("Both tensors have to be given!")
    if strict_checks:
        tensor_shape_a = tuple([tensor.shape[i] for i in a_legs])
        tensor_shape_b = tuple([tensor.shape[i] for i in b_legs])
        a_shape = a_tensor.shape[0:-1]
        b_shape = b_tensor.shape[1:]
        if tensor_shape_a != a_shape:
            raise ValueError("A tensor not compatible!")
        if tensor_shape_b != b_shape:
            raise ValueError("B tensor not compatible!")
    if a_tensor.ndim != len(a_legs) + 1:
        raise ValueError("A tensor has wrong number of legs!")
    if b_tensor.ndim != len(b_legs) + 1:
        raise ValueError("B tensor has wrong number of legs!")
    if a_tensor.shape[-1] != b_tensor.shape[0]:
        raise ValueError("A and B tensor not compatible!")
    return (a_tensor, b_tensor)