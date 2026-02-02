"""
This module implements the randomised SVD from

"Finding structure with randomness: Probabilistic algorithms for constructing
approximate matrix decompositions" by Halko, Martinsson, Tropp (2011).

The code is adapted from the implementation at
https://github.com/gwgundersen/randomized-svd
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from enum import Enum

from numpy.linalg import svd, qr

from ..crandn import crandn

if TYPE_CHECKING:
    import numpy.typing as npt

class SubspaceIterationNumber(Enum):
    """
    Enum for specifying the number of subspace iterations in the randomized SVD.

    Attributes:
        NONE (int): No subspace iterations.
        AUTO (str): Automatically determine the number of subspace iterations.
            This for unless the number of ranks desired is smaller than
            `0.1 * min(matrix.shape)`, in which case 7 iterations are used.
            This is the same as in the sklearn implementation.
    """
    NONE = 0
    AUTO = "auto"

    def iteration_number(self,
                         matrix: npt.NDArray,
                         max_rank: int
                         ) -> int:
        """
        Determines the number of subspace iterations based on the enum value.

        Args:
            matrix (npt.NDArray): The input matrix.
            max_rank (int): The maximum rank of the decomposition.
        
        Returns:
            int: The number of subspace iterations to perform.
        """
        if self is SubspaceIterationNumber.AUTO:
            min_dim = min(matrix.shape)
            if max_rank < 0.1 * min_dim:
                n_subspace_iters = 7
            else:
                n_subspace_iters = 4
        elif self is SubspaceIterationNumber.NONE:
            n_subspace_iters = 0
        else:
            raise ValueError(f"Unsupported SubspaceIterationNumber value: {self}!")
        return n_subspace_iters

def randomized_svd(matrix: npt.NDArray,
                   max_rank: int,
                   n_oversamples: int | None = None,
                   n_subspace_iters: int | SubspaceIterationNumber = SubspaceIterationNumber.AUTO,
                   seed: int | None = None
                   ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Computes the truncated randomized SVD of a matrix.

    Args:
        matrix (npt.NDArray): The input matrix to decompose.
        max_rank (int): The maximum rank of the decomposition.
        n_oversamples (int | None, optional): The number of oversampling vectors to use.
            Defaults to None, which sets it to `max_rank`.
        n_subspace_iters (int | SubspaceIterationNumber, optional): The number of
            subspace iterations to perform. If set to `SubspaceIterationNumber.AUTO`,
            the number of iterations is determined automatically based on the size
            of the input matrix and the desired rank. Defaults to
            `SubspaceIterationNumber.AUTO`.
        seed (int | None, optional): The random seed to use for generating
            the random matrix. Defaults to None.
    
    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]: The left singular vectors,
            the singular values, and the right singular vectors of the input matrix.
            The shapes are (m, max_rank), (max_rank,), and (max_rank, n) respectively,
            where m and n are the dimensions of the input matrix.
    """
    if n_oversamples is None:
        n_oversamples = max_rank
    num_samples = max_rank + n_oversamples
    if isinstance(n_subspace_iters, SubspaceIterationNumber):
        n_subspace_iters = n_subspace_iters.iteration_number(matrix, max_rank)
    approx_range = find_range(matrix, num_samples, n_subspace_iters,
                                seed=seed)
    # Find the SVD in the subspace
    small_mat = approx_range.T.conj() @ matrix
    u_tilde, s, vt = svd(small_mat)
    u = approx_range @ u_tilde
    # Truncate to the desired rank
    u = u[:, :max_rank]
    s = s[:max_rank]
    vt = vt[:max_rank, :]
    return u, s, vt

def randomised_qb(matrix: npt.NDArray,
                  desired_rank: int,
                  n_subspace_iters: int | SubspaceIterationNumber = SubspaceIterationNumber.NONE,
                  seed: int | None = None
                    ) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Computes a low-rank QB decomposition of the input matrix using randomized methods.

    Args:
        matrix (npt.NDArray): The input matrix to decompose.
        desired_rank (int): The desired rank of the decomposition.
        n_subspace_iters (int | SubspaceIterationNumber, optional): The number of
            subspace iterations to perform. If set to `SubspaceIterationNumber.AUTO`,
            the number of iterations is determined automatically based on the size
            of the input matrix and the desired rank. Defaults to
            `SubspaceIterationNumber.NONE`.
        seed (int | None, optional): The random seed to use for generating
            the random matrix. Defaults to None.

    Returns:
        tuple[npt.NDArray, npt.NDArray]: The orthonormal matrix Q and the
            matrix B such that matrix ≈ Q @ B. The shapes are (m, desired_rank)
            and (desired_rank, n) respectively, where m and n are the dimensions
            of the input matrix.
    """
    if isinstance(n_subspace_iters, SubspaceIterationNumber):
        n_subspace_iters = n_subspace_iters.iteration_number(matrix, desired_rank)
    approx_range = find_range(matrix,
                              desired_rank,
                              n_subspace_iters,
                              seed=seed)
    b_mat = approx_range.conj().T @ matrix
    return approx_range, b_mat

def find_range(matrix: npt.NDArray,
               num_samples: int,
               n_subspace_iters: int,
               seed: int | None = None
               ) -> npt.NDArray:
    """
    Finds an orthonormal matrix whose range approximates the range of the input matrix.

    Args:
        matrix (npt.NDArray): The input matrix.
        num_samples (int): The number of random samples to use.
        n_subspace_iters (int): The number of subspace iterations to perform.
        seed (int | None, optional): The random seed to use for generating
            the random matrix. Defaults to None.

    Returns:
        npt.NDArray: An orthonormal matrix whose range approximates the range
            of the input matrix.
    """
    _, size = matrix.shape
    rand_mat = crandn((size, num_samples), seed=seed)
    sample_mat = matrix @ rand_mat
    if n_subspace_iters > 0:
        approx_range = subspace_iteration(matrix, sample_mat, n_subspace_iters)
    else:
        approx_range = orth_basis(sample_mat)
    return approx_range

def subspace_iteration(matrix: npt.NDArray,
                       sample_mat: npt.NDArray,
                       n_iters: int
                       ) -> npt.NDArray:
    """
    Performs subspace iterations to improve the approximation of the range of the input matrix.

    Args:
        matrix (npt.NDArray): The input matrix.
        sample_mat (npt.NDArray): The initial sample matrix.
        n_iters (int): The number of subspace iterations to perform.

    Returns:
        npt.NDArray: The improved sample matrix after subspace iterations.
    """
    orth_mat = orth_basis(sample_mat)
    for _ in range(n_iters):
        new_samp_mat = orth_basis(matrix.conj().T @ orth_mat)
        orth_mat = orth_basis(matrix @ new_samp_mat)
    return orth_mat

def orth_basis(matrix: npt.NDArray) -> npt.NDArray:
    """
    Computes an orthonormal basis for the range of the input matrix using QR decomposition.

    Args:
        matrix (npt.NDArray): The input matrix.
    
    Returns:
        npt.NDArray: An orthonormal basis for the range of the input matrix.
    """
    q, _ = qr(matrix)
    return q
