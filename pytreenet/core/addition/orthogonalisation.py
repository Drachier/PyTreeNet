"""
Orthogonalisation functions for TTNS using the addition module.
"""
from __future__ import annotations
from typing import List, Union
import numpy as np
import scipy
from copy import deepcopy

from ...ttns import TreeTensorNetworkState
from ...ttno import TTNO
from ...operators.hamiltonian import Hamiltonian
from ...contractions.state_operator_contraction import get_matrix_element
from ...util.tensor_splitting import SVDParameters

from .addition import AdditionMethod
from .linear_combination import LinearCombination
from ...ttns.ttns_ttno.application import ApplicationMethod
from ...core.truncation.truncating import TruncationMethod

__all__ = [
    'orthogonalise_gram_schmidt',
    'orthogonalise_cholesky',
    'orthogonalise_gep',
    'orthogonalise_to',
    'linear_combination',
]

# TODO: Test properly

def orthogonalise_gram_schmidt(ttns_list: List[TreeTensorNetworkState],
                              max_bond_dim: int,
                              num_sweeps: int
                              ) -> List[TreeTensorNetworkState]:
    """
    Gram-Schmidt orthogonalisation of a list of TTNS.

    Args:
        ttns_list (List[TreeTensorNetworkState]): List of TTNS to
            orthogonalise.
        max_bond_dim (int): Maximum bond dimension of the resulting TTNS.
        num_sweeps (int): Number of sweeps for the variational fitting.

    Returns:
        List[TreeTensorNetworkState]: List of orthogonalised TTNS.
    """
    ttns_list = deepcopy(ttns_list)
    for i in range(1, len(ttns_list)):
        ttns_list[i] = orthogonalise_to(ttns_list[i],
                                       ttns_list[:i],
                                       max_bond_dim,
                                       num_sweeps)
        root_id = ttns_list[i].root_id
        assert root_id is not None
        ttns_list[i].canonical_form(root_id)
        ttns_list[i].normalize()
    return ttns_list


def orthogonalise_cholesky(ttns_list: List[TreeTensorNetworkState],
                          max_bond_dim: int,
                          num_sweeps: int
                          ) -> List[TreeTensorNetworkState]:
    """
    Orthogonalises a list of TTNS using the Cholesky decomposition.

    Args:
        ttns_list (List[TreeTensorNetworkState]): List of TTNS to
            orthogonalise.
        max_bond_dim (int): Maximum bond dimension of the resulting TTNS.
        num_sweeps (int): Number of sweeps for the variational fitting.
    
    Returns:
        List[TreeTensorNetworkState]: List of orthogonalised TTNS.
    """
    ttns_list_return = deepcopy(ttns_list)
    ovp = np.zeros((len(ttns_list), len(ttns_list)),
                  dtype=np.complex128)
    for i, ttns in enumerate(ttns_list):
        ovp[i,i] = ttns.scalar_product()
        for j in range(i+1, len(ttns_list)):
            ovp[i,j] = ttns_list[j].scalar_product(ttns_list[i])
            ovp[j,i] = ovp[i,j].conjugate()
    e,v = np.linalg.eigh(ovp)
    vs = np.where(e > 1e-12, 1.0 / np.sqrt(e), 0.0)
    L_inv = v @ np.diag(vs) @ v.T.conj()
    for i in range(len(ttns_list)):
        dtype = ttns_list[i].tensors[ttns_list[i].root_id].dtype
        ttns_list_return[i] = linear_combination(ttns_list,
                                                L_inv[:,i].tolist(),
                                                max_bond_dim,
                                                num_sweeps,
                                                dtype)
    return ttns_list_return


def orthogonalise_gep(ttno: TTNO,
                     ttns_list: List[TreeTensorNetworkState],
                     max_bond_dim: int,
                     num_sweeps: int
                     ) -> List[TreeTensorNetworkState]:
    """
    Orthogonalises a list of TTNS by solving generalised eigenvalue problem.

    Args:
        ttno (TTNO): The TTNO to use for the orthogonalisation.
        ttns_list (List[TreeTensorNetworkState]): List of TTNS to
            orthogonalise.
        max_bond_dim (int): Maximum bond dimension of the resulting TTNS.
        num_sweeps (int): Number of sweeps for the variational fitting.

    Returns:
        List[TreeTensorNetworkState]: List of orthogonalised TTNS.
    """
    ttns_list_return = deepcopy(ttns_list)
    ovp = np.zeros((len(ttns_list), len(ttns_list)), dtype=np.complex128)
    h = np.zeros((len(ttns_list), len(ttns_list)), dtype=np.complex128)
    for i, ttns in enumerate(ttns_list):
        ovp[i,i] = ttns.scalar_product().real
        h[i,i] = ttns.operator_expectation_value(ttno).real
        for j in range(i+1, len(ttns_list)):
            ovp[i,j] = ttns_list[j].scalar_product(ttns_list[i]).real
            ovp[j,i] = ovp[i,j]
            h[i,j] = get_matrix_element(ttns.conjugate(),
                                       ttno,
                                       ttns_list[j]).real
            h[j,i] = h[i,j]
    # Solve the generalized eigenvalue problem
    _, ev = scipy.linalg.eigh(h, ovp)
    ev = ev.real
    for i in range(len(ttns_list)):
        ttns_list_return[i] = linear_combination(ttns_list,
                                                ev[:,i],
                                                max_bond_dim,
                                                num_sweeps)
    return ttns_list_return


def orthogonalise_to(ttns: TreeTensorNetworkState,
                    state_list: Union[List[TreeTensorNetworkState], List[str]],
                    max_bond_dim: int,
                    num_sweeps: int
                    ) -> TreeTensorNetworkState:
    """
    Orthogonalises a TTNS to a list of TTNS.

    Args:
        ttns (TreeTensorNetworkState): The TTNS to orthogonalise.
        state_list (Union[List[TreeTensorNetworkState], List[str]]): The list of
            TTNS to orthogonalise to. If a list of strings is provided, the
            strings are interpreted as file paths to load the TTNS from.
        max_bond_dim (int): The maximum bond dimension of the resulting TTNS.
        num_sweeps (int): The number of sweeps for the variational fitting.
    Returns:
        TreeTensorNetworkState: The orthogonalised TTNS.
    """
    if len(state_list) == 0:
        return ttns
    if isinstance(state_list, list) and isinstance(state_list[0], str):
        state_list = [TreeTensorNetworkState().load(path)
                     for path in state_list]
    coeffs = [1.0 + 1.0j]
    ttns_list = [ttns]
    for state in state_list:
        assert isinstance(state, TreeTensorNetworkState)
        overlap = ttns.scalar_product(state)
        if abs(overlap) > 1e-4:
            coeffs.append(-1 * overlap)
            ttns_list.append(state)
    dtype = ttns.tensors[ttns.root_id].dtype
    return linear_combination(ttns_list, coeffs, max_bond_dim, num_sweeps, dtype)


def linear_combination(ttns_list: List[TreeTensorNetworkState],
                      coeffs: Union[float, complex, List[float], List[complex]],
                      max_bond_dim: int,
                      **kwargs
                      ) -> TreeTensorNetworkState:
    """
    Returns a linear combination of a list of TTNS.
    
    This implementation uses the LinearCombination class from core.addition
    with the DIRECT_TRUNCATE method.
    
    Args:
        ttns_list (List[TreeTensorNetworkState]): The list of TTNS to combine.
        coeffs (Union[float, complex, List[float], List[complex]]): The coefficients
            of the linear combination.
        max_bond_dim (int): The maximum bond dimension of the resulting TTNS.
        **kwargs: For backward compatibility, additional keyword arguments
            are accepted but ignored.

    Returns:
        TreeTensorNetworkState: The linear combination of the TTNS.
    """
    
    if isinstance(coeffs, (float, complex)):
        coeffs = [coeffs] * len(ttns_list)
    
    # Filter out small coefficients
    abs_coeffs = [abs(coeff) for coeff in coeffs]
    ordering = np.argsort(abs_coeffs)[::-1]
    mask = np.array([abs_coeffs[i] >= 1e-4 for i in ordering])
    ordering = ordering[mask]
    ttns_filtered = [ttns_list[i] for i in ordering]
    coeffs_filtered = [coeffs[i] for i in ordering]
    
    # Use LinearCombination with DIRECT_TRUNCATE method
    lc = LinearCombination(
        ttnss=ttns_filtered,
        ttnos=None,
        coefficients=coeffs_filtered
    )
    
    # Use DIRECT_TRUNCATE addition method with SVD parameters
    svd_params = SVDParameters(max_bond_dim, 1e-10, 1e-10)
    
    result = lc.compute(
        ApplicationMethod.DIRECT_TRUNCATE,
        AdditionMethod.DIRECT_TRUNCATE,
        args_add=(TruncationMethod.SVD, ),
        kwargs_add={'params': svd_params}
    )
    
    return result
