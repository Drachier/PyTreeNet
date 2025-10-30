from __future__ import annotations
from typing import List, Union

from copy import deepcopy

import numpy as np
import scipy

from .tensor_splitting import SVDParameters

from ..ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TTNO
from ..dmrg.variational_fitting import VariationalFitting
from ..operators.hamiltonian import Hamiltonian
from ..contractions.state_operator_contraction import get_matrix_element
from ..contractions.contraction_util import get_equivalent_legs
from ..util.tensor_splitting import SVDParameters

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
    L_inv = v@np.diag(vs)@v.T.conj()
    for i in range(len(ttns_list)):
        dtype = ttns_list[i].tensors[ttns_list[i].root_id].dtype
        ttns_list_return[i] = linear_combination(ttns_list,
                                                 L_inv[:,i],
                                                 max_bond_dim,
                                                 num_sweeps,
                                                 dtype)
    return ttns_list_return

def orthogonalise_gep(ttno: TTNO,
                      ttns_list: List[TreeTensorNetworkState],
                      max_bond_dim: int,
                      num_sweeps:int
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
        ovp[i,i] = ttns.scalar_product()
        h[i,i] = ttns.operator_expectation_value(ttno)
        for j in range(i+1, len(ttns_list)):
            ovp[i,j] = ttns_list[j].scalar_product(ttns_list[i])
            ovp[j,i] = ovp[i,j]
            h[i,j] = get_matrix_element(ttns.conjugate(),
                                        ttno,
                                        ttns_list[j])
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

def linear_combination(ttns: List[TreeTensorNetworkState],
                       coeffs: Union[float, complex, List[float], List[complex], np.ndarray],
                       max_bond_dim: int,
                       num_sweeps: int = 10,
                       dtype: np.dtype = np.float64
                       ) -> TreeTensorNetworkState:
    """
    Returns a linear combination of a list of TTNS.
    
    Args:
        ttns (List[TreeTensorNetworkState]): The list of TTNS to combine.
        coeffs (Union[float, complex, List[float], List[complex], np.ndarray]): The coefficients of the linear combination.
        max_bond_dim (int): The maximum bond dimension of the resulting TTNS.
        num_sweeps (int): The number of sweeps for the variational fitting.

    Returns:
        TreeTensorNetworkState: The linear combination of the TTNS.
    """
    identity_ttno = TTNO.from_hamiltonian(Hamiltonian.identity_like(ttns[0],dtype=dtype),
                                          ttns[0],
                                          dtype=dtype)
    if isinstance(coeffs, (float, complex)):
        coeffs = [coeffs]*len(ttns)
    elif isinstance(coeffs, np.ndarray):
        assert coeffs.shape == (len(ttns),), "The coefficients must be a list of length the number of TTNS."
        coeffs = coeffs.tolist()
    assert isinstance(coeffs, list)
    abs_coeffs = [abs(coeff) for coeff in coeffs]
    ordering = np.argsort(abs_coeffs)[::-1]
    # Filter out small coefficients using a mask
    mask = np.array([abs_coeffs[i] >= 1e-4 for i in ordering])
    if not mask.any():
        mask[0] = True
    ordering = ordering[mask]
    ttns = [ttns[i] for i in ordering]
    coeffs = [coeffs[i] for i in ordering]
    y = deepcopy(ttns[0])
    # y.canonical_form(y.root_id)
    # y.pad_bond_dimensions(max_bond_dim)
    # y.normalize()
    varfit = VariationalFitting([identity_ttno]*len(ttns),
                                deepcopy(ttns),
                                y,
                                num_sweeps, 500,
                                SVDParameters(max_bond_dim, 1e-8, 1e-8),
                                "one-site",
                                coeffs,
                                residual_rank=2,
                                dtype=dtype)
    varfit.run()
    # varfit.y.normalize()
    return varfit.y

def add(ttns1: TreeTensorNetworkState,
        ttns2: TreeTensorNetworkState,
        c1: Union[int, float, complex] = 1.0,
        c2: Union[int, float, complex] = 1.0,
        svd_params: SVDParameters | None = None
        ) -> TreeTensorNetworkState:
    """
    Adds two TreeTensorNetworkStates.

    Args:
        ttns1 (TreeTensorNetworkState): The first TreeTensorNetworkState.
        ttns2 (TreeTensorNetworkState): The second TreeTensorNetworkState.
        c1 (Union[int, float, complex]): The scaling factor for the first
            TreeTensorNetworkState.
        c2 (Union[int, float, complex]): The scaling factor for the second
            TreeTensorNetworkState.
        svd_params (SVDParameters | None): The SVD parameters for the truncation.

    Returns:
        TreeTensorNetworkState: The resulting TreeTensorNetworkState.
    """
    if svd_params is None:
        svd_params = SVDParameters()
    computation_order = ttns1.linearise() # Getting a linear list of all identifiers
    rest_ttns = deepcopy(ttns1)
    for node_id in computation_order[:-1]: # The last one is the root node
        _, legs_2 = get_equivalent_legs(ttns1.nodes[node_id], ttns2.nodes[node_id])
        legs_2.append(-1) # appending the physical leg
        t1 = ttns1.tensors[node_id]
        t2 = (ttns2.tensors[node_id]).transpose(legs_2)
        tensor = block_diag_list([(t1), (t2)])
        if np.isnan(tensor).any() or np.isinf(tensor).any():
            # decide whether to drop rows/cols or impute
            tensor = np.nan_to_num(tensor)
        rest_ttns.nodes[node_id]._reset_permutation()  
        rest_ttns.nodes[node_id]._shape = tensor.shape
        assert rest_ttns.nodes[node_id].shape == tensor.shape, "The shape of the node is not the same as the tensor."
        
        rest_ttns.tensors[node_id] = tensor
        assert rest_ttns.tensors[node_id].shape == tensor.shape, "The shape of the node is not the same as the tensor."
    _, legs_2 = get_equivalent_legs(ttns1.nodes[ttns1.root_id], ttns2.nodes[ttns2.root_id])
    legs_2.append(-1)
    assert ttns1.root_id == computation_order[-1], "The root id is not the last in the computation order!"
    t1 = ttns1.tensors[ttns1.root_id]
    t2 = (ttns2.tensors[ttns2.root_id]).transpose(legs_2)

    tensor = block_diag_list([c1*(t1), c2*(t2)])
    if np.isnan(tensor).any() or np.isinf(tensor).any():
        # decide whether to drop rows/cols or impute
        tensor = np.nan_to_num(tensor)
    rest_ttns.nodes[rest_ttns.root_id]._reset_permutation()
    
    rest_ttns.nodes[rest_ttns.root_id]._shape = tensor.shape
    rest_ttns.tensors[rest_ttns.root_id] = tensor
    rest_ttns.orthogonality_center_id = None
   
    return rest_ttns

def block_diag_list(tensors: Sequence[np.ndarray]) -> np.ndarray:
    """
    Block–diagonal sum of a list of (d+1)-D tensors.

    Args
    ----
    tensors : list/tuple of arrays, each with shape
              (n1_i, n2_i, …, nd_i, p)

    Returns
    -------
    out     : single array with shape
              (sum_i n1_i, …, sum_i nd_i, p)
    """
    # ---------- basic checks -----------------------------------------------
    if len(tensors) == 0:
        raise ValueError("The list of tensors must not be empty.")

    ref_ndim = tensors[0].ndim
    ref_channels = tensors[0].shape[-1]

    for idx, t in enumerate(tensors):
        if t.ndim != ref_ndim:
            raise ValueError(f"Tensor {idx} has ndim {t.ndim}, "
                             f"expected {ref_ndim}.")
        if t.shape[-1] != ref_channels:
            raise ValueError(f"Tensor {idx} has channel dimension "
                             f"{t.shape[-1]}, expected {ref_channels}.")

    d = ref_ndim - 1                                      # spatial rank

    # ---------- overall output shape ---------------------------------------
    spatial_sizes = [tuple(t.shape[:-1]) for t in tensors]
    totals = tuple(sum(sz[k] for sz in spatial_sizes)     # per‑axis sum
                   for k in range(d)) + (ref_channels,)

    out = np.zeros(totals, dtype=np.result_type(*tensors))

    # ---------- paste every tensor at its own diagonal offset --------------
    offset = [0]*d                                          # running prefix sum
    for t in tensors:
        # build slice(offset[k], offset[k]+size_k) for every spatial axis
        slc = tuple(slice(offset[k], offset[k]+t.shape[k]) for k in range(d)) \
              + (slice(None),)                              # keep all channels

        out[slc] = t                                        # copy block
        offset = [offset[k] + t.shape[k] for k in range(d)] # advance offsets

    return out

def scale(ttns: TreeTensorNetworkState, c: Union[int, float, complex]) -> TreeTensorNetworkState:
    """
    Scales a TreeTensorNetworkState by a constant factor.
    """
    return ttns.scale(c, inplace=False)
