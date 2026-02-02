"""
This module implements the density matrix based truncation methods for TTNs.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...contractions.local_contr import LocalContraction
from ...util.tensor_splitting import (SVDParameters,
                                      truncated_tensor_svd)
from ...ttns.ttns import TTNS
from ...contractions.state_state_contraction import build_full_subtree_cache

if TYPE_CHECKING:
    import numpy.typing as npt
    from ..node import Node

def density_matrix_truncation(ttns: TTNS,
                              svd_params: SVDParameters
                              ) -> TTNS:
    """
    Truncates the TTNS using density matrix based truncation.

    Args:
        ttns (TTNS): The tree tensor network state to truncate.
        svd_params (SVDParameters): The SVD parameters for the truncation.

    Returns:
        TTNS: The truncated tree tensor network state.
    """
    # First we need to build the full subtree cache
    cache = build_full_subtree_cache(ttns)
    # Next we find the new tensors for the TTNS
    new_tensors = _find_new_ttns_tensors(ttns,
                                         cache,
                                         svd_params)
    # Finally we build the new TTNS
    new_ttns = TTNS.from_tensors(ttns, new_tensors)
    return new_ttns

def _find_new_ttns_tensors(ttns: TTNS,
                           cache: PartialTreeCachDict,
                           svd_params: SVDParameters
                           ) -> dict[str, npt.NDArray]:
    """
    Finds the new tensors for the TTNS after density matrix truncation.

    Args:
        ttns (TTNS): The tree tensor network state.
        cache (PartialTreeCachDict): The contracted subtrees.
        svd_params (SVDParameters): The SVD parameters for the truncation.
    
    Returns:
        dict[str, npt.NDArray]: The new tensors for the TTNS.
    """
    new_tensors = {}
    order = ttns.linearise()
    r_matrices = PartialTreeCachDict()
    for node_id in order[:-1]: # Root needs special treatment
        node_tensor = ttns[node_id]
        parent_id = node_tensor[0].parent
        assert parent_id is not None
        subtree_tensor = cache.get_entry(parent_id,
                                         node_id)
        new_tensor, r_matrix = _node_evaluation(node_tensor,
                                                subtree_tensor,
                                                r_matrices,
                                                svd_params)
        new_tensors[node_id] = new_tensor
        r_matrices.add_entry(node_id,
                             parent_id,
                             r_matrix)
    # Now do the root
    root_id = ttns.root_id
    assert root_id == order[-1]
    assert root_id is not None
    new_root_tensor = _root_evaluation(ttns[root_id],
                                       r_matrices)
    new_tensors[root_id] = new_root_tensor
    return new_tensors

def _root_evaluation(root_node_tensor: tuple[Node, npt.NDArray],
                     r_matrices: PartialTreeCachDict
                     ) -> npt.NDArray:
    """
    Evaluates the new tensor for the root node after density matrix truncation.

    Args:
        root_node_tensor (tuple[Node, npt.NDArray]): The root node and its tensor.
        r_matrices (PartialTreeCachDict): The R matrices.

    Returns:
        npt.NDArray: The new tensor for the root node.
    """
    # We need to contract all r_matrices into the root tensor
    local_contr = LocalContraction([root_node_tensor],
                                   r_matrices)
    return local_contr()

def _node_evaluation(node_tensor: tuple[Node, npt.NDArray],
                     subtree_tensor: npt.NDArray,
                     r_matrices: PartialTreeCachDict,
                     svd_params: SVDParameters
                     ) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Evaluates the new tensor for a node after density matrix truncation.

    Args:
        node_tensor (tuple[Node, npt.NDArray]): The node and its tensor.
        subtree_tensor (npt.NDArray): The contracted subtree tensor.
        r_matrices (PartialTreeCachDict): The R matrices.
        svd_params (SVDParameters): The SVD parameters for the truncation.
    
    Returns:
        tuple[npt.NDArray, npt.NDArray]: The new tensor and the R matrix.
    """
    # First we contract all the r_matrices into the subtree tensor
    ignored_leg = node_tensor[0].parent
    assert ignored_leg is not None
    local_contr = LocalContraction([node_tensor],
                                   r_matrices,
                                   ignored_leg=ignored_leg)
    lower_contr_tensor = local_contr()
    upper_contr_tensor = lower_contr_tensor.conj()
    # Now we form the density matrix by contracting with the subtree tensor
    tensor = np.tensordot(subtree_tensor, lower_contr_tensor,
                          axes=([0],[0]))
    tensor = np.tensordot(tensor, upper_contr_tensor,
                          axes=([0],[0]))
    # Note the order of axes here
    # Now the first half of the legs are the output legs, the second half
    # are the input legs of the reduced density matrix
    half = tensor.ndim // 2
    out_legs = tuple(range(half))
    in_legs = tuple(range(half, tensor.ndim))
    u, _, uh = truncated_tensor_svd(tensor,
                                    out_legs,
                                    in_legs,
                                    svd_params)
    # Now u has legs order (neighs, phys, new_bond) and
    # uh has (new_bond, neighs, phys)
    # We perform the final contraction to get the R matrix
    uh_legs = tuple(range(1, uh.ndim)) # Skip new bond leg
    low_tens_legs = tuple(range(1, lower_contr_tensor.ndim))
    r_matrix = np.tensordot(lower_contr_tensor,
                            uh,
                            axes=(low_tens_legs, uh_legs))
    # As a final step, we need to move the new leg of u to the correct position
    # to have the same leg order as the original tensor
    last_leg = node_tensor[1].ndim - 1
    perm = [last_leg] + list(range(last_leg))
    new_tensor = np.transpose(u, axes=perm)
    return new_tensor, r_matrix
