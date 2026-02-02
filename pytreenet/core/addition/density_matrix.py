"""
Implements the density matrix addition for TTNs.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..ttn import TreeTensorNetwork
from ...contractions.state_state_contraction import build_full_subtree_cache
from ...util.tensor_splitting import SVDParameters, truncated_tensor_svd
from ..node import relative_leg_permutation
from ...contractions.local_contr import LocalContraction

if TYPE_CHECKING:
    from ...contractions.tree_cach_dict import PartialTreeCachDict
    from ..node import Node
    import numpy.typing as npt

def density_matrix_addition(ttns: list[TreeTensorNetwork],
                            truncation_parameters: SVDParameters
                            ) -> TreeTensorNetwork:
    """
    Adds multiple TTNs using density matrix based addition.

    Args:
        ttns (list[TreeTensorNetwork]): The list of TTNs to add.
        truncation_parameters (SVDParameters): The SVD parameters for
            truncation.

    Returns:
        TreeTensorNetwork: The resulting TTN after addition.
    """
    if not ttns:
        errstr = "The list of TTNs to add cannot be empty!"
        raise ValueError(errstr)
    if len(ttns) == 1:
        return ttns[0]
    caches: list[PartialTreeCachDict] = []
    for ttn in ttns:
        cache = build_full_subtree_cache(ttn)
        caches.append(cache)
    ref_ttn = ttns[0]
    new_tensors = _find_new_tensors(ttns,
                                    caches,
                                    truncation_parameters)
    new_ttn = TreeTensorNetwork.from_tensors(ref_ttn,
                                             new_tensors)
    return new_ttn

def _find_new_tensors(ttns: list[TreeTensorNetwork],
                      caches: list[PartialTreeCachDict],
                      svd_params: SVDParameters
                      ) -> dict[str, np.ndarray]:
    """
    Finds the new tensors for the added TTN after density matrix addition.

    Args:
        ttns (list[TreeTensorNetwork]): The list of TTNs.
        caches (list[PartialTreeCachDict]): The list of caches for the TTNs.
        svd_params (SVDParameters): The SVD parameters for truncation.

    Returns:
        dict[str, np.ndarray]: The new tensors for the added TTN.
    """
    new_tensors = {}
    ref_ttn = ttns[0]
    order = ref_ttn.linearise()
    r_matrices = [PartialTreeCachDict() for _ in ttns]
    for node_id in order[:-1]: # Root needs special treatment
        nodes_tensors = [ttn[node_id] for ttn in ttns]
        parent_id = nodes_tensors[0][0].parent
        assert parent_id is not None
        subtree_tensors = [cache.get_entry(parent_id, node_id)
                           for cache in caches]
        new_tensor, r_matrix = _node_evaluation(nodes_tensors,
                                                subtree_tensors,
                                                r_matrices,
                                                svd_params)
        for r_mats, r_mat in zip(r_matrices, r_matrix):
            r_mats.add_entry(node_id,
                             parent_id,
                             r_mat)
        new_tensors[node_id] = new_tensor
    # Now do the root
    root_id = ref_ttn.root_id
    assert root_id == order[-1]
    assert root_id is not None
    nodes_tensors = [ttn[root_id] for ttn in ttns]
    new_root_tensor = _root_evaluation(nodes_tensors,
                                       r_matrices)
    new_tensors[root_id] = new_root_tensor
    return new_tensors

def _root_evaluation(nodes_tensors: list[tuple[Node,npt.NDArray]],
                     r_matrices: list[PartialTreeCachDict]
                     ) -> npt.NDArray:
    """
    Evaluates the new tensor for the root node after density matrix addition.

    Args:
        nodes_tensors (list[tuple[Node,npt.NDArray]]): The tensors of the root node from each TTN.
        r_matrices (list[PartialTreeCachDict]): The R matrices from previous nodes.
    
    Returns:
        npt.NDArray: The new tensor for the root node.
    """
    ref_node = nodes_tensors[0][0]
    dm_tensors = []
    for node_tensor, r_mats in zip(nodes_tensors, r_matrices):
        node, tensor = node_tensor
        perm = relative_leg_permutation(node,
                                        ref_node)
        tensor = np.transpose(tensor, axes=perm)
        local_contr = LocalContraction([(node, tensor)],
                                       r_mats)
        dm_tensor = local_contr()
        dm_tensors.append(dm_tensor)
    total_dm_tensor = np.sum(dm_tensors, axis=0)
    return total_dm_tensor

def _node_evaluation(nodes_tensors: list[tuple[Node,npt.NDArray]],
                     subtree_tensors: list[npt.NDArray],
                     r_matrices: list[PartialTreeCachDict],
                     svd_params: SVDParameters
                     ) -> tuple[npt.NDArray, list[npt.NDArray]]:
    """
    Evaluates a single node during density matrix addition.

    Args:
        nodes_tensors (list[tuple[Node,npt.NDArray]]): The tensors of the node from each TTN.
        subtree_tensors (list[npt.NDArray]): The contracted subtree tensors.
        r_matrices (list[PartialTreeCachDict]): The R matrices from previous nodes.
        svd_params (SVDParameters): The SVD parameters for truncation.

    Returns:
        tuple[npt.NDArray, list[npt.NDArray]]: The new tensor for the node and the R matrix for each TTN.
    """
    ref_node, ref_tensor = nodes_tensors[0]
    parent_id = ref_node.parent
    assert parent_id is not None
    dm_tensors = []
    # Cache result for speed up
    lower_contr_tensors = []
    for node_tensor, subtree_tensor, r_mats in zip(nodes_tensors,
                                               subtree_tensors,
                                               r_matrices):
        node, tensor = node_tensor
        perm = relative_leg_permutation(node,
                                        ref_node)
        tensor = np.transpose(tensor, axes=perm)
        local_contr = LocalContraction([(node, tensor)],
                                       r_mats,
                                       ignored_leg=parent_id)
        lower_contr_tensor = local_contr()
        upper_contr_tensor = lower_contr_tensor.conj()
        lower_contr_tensors.append(lower_contr_tensor)
        # Now we form the density matrix by contracting with the subtree tensor
        dm_tensor = np.tensordot(subtree_tensor, lower_contr_tensor,
                                axes=([0],[0]))
        dm_tensor = np.tensordot(dm_tensor, upper_contr_tensor,
                                axes=([0],[0]))
        dm_tensors.append(dm_tensor)
    # Now we sum up the density matrices
    total_dm_tensor = np.sum(dm_tensors, axis=0)
    # Note the order of axes here
    # Now the first half of the legs are the output legs, the second half
    # are the input legs of the reduced density matrix
    half = total_dm_tensor.ndim // 2
    out_legs = tuple(range(half))
    in_legs = tuple(range(half, total_dm_tensor.ndim))
    u, _, uh = truncated_tensor_svd(total_dm_tensor,
                                    out_legs,
                                    in_legs,
                                    svd_params)
    # We need to move the new leg of u to the correct position
    # to have the same leg order as the original tensor
    last_leg = ref_tensor.ndim - 1
    perm = [last_leg] + list(range(last_leg))
    new_tensor = np.transpose(u, axes=perm)
    # As a final step, we need to compute the R matrices for each TTN
    new_r_matrices = []
    for lower_contr_tensor in lower_contr_tensors:
        uh_legs = tuple(range(1, uh.ndim)) # Skip new bond leg
        low_tens_legs = tuple(range(1, lower_contr_tensor.ndim))
        r_matrix = np.tensordot(lower_contr_tensor,
                                uh,
                                axes=(low_tens_legs, uh_legs))
        new_r_matrices.append(r_matrix)
    return new_tensor, new_r_matrices
