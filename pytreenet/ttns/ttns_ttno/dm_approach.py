"""
Implements the density matrix approach to contract a TTNS and TTNO.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from copy import deepcopy

import numpy as np

from ..ttns import TTNS
from ...core.tree_structure import LinearisationMode
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...contractions.local_contr import LocalContraction
from ...util.tensor_splitting import (SVDParameters,
                                      truncated_tensor_svd)
from ...core.node import Node
from ...util.std_utils import identity_mapping

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...ttno.ttno_class import TTNO

__all__ = ["dm_ttns_ttno_application"]

def dm_ttns_ttno_application(ttns: TTNS,
                             ttno: TTNO,
                             id_trafo: None | Callable = None,
                             svd_params: SVDParameters = SVDParameters()
                             ) -> TTNS:
    """
    Apply a TTNO to a TTNS via the density matrix based algorithm.

    Details for the MPS version may be found under
        https://tensornetwork.org/mps/algorithms/denmat_mpo_mps/.
    
    Args:
        ttns (TTNS): The TTNS to contract.
        ttno (TTNO): The TTNO to contract.
        id_trafo (Callable, optional): A function that transforms the TTNS`s
            node identifiers to the TTNO`s node identifiers. If None, the
            identity function is used. Defaults to None.
        svd_params (SVDParameters): The parameters for the decomposition.
    
    Returns:
        TTNS: The TTNS approximating the TTNS that is yielded by the
            contraction of the TTNS with the TTNO.

    """
    if id_trafo is None:
        id_trafo = identity_mapping
    subtree_cache = build_full_subtree_cache(ttns,
                                             ttno,
                                             id_trafo)
    new_tensors = find_new_ttns_tensors(ttns,
                                        ttno,
                                        subtree_cache,
                                        id_trafo,
                                        svd_params)
    new_ttns = TTNS.from_tensors(ttns, new_tensors)
    return new_ttns

def build_full_subtree_cache(ttns: TTNS,
                             ttno: TTNO,
                             id_trafo: Callable
                             ) -> PartialTreeCachDict:
    """
    Contracts all subtrees of the TTNS-TTNO-TTNO.conj-TTNS.conj contraction.

    This means all subtrees pointing up and down the tree.

    Args:
        ttns (TTNS): The TTNS to contract.
        ttno (TTNO): The TTNO to contract.
        id_trafo (Callable,): A function that transforms the TTNS`s
            node identifiers to the TTNO`s node identifiers.
        
    Returns:
        PartialTreeCachDict: A dictionary containing all partial contractions.
    """
    cache = PartialTreeCachDict()
    lin_order = ttns.linearise()[:-1]  # Exclude root
    # Get envs upwards
    for node_id in lin_order:
        nodes_tensors = _prepare_contraction_nts(node_id,
                                                 ttns,
                                                 ttno,
                                                 id_trafo)
        id_trafos = [identity_mapping, id_trafo, id_trafo, identity_mapping]
        ket_node = nodes_tensors[0][0]
        ignored_leg = ket_node.parent
        assert ignored_leg is not None
        local_contr = LocalContraction(nodes_tensors,
                                       cache,
                                       ignored_leg=ignored_leg,
                                       id_trafos=id_trafos)
        local_contr.contract_into_cache()
    # At this point all upwards envs are in the cache, so everything towards
    # the root.
    # Now we go back down.
    lin_order = ttns.linearise(mode=LinearisationMode.PARENTS_FIRST)
    for node_id in lin_order:
        ket_node = ttns.nodes[node_id]
        if not ket_node.is_leaf():
            # We don't need to compute anything for leaves, as they
            # don't have any downwards envs.
            for child_id in ket_node.children:
                nodes_tensors = _prepare_contraction_nts(node_id,
                                                         ttns,
                                                         ttno,
                                                         id_trafo)
                id_trafos = [identity_mapping, id_trafo, id_trafo, identity_mapping]
                ignored_leg = child_id
                local_contr = LocalContraction(nodes_tensors,
                                               cache,
                                               ignored_leg=ignored_leg,
                                               id_trafos=id_trafos)
                local_contr.contract_into_cache()
    return cache

def find_new_ttns_tensors(ttns: TTNS,
                          ttno: TTNO,
                          subtree_cache: PartialTreeCachDict,
                          id_trafo: Callable,
                          svd_params: SVDParameters
                          ) -> dict[str, npt.NDArray]:
    """
    Finds the new tensors for the TTNS after the contraction with the TTNO.

    Args:
        ttns (TTNS): The TTNS to contract.
        ttno (TTNO): The TTNO to contract.
        subtree_cache (PartialTreeCachDict): A cache containing the
            contractions of all subtrees.
        id_trafo (Callable): A function that transforms the TTNS`s
            node identifiers to the TTNO`s node identifiers.
        svd_params (SVDParameters): The parameters for the decomposition.

    Returns:
        dict[str, npt.NDArray]: A dictionary containing the new tensors for the
            TTNS after the contraction with the TTNO.
    """
    order = ttns.linearise()
    new_tensors = {}
    half_subtree_cache = PartialTreeCachDict()
    for node_id in order[:-1]:
        ket_node_tensor = ttns[node_id]
        op_node_tensor = ttno[id_trafo(node_id)]
        parent_id = ket_node_tensor[0].parent
        assert parent_id is not None
        subtree_tensor = subtree_cache.get_entry(parent_id,
                                                 node_id)
        new_tensor, half_tensor = _node_evaluation(ket_node_tensor,
                                                   op_node_tensor,
                                                   subtree_tensor,
                                                   half_subtree_cache,
                                                   id_trafo,
                                                   svd_params)
        new_tensors[node_id] = new_tensor
        half_subtree_cache.add_entry(node_id,
                                     parent_id,
                                     half_tensor)
    root_id = ttns.root_id
    assert root_id is not None and order[-1] == root_id
    ket_node_tensor = ttns[root_id]
    op_node_tensor = ttno[id_trafo(root_id)]
    new_tensor = _root_evaluation(ket_node_tensor,
                                  op_node_tensor,
                                  half_subtree_cache,
                                  id_trafo)
    new_tensors[root_id] = new_tensor
    return new_tensors

def _prepare_contraction_nts(node_id: str,
                             ttns: TTNS,
                             ttno: TTNO,
                             id_trafo: Callable
                             ) -> list[tuple[Node, npt.NDArray]]:
    """
    Prepares the node-tensor pairs for the local contraction.

    Args:
        node_id (str): The identifier of the node to prepare the node-tensor
            pairs for.
        ttns (TTNS): The TTNS to contract.
        ttno (TTNO): The TTNO to contract.
        id_trafo (Callable): A function that transforms the TTNS`s
            node identifiers to the TTNO`s node identifiers.

    Returns:
        list[tuple[Node, npt.NDArray]]: A list of node-tensor pairs in the
            order [ket, operator, operator_conj, bra].
    """
    ket_nt = ttns[node_id]
    op_nt = ttno[id_trafo(node_id)]
    bra_nt = (ket_nt[0], ket_nt[1].conj())
    op_conj_nt = _prepare_operator_conj_nt(op_nt)
    return [ket_nt, op_nt, op_conj_nt, bra_nt]

def _prepare_operator_conj_nt(op_nt: tuple[Node, npt.NDArray]
                              ) -> tuple[Node, npt.NDArray]:
    """
    Prepares the conjugated operator node-tensor pair for the local
    contraction.
    """
    op_conj_node = deepcopy(op_nt[0])
    op_conj_node.operator_transpose()
    op_conj_tensor = op_conj_node.transpose_tensor(op_nt[1]).conj()
    return (op_conj_node, op_conj_tensor)

def _root_evaluation(ket_node_tensor: tuple[Node, npt.NDArray],
                        op_node_tensor: tuple[Node, npt.NDArray],
                        half_subtree_cache: PartialTreeCachDict,
                        id_trafo: Callable
                     ) -> npt.NDArray:
    """
    Evaluates the new root node.

    Args:
        ket_node_tensor (tuple[Node, npt.NDArray]): The ket node-tensor pair.
        op_node_tensor (tuple[Node, npt.NDArray]): The operator node-tensor
            pair.
        half_subtree_cache (PartialTreeCachDict): A cache containing the
            contractions of all subtrees below this node.
        id_trafo (Callable): A function that transforms the TTNS`s
            node identifiers to the TTNO`s node identifiers.
    
    Returns:
        npt.NDArray: The new root tensor. Corresponds to M_1 in
            https://tensornetwork.org/mps/algorithms/denmat_mpo_mps/.
    """
    # Now we contract a half environment to every neighbour
    local_contr = LocalContraction([ket_node_tensor,
                                    op_node_tensor],
                                   half_subtree_cache,
                                   id_trafos=[identity_mapping, id_trafo])
    # Due to the final transpose in the local contraction. The legs will all
    # be at the right position, i.e. the subtree of a neighbour has one open
    # leg. This open legs will be a the same position as the neighbour.
    return local_contr.contract_all()

def _node_evaluation(ket_node_tensor: tuple[Node, npt.NDArray],
                        op_node_tensor: tuple[Node, npt.NDArray],
                        subtree_tensor: npt.NDArray,
                        half_subtree_cache: PartialTreeCachDict,
                        id_trafo: Callable,
                        svd_params: SVDParameters
                        ) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Evaluates the optimisation of a node.

    Args:
        ket_node_tensor (tuple[Node, npt.NDArray]): The ket node-tensor pair.
        op_node_tensor (tuple[Node, npt.NDArray]): The operator node-tensor
            pair.
        subtree_tensor (npt.NDArray): The subtree tensor to contract with.
        half_subtree_cache (PartialTreeCachDict): A cache containing the
            contractions of all subtrees below this node.
        id_trafo (Callable): A function that transforms the TTNS`s
            node identifiers to the TTNO`s node identifiers.
        svd_params (SVDParameters): The parameters for the decomposition.
    
    Returns:
        tuple[npt.NDArray, npt.NDArray]: The first tensor is the tensor in the
            new TTNS, and the second tensor is the subtree tensor of the lower
            half of the contraction (only ket and op). This corresponds to
            U_i and C_i in https://tensornetwork.org/mps/algorithms/denmat_mpo_mps/.
    """
    # First we perform the lower contraction
    ignored_leg = ket_node_tensor[0].parent
    assert ignored_leg is not None
    local_contr = LocalContraction([ket_node_tensor,
                                    op_node_tensor],
                                   half_subtree_cache,
                                   ignored_leg=ignored_leg,
                                   id_trafos=[identity_mapping, id_trafo])
    lower_contr_tensor = local_contr()
    upper_contr_tensor = lower_contr_tensor.conj()
    # Now we perform the contraction with the subtree tensor
    # The first two open legs of the lower tensor are the legs towards the
    # parent, which is exactly what we want to contract with the subtree tensor
    tensor = np.tensordot(subtree_tensor,
                          lower_contr_tensor,
                          axes=([0,1],[0,1]))
    tensor = np.tensordot(tensor,
                          upper_contr_tensor,
                          axes=([0,1],[1,0])) # Note the order of axes here
    # Now the first half of the legs are the output legs, the second half
    # are the input legs of the reduced density matrix
    half = tensor.ndim // 2
    out_legs = tuple(range(half))
    in_legs = tuple(range(half, tensor.ndim))
    u, _, uh = truncated_tensor_svd(tensor,
                                    out_legs,
                                    in_legs,
                                    svd_params)
    # The order of the order of the legs in uh is (new, neighs, phys) as it
    # is not changed in the svd.
    # The same is true for lower_contr_tensor, but the order of the legs
    # is (parent0, parent1, neighs, phys).
    uh_legs = tuple(range(1,uh.ndim))
    low_legs = tuple(range(2,lower_contr_tensor.ndim))
    low_half_tensor = np.tensordot(lower_contr_tensor,
                                   uh,
                                   axes=(low_legs, uh_legs))
    u = _adjust_new_ttns_tensor(u)
    return u, low_half_tensor

def _adjust_new_ttns_tensor(ttns_tensor: npt.NDArray) -> npt.NDArray:
    """
    Adjusts the new TTNS tensor to be in the correct shape in the TTNS.

    Args:
        ttns_tensor (npt.NDArray): The new TTNS tensor.

    Returns:
        npt.NDArray: The adjusted TTNS tensor.
    """
    # The new tensor has the shape (neighs, phys, new_bond)
    # The new bond is the original parent bond and thus must be moved to the
    # front
    perm = [ttns_tensor.ndim - 1] + list(range(ttns_tensor.ndim - 1))
    return ttns_tensor.transpose(perm)
