"""
This module implements the half density matrix approach for the TTNO application.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..ttns import TTNS
from ...core.tree_structure import LinearisationMode
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...contractions.local_contr import LocalContraction, FinalTransposition
from ...util.tensor_splitting import (tensor_qr_decomposition,
                                      contr_truncated_svd_splitting,
                                      SVDParameters,
                                      ContractionMode)
from ...core.node import Node
from ...util.std_utils import identity_mapping

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...ttno.ttno_class import TTNO

__all__ = ["half_dm_ttns_ttno_application"]

def half_dm_ttns_ttno_application(ttns: TTNS,
                                  ttno: TTNO,
                                  id_trafo: Callable = identity_mapping,
                                  svd_params: SVDParameters | None = None
                                  ) -> TTNS:
    """
    Applies a TTNO to a TTNS using the half density matrix approach.

    Args:
        ttns (TTNS): The TTNS to apply the operator to.
        ttno (TTNO): The TTNO to apply.
        id_trafo (Callable, optional): A function that transforms the TTNS`s
            node identifiers to the TTNO`s node identifiers. Defaults to
            identity_mapping.
        svd_params (SVDParameters | None, optional): The parameters for
            the SVD truncation. If None, default parameters will be used.
            Defaults to None.
    
    Returns:
        TTNS: The resulting TTNS after applying the operator.
    """
    if svd_params is None:
        svd_params = SVDParameters()
    # First build the full subtree cache
    subtree_cache = build_full_subtree_cache(ttns,
                                             ttno,
                                             id_trafo,
                                             svd_params)
    # Now find the new tensors
    new_tensors = find_new_tensors(ttns,
                                   ttno,
                                   subtree_cache,
                                   id_trafo)
    new_ttns = TTNS.from_tensors(ttns, new_tensors)
    return new_ttns                 

def build_full_subtree_cache(ttns: TTNS,
                             ttno: TTNO,
                             id_trafo: Callable,
                             svd_params: SVDParameters
                             ) -> PartialTreeCachDict:
    """
    Builds a full subtree cache for the half density matrix approach.

    Here the singular value decomposition will determine the bond dimensions.

    Args:
        ttns (TTNS): The TTNS to build the cache for.
        ttno (TTNO): The TTNO to build the cache for.
        id_trafo (Callable): The identity transformation to use.
        svd_params (SVDParameters): The SVD parameters to use.

    Returns:
        PartialTreeCachDict: The built subtree cache.
    """
    cache = PartialTreeCachDict()
    # Get envs upward
    lin_order = ttns.linearise()[:-1] # Exclude root
    id_trafos = [identity_mapping, id_trafo]
    for node_id in lin_order:
        ket_node_tensor = ttns[node_id]
        op_node_tensor = ttno[id_trafo(node_id)]
        ignored_leg = ket_node_tensor[0].parent
        assert ignored_leg is not None
        local_contr = LocalContraction([ket_node_tensor,
                                        op_node_tensor],
                                        cache,
                                        id_trafos=id_trafos,
                                        ignored_leg=ignored_leg)
        new_tensor = local_contr.contract_all(transpose_option=FinalTransposition.IGNOREDFIRST)
        # We can assume the first leg of out tensor is the ignored leg
        # TODO: Make use of the node structure to identify the legs
        new_tensor = _truncate_subtree_tensor(new_tensor,
                                             svd_params)
        # Now the tensor should have the new leg as last leg and the legs
        # pointing to the next node as the first two legs, which is exactly
        # what we want.
        cache.add_entry(node_id, ignored_leg, new_tensor)
    # At this point all upwards envs are in the cache, so everything towards
    # the root.
    # Now we go back down.
    lin_order = ttns.linearise(mode=LinearisationMode.PARENTS_FIRST)
    id_trafos = [identity_mapping, id_trafo]
    for node_id in lin_order:
        ket_node_tensor = ttns[node_id]
        if not ket_node_tensor[0].is_leaf():
            op_node_tensor = ttno[id_trafo(node_id)]
            for child_id in ket_node_tensor[0].children:

                local_contr = LocalContraction([ket_node_tensor,
                                                op_node_tensor],
                                                cache,
                                                id_trafos=id_trafos,
                                                ignored_leg=child_id)
                new_tensor = local_contr.contract_all(transpose_option=FinalTransposition.IGNOREDFIRST)
                new_tensor = _truncate_subtree_tensor(new_tensor,
                                                     svd_params)
                # Now the tensor should have the new leg as last leg and the legs
                # pointing to the next node as the first two legs, which is exactly
                # what we want.
                cache.add_entry(node_id, child_id, new_tensor)
    return cache

def _truncate_subtree_tensor(new_tensor: npt.NDArray,
                              svd_params: SVDParameters
                              ) -> npt.NDArray:
    """
    Truncate the obtained tensor using a truncated SVD on the open legs.

    Args:
        new_tensor (npt.NDArray): The tensor to truncate.
        svd_params (SVDParameters): The SVD parameters to use.

    Returns:
        npt.NDArray: The truncated tensor. The new leg is the last leg,
            the legs pointing to the next node are the first two legs.
    """
    v_legs = (0,1)
    u_legs = tuple(range(2, new_tensor.ndim))
    truncated, _ = contr_truncated_svd_splitting(new_tensor,
                                                v_legs,
                                                u_legs,
                                                svd_params=svd_params,
                                                contr_mode=ContractionMode.UCONTR)
    # Now the tensor should have the new leg as last leg and the legs
    # pointing to the next node as the first two legs, which is exactly
    # what we want.
    return truncated

def find_new_tensors(ttns: TTNS,
                     ttno: TTNO,
                     subtree_cache: PartialTreeCachDict,
                     id_trafo: Callable
                     ) -> dict[str, np.ndarray]:
    """
    Finds new tensors for the TTNS using the subtree cache.

    Args:
        ttns (TTNS): The TTNS to find new tensors for.
        ttno (TTNO): The TTNO to use for the contractions.
        subtree_cache (PartialTreeCachDict): The subtree cache.
        id_trafo (Callable): A function to transform node identifiers.
    
    Returns:
        dict[str, np.ndarray]: A dictionary mapping node identifiers to
            their new tensors.
    """
    new_tensors = {}
    order = ttns.linearise()
    half_subtree_cache = PartialTreeCachDict()
    for node_id in order[:-1]: # Exclude root
        ket_node_tensor = ttns[node_id]
        op_node_tensor = ttno[id_trafo(node_id)]
        parent_id = ket_node_tensor[0].parent
        assert parent_id is not None
        parent_subtree = subtree_cache.get_entry(parent_id,
                                                 node_id)
        new_tensor, half_subtree = _node_evaluation(ket_node_tensor,
                                                    op_node_tensor,
                                                    parent_subtree,
                                                    half_subtree_cache,
                                                    id_trafo)
        new_tensors[node_id] = new_tensor
        half_subtree_cache.add_entry(node_id,
                                     parent_id,
                                     half_subtree)
    # Finally the root
    root_id = ttns.root_id
    assert root_id is not None and root_id == order[-1]
    ket_node_tensor = ttns[root_id]
    op_node_tensor = ttno[id_trafo(root_id)]
    new_tensor = _root_evaluation(ket_node_tensor,
                                  op_node_tensor,
                                  half_subtree_cache,
                                  id_trafo)
    new_tensors[root_id] = new_tensor
    return new_tensors

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
        npt.NDArray: The new root tensor. Corresponds to eta^(1) in the
            paper.
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

def _node_evaluation(ket_node_tensor: tuple[Node,npt.NDArray],
                     op_node_tensor: tuple[Node,npt.NDArray],
                     subtree_tensor: npt.NDArray,
                     half_subtree_cache: PartialTreeCachDict,
                     id_trafo: Callable
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
            eta^(i) and the pink tensor in the paper.
    """
    ket_node = ket_node_tensor[0]
    parent_id = ket_node.parent
    assert parent_id is not None
    half_subtree_cache.add_entry(parent_id,
                                    ket_node.identifier,
                                    subtree_tensor)
    loc_contr = LocalContraction([ket_node_tensor,
                                    op_node_tensor],
                                    half_subtree_cache,
                                    id_trafos=[identity_mapping, id_trafo])
    effective_tensor = loc_contr()
    # TODO: We could in principle delete some things now.
    r_legs = (ket_node_tensor[0].parent_leg, )
    q_legs = tuple([leg for leg in range(ket_node_tensor[0].nlegs())
                if leg != r_legs[0]])
    q, _ = tensor_qr_decomposition(effective_tensor,
                                    q_legs,
                                    r_legs)
    new_tensor = _adjust_new_ttns_tensor(q, ket_node_tensor[0])
    ignored_leg = ket_node_tensor[0].parent
    assert ignored_leg is not None
    nodes_tensors = [ket_node_tensor,
                        op_node_tensor,
                        (ket_node_tensor[0], new_tensor.conj())] # Note the conjugate
    id_trafos_2 = [identity_mapping, id_trafo, identity_mapping]
    loc_contr = LocalContraction(nodes_tensors,
                                half_subtree_cache,
                                ignored_leg=ignored_leg,
                                id_trafos=id_trafos_2)
    half_subtree = loc_contr()
    return new_tensor, half_subtree

def _adjust_new_ttns_tensor(ttns_tensor: npt.NDArray,
                            ttns_node: Node
                            ) -> npt.NDArray:
    """
    Adjusts the new TTNS tensor to be in the correct shape in the TTNS.

    Args:
        ttns_tensor (npt.NDArray): The new TTNS tensor.
        ttns_node (Node): The TTNS node corresponding to the tensor.

    Returns:
        npt.NDArray: The adjusted TTNS tensor.
    """
    # The new tensor has the shape (neighs, phys, new_bond)
    # The new bond is the original parent bond and must be moved to the
    # correct position.
    new_leg = ttns_tensor.ndim - 1
    parent_leg = ttns_node.parent_leg
    if parent_leg == new_leg:
        return ttns_tensor
    perm = list(range(new_leg))
    perm.insert(parent_leg, new_leg)
    return ttns_tensor.transpose(perm)
