"""
Implements the succesive randomized compression algorithm for TTN.

This was introduced for MPS in https://arxiv.org/abs/2504.06475
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import numpy as np

from ...random.random_matrices import crandn
from ..ttns import TTNS
from ...core.tree_structure import LinearisationMode
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...contractions.local_contr import LocalContraction
from ...util.tensor_splitting import tensor_qr_decomposition
from ...operators.common_operators import copy_tensor
from ...core.node import Node
from ...util.std_utils import identity_mapping

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...ttno.ttno_class import TTNO

__all__ = ["src_ttns_ttno_application"]

def src_ttns_ttno_application(ttns: TTNS,
                              ttno: TTNO,
                              desired_dimension: int,
                              id_trafo: Callable | None = None,
                              seed: int | None = None
                              ) -> TTNS:
    """
    Applies the TTNO to the TTNS using the succesive randomized compression
    algorithm.

    Args:
        ttns (TTNS): The TTNS to apply the TTNO to.
        ttno (TTNO): The TTNO to apply.
        desired_dimension (int): The desired dimension of the resulting TTNS.
        id_trafo (Callable): A function that transforms the TTNS`s node
            identifiers to the TTNO`s node identifiers. If None, the identity
            mapping is used. Defaults to None.
        seed (int | None, optional): Seed for the random number generator.
            Defaults to None.

    Returns:
        TTNS: The resulting TTNS after applying the TTNO.
    """
    if id_trafo is None:
        id_trafo = identity_mapping
    random_ttns = generate_random_matrices(ttns,
                                           desired_dimension,
                                           seed=seed)
    subtree_cache = build_full_subtree_cache(ttns,
                                             ttno,
                                             random_ttns,
                                             id_trafo)
    new_tensors = find_new_tensors(ttns,
                                   ttno,
                                   subtree_cache,
                                   id_trafo)
    new_ttns = TTNS.from_tensors(ttns, new_tensors)
    return new_ttns

def generate_random_matrices(ttns: TTNS,
                             desired_dimension: int,
                             seed: int | None = None
                             ) -> TTNS:
    """
    Generates a TTNS with random tensors of compatible dimensions.

    Args:
        ttns (TTNS): The TTNS to base the random TTNS on.
        desired_dimension (int): The desired dimension of the random tensors.

    Returns:
        TTNS: A TTNS representing the Kati-Rao-Product of the input tree
            structure with random tensors.
    """
    tensors = {}
    for node_id, node in ttns.nodes.items():
        input_dims = node.open_dimensions()
        desired_shape = [desired_dimension] + input_dims
        rand_tensor = crandn(tuple(desired_shape), seed=seed)
        copy_t = copy_tensor(node.nlegs(), desired_dimension)
        rand_tensor = np.tensordot(copy_t,
                                   rand_tensor,
                                   axes=(0,0))
        tensors[node_id] = rand_tensor
        seed = None if seed is None else seed + 1
    rand_ttns = TTNS.from_tensors(ttns, tensors)
    return rand_ttns

def build_full_subtree_cache(ttns: TTNS,
                             ttno: TTNO,
                             random_ttns: TTNS,
                             id_trafo: Callable
                             ) -> PartialTreeCachDict:
    """
    Builds a full subtree cache for the given TTNS and TTNO with random
    tensors.

    Args:
        ttns (TTNS): The TTNS to build the cache for.
        ttno (TTNO): The TTNO to use for the contractions.
        random_ttns (TTNS): The TTNS with random tensors.
        id_trafo (Callable): A function to transform node identifiers.

    Returns:
        PartialTreeCachDict: The full subtree cache.
    """
    cache = PartialTreeCachDict()
    # Get envs upward
    lin_order = ttns.linearise()[:-1] # Exclude root
    id_trafos = [identity_mapping, id_trafo, identity_mapping]
    for node_id in lin_order:
        ket_node_tensor = ttns[node_id]
        bra_node_tensor = random_ttns[node_id]
        op_node_tensor = ttno[id_trafo(node_id)]
        ignored_leg = ket_node_tensor[0].parent
        assert ignored_leg is not None
        local_contr = LocalContraction([ket_node_tensor,
                                        op_node_tensor,
                                        bra_node_tensor],
                                        cache,
                                        ignored_leg=ignored_leg,
                                        id_trafos=id_trafos)
        local_contr.contract_into_cache()
    # At this point all upwards envs are in the cache, so everything towards
    # the root.
    # Now we go back down.
    lin_order = ttns.linearise(mode=LinearisationMode.PARENTS_FIRST)
    for node_id in lin_order:
        ket_node_tensor = ttns[node_id]
        if not ket_node_tensor[0].is_leaf():
            bra_node_tensor = random_ttns[node_id]
            op_node_tensor = ttno[id_trafo(node_id)]
            for child_id in ket_node_tensor[0].children:
                local_contr = LocalContraction([ket_node_tensor,
                                                op_node_tensor,
                                                bra_node_tensor],
                                                cache,
                                                ignored_leg=child_id,
                                                id_trafos=id_trafos)
                local_contr.contract_into_cache()
    return cache

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
