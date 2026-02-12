"""
This module implements the half density matrix approach for the TTNO application.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from copy import deepcopy

import numpy as np
from numpy._typing import NDArray

from ..ttns import TTNS
from ...core.tree_structure import LinearisationMode
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...contractions.local_contr import LocalContraction, FinalTransposition
from ...util.tensor_splitting import (tensor_qr_decomposition,
                                      contr_truncated_svd_splitting,
                                      SVDParameters,
                                      ContractionMode)
from ...core.node import Node, relative_leg_permutation
from ...util.std_utils import identity_mapping
from .abtract_lc_class import AbstractLinearCombination

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...ttno.ttno_class import TTNO

__all__ = ["half_dm_ttns_ttno_application"]

class HalfDMTTNOApplication(AbstractLinearCombination):
    """
    A class for applying a TTNO to a TTNS using the half density matrix approach.
    """

    def __init__(self,
                 ttnss: list[TTNS] | TTNS,
                 ttnos: list[TTNO | None] | TTNO | None = None,
                 id_trafos_ttns: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                 id_trafos_ttnos: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                 svd_params: SVDParameters | None = None
                 ) -> None:
        """
        Initialises the HalfDMTTNOApplication.

        Args:
            ttnss (list[TTNS] | TTNS): The TTNSs to apply the TTNOs to. If a
                single TTNS is given, it is treated as a list of length one.
            ttnos (list[TTNO | None] | TTNO | None, optional): The TTNOs to apply to
                the TTNSs. If a single TTNO is given, it is treated as a list of
                length one, and applied to all TTNSs in ttnss. If None is given,
                the sum of the TTNSs is computed.
                Defaults to None.
            id_trafos_ttns (list[Callable[[str],str]] | Callable[[str],str], optional):
                The identifier transformation functions for the TTNSs. The i-th
                function transforms the node identifiers of the 0-th TTNS to the node
                identifiers of the i-th TTNS. If a single function is given, it is
                treated as a list of length one, and applied to all TTNSs in ttnss.
                Defaults to identity_mapping.
            id_trafos_ttnos (list[Callable[[str],str]] | Callable[[str],str], optional):
                The identifier transformation functions for the TTNOs. The i-th
                function transforms the node identifiers of the 0-th TTNS to the node
                identifiers of the i-th TTNO. If a single function is given, it is
                treated as a list of length one, and applied to all TTNOs in ttnos.
                Defaults to identity_mapping.
            svd_params (SVDParameters | None, optional): The parameters for the
                decomposition. If None is given, the default parameters are used.
                Defaults to None.
        """
        super().__init__(ttnss,
                         ttnos=ttnos,
                         id_trafos_ttns=id_trafos_ttns,
                         id_trafos_ttnos=id_trafos_ttnos)
        if svd_params is None:
            svd_params = SVDParameters()
        self._svd_params: SVDParameters = svd_params
        self._subtree_caches = [PartialTreeCachDict()
                                for _ in range(self.num_ttns())]

    def build_full_subtree_cache(self,
                                 index: int
                                 ) -> PartialTreeCachDict:
        """
        Build the full subtree cache for the given index.
        """
        ttns = self._ttnss[index]
        if self.ttno_applied(index):
            ttno = self._ttnos[index]
            id_trafo = self.get_id_trafos_ttns_ttno(index)
            cache = build_full_subtree_cache(ttns,
                                             ttno,
                                             id_trafo,
                                             self._svd_params)
        else:
            cache = build_full_subtree_cache_state_only(ttns,
                                                        self._svd_params)
        return cache

    def _node_evaluation(self,
                         node_id: str,
                         r_tensors_caches: list[PartialTreeCachDict]
                         ) -> NDArray:
        """
        Evaluates the new tensor for the given node and index.
        """
        effective_tensors: list[npt.NDArray] = []
        base_node = self.get_base_node(0, node_id)
        for i in range(self.num_ttns()):
            non_base_node_id = self.base_id_to_ttns(node_id, i)
            ket_node_tensor = self._ttnss[i][non_base_node_id]
            parent_id = ket_node_tensor[0].parent
            assert parent_id is not None
            node_tensors = [ket_node_tensor]
            id_trafos = [identity_mapping]
            if self.ttno_applied(i):
                id_trafo_ttns_ttno = self.get_id_trafos_ttns_ttno(i)
                ttno_node_id = id_trafo_ttns_ttno(non_base_node_id)
                op_node_tensor = self.get_ttno_node_tensor(i,
                                                           ttno_node_id)
                node_tensors.append(op_node_tensor)
                id_trafos.append(id_trafo_ttns_ttno)
            r_tensor_cache = r_tensors_caches[i]
            # We can temporarily add the subtree tensor to this cache, as it
            # fits with the contraction structre. In turn we don't need
            # an ignored leg
            subtree_tensor = self._subtree_caches[i].get_entry(parent_id,
                                                    non_base_node_id)
            r_tensor_cache.add_entry(parent_id,
                                     non_base_node_id,
                                     subtree_tensor)
            local_contr = LocalContraction(node_tensors,
                                           r_tensor_cache,
                                           id_trafos=id_trafos)
            effective_tensor = local_contr()
            r_tensor_cache.delete_entry(parent_id,
                                        non_base_node_id)
            # Now we need to bring the effective tensor in the same order as the
            # base TTNS node
            perm = relative_leg_permutation(ket_node_tensor[0],
                                            base_node,
                                            modify_function=self.get_ttns_base_id_map(i))
            effective_tensor = effective_tensor.transpose(perm)
            effective_tensors.append(effective_tensor)
        # Now that we have all the effective tensors, we need to 
        # concatenate them to allow for a larger bond when summing them.
        # They will differ in  the dimension towards the parent.
        effective_tensor = np.concatenate(effective_tensors,
                                          axis=base_node.parent_leg)
        # Nowe we need to decompose to obtain an isometric tensor
        # Luckily we can use the same leg in both cases.
        r_legs = (base_node.parent_leg, )
        q_legs = tuple([leg for leg in range(base_node.nlegs())
                    if leg != r_legs[0]])
        if len(effective_tensors) == 1:
            # In this case no bond enlargement was needed, so we can just qr.
            # Perform the QR-decomposition
            # Note that the R can be thrown away.
            q, _ = tensor_qr_decomposition(effective_tensor,
                                            q_legs,
                                            r_legs)
        else:
            # In this case the bond dimension was enlarged, so we need to truncate it
            # back down.
            q, _ = contr_truncated_svd_splitting(effective_tensor,
                                                         q_legs,
                                                         r_legs,
                                                         svd_params=self._svd_params)
        # Now we need to adjust the leg order of the output
        new_tensor = _adjust_new_ttns_tensor(q, base_node)
        # Now we need to compute the new r-tensors
        for i in range(self.num_ttns()):
            non_base_node_id = self.base_id_to_ttns(node_id, i)
            ket_node_tensor = self.get_ttns_node_tensor(i, non_base_node_id)
            ignored_leg = ket_node_tensor[0].parent
            assert ignored_leg is not None
            node_tensors = [ket_node_tensor]
            id_trafos = [identity_mapping]
            if self.ttno_applied(i):
                id_trafo_ttns_ttno = self.get_id_trafos_ttns_ttno(i)
                ttno_node_id = id_trafo_ttns_ttno(non_base_node_id)
                op_node_tensor = self.get_ttno_node_tensor(i,
                                                           ttno_node_id)
                node_tensors.append(op_node_tensor)
                id_trafos.append(id_trafo_ttns_ttno)
            # We also need to add the new tensor to the contraction
            node_tensors.append((base_node, new_tensor.conj())) # Note the conjugate
            id_trafos.append(self.get_ttns_base_id_map(i))
            local_contr = LocalContraction(node_tensors,
                                           r_tensors_caches[i],
                                           id_trafos=id_trafos,
                                           ignored_leg=ignored_leg)
            r_tensor = local_contr()
            r_tensors_caches[i].add_entry(non_base_node_id,
                                          ignored_leg,
                                          r_tensor)
        return new_tensor

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
    appl_obj = HalfDMTTNOApplication(ttns,
                                     ttnos=ttno,
                                     id_trafos_ttnos=id_trafo,
                                     svd_params=svd_params)
    return appl_obj()

def half_dm_linear_combination(ttnss: list[TTNS],
                          ttnos: list[TTNO | None],
                          id_trafos_ttns: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                          id_trafos_ttnos: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                          svd_params: SVDParameters = SVDParameters()
                            ) -> TTNS:
    """
    Computes a linear combination of the given TTNSs with the given TTNOs via the
    half density matrix based algorithm.

    Args:
        ttnss (list[TTNS]): The TTNSs to combine.
        ttnos (list[TTNO]): The TTNOs to apply to the TTNSs. The i-th TTNO is
            applied to the i-th TTNS.
        id_trafos_ttns (list[Callable[[str],str]] | Callable[[str],str], optional):
            The identifier transformation functions for the TTNSs. The i-th
            function transforms the node identifiers of the 0-th TTNS to the node
            identifiers of the i-th TTNS. If a single function is given, it is
            treated as a list of length one, and applied to all TTNSs in ttnss.
            Defaults to identity_mapping.
        id_trafos_ttnos (list[Callable[[str],str]] | Callable[[str],str], optional):
            The identifier transformation functions for the TTNOs. The i-th
            function transforms the node identifiers of the 0-th TTNS to the node
            identifiers of the i-th TTNO. If a single function is given, it is
            treated as a list of length one, and applied to all TTNOs in ttnos.
            Defaults to identity_mapping.
        svd_params (SVDParameters, optional): The parameters for the
            decomposition. Defaults to SVDParameters().
        
    Returns:
        TTNS: The result of the linear combination.
    """
    appl_obj = HalfDMTTNOApplication(ttnss,
                                     ttnos=ttnos,
                                     id_trafos_ttns=id_trafos_ttns,
                                     id_trafos_ttnos=id_trafos_ttnos,
                                     svd_params=svd_params)
    new_ttns = appl_obj()
    return new_ttns

def half_dm_addition(ttnss: list[TTNS],
                id_trafos_ttns: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                svd_params: SVDParameters = SVDParameters()
                ) -> TTNS:
    """
    Computes the sum of the given TTNSs via the half density matrix based algorithm.

    Args:
        ttnss (list[TTNS]): The TTNSs to sum.
        id_trafos_ttns (list[Callable[[str],str]] | Callable[[str],str], optional):
            The identifier transformation functions for the TTNSs. The i-th
            function transforms the node identifiers of the 0-th TTNS to the node
            identifiers of the i-th TTNS. If a single function is given, it is
            treated as a list of length one, and applied to all TTNSs in ttnss.
            Defaults to identity_mapping.
        svd_params (SVDParameters, optional): The parameters for the
            decomposition. Defaults to SVDParameters().
    
    Returns:
        TTNS: The sum of the TTNSs.
    """
    appl_obj = HalfDMTTNOApplication(ttnss,
                                     id_trafos_ttns=id_trafos_ttns,
                                     svd_params=svd_params)
    new_ttns = appl_obj()
    return new_ttns

def build_full_subtree_cache(ttns: TTNS,
                             ttno: TTNO | None = None,
                             id_trafo: Callable | None = None,
                             svd_params: SVDParameters = SVDParameters()
                             ) -> PartialTreeCachDict:
    """
    Builds a full subtree cache for the half density matrix approach.

    Here the singular value decomposition will determine the bond dimensions.

    Args:
        ttns (TTNS): The TTNS to build the cache for.
        ttno (TTNO | None): The TTNO to build the cache for. If None is given,
            the cache will be built only for the state, i.e. without the operator.
        id_trafo (Callable | None): The identity transformation to use.
        svd_params (SVDParameters): The SVD parameters to use.

    Returns:
        PartialTreeCachDict: The built subtree cache.
    """
    use_ttno = ttno is not None
    cache = PartialTreeCachDict()
    # Get envs upward
    id_trafos = [identity_mapping]
    if use_ttno:
        if id_trafo is None:
            id_trafo = identity_mapping
        id_trafos.append(id_trafo)
    lin_order = ttns.linearise()[:-1] # Exclude root
    for node_id in lin_order:
        ket_node_tensor = ttns[node_id]
        node_tensors = [ket_node_tensor]
        if use_ttno:
            op_node_tensor = ttno[id_trafo(node_id)]
            node_tensors.append(op_node_tensor)
        ignored_leg = ket_node_tensor[0].parent
        assert ignored_leg is not None
        local_contr = LocalContraction(node_tensors,
                                        cache,
                                        id_trafos=id_trafos,
                                        ignored_leg=ignored_leg)
        new_tensor = local_contr.contract_all(transpose_option=FinalTransposition.IGNOREDFIRST)
        # We can assume the first leg of out tensor is the ignored leg
        # TODO: Make use of the node structure to identify the legs
        new_tensor = _truncate_subtree_tensor(new_tensor,
                                              use_ttno,
                                             svd_params)
        # Now the tensor should have the new leg as last leg and the legs
        # pointing to the next node as the first two legs, which is exactly
        # what we want.
        cache.add_entry(node_id, ignored_leg, new_tensor)
    # At this point all upwards envs are in the cache, so everything towards
    # the root.
    # Now we go back down.
    lin_order = ttns.linearise(mode=LinearisationMode.PARENTS_FIRST)
    for node_id in lin_order:
        ket_node_tensor = ttns[node_id]
        node_tensors = [ket_node_tensor]
        if not ket_node_tensor[0].is_leaf():
            if use_ttno:
                op_node_tensor = ttno[id_trafo(node_id)]
                node_tensors.append(op_node_tensor)
            for child_id in ket_node_tensor[0].children:
                local_contr = LocalContraction(node_tensors,
                                                cache,
                                                id_trafos=id_trafos,
                                                ignored_leg=child_id)
                new_tensor = local_contr.contract_all(transpose_option=FinalTransposition.IGNOREDFIRST)
                new_tensor = _truncate_subtree_tensor(new_tensor,
                                                      use_ttno,
                                                      svd_params)
                # Now the tensor should have the new leg as last leg and the legs
                # pointing to the next node as the first two legs, which is exactly
                # what we want.
                cache.add_entry(node_id, child_id, new_tensor)
    return cache

def build_full_subtree_cache_state_only(ttns: TTNS,
                                        svd_params: SVDParameters = SVDParameters()
                                        ) -> PartialTreeCachDict:
    """
    Builds a full subtree cache for the half density matrix approach, but only
    for the state, i.e. without the operator.

    Here the singular value decomposition will determine the bond dimensions.

    Args:
        ttns (TTNS): The TTNS to build the cache for.
        svd_params (SVDParameters): The SVD parameters to use.

    Returns:
        PartialTreeCachDict: The built subtree cache.
    """
    return build_full_subtree_cache(ttns,
                                   ttno=None,
                                   id_trafo=None,
                                   svd_params=svd_params)

def _truncate_subtree_tensor(new_tensor: npt.NDArray,
                             use_ttno: bool,
                              svd_params: SVDParameters
                              ) -> npt.NDArray:
    """
    Truncate the obtained tensor using a truncated SVD on the open legs.

    Args:
        new_tensor (npt.NDArray): The tensor to truncate.
        use_ttno (bool): Whether the TTNO is used. This determines how many
            legs towards the ignored node exist.
        svd_params (SVDParameters): The SVD parameters to use.

    Returns:
        npt.NDArray: The truncated tensor. The new leg is the last leg,
            the legs pointing to the next node are the first two legs.
    """
    num_ignored_legs = 2 if use_ttno else 1
    v_legs = tuple(range(num_ignored_legs))
    u_legs = tuple(range(num_ignored_legs, new_tensor.ndim))
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
