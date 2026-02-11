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
from ...contractions.state_state_contraction import build_full_subtree_cache as build_full_subtree_cache_state_only
from ...util.tensor_splitting import (SVDParameters,
                                      truncated_tensor_svd)
from ...core.node import (Node,
                          relative_leg_permutation)
from ...util.std_utils import identity_mapping
from .abtract_lc_class import AbstractLinearCombination

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...ttno.ttno_class import TTNO

__all__ = ["dm_ttns_ttno_application"]

class DMTTNOApplication(AbstractLinearCombination):
    """
    A class to apply TTNOs to TTNSs via the density matrix based algorithm.
    """

    def __init__(self,
                 ttnss: list[TTNS] | TTNS,
                 ttnos: list[TTNO | None] | TTNO | None = None,
                 id_trafos_ttns: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                 id_trafos_ttnos: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                 svd_params: SVDParameters | None = None
                 ) -> None:
        """
        Initialises the DMTTNOApplication.

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

    def __call__(self) -> TTNS:
        """
        Computes the result TTNS.

        Returns:
            TTNS: The result TTNS.
        """
        self._subtree_caches = self.build_full_subtree_caches()
        new_tensors = self.find_new_ttns_tensors()
        new_ttns = TTNS.from_tensors(self.get_base_ttns(), new_tensors)
        return new_ttns

    def build_full_subtree_caches(self) -> list[PartialTreeCachDict]:
        """
        Builds the full subtree caches for all TTNS-TTNO pairs.

        Returns:
            list[PartialTreeCachDict]: A list of subtree caches for all
                TTNS-TTNO pairs.
        """
        caches = [self.build_full_subtree_cache(i)
                  for i in range(self.num_ttns())]
        return caches

    def build_full_subtree_cache(self,
                                 index: int
                                 ) -> PartialTreeCachDict:
        """
        Builds the full subtree cache for the TTNS-TTNO pair at the given index.

        Args:
            index (int): The index of the TTNS-TTNO pair.
        
        Returns:
            PartialTreeCachDict: The subtree cache for the TTNS-TTNO pair.
        """
        ttns = self._ttnss[index]
        ttno = self._ttnos[index]
        if ttno is None:
            cache = build_full_subtree_cache_state_only(ttns)
        else:
            id_trafo = self.get_id_trafos_ttns_ttno(index)
            cache = build_full_subtree_cache(ttns,
                                            ttno,
                                            id_trafo)
        return cache

    def find_new_ttns_tensors(self) -> dict[str, npt.NDArray]:
        """
        Finds the new tensors for the result TTNS.

        Returns:
            dict[str, npt.NDArray]: A dictionary containing the new tensors for the
                result TTNS.
        """
        new_tensors = {}
        order = self.get_base_ttns().linearise()
        r_tensors_caches = [PartialTreeCachDict()
                            for _ in range(self.num_ttns())]
        for node_id in order[:-1]: # Root is handled separately
            new_tensor = self._node_evaluation(node_id,
                                               r_tensors_caches)
            new_tensors[node_id] = new_tensor
        root_tensor = self._root_evaluation(r_tensors_caches)
        new_tensors[order[-1]] = root_tensor
        return new_tensors

    def _root_evaluation(self,
                         r_tensor_caches: list[PartialTreeCachDict]
                         ) -> npt.NDArray:
        """
        Evaluates the new root tensor.

        Args:
            r_tensor_caches (list[PartialTreeCachDict]): A list of caches for
                the contractions of all subtrees below the root for all
                TTNS-TTNO pairs.

        Returns:
            npt.NDArray: The new root tensor.
        """
        # The root is a special case, as we don't have a parent leg to contract
        # with, so we just contract the whole subtree tensor with the local
        # contraction of the root node.
        local_contr_tensors: list[npt.NDArray] = []
        root_id = self.get_base_ttns().root_id
        assert root_id is not None
        for i in range(self.num_ttns()):
            non_base_root_id = self.base_id_to_ttns(root_id, i)
            ket_node_tensor = self.get_ttns_node_tensor(i, non_base_root_id)
            node_tensors = [ket_node_tensor]
            id_trafos = [identity_mapping]
            if self.ttno_applied(i):
                op_node_tensor = self.get_ttno_node_tensor(i, non_base_root_id)
                node_tensors.append(op_node_tensor)
                ttns_ttno_trafo = self.get_id_trafos_ttns_ttno(i)
                id_trafos.append(ttns_ttno_trafo)
            r_tensor_cache = r_tensor_caches[i]
            local_contr = LocalContraction(node_tensors,
                                            r_tensor_cache,
                                            id_trafos=id_trafos)
            local_contr_tensor = local_contr.contract_all()
            local_contr_tensors.append(local_contr_tensor)
        new_tensor = np.sum(local_contr_tensors, axis=0)
        return new_tensor

    def _node_evaluation(self,
                         node_id: str,
                         r_tensor_caches: list[PartialTreeCachDict]
                         ) -> npt.NDArray:
        """
        Evaluates the new tensor for the given node and index.

        Args:
            node_id (str): The identifier of the node to evaluate. This is not
                the identifier in the base TTNS!
            r_tensor_caches (list[PartialTreeCachDict]): A list of caches for
                the contractions of all subtrees below this node for all
                TTNS-TTNO pairs. Note, that these will be modified by adding
                the new contractions for this node.

        Returns:
            npt.NDArray: The tensor in the new TTNS.
        """
        red_dms: list[npt.NDArray] = []
        # It is efficient to reuse the contraction result
        lower_contr_tensors: list[npt.NDArray] = []
        for i in range(self.num_ttns()):
            non_base_node_id = self.base_id_to_ttns(node_id, i)
            ket_node_tensor = self.get_ttns_node_tensor(i, non_base_node_id)
            parent_id = ket_node_tensor[0].parent
            assert parent_id is not None
            subtree_tensor = self._subtree_caches[i].get_entry(parent_id,
                                                                non_base_node_id)
            red_dm, lower_contr_tensor = self.get_reduced_density_matrix(i,
                                                                         non_base_node_id,
                                                                         subtree_tensor,
                                                                         r_tensor_caches[i])
            red_dms.append(red_dm)
            lower_contr_tensors.append(lower_contr_tensor)
        # Now we can find the combined density matrix
        combined_red_dm = np.sum(red_dms, axis=0)
        # Remember, the first half of the legs are the output legs,
        # the second half are the input legs. We want to split in the middle
        half = combined_red_dm.ndim // 2
        out_legs = tuple(range(half))
        in_legs = tuple(range(half, combined_red_dm.ndim))
        u, _, uh = truncated_tensor_svd(combined_red_dm,
                                        out_legs,
                                        in_legs,
                                        self._svd_params)
        base_node = self.get_base_node(0, node_id)
        for i in range(self.num_ttns()):
            # The order of the order of the legs in uh is (new, neighs, phys) as it
            # is not changed in the svd.
            # The same is true for lower_contr_tensor, but the order of the legs
            # is (parent0, parent1, neighs, phys).
            # We now need to find the new r_tensors
            uh_legs = tuple(range(1,uh.ndim))
            non_base_node = self.get_ttns_node_tensor(i, non_base_node_id)[0]
            low_legs = relative_leg_permutation(non_base_node,
                                                base_node,
                                                modify_function=self.get_ttns_base_id_map(i),
                                                include_parent=False
                                                )
            
            if self.ttno_applied(i):
                # In this case there is an additional leg towards the parent in the
                # lower_contr_tensor, the one coming from the operator tensor.
                offset = 1
            else:
                # In this case all legs on the lower_contr_tensor are in the same order as
                # in the corresponding TTNS
                offset = 0
            low_legs = tuple([leg + offset for leg in low_legs])
            r_tensor = np.tensordot(lower_contr_tensors[i],
                                    uh,
                                    axes=(low_legs, uh_legs))
            r_tensor_caches[i].add_entry(non_base_node_id,
                                         non_base_node.parent,
                                         r_tensor)
        new_tensor = _adjust_new_ttns_tensor(u)
        return new_tensor

    def get_reduced_density_matrix(self,
                                   index: int,
                                   node_id: str,
                                   subtree_tensor: npt.NDArray,
                                   r_tensor_cache: PartialTreeCachDict,
                                   ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Gets the reduced density matrix for the given node for the given
        TTNS-TTNO pair.

        Args:
            index (int): The index of the TTNS-TTNO pair.
            node_id (str): The identifier of the node to get the reduced density
                matrix for of the considered TTNS. This is not the identifier in 
                the base TTNS!
            subtree_tensor (npt.NDArray): The subtree tensor to contract the
                parent legs with.
            r_tensor_cache (PartialTreeCachDict): A cache containing the
                contractions of all subtrees below this node.

        Returns:
            tuple[npt.NDArray, npt.NDArray]: The reduced density matrix adapted
                to the leg order of the base TTNS, and the lower contraction
                result.
        """
        ket_node_tensor = self._ttnss[index][node_id]
        node_tensors = [ket_node_tensor]
        id_trafos = [identity_mapping]
        if self.ttno_applied(index):
            id_trafo = self.get_id_trafos_ttns_ttno(index)
            op_node_tensor = self._ttnos[index][id_trafo(node_id)]
            node_tensors.append(op_node_tensor)
            ttns_ttno_trafo = self.get_id_trafos_ttns_ttno(index)
            id_trafos.append(ttns_ttno_trafo)
        ignored_leg = ket_node_tensor[0].parent
        assert ignored_leg is not None
        local_contr = LocalContraction(node_tensors,
                                        r_tensor_cache,
                                        ignored_leg=ignored_leg,
                                        id_trafos=id_trafos)
        lower_contr_tensor = local_contr()
        upper_contr_tensor = lower_contr_tensor.conj()
        # Now we perform the contraction with the subtree tensor
        if self.ttno_applied(index):
            # The first two open legs of the lower tensor are the legs towards the
            # parent, which is exactly what we want to contract with the subtree tensor
            legs_1 = ([0,1], [0,1])
            legs_2 = ([0,1], [1,0]) # Note the order of axes here
        else:
            # If we don't have a ttno, we only have one leg towards the parent.
            legs_1 = ([0], [0])
            legs_2 = ([0], [0])
        tensor = np.tensordot(subtree_tensor,
                            lower_contr_tensor,
                            axes=legs_1)
        tensor = np.tensordot(tensor,
                            upper_contr_tensor,
                            axes=legs_2)
        # Now the first half of the legs are the output legs, the second half
        # are the input legs of the reduced density matrix
        # We also must adapt the leg order to fit with the base tensor.
        children_permutation = relative_leg_permutation(ket_node_tensor[0],
                                                        self.get_base_node(index,node_id),
                                                        self.get_ttns_base_id_map(index),
                                                        include_parent=False)
        # The parent leg is the first, which we don't have anymore, so -1
        children_permutation = [leg - 1 for leg in children_permutation]
        transpose_map = children_permutation
        half_legs = len(transpose_map)
        transpose_map += [i + half_legs for i in transpose_map]
        return tensor.transpose(transpose_map), lower_contr_tensor


def dm_ttns_ttno_application(ttns: TTNS,
                             ttno: TTNO,
                             id_trafo: Callable = identity_mapping,
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
            node identifiers to the TTNO`s node identifiers. Defaults to
            the identity_mapping.
        svd_params (SVDParameters): The parameters for the decomposition.
    
    Returns:
        TTNS: The TTNS approximating the TTNS that is yielded by the
            contraction of the TTNS with the TTNO.

    """
    appl_obj = DMTTNOApplication(ttns,
                                 ttnos=ttno,
                                 id_trafos_ttnos=id_trafo,
                                 svd_params=svd_params)
    new_ttns = appl_obj()
    return new_ttns

def dm_linear_combination(ttnss: list[TTNS],
                          ttnos: list[TTNO | None],
                          id_trafos_ttns: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                          id_trafos_ttnos: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                          svd_params: SVDParameters = SVDParameters()
                            ) -> TTNS:
    """
    Computes a linear combination of the given TTNSs with the given TTNOs via the
    density matrix based algorithm.

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
    appl_obj = DMTTNOApplication(ttnss,
                                 ttnos=ttnos,
                                 id_trafos_ttns=id_trafos_ttns,
                                 id_trafos_ttnos=id_trafos_ttnos,
                                 svd_params=svd_params)
    new_ttns = appl_obj()
    return new_ttns

def dm_addition(ttnss: list[TTNS],
                id_trafos_ttns: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                svd_params: SVDParameters = SVDParameters()
                ) -> TTNS:
    """
    Computes the sum of the given TTNSs via the density matrix based algorithm.

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
    appl_obj = DMTTNOApplication(ttnss,
                                 id_trafos_ttns=id_trafos_ttns,
                                 svd_params=svd_params)
    new_ttns = appl_obj()
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
