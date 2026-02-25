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
from ...util.tensor_splitting import (tensor_qr_decomposition,
                                      SVDParameters)
from ...operators.common_operators import copy_tensor
from ...core.node import Node
from ...util.std_utils import identity_mapping
from .abtract_lc_class import AbtractLCwithTempTensors

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...ttno.ttno_class import TTNO

__all__ = ["src_ttns_ttno_application"]

class SRCTTNOApplication(AbtractLCwithTempTensors):
    """
    A class for applying a TTNO to a TTNS using the succesive randomized compression
    algorithm.
    """

    def __init__(self,
                 ttnss: list[TTNS] | TTNS,
                 ttnos: list[TTNO | None] | TTNO | None = None,
                 id_trafos_ttns: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                 id_trafos_ttnos: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                 svd_params: SVDParameters | None = None,
                 desired_dimension: int | None = None,
                 seed: int | None = None
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
            svd_params (SVDParameters | None, optional): The parameters for the SVD
                decomposition used to truncate the temporary tensors. If None is given,
                default parameters are used. Defaults to None.
            desired_dimension (int | None, optional): The desired dimension of the
                for the randomly generated matrices. If None is given, the dimension
                in the SVD parameters is used. Defaults to None.
            seed (int | None, optional): The seed for the random number generator used
                to generate the random matrices. If None is given, no seed is set.
                Defaults to None.
        """
        super().__init__(ttnss,
                         ttnos=ttnos,
                         id_trafos_ttns=id_trafos_ttns,
                         id_trafos_ttnos=id_trafos_ttnos,
                         svd_params=svd_params)
        if desired_dimension is None:
            desired_dimension = self._svd_params.max_bond_dim
        self._desired_dimension = desired_dimension
        self._seed = seed
        self._rand_ttns = TTNS()

    def build_full_subtree_caches(self) -> list[PartialTreeCachDict]:
        # Notably we can reuse the random TTNS for all terms.
        self._rand_ttns = generate_random_matrices(self.get_base_ttns(),
                                                    self._desired_dimension,
                                                    seed=self._seed)
        return super().build_full_subtree_caches()

    def build_full_subtree_cache(self,
                                 index: int
                                 ) -> PartialTreeCachDict:
        """
        Build the full subtree cache for the given index.
        """
        ttns = self._ttnss[index]
        # Note that the random TTNS has the same identifiers as the base TTNs.
        trafo_to_random = self.get_ttns_base_id_map(index)
        if self.ttno_applied(index):
            ttno = self._ttnos[index]
            id_trafo = self.get_id_trafos_ttns_ttno(index)
            cache = build_full_subtree_cache(ttns,
                                             ttno,
                                             self._rand_ttns,
                                             id_trafo=id_trafo,
                                             id_trafo_to_random=trafo_to_random)
        else:
            cache = build_full_subtree_cache_state_only(ttns,
                                                        self._rand_ttns,
                                                        id_trafo_to_random=trafo_to_random)
        return cache

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
    appl_obj = SRCTTNOApplication(ttns,
                                  ttnos=ttno,
                                  desired_dimension=desired_dimension,
                                  id_trafos_ttnos=id_trafo if id_trafo is not None else identity_mapping,
                                  seed=seed
                                  )
    return appl_obj()

def src_linear_combination(ttnss: list[TTNS],
                          ttnos: list[TTNO | None],
                          id_trafos_ttns: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                          id_trafos_ttnos: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                          svd_params: SVDParameters = SVDParameters(),
                          desired_dimension: int | None = None,
                          seed: int | None = None
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
        desired_dimension (int | None, optional): The desired dimension of the
            resulting TTNS. If None is given, the dimension in svd_params is used.
            Defaults to None.
        seed (int | None, optional): Seed for the random number generator used to
            generate the random matrices. Defaults to None.
        
    Returns:
        TTNS: The result of the linear combination.
    """
    appl_obj = SRCTTNOApplication(ttnss,
                                  ttnos=ttnos,
                                  id_trafos_ttns=id_trafos_ttns,
                                  id_trafos_ttnos=id_trafos_ttnos,
                                  svd_params=svd_params,
                                  desired_dimension=desired_dimension,
                                  seed=seed
                                  )
    return appl_obj()

def src_addition(ttnss: list[TTNS],
                 id_trafos_ttns: list[Callable[[str],str]] | Callable = identity_mapping,
                 svd_params: SVDParameters = SVDParameters(),
                 desired_dimension: int | None = None,
                 seed: int | None = None
                 ) -> TTNS:
    """
    Computes the sum of the given TTNSs via the succesive randomized compression
    algorithm.

    Args:
        ttnss (list[TTNS]): The TTNSs to add.
        id_trafos_ttns (list[Callable[[str],str]] | Callable, optional):
            The identifier transformation functions for the TTNSs. The i-th
            function transforms the node identifiers of the 0-th TTNS to the node
            identifiers of the i-th TTNS. If a single function is given, it is
            treated as a list of length one, and applied to all TTNSs in ttnss.
            Defaults to identity_mapping.
        svd_params (SVDParameters, optional): The parameters for the
            decomposition. Defaults to SVDParameters().
        desired_dimension (int | None, optional): The desired dimension of the
            resulting TTNS. If None is given, the dimension in svd_params is used.
            Defaults to None.
        seed (int | None, optional): Seed for the random number generator used to
            generate the random matrices. Defaults to None.
    
    Returns:
        TTNS: The sum of the input TTNSs.
    """
    appl_obj = SRCTTNOApplication(ttnss,
                                  id_trafos_ttns=id_trafos_ttns,
                                  svd_params=svd_params,
                                  desired_dimension=desired_dimension,
                                  seed=seed
                                  )
    return appl_obj()

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
                             ttno: TTNO | None,
                             random_ttns: TTNS,
                             id_trafo: Callable,
                             id_trafo_to_random: Callable = identity_mapping
                             ) -> PartialTreeCachDict:
    """
    Builds a full subtree cache for the given TTNS and TTNO with random
    tensors.

    Args:
        ttns (TTNS): The TTNS to build the cache for.
        ttno (TTNO | None): The TTNO to use for the contractions.
            If None is given, only the state is considered in the creation.
        random_ttns (TTNS): The TTNS with random tensors.
        id_trafo (Callable): A function to transform node identifiers.
        id_trafo_to_random (Callable): A function to transform node
            that transforms the TTNS`s node identifiers to the random TTNS`s
            node identifiers. Defaults to identity_mapping.

    Returns:
        PartialTreeCachDict: The full subtree cache.
    """
    use_ttno = ttno is not None
    cache = PartialTreeCachDict()
    # Get envs upward
    lin_order = ttns.linearise()[:-1] # Exclude root
    if use_ttno:
        id_trafos = [identity_mapping, id_trafo, id_trafo_to_random]
    else:
        id_trafos = [identity_mapping, id_trafo_to_random]
    for node_id in lin_order:
        node_tensors = []
        ket_node_tensor = ttns[node_id]
        node_tensors.append(ket_node_tensor)
        if use_ttno:
            op_node_tensor = ttno[id_trafo(node_id)]
            node_tensors.append(op_node_tensor)
        bra_node_tensor = random_ttns[id_trafo_to_random(node_id)]
        node_tensors.append(bra_node_tensor)
        ignored_leg = ket_node_tensor[0].parent
        assert ignored_leg is not None
        local_contr = LocalContraction(node_tensors,
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
            node_tensors = [ket_node_tensor]
            if use_ttno:
                op_node_tensor = ttno[id_trafo(node_id)]
                node_tensors.append(op_node_tensor)
            bra_node_tensor = random_ttns[id_trafo_to_random(node_id)]
            node_tensors.append(bra_node_tensor)
            for child_id in ket_node_tensor[0].children:
                local_contr = LocalContraction(node_tensors,
                                                cache,
                                                ignored_leg=child_id,
                                                id_trafos=id_trafos)
                local_contr.contract_into_cache()
    return cache

def build_full_subtree_cache_state_only(ttns: TTNS,
                                       random_ttns: TTNS,
                                       id_trafo_to_random: Callable = identity_mapping
                                       ) -> PartialTreeCachDict:
    """
    Builds a full subtree cache for the given TTNS with random tensors, only
    considering the state.

    Args:
        ttns (TTNS): The TTNS to build the cache for.
        random_ttns (TTNS): The TTNS with random tensors.
        id_trafo_to_random (Callable): A function to transform node
            that transforms the TTNS`s node identifiers to the random TTNS`s
            node identifiers. Defaults to identity_mapping.

    Returns:
        PartialTreeCachDict: The full subtree cache.
    """
    return build_full_subtree_cache(ttns,
                                    None,
                                    random_ttns,
                                    id_trafo=identity_mapping,
                                    id_trafo_to_random=id_trafo_to_random)
