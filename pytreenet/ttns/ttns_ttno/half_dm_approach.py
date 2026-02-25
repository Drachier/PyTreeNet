"""
This module implements the half density matrix approach for the TTNO application.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable

from ..ttns import TTNS
from ...core.tree_structure import LinearisationMode
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...contractions.local_contr import LocalContraction, FinalTransposition
from ...util.tensor_splitting import (contr_truncated_svd_splitting,
                                      SVDParameters,
                                      ContractionMode)
from ...util.std_utils import identity_mapping
from .abtract_lc_class import AbtractLCwithTempTensors

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...ttno.ttno_class import TTNO

__all__ = ["half_dm_ttns_ttno_application"]

class HalfDMTTNOApplication(AbtractLCwithTempTensors):
    """
    A class for applying a TTNO to a TTNS using the half density matrix approach.
    """

    def __init__(self,
                 ttnss: list[TTNS] | TTNS,
                 ttnos: list[TTNO | None] | TTNO | None = None,
                 id_trafos_ttns: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                 id_trafos_ttnos: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                 svd_params: SVDParameters | None = None,
                 cache_svd_params: SVDParameters | None = None
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
            cache_svd_params (SVDParameters | None, optional): The parameters for the
                SVD truncation in the cache building. If None is given, the same
                parameters as for the main decomposition are used. Defaults to None.
        """
        super().__init__(ttnss,
                         ttnos=ttnos,
                         id_trafos_ttns=id_trafos_ttns,
                         id_trafos_ttnos=id_trafos_ttnos,
                         svd_params=svd_params)
        if cache_svd_params is None:
            cache_svd_params = self._svd_params
        self._cache_svd_params = cache_svd_params

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
                                             self._cache_svd_params)
        else:
            cache = build_full_subtree_cache_state_only(ttns,
                                                        self._cache_svd_params)
        return cache

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
