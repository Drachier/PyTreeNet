"""
Implements an abstract class for linear combinations of tree tensor networks.
This class mostly deals with the bookkeeping of the TTNSs, TTNOs, and identifier
transformations between them.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from abc import ABC, abstractmethod

import numpy as np

from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...contractions.local_contr import LocalContraction
from ...util.std_utils import (identity_mapping,
                               inverse_bijective_finite_map)
from ...ttns import TTNS
from ...ttno import TTNO
from ...core.node import relative_leg_permutation

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...core.node import Node

class AbstractLinearCombination(ABC):
    """
    A class to apply TTNOs to TTNSs via the density matrix based algorithm.
    """

    def __init__(self,
                 ttnss: list[TTNS] | TTNS,
                 ttnos: list[TTNO | None] | TTNO | None = None,
                 id_trafos_ttns: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                 id_trafos_ttnos: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                 ) -> None:
        """
        Initialises the AbstractLinearCombination.

        Args:
            ttnss (list[TTNS] | TTNS): The TTNSs to apply the TTNOs to. If a
                single TTNS is given, it is treated as a list of length one.
            ttnos (list[TTNO | None] | TTNO | None, optional): The TTNOs to apply to
                the TTNSs. If a single TTNO is given, it is treated as a list of
                length one, and applied to all TTNSs in ttnss. If None is given,
                the sum of the TTNSs is computed. If a TTNO is given as None,
                the TTNO is assumed to be the identity operator.
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
        """
        if isinstance(ttnss, TTNS):
            ttnss = [ttnss]
        self._ttnss = ttnss
        if ttnos is None:
            ttnos = [None] * len(ttnss)
        elif isinstance(ttnos, TTNO):
            ttnos = [ttnos] * len(ttnss)
        self._ttnos: list[TTNO, None] = ttnos
        if isinstance(id_trafos_ttns, Callable):
            id_trafos_ttns = [id_trafos_ttns] * len(ttnss)
        self._id_trafos_ttns: list[Callable[[str],str]] = id_trafos_ttns
        if isinstance(id_trafos_ttnos, Callable):
            id_trafos_ttnos = [id_trafos_ttnos] * len(ttnss)
        self._id_trafos_ttnos: list[Callable[[str],str]] = id_trafos_ttnos
        self._inverse_id_trafos_ttns: list[Callable[[str],str] | None] = [None] * len(ttnss)
        self._inverse_id_trafos_ttnos: list[Callable[[str],str] | None] = [None] * len(ttnss)
        self._id_trafos_ttns_ttno: list[Callable[[str],str] | None] = [None] * len(ttnss)
        self._id_trafos_ttno_ttns: list[Callable[[str],str] | None] = [None] * len(ttnss)

        self._subtree_caches = [PartialTreeCachDict()
                                for _ in range(self.num_ttns())]

    def ttno_applied(self,
                     index: int
                     ) -> bool:
        """
        Checks whether a TTNO is applied to the i-th TTNS.

        Args:
            index (int): The index of the TTNS to check.

        Returns:
            bool: True if a TTNO is applied to the i-th TTNS, False otherwise.
        """
        return self._ttnos[index] is not None

    def base_id_to_ttns(self,
                        identifier: str,
                        index: int
                        ) -> str:
        """
        Transforms a node identifier from the base TTNS to the i-th TTNS.

        Args:
            identifier (str): The node identifier in the base TTNS.
            index (int): The index of the TTNS to transform to.
        
        Returns:
            str: The node identifier in the i-th TTNS.
        """
        return self._id_trafos_ttns[index](identifier)

    def base_id_to_ttno(self,
                        identifier: str,
                        index: int
                        ) -> str:
        """
        Transforms a node identifier from the base TTNS to the i-th TTNO.

        Args:
            identifier (str): The node identifier in the base TTNS.
            index (int): The index of the TTNO to transform to.
        
        Returns:
            str: The node identifier in the i-th TTNO.
        """
        return self._id_trafos_ttnos[index](identifier)

    def get_ttns_base_id_map(self,
                                 index: int
                                    ) -> Callable[[str], str]:
        """
        Builds the identifier transformation function from the i-th TTNS to the
        base TTNS.

        Args:
            index (int): The index of the TTNS to build the transformation for.

        Returns:
            Callable[[str], str]: The identifier transformation function from
            the i-th TTNS to the base TTNS.
        """
        if self._inverse_id_trafos_ttns[index] is None:
            inverse = inverse_bijective_finite_map(self._id_trafos_ttns[index],
                                                   self._ttnss[0].nodes.keys())
            self._inverse_id_trafos_ttns[index] = inverse
        return self._inverse_id_trafos_ttns[index]

    def ttns_id_to_base(self,
                        identifier: str,
                        index: int
                        ) -> str:
        """
        Transforms a node identifier from the i-th TTNS to the base TTNS.

        Args:
            identifier (str): The node identifier in the i-th TTNS.
            index (int): The index of the TTNS to transform from.

        Returns:
            str: The node identifier in the base TTNS.
        """
        inverse = self.get_ttns_base_id_map(index)
        return inverse(identifier)

    def get_ttno_base_id_map(self,
                                 index: int
                                    ) -> Callable[[str], str]:
        """
        Builds the identifier transformation function from the i-th TTNO to the
        base TTNS.

        Args:
            index (int): The index of the TTNO to build the transformation for.

        Returns:
            Callable[[str], str]: The identifier transformation function from
            the i-th TTNO to the base TTNS.
        """
        if self._inverse_id_trafos_ttnos[index] is None:
            inverse = inverse_bijective_finite_map(self._id_trafos_ttnos[index],
                                                   self._ttnss[0].nodes.keys())
            self._inverse_id_trafos_ttnos[index] = inverse
        return self._inverse_id_trafos_ttnos[index]

    def ttno_id_to_base(self,
                        identifier: str,
                        index: int
                        ) -> str:
        """
        Transforms a node identifier from the i-th TTNO to the base TTNS.

        Args:
            identifier (str): The node identifier in the i-th TTNO.
            index (int): The index of the TTNO to transform from.
    
        Returns:
            str: The node identifier in the base TTNS.
        """
        inverse = self.get_ttno_base_id_map(index)
        return inverse(identifier)

    def get_id_trafos_ttns_ttno(self,
                                   index: int
                                    ) -> Callable[[str], str]:
        """
        Builds the identifier transformation functions between the i-th TTNS
        and the i-th TTNO.

        Args:
            index (int): The index of the TTNS and TTNO to build the
                transformations for.

        Returns:
            Callable[[str], str]: The identifier transformation function from
            the i-th TTNS to the i-th TTNO.
        """
        if self._id_trafos_ttns_ttno[index] is None:
            id_trafos_ttns_ttno = {node_id: self.base_id_to_ttno(self.ttns_id_to_base(node_id, index), index)
                                   for node_id in self._ttnss[index].nodes.keys()}
            self._id_trafos_ttns_ttno[index] = lambda x: id_trafos_ttns_ttno[x]
        return self._id_trafos_ttns_ttno[index]

    def ttns_id_to_ttno(self,
                        identifier: str,
                        index: int
                        ) -> str:
        """
        Transforms a node identifier from the i-th TTNS to the i-th TTNO.

        Args:
            identifier (str): The node identifier in the i-th TTNS.
            index (int): The index of the TTNS and TTNO to transform between.

        Returns:
            str: The node identifier in the i-th TTNO.
        """
        id_trafos_ttns_ttno = self.get_id_trafos_ttns_ttno(index)
        return id_trafos_ttns_ttno(identifier)

    def get_id_trafos_ttno_ttns(self,
                                   index: int
                                    ) -> Callable[[str], str]:
        """
        Builds the identifier transformation functions between the i-th TTNO
        and the i-th TTNS.

        Args:
            index (int): The index of the TTNS and TTNO to build the
                transformations for.

        Returns:
            Callable[[str], str]: The identifier transformation function from
            the i-th TTNO to the i-th TTNS.
        """
        if self._id_trafos_ttno_ttns[index] is None:
            id_trafos_ttno_ttns = {node_id: self.base_id_to_ttns(self.ttno_id_to_base(node_id, index), index)
                                   for node_id in self._ttnss[index].nodes.keys()}
            self._id_trafos_ttno_ttns[index] = lambda x: id_trafos_ttno_ttns[x]
        return self._id_trafos_ttno_ttns[index]

    def ttno_id_to_ttns(self,
                        identifier: str,
                        index: int
                        ) -> str:
        """
        Transforms a node identifier from the i-th TTNO to the i-th TTNS.

        Args:
            identifier (str): The node identifier in the i-th TTNO.
            index (int): The index of the TTNS and TTNO to transform between.

        Returns:
            str: The node identifier in the i-th TTNS.
        """
        id_trafos_ttno_ttns = self.get_id_trafos_ttno_ttns(index)
        return id_trafos_ttno_ttns(identifier)

    def get_base_ttns(self) -> TTNS:
        """
        Gets the base TTNS, i.e. the first TTNS in the list of TTNSs.

        Returns:
            TTNS: The base TTNS.
        """
        return self._ttnss[0]

    def get_base_node(self,
                      index: int,
                      node_id: str
                      ) -> Node:
        """
        Gets the node in the base TTNS corresponding to the node with the given
        identifier in the i-th TTNS.

        Args:
            index (int): The index of the TTNS to get the node from.
            node_id (str): The identifier of the node in the i-th TTNS.

        Returns:
            Node: The node in the base TTNS corresponding to the node with the given
            identifier in the i-th TTNS.
        """
        base_id = self.ttns_id_to_base(node_id, index)
        return self.get_base_ttns().nodes[base_id]

    def get_ttns_node_tensor(self,
                             index: int,
                             node_id: str
                             ) -> tuple[Node, npt.NDArray]:
        """
        Gets the node and tensor in the i-th TTNS corresponding to the node with the given
        identifier in the i-th TTNS.

        Args:
            index (int): The index of the TTNS to get the node from.
            node_id (str): The identifier of the node in the i-th TTNS.

        Returns:
            tuple[Node, npt.NDArray]: The node and tensor in the i-th TTNS corresponding
                to the node with the given identifier in the i-th TTNS.
        """
        return self._ttnss[index][node_id]

    def get_ttno_node_tensor(self,
                             index: int,
                             node_id: str
                             ) -> tuple[Node, npt.NDArray]:
        """
        Gets the node and tensor in the i-th TTNO corresponding to the node with the given
        identifier in the i-th TTNO.

        Args:
            index (int): The index of the TTNO to get the node from.
            node_id (str): The identifier of the node in the i-th TTNO.

        Returns:
            tuple[Node, npt.NDArray]: The node and tensor in the i-th TTNO corresponding
                to the node with the given identifier in the i-th TTNO.
        """
        if self._ttnos is None:
            raise ValueError("The TTNOs have not been set for this linear combination!")
        return self._ttnos[index][node_id]

    def num_ttns(self) -> int:
        """
        Gets the number of TTNSs in the linear combination.

        Returns:
            int: The number of TTNSs in the linear combination.
        """
        return len(self._ttnss)
    
    @abstractmethod
    def build_full_subtree_cache(self,
                                index: int
                                ) -> PartialTreeCachDict:
        """
        Builds the full subtree cache for the i-th TTNS and TTNO.

        Args:
            index (int): The index of the TTNS and TTNO to build the subtree cache for.
        
        Returns:
            PartialTreeCachDict: The full subtree cache for the i-th TTNS and TTNO.
        """
        pass

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

    @abstractmethod
    def _node_evaluation(self,
                         node_id: str,
                         r_tensors_caches: list[PartialTreeCachDict]
                         ) -> npt.NDArray:
        """
        Evaluates the new tensor for a non-root node in the result TTNS.

        Args:
            node_id (str): The identifier of the node to evaluate the tensor for.
            r_tensors_caches (list[PartialTreeCachDict]): The subtree caches for
                the TTNSs and TTNOs in the linear combination.
        
        Returns:
            npt.NDArray: The new tensor for the node with the given identifier in
                the result TTNS.
        """
        pass

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
        base_root_node = self.get_base_node(0, root_id)
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
            transpose_map = relative_leg_permutation(ket_node_tensor[0],
                                                     base_root_node,
                                                     modify_function=self.get_ttns_base_id_map(i))
            local_contr_tensor = local_contr_tensor.transpose(transpose_map) 
            local_contr_tensors.append(local_contr_tensor)
        new_tensor = np.sum(local_contr_tensors, axis=0)
        return new_tensor

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
