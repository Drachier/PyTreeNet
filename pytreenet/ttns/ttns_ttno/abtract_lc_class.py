"""
Implements an abstract class for linear combinations of tree tensor networks.
This class mostly deals with the bookkeeping of the TTNSs, TTNOs, and identifier
transformations between them.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable

from ...util.std_utils import (identity_mapping,
                               inverse_bijective_finite_map)
from ...ttns import TTNS
from ...ttno import TTNO

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...core.node import Node

class AbstractLinearCombination:
    """
    A class to apply TTNOs to TTNSs via the density matrix based algorithm.
    """

    def __init__(self,
                 ttnss: list[TTNS] | TTNS,
                 ttnos: list[TTNO] | TTNO | None = None,
                 id_trafos_ttns: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                 id_trafos_ttnos: list[Callable[[str],str]] | Callable[[str],str] = identity_mapping,
                 ) -> None:
        """
        Initialises the AbstractLinearCombination.

        Args:
            ttnss (list[TTNS] | TTNS): The TTNSs to apply the TTNOs to. If a
                single TTNS is given, it is treated as a list of length one.
            ttnos (list[TTNO] | TTNO | None, optional): The TTNOs to apply to
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
        """
        if isinstance(ttnss, TTNS):
            ttnss = [ttnss]
        self._ttnss = ttnss
        if ttnos is None:
            ttnos = [None] * len(ttnss)
        elif isinstance(ttnos, TTNO):
            ttnos = [ttnos] * len(ttnss)
        self._ttnos: list[TTNO | None] = ttnos
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
