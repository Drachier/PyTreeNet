from __future__ import annotations
from typing import Union, List

import numpy as np

from ...ttns import TreeTensorNetworkState
from ...ttno.ttno import TTNO
from ...graph_node import GraphNode

from .cached_tensor import CachedSiteTensor
from ...contractions.tree_cach_dict import PartialTreeChachDict

class PartialTreeCache():
    """
    A class representing the contracted partial tree which starts at the node
     with node_id and where the open legs point towards the node with
     identifier pointing_to_node. Contracted means that in the TDVP picture
     the all tensors that are in the bra, ket, and Hamiltonian are contracted
     with one another and the other tensors in the subtree.
    """

    def __init__(self, node: GraphNode, ham_node: GraphNode, pointing_to_node: str,
                 tensor: np.ndarray) -> None:
        self.node = node
        self.ham_node = ham_node
        self.pointing_to_node = pointing_to_node
        self.tensor = tensor

    def close_to(self, other: PartialTreeCache) -> bool:
        """
        Checks if the two PartialTreeCaches have the same node_ids and
         close tensors

        Args:
            other (PartialTreeCache): A PartialTreeCache to compare to.

        Returns:
            bool: True if the tensors are close, and the identifiers are
             equal.
        """
        if self.node.identifier != other.node.identifier:
            return False
        if self.pointing_to_node != other.pointing_to_node:
            return False
        if not np.allclose(self.tensor, other.tensor):
            return False
        return True

    @classmethod
    def for_all_nodes(cls, node_id: str, next_node_id: str,
                      state: TreeTensorNetworkState,
                      hamiltonian: TTNO,
                      partial_tree_cache: Union[PartialTreeChachDict,None] = None) -> PartialTreeCache:
        """
        Generates the PartialTreeCache for a given node and direction for the
         open legs to point to. This means it is automatically chosen, if the
         cache is constructed for a leaf or with other tensors already cached.

        Args:
            node_id (str): The identifier of the node to which this cache
             corresponds.
            next_node_id (str): The identifier of the node to which the open
             legs of the tensor point.
            state (TreeTensorNetworkState): A state to be used for the
             computation in TTNS form.
            hamiltonian (TTNO): A Hamiltonian to be used for the computation
             in TTNO form.
            partial_tree_cache (Union[PartialTreeChachDict,None], optional): 
             Potentially a dictionary of already computed tensors. Defaults to
              None.

        Raises:
            ValueError: If the node is not a leaf and no partial tree cach
             dict is given.

        Returns:
            PartialTreeCache: The contracted partial tree.
                         _____
                    ____|     |
                        |  A* |
                        |_____|
                           |
                           |
                         __|__
                    ____|     |
                        |  H  |
                        |_____|
                           |
                           |
                         __|__
                    ____|     |
                        |  A  |
                        |_____|
        """
        node = state.nodes[node_id]
        if node.is_leaf():
            return cls.for_leaf(node_id,
                                state,
                                hamiltonian)
        if partial_tree_cache is None:
            errstr = "For a general node the partial tree cache cannot be None!"
            raise ValueError(errstr)
        return cls.with_existing_cache(node_id,
                                       next_node_id,
                                       partial_tree_cache,
                                       state,
                                       hamiltonian)
    @classmethod
    def for_leaf(cls, node_id: str,
                 state: TreeTensorNetworkState,
                 hamiltonian: TTNO) -> PartialTreeCache:
        """
        If the current subtree ends and starts at a leaf, only the three
         tensors corresponding to that site must be contracted. Furthermore,
         the retained legs must point towards the leaf's parent.

        Args:
            node_id (str): Identifier of the leaf node
            state (TreeTensorNetworkState): The TTNS representing the state.
            hamiltonian (TTNO): The TTNO representing the Hamiltonian.

        Returns:
            PartialTreeCache: The contracted partial tree.
                         _____
                    ____|     |
                        |  A* |
                        |_____|
                           |
                           |
                         __|__
                    ____|     |
                        |  H  |
                        |_____|
                           |
                           |
                         __|__
                    ____|     |
                        |  A  |
                        |_____|
        """
        ket_node, ket_tensor = state[node_id]
        ham_node, ham_tensor = hamiltonian[node_id]
        site_triple_tensor = CachedSiteTensor(ket_node, ham_node,
                                        ket_tensor, ham_tensor)
        site_triple_tensor = site_triple_tensor.contract_tensor_sandwich()
        parent_node_id = ket_node.parent
        return PartialTreeCache(ket_node, ham_node,
                                parent_node_id, site_triple_tensor)

    @classmethod
    def with_existing_cache(cls, node_id: str, next_node_id: str,
                            partial_tree_cache: PartialTreeChachDict,
                            state: TreeTensorNetworkState,
                            hamiltonian: TTNO) -> PartialTreeCache:
        """
        Creates this cache using the contracted partial tree tensors already
         cached.

        Args:
            node_id (str): Identifier of the node this partial tree ends at.
            next_node_id (str): The identifier to which the open legs should
             point.
            partial_tree_cache (PartialTreeChachDict): The tensors already
             cached.
            state (TreeTensorNetworkState): The state on which to perform
             this action in TTNS form.
            hamiltonian (TTNO): The Hamiltonian on with which to compute
             the tensor contraction in TTNO form.

        Returns:
            PartialTreeCache: _description_
        """
        ket_node, ket_tensor = state[node_id]
        ham_node, ham_tensor = hamiltonian[node_id]
        cached_tensor = PartialTreeCache(ket_node, ham_node,
                                         next_node_id, ket_tensor)
        next_node_index = ket_node.neighbour_index(next_node_id)
        cached_tensor._contract_all_but_one_neighbouring_cache(next_node_index,
                                                               partial_tree_cache)
        cached_tensor._contract_hamiltonian_tensor(ham_tensor)
        cached_tensor._contract_bra_tensor(next_node_index,
                                           ket_tensor.conj())
        return cached_tensor

    def _contract_neighbour_cache_to_ket(self,
                                         neighbour_id: str,
                                         next_node_index: int,
                                         partial_tree_cache: PartialTreeChachDict):
        """
        Contracts a cached contracted tree C that originates at the neighbour
         node with the tensor of this node, which is currently a ket node A,
         possibly with other neighbour caches already contracted.
                                    ______
                                   |      |
                            _______|      |
                                   |      |
                                   |      |
                                   |      |
                                   |      |
                            _______|      |
                                   |  C   |
                                   |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|

        Args:
            neighbour_id (str): The identifier of the neighbour
            next_node_index (int): The index value of the neighbour to which
             the open legs should point.
            partial_tree_cache (PartialTreeChachDict): All the cached partial
             trees.
        """
        cached_neighbour_tensor = partial_tree_cache.get_cached_tensor(neighbour_id,
                                                                    self.node.identifier)
        neighbour_index = self.node.neighbour_index(neighbour_id)
        if neighbour_index > next_node_index:
            tensor_index_to_neighbour = 1
        elif neighbour_index < next_node_index:
            tensor_index_to_neighbour = 0
        else:
            raise AssertionError("Next Node shouldn't be touched!")
        self.tensor = np.tensordot(self.tensor, cached_neighbour_tensor,
                                    axes=([tensor_index_to_neighbour],[0]))

    def _contract_all_but_one_neighbouring_cache(self,
                                                 next_node_index: int,
                                                 partial_tree_cache: PartialTreeChachDict):
        """
        Contracts all cached contracted trees C that originates at neighbour
         nodes with the tensor of this node, which is currently the ket node A.
                                    ______
                                   |      |
                            _______|      |
                                   |      |
                                   |      |
                                   |      |
                                   |      |
                            _______|      |
                                   |  C   |
                                   |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|

        Args:
            next_node_index (int): The index value of the neighbour to which
             the open legs should point.
            partial_tree_cache (PartialTreeChachDict): All the cached partial trees.
        """
        neighbours_wo_next = [n_id for i, n_id in enumerate(self.node.neighbouring_nodes())
                              if i != next_node_index]
        for neighbour_id in neighbours_wo_next:
            self._contract_neighbour_cache_to_ket(neighbour_id,
                                                  next_node_index,
                                                  partial_tree_cache)

    def find_ham_tensor_legs(self) -> List[int]:
        """
        Finds the legs of the hamiltonian tensor that are to be contracted
         with the tensor aquired by contracting all neighbouring caches.

        Returns:
            List[int]: [physical_leg, leg_to_parent, leg_to_state_c1, leg_to_state_c2, ...]
        """
        ham_legs = [self._node_operator_input_leg()] # Physical leg is first
        for neighbour_id in self.node.neighbouring_nodes():
            if neighbour_id != self.pointing_to_node:
                if neighbour_id == self.node.parent:
                    ham_legs.append(0)
                else:
                    leg = self.ham_node.neighbour_index(neighbour_id)
                    ham_legs.append(leg)
        return ham_legs

    def _contract_hamiltonian_tensor(self, ham_tensor: np.ndarray):
        """
        Contracts the hamiltonian tensor with the saved tensor. The saved
         tensor has all the other neighbouring cached subtrees contracted to
         it already.
                                    ______
                                   |      |
                            _______|      |
                                   |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  H  |    |  C   |
                        |_____|    |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|

        Args:
            ham_tensor (np.ndarray): The Hamiltonian tensor corresponding to
             this site.
        """
        num_neighbours = self.node.nneighbours()
        legs_tensor = [1]
        legs_tensor.extend(range(2,2*num_neighbours,2))
        legs_ham_tensor = self.find_ham_tensor_legs()
        self.tensor = np.tensordot(self.tensor, ham_tensor,
                                   axes=(legs_tensor, legs_ham_tensor))

    def _contract_bra_tensor(self,
                             next_node_index: int,
                             bra_tensor: np.ndarray):
        """
        Contracts the bra tensor with the saved tensor. The saved
         tensor has all the other neighbouring cached subtrees contracted to
         it already.
                         _____      ______
                    ____|     |____|      |
                        |  A* |    |      |
                        |_____|    |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  H  |    |  C   |
                        |_____|    |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|

        Args:
            next_node_index (int): The index value of the neighbour to which
             the open legs should point.
            ham_tensor (np.ndarray): The bra tensor corresponding to
             this site.
        """
        num_neighbours = self.node.nneighbours()
        legs_tensor = list(range(1,num_neighbours))
        legs_tensor.append(num_neighbours+1)
        legs_bra_tensor = list(range(next_node_index))
        legs_bra_tensor.extend(range(next_node_index+1,num_neighbours+1))
        self.tensor = np.tensordot(self.tensor, bra_tensor,
                                   axes=(legs_tensor, legs_bra_tensor))

    def _node_operator_input_leg(self) -> int:
        """
        Finds the leg of a node of the hamiltonian corresponding to the input.

        Returns:
            int: The index of the leg corresponding to input.
        """
        # Corr ket leg
        return self.node.nneighbours() + 1

    def _node_operator_output_leg(self) -> int:
        """
        Finds the leg of a node of the hamiltonian corresponding to the
         output.
        
        Returns:
            int: The index of the leg corresponding to output.
        """
        # Corr bra leg
        return self.node.nneighbours()
