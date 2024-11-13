
from __future__ import annotations
from typing import Union, Dict, Tuple, List

from numpy import ndarray, allclose

from .state_operator_contraction import contract_any
from .tree_cach_dict import PartialTreeCachDict
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator

class SandwichCache(PartialTreeCachDict):
    """
    In many cases a caching of the partial contractions of a TTNO sandwiched
    between two TTNS is useful.

    Attributes:
        state (TreeTensorNetworkState): The state of the system.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian of
            the system.

    """

    def __init__(self,
                 state: TreeTensorNetworkState,
                 hamiltonian: TreeTensorNetworkOperator,
                 dictionary: Union[Dict[Tuple[str,str],ndarray],None] = None
                 ) -> None:
        """
        Initializes the SandwichCache.

        Args:
            state (TreeTensorNetworkState): The state of the system.
            hamiltonian (TreeTensorNetworkOperator): The Hamiltonian of the
             system.
            dictionary (Union[Dict[Tuple[str,str],ndarray],None], optional): A
             dictionary that contains the initial entries. Defaults to None.
        
        """
        super().__init__(dictionary)
        self.state = state
        self.hamiltonian = hamiltonian

    def close_to(self, other: SandwichCache) -> bool:
        """
        Checks if the other cache is close to this cache.

        Args:
            other (SandwichCache): The other cache to compare with.

        Returns:
            bool: True if the other cache is close to this cache.
        """
        dict_same = super().close_to(other)
        states_close = self.state == other.state
        hamiltonians_close = self.hamiltonian == other.hamiltonian
        return states_close and hamiltonians_close and dict_same

    def shapes(self) -> Dict[Tuple[str,str],Tuple[int]]:
        """
        Returns the shapes of the tensors in the cache.

        Returns:
            Dict[Tuple[str,str],Tuple[int]]: A dictionary with the shapes
                of the tensors in the cache.
        """
        return {key: self[key].shape for key in self}

    def update_tree_cache(self, node_id: str, next_node_id: str):
        """
        Updates a tree tensor for given node identifiers.
        
        Updates the tree cache tensor that ends in the node with identifier
        `node_id` and has open legs pointing towards the neighbour node with
        identifier `next_node_id`.

        Args:
            node_id (str): The identifier of the node to which this cache
                corresponds.
            next_node_id (str): The identifier of the node to which the open
                legs of the tensor point.

        """
        update_tree_cache(self, self.state, self.hamiltonian,
                            node_id, next_node_id)

    @classmethod
    def init_cache_but_one(cls,
                            state: TreeTensorNetworkState,
                            hamiltonian: TreeTensorNetworkOperator,
                            left_out_id: str) -> SandwichCache:
        """
        Initialises the caching for the partial trees. 
        
        This means all the partial trees that are not the left out node
        have the bra, ket, and hamiltonian tensor corresponding
        to this node contracted and saved into the cache dictionary.

        Args:
            state (TreeTensorNetworkState): The state of the system.
            hamiltonian (TreeTensorNetworkOperator): The Hamiltonian of the
                system.
            left_out_id (str): The identifier of the node that is not
                contracted.
        """
        cache = cls(state, hamiltonian)
        rev_update_path, next_node_id_dict = _find_caching_path(state,
                                                                left_out_id)
        for node_id in rev_update_path[:-1]:
            next_node_id = next_node_id_dict[node_id]
            cache.update_tree_cache(node_id, next_node_id)
        return cache

def update_tree_cache(cache: SandwichCache,
                      state: TreeTensorNetworkState,
                      hamiltonian: TreeTensorNetworkOperator,
                      node_id: str,
                      next_node_id: str):
    """
    Updates the cache tensor for given node identifiers.

    Args:
    cache (SandwichCache): The cache to update.
    state (TreeTensorNetworkState): The tree tensor network state to be
        used for the update.
    hamiltonian (TreeTensorNetworkOperator): The Hamiltonian to be used
        for the update.
    node_id (str): The node at which the contraction should happen.
    next_node_id (str): The node to which the open legs of the tensor
        should point.

    """
    new_tensor = contract_any(node_id, next_node_id,
                                state, hamiltonian,
                                cache)
    cache.add_entry(node_id, next_node_id, new_tensor)

def _find_caching_path(state: TreeTensorNetworkState,
                       left_out_id: str
                        ) -> Tuple[List[str], Dict[str,str]]:
    """
    Finds the path used to cache the contracted subtrees initially.

    Args:
        state (TreeTensorNetworkState): The state of the system.
        left_out_id (str): The identifier fo whose node no caching is
            performed.

    Returns:
        List[str]: The path along which to update.
        Dict[str,str]: A dictionary with node_ids. If we compute at the
            key identifier, the legs of the cached tensor should point
            towards the value identifier node.

    """
    initial_path = state.find_path_to_root(left_out_id)
    initial_path.reverse()
    caching_path = []
    next_id_dict = {node_id: initial_path[i+1]
                    for i, node_id in enumerate(initial_path[:-1])}
    for node_id in initial_path:
        _find_caching_path_rec(node_id, caching_path,
                                    next_id_dict, initial_path,
                                    state)
    return (caching_path, next_id_dict)

def _find_caching_path_rec(node_id: str,
                            caching_path: List[str],
                            next_id_dict: Dict[str,str],
                            initial_path: List[str],
                            state: TreeTensorNetworkState) -> None:
    """
    Runs through the subranch starting at node_id and adds the the branch
    to the path starting with the leafs.

    Args:
        node_id (str): The identifier of the current node.
        caching_path (List[str]): The list in which the path is saved.
        Dict[str,str]: A dictionary with node_ids. If we compute at the
            key identifier, the legs of the cached tensor should point
            towards the value identifier node.
        initial_path (List[str]): The path to the root.
        state (TreeTensorNetworkState): The state of the system.

    """
    node = state.nodes[node_id]
    new_children = [node_id for node_id in node.children
                    if node_id not in initial_path]
    for child_id in new_children:
        _find_caching_path_rec(child_id, caching_path,
                                    next_id_dict, initial_path,
                                    state)
    if node_id not in next_id_dict and node_id != initial_path[-1]:
        # The root can never appear here, since it already appeared before
        assert node.parent is not None
        next_id_dict[node_id] = node.parent
    caching_path.append(node_id)
