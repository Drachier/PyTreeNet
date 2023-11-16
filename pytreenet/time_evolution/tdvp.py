"""
Implements the time-dependent variational principle (TDVP) for tree tensor
networks.

Reference:
    [1] D. Bauernfeind, M. Aichhorn; "Time Dependent Variational Principle for Tree
        Tensor Networks", DOI: 10.21468/SciPostPhys.8.2.024
"""
from __future__ import annotations
from typing import Union, List, Tuple, Dict
from copy import deepcopy

import numpy as np

from ..leg_specification import LegSpecification
from .time_evolution import TimeEvolution, time_evolve
from ..tensor_util import tensor_matricization
from ..ttns import TreeTensorNetworkState
from ..ttno.ttno import TTNO
from ..operators.tensorproduct import TensorProduct
from ..ttn_exceptions import NoConnectionException
from .tdvp_util.partial_tree_cache import PartialTreeCache
from .tdvp_util.tree_chach_dict import PartialTreeChachDict

class TDVPAlgorithm(TimeEvolution):
    """
    The general abstract class of a TDVP algorithm. Subclasses the general
     evolution.
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO,
                 time_step_size: float, final_time: float,
                 operators: Union[TensorProduct, List[TensorProduct]]) -> None:
        """
        Initilises an instance of a TDVP algorithm.

        Args:
            intial_state (TreeTensorNetworkState): The initial state of the
             system.
            hamiltonian (TTNO): The Hamiltonian in TTNO form under which to
             time-evolve the system.
            time_step_size (float): The size of one time-step.
            final_time (float): The final time until which to run the evolution.
            operators (Union[TensorProduct, List[TensorProduct]]): Operators
             to be measured during the time-evolution.
        """
        assert len(initial_state.nodes) == len(hamiltonian.nodes)
        self.hamiltonian = hamiltonian
        super().__init__(initial_state, time_step_size, final_time, operators)
        self.update_path = self._find_tdvp_update_path()
        self.orthogonalization_path = self._find_tdvp_orthogonalization_path(self.update_path)
        self._orthogonalize_init()

        # Caching for speed up
        self.partial_tree_cache = PartialTreeChachDict()
        self._init_partial_tree_cache()
        self._cached_distances = {node_id: self.state.distance_to_node(node_id)
                                       for node_id in self.state.nodes}
        self.neighbouring_nodes = {node_id: deepcopy(self.state.nodes[node_id].neighbouring_nodes())
                                    for node_id in self.state.nodes}

    def _find_tdvp_update_path(self) -> List[str]:
        """
        Returns a list of all nodes - ordered so that a TDVP update along
        this path requires a minimal amount of orthogonalizations.
        """
        # Start with leaf furthest from root.
        assert self.state.root_id is not None
        distances_from_root = self.state.distance_to_node(self.state.root_id)
        start = max(distances_from_root, key=distances_from_root.get)
        if len(self.state.nodes.keys()) < 2:
            update_path = [start]
        else:
            # Move from start to root.
            # Start might not be exactly start, but another leaf with same distance.
            sub_path = self._find_tdvp_path_from_leaves_to_root(start)
            update_path = [] + sub_path
            branch_roots = [x for x in self.state.root[0].children
                            if x not in update_path]
            sub_paths = []
            for branch_root in branch_roots:
                sp = self._find_tdvp_path_from_leaves_to_root(branch_root)
                sp.reverse()
                sub_paths.append(sp)
            sub_paths.sort(key=lambda x: -len(x))
            if len(sub_paths) == 0:
                update_path += [self.state.root_id]
            for i, sub_path in enumerate(sub_paths):
                if i == len(sub_paths) - 1:
                    update_path += [self.state.root_id]
                update_path += sub_path
        return update_path

    def _find_tdvp_path_from_leaves_to_root(self,
                                            any_child: str) -> List[str]:
        """
        Finds the path from a leaf to the root.
        """
        path_from_child_to_root = self.state.find_path_to_root(any_child)
        branch_origin = path_from_child_to_root[-2]
        path = self._find_tdvp_path_for_branch(branch_origin, [])
        return path

    def _find_tdvp_path_for_branch(self, branch_origin: str,
                                   path: Union[None, List[str]]=None) -> List[str]:
        """
        Finds a path for a branch, i.e. a subtree starting from a node in a
         path

        Args:
            branch_origin (str): The origin of the new branch
            path (Union[None, List[str]], optional): The path already found.
             Defaults to None.

        Returns:
            List[str]: A path of node identifiers.
        """
        if path is None:
            path = []
        node_id = branch_origin
        children = self.state.nodes[node_id].children
        for child in children:
            path = self._find_tdvp_path_for_branch(child, path)
        path.append(node_id)
        return path

    def _orthogonalize_init(self, force_new: bool=False):
        """
        Orthogonalises the state to the start of the tdvp update path.
         If the state is already orthogonalised, the orthogonalisation center
         is moved to the start of the update path.

        Args:
            force_new (bool, optional): If True a complete orthogonalisation
             is always enforced, instead of moving the orthogonality center.
             Defaults to False.
        """
        if self.state.orthogonality_center_id is None or force_new:
            self.state.orthogonalize(self.update_path[0])
        else:
            self.state.move_orthogonalization_center(self.update_path[0])

    def _find_tdvp_orthogonalization_path(self,
                                          update_path: List[str]) -> List[str]:
        """
        The path along which to orthogonalise during the tdvp algorithm.

        Args:
            update_path (List[str]): The path along which tdvp updates sites.

        Returns:
            List[str]: _description_
        """
        orthogonalization_path = []
        for i in range(len(update_path)-1):
            sub_path = self.state.path_from_to(update_path[i], update_path[i+1])
            orthogonalization_path.append(sub_path[1::])
        return orthogonalization_path

    def _find_caching_path(self) -> Tuple[List[str], Dict[str,str]]:
        """
        Finds the path used to cache the contracted subtrees initially.

        Returns:
            List[str]: The path along which to update.
            Dict[str,str]: A dictionary with node_ids. If we compute at the
             key identifier, the legs of the cached tensor should point
             towards the value identifier node.
        """
        initial_path = self.state.find_path_to_root(self.update_path[0])
        initial_path.reverse()
        caching_path = []
        next_id_dict = {node_id: initial_path[i+1]
                        for i, node_id in enumerate(initial_path[:-1])}
        for node_id in initial_path:
            self._find_caching_path_rec(node_id, caching_path,
                                        next_id_dict, initial_path)
        return (caching_path, next_id_dict)

    def _find_caching_path_rec(self, node_id: str,
                               caching_path: List[str],
                               next_id_dict: Dict[str,str],
                               initial_path: List[str]):
        """
        Runs through the subranch starting at node_id and adds the the branch
         to the path starting with the leafs.

        Args:
            node_id (str): The identifier of the current node.
            caching_path (List[str]): The list in which the path is saved.
            Dict[str,str]: A dictionary with node_ids. If we compute at the
             key identifier, the legs of the cached tensor should point
             towards the value identifier node.
        """
        node = self.state.nodes[node_id]
        new_children = [node_id for node_id in node.children
                        if node_id not in initial_path]
        for child_id in new_children:
            self._find_caching_path_rec(child_id, caching_path,
                                        next_id_dict, initial_path)
        if node_id not in next_id_dict and node_id != initial_path[-1]:
            # The root can never appear here, since it already appeared before
            assert node.parent is not None
            next_id_dict[node_id] = node.parent
        caching_path.append(node_id)

    def _init_partial_tree_cache(self):
        """
        Initialises the caching for the partial trees. 
         This means all the partial trees that are not the starting node of
         the tdvp path are cached.
        """
        rev_update_path, next_node_id_dict = self._find_caching_path()
        for node_id in rev_update_path[:-1]:
            next_node_id = next_node_id_dict[node_id]
            self.update_tree_cache(node_id, next_node_id)

    def update_tree_cache(self, node_id: str, next_node_id: str):
        """
        Updates the tree cache tensor that ends in the node with
         identifier `node_id` and has open legs pointing towards
         the neighbour node with identifier `next_node_id`.

        Args:
            node_id (str): The identifier of the node to which this cache
             corresponds.
            next_node_id (str): The identifier of the node to which the open
             legs of the tensor point.
        """
        new_cache = PartialTreeCache.for_all_nodes(node_id, next_node_id,
                                                   self.state,
                                                   self.hamiltonian,
                                                   self.partial_tree_cache)
        self.partial_tree_cache.add_entry(node_id, next_node_id, new_cache)

    def _contract_all_except_node(self,
                                  target_node_id: str) -> np.ndarray:
        """
        Uses the cached trees to contract the bra, ket, and hamiltonian
         tensors for all nodes in the trees apart from the given target node.
         All the resulting tensors are contracted to the hamiltonian tensor
         corresponding to the target node.

        Args:
            target_node_id (str): The node which is not to be part of the
             contraction.
        
        Returns:
            np.ndarray: The tensor resulting from the contraction:
                 _____       out         _____
                |     |____n-1    0_____|     |
                |     |                 |     |
                |     |        |n       |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|      |_____|     |
                |     |    |   H  |     |     |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |        |2n+1    |     |
                |     |                 |     |
                |     |_____       _____|     |
                |_____|  2n         n+1 |_____|
                              in

                where n is the number of neighbours of the node.
        """
        target_node = self.state.nodes[target_node_id]
        neighbours = target_node.neighbouring_nodes()
        tensor = self.hamiltonian.tensors[target_node_id]
        for neighbour_id in neighbours:
            chached_tensor = self.partial_tree_cache.get_cached_tensor(neighbour_id,
                                                                       target_node_id)
            tensor = np.tensordot(tensor, chached_tensor,
                                  axes=(([0],[1])))
        # Transposing to have correct leg order
        axes = [i+1 for i in range(2,2*len(neighbours)+2,2)]
        axes.append(0)
        axes.extend(range(2,2*len(neighbours)+2,2))
        axes.append(1)
        tensor = np.transpose(tensor, axes=axes)
        return tensor

    def _get_effective_site_hamiltonian(self,
                                        node_id: str) -> np.ndarray:
        """
        Obtains the effective site Hamiltonian as defined in Ref. [1]
         Eq. (16a) as a matrix.

        Args:
            node_id (str): The node idea centered in the effective Hamiltonian

        Returns:
            np.ndarray: The effective site Hamiltonian
        """
        tensor = self._contract_all_except_node(node_id)
        output_legs = tuple(range(0,tensor.ndim//2))
        input_legs = tuple(range(tensor.ndim//2,tensor.ndim))
        return tensor_matricization(tensor, output_legs, input_legs,
                                    correctly_ordered=False)

    def _update_cache_after_split(self, node_id: str, next_node_id: str):
        """
        Updates the cached tensor after splitting a tensor at node_id
         towards next_node_id.

        Args:
            node_id (str): Node to update
            next_node_id (str): Next node to which the link is found
        """
        link_id = self.create_link_id(node_id, next_node_id)
        new_cache = PartialTreeCache.for_all_nodes(node_id, link_id,
                                                   self.state,
                                                   self.hamiltonian,
                                                   self.partial_tree_cache)
        new_cache.pointing_to_node = next_node_id
        self.partial_tree_cache.add_entry(node_id, next_node_id, new_cache)

    def _split_updated_site(self,
                            node_id: str,
                            next_node_id: str):
        """
        Splits the tensor at site node_id and obtains the tensor linking
         this node and the node of next_node_id from a QR decomposition.

        Args:
            node_id (str): Node to update
            next_node_id (str): Next node to which the link is found
        """
        node = self.state.nodes[node_id]
        if node.is_parent_of(next_node_id):
            q_children = deepcopy(node.children)
            q_children.remove(next_node_id)
            q_legs = LegSpecification(node.parent,
                                      q_children,
                                      node.open_legs)
            r_legs = LegSpecification(None, [next_node_id], [])
        elif node.is_child_of(next_node_id):
            q_legs = LegSpecification(None,
                                      deepcopy(node.children),
                                      node.open_legs)
            r_legs = LegSpecification(node.parent, [], [])
        else:
            errstr = f"Nodes {node_id} and {next_node_id} are not connected!"
            raise NoConnectionException(errstr)
        link_id = self.create_link_id(node_id, next_node_id)
        self.state.split_node_qr(node_id, q_legs, r_legs,
                                 q_identifier=node.identifier,
                                 r_identifier=link_id)
        self._update_cache_after_split(node_id, next_node_id)

    def _get_effective_link_hamiltonian(self, node_id: str,
                                        next_node_id: str) -> np.ndarray:
        """
        Obtains the effective link Hamiltonian as defined in Ref. [1]
         Eq. (16b) as a matrix.

        Args:
            node_id (str): The last node that was centered in the effective
             Hamiltonian.
            next_node_id (str): The next node to go to. The link for which
             this effective Hamiltonian is constructed is between the two
             nodes.

        Returns:
            np.ndarray: The effective link Hamiltonian
                 _____       out         _____
                |     |____1      0_____|     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |_________________|     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |_____       _____|     |
                |_____|  2         3    |_____|
                              in
        """
        link_id = self.create_link_id(node_id, next_node_id)
        target_node = self.state.nodes[link_id]
        assert not target_node.is_root()
        assert len(target_node.children) == 1
        new_cache_tensor = self.partial_tree_cache.get_cached_tensor(node_id, next_node_id)
        # We get the cached tensor of the other neighbour of the link
        other_cache_tensor = self.partial_tree_cache.get_cached_tensor(next_node_id, node_id)
        # Contract the Hamiltonian legs
        if target_node.is_parent_of(node_id):
            tensor = np.tensordot(other_cache_tensor,
                                  new_cache_tensor,
                                  axes=(1,1))
        else:
            tensor = np.tensordot(new_cache_tensor,
                                  other_cache_tensor,
                                  axes=(1,1))
        tensor = np.transpose(tensor, axes=[1,3,0,2])
        return tensor_matricization(tensor, (0,1), (2,3),
                                    correctly_ordered=True)

    def _update_site(self, node_id: str,
                     half_time_step: bool = False):
        """
        Updates a single site using the effective Hamiltonian for that site.

        Args:
            node_id (str): The identifier of the site to update.
            half_time_step (bool, optional): Use only a half the time step. 
             Defaults to False.
        """
        hamiltonian_eff_site = self._get_effective_site_hamiltonian(node_id)
        psi = self.state.tensors[node_id]
        if half_time_step is True:
            self.state.tensors[node_id] = time_evolve(psi,
                                                      hamiltonian_eff_site,
                                                      self.time_step_size / 2,
                                                      forward=True)
        else:
            self.state.tensors[node_id] = time_evolve(psi,
                                                      hamiltonian_eff_site,
                                                      self.time_step_size,
                                                      forward=True)

    def _update_link(self, node_id: str,
                     next_node_id: str,
                     half_time_step: bool = False):
        """
        Updates a link tensor between two nodes using the effective link
         Hamiltonian.

        Args:
            node_id (str): The node from which the link tensor originated.
            next_node_id (str): The other tensor the link connects to.
            half_time_step (bool, optional): Use only half a time step.
             Defaults to False.
        """
        assert self.state.orthogonality_center_id == node_id
        self._split_updated_site(node_id, next_node_id)
        link_id = self.create_link_id(node_id, next_node_id)
        link_tensor = self.state.tensors[link_id]
        hamiltonian_eff_link = self._get_effective_link_hamiltonian(node_id,
                                                                    next_node_id)
        if half_time_step is True:
            link_tensor = time_evolve(link_tensor,
                                      hamiltonian_eff_link,
                                      self.time_step_size / 2,
                                      forward=False)
        else:
            link_tensor = time_evolve(link_tensor,
                                      hamiltonian_eff_link,
                                      self.time_step_size,
                                      forward=False)
        self.state.contract_nodes(link_id, next_node_id,
                                  new_identifier=next_node_id)
        self.state.orthogonality_center_id = next_node_id
    


    @staticmethod
    def create_link_id(node_id: str, next_node_id: str) -> str:
        """
        Creates the identifier of a link node after a split happened.
        """
        return "link_" + node_id + "_with_" + next_node_id
