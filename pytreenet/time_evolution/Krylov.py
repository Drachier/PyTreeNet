from __future__ import annotations
from typing import List, Union, Any, Dict

from copy import deepcopy
from math import modf

import numpy as np
from tqdm import tqdm
from ..util.tensor_util import tensor_matricisation_half, SplitMode
from ..ttns import TreeTensorNetworkState
from ..ttno.ttno import TTNO
from ..contractions.tree_cach_dict import PartialTreeCachDict
from ..contractions.state_operator_contraction import contract_any
from .time_evo_util.update_path import TDVPUpdatePathFinder
from ..ttns import TreeTensorNetworkState
from ..util.tensor_util import SplitMode




def apply_hamiltonian(psi: np.ndarray, hamiltonian: np.ndarray) -> np.ndarray:
    shape = psi.shape
    HC = hamiltonian @ psi.flatten()
    return np.reshape(HC, shape)



class Krylov1:

    def __init__(self, initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO,
                 **kwargs) -> None:
        
        assert len(initial_state.nodes) == len(hamiltonian.nodes)
        self.state = deepcopy(initial_state)
        self.initial_state = initial_state
        self.hamiltonian = hamiltonian

        self.update_path = self._finds_update_path()
        self.orthogonalization_path = self._find_tdvp_orthogonalization_path(self.update_path)
        self._orthogonalize_init()

        # Caching for speed up
        self.partial_tree_cache = PartialTreeCachDict()
        self._init_partial_tree_cache()

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
            self.state.canonical_form(self.update_path[0],
                                     max_bond_dim=np.inf , rel_tol=-np.inf, total_tol=-np.inf)
        else:
            self.state.move_orthogonalization_center(self.update_path[0],
                                                     max_bond_dim=np.inf , rel_tol=-np.inf, total_tol=-np.inf)

    def _find_tdvp_orthogonalization_path(self,
                                          update_path: List[str]) -> List[List[str]]:
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

    def _finds_update_path(self) -> List[str]:
        """
        Finds the update path for this TDVP Algorithm.
         Overwrite to create custom update paths.

        Returns:
            List[str]: The order in which the nodes in the TTN should be time
             evolved.
        """
        return TDVPUpdatePathFinder(self.initial_state).find_path()

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
        new_tensor = contract_any(node_id, next_node_id,
                                  self.state, self.hamiltonian,
                                  self.partial_tree_cache)
        self.partial_tree_cache.add_entry(node_id, next_node_id, new_tensor)

    def _find_tensor_leg_permutation(self, node_id: str) -> Tuple[int,...]:
        """
        After contracting all the cached tensors to the site Hamiltonian, the
         legs of the resulting tensor are in the order of the Hamiltonian TTNO.
         However, they need to be permuted to match the legs of the site's
         state tensor. Such that the two can be easily contracted.
        """
        state_node = self.state.nodes[node_id]
        hamiltonian_node = self.hamiltonian.nodes[node_id]
        permutation = []
        for neighbour_id in state_node.neighbouring_nodes():
            hamiltonian_index = hamiltonian_node.neighbour_index(neighbour_id)
            permutation.append(hamiltonian_index)
        output_legs = []
        input_legs = []
        for hamiltonian_index in permutation:
            output_legs.append(2*hamiltonian_index+3)
            input_legs.append(2*hamiltonian_index+2)
        output_legs.append(0)
        input_legs.append(1)
        output_legs.extend(input_legs)
        return tuple(output_legs)

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
        target_node = self.hamiltonian.nodes[target_node_id]
        neighbours = target_node.neighbouring_nodes()
        tensor = self.hamiltonian.tensors[target_node_id]
        for neighbour_id in neighbours:
            cached_tensor = self.partial_tree_cache.get_entry(neighbour_id,
                                                                target_node_id)
            tensor = np.tensordot(tensor, cached_tensor,
                                  axes=((0,1)))
        # Transposing to have correct leg order
        axes = self._find_tensor_leg_permutation(target_node_id)
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
        return tensor_matricisation_half(tensor)

    def _update_site(self, node_id: str):
        """
        Updates a single site using the effective Hamiltonian for that site.

        Args:
            node_id (str): The identifier of the site to update.
            time_step_factor (float, optional): A factor that should be
             multiplied with the internal time step size. Defaults to 1.
        """
        hamiltonian_eff_site = self._get_effective_site_hamiltonian(node_id)
        psi = self.state.tensors[node_id]
        self.state.tensors[node_id] = apply_hamiltonian(psi,hamiltonian_eff_site)

    def _move_orth_and_update_cache_for_path(self, path: List[str]):
        """
        Moves the orthogonalisation center and updates all required caches
         along a given path.

        Args:
            path (List[str]): The path to move from. Should start with the
             orth center and end at with the final node. If the path is empty
             or only the orth center nothing happens.
        """
        if len(path) == 0:
            return
        assert self.state.orthogonality_center_id == path[0]
        for i, node_id in enumerate(path[1:]):
            self.state.move_orthogonalization_center(node_id,
                                                     max_bond_dim=np.inf , rel_tol=-np.inf, total_tol=-np.inf)
            previous_node_id = path[i] # +0, because path starts at 1.
            self.update_tree_cache(previous_node_id, node_id)


class Krylov(Krylov1):
    """
    Args:
        initial_state (TreeTensorNetworkState): The initial state of the system.
        hamiltonian (TTNO): The Hamiltonian of the system.
    """

    def __init__(self,initial_state,hamiltonian) -> None:
        self.results = None
        super().__init__(initial_state,hamiltonian)

    def run(self, num_steps: int):
        """
            At each step, one site projector * hamiltonian(TTNO) is applied to the state(TTN).
            results = [initial_state, Krylov basis 1, state2, ..., Krylov basis num_steps]    
        """
        self.results = list(np.zeros(num_steps + 2))
        self.results[0] = self.initial_state
        for i in tqdm(range(num_steps+1)):
                self.apply_1sproj_H()
                self.results[i+1] = self.state
                # orthogonalize self.results[i+1] against self.results[i+1]
        self.results = self.results[:-1] 
                
    def _assert_orth_center(self, node_id: str):
        errstr = f"Node {node_id} is not the orthogonality center! It should be!"
        assert self.state.orthogonality_center_id == node_id, errstr

    def _update_site_and_move_orth(self, node_id: str, update_index: int):
        assert update_index < len(self.orthogonalization_path)
        self._update_site(node_id)
        next_node_id = self.orthogonalization_path[update_index][0]
        self._move_orth_and_update_cache_for_path([node_id, next_node_id])

    def _first_update(self, node_id: str):
        self._assert_orth_center(node_id)
        self._update_site_and_move_orth(node_id, 0)

    def _normal_update(self, node_id: str, update_index: int):
        current_orth_path = self.orthogonalization_path[update_index-1]
        self._move_orth_and_update_cache_for_path(current_orth_path)
        self._update_site_and_move_orth(node_id, update_index)

    def _reset_for_next_time_step(self):
        # Orthogonalise for next time step
        self.state.move_orthogonalization_center(self.update_path[0],
                                                 max_bond_dim=np.inf , rel_tol=-np.inf, total_tol=-np.inf)
        # We have to recache all partial tree tensors
        self._init_partial_tree_cache()

    def _final_update(self, node_id: str):
        if len(self.orthogonalization_path) > 0: # Not for the special case of one node
            current_orth_path = self.orthogonalization_path[-1]
            self._move_orth_and_update_cache_for_path(current_orth_path)
        self._update_site(node_id)
        self._reset_for_next_time_step()

    def apply_1sproj_H(self):
        for i, node_id in enumerate(self.update_path):
            if i == len(self.update_path)-1:
                self._final_update(node_id)
            elif i == 0:
                self._first_update(node_id)
            else:
                self._normal_update(node_id, i)     
        self.state = deepcopy(self.state)

                                        














class Krylovv:

    def __init__(self,initial_state,hamiltonian) -> None:
        self.results = None
        self.initial_state = initial_state
        self.hamiltonian = hamiltonian
        self.state = deepcopy(initial_state)


    def run(self, num_steps: int):
        self.results = list(np.zeros(num_steps + 1))
        self.results[0] = self.state
        for i in tqdm(range(num_steps)):
                self.apply_1sproj_H()
                self.results[i+1] = self.state
                # orthogonalize self.results[i+1] against self.results[i+1]

    def apply_1sproj_H(self):
        for node_id in self.state.nodes:
            n = self.state.nodes[node_id].nneighbours()
            tensor = np.tensordot(self.state.tensors[node_id], 
                                  self.hamiltonian.tensors[node_id], 
                                  axes=(-1, n+1))
        list1 = list(range(n))
        list2 = list(range(n, 2 * n))
        legs = list(zip(list1, list2))
        shape = []
        for legs in legs:
            shape.append(tensor.shape[legs[0]] * tensor.shape[legs[1]])
            shape.append(tensor.shape[-1])
        self.state.tensors[node_id] = np.reshape(tensor, shape) 
        self.state = deepcopy(self.state)

                                        