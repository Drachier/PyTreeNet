from __future__ import annotations
from typing import Union, List, Tuple, Dict

import numpy as np

from ...util.tensor_util import tensor_matricisation_half
from ...util.tensor_splitting import SplitMode
from ...ttns import TreeTensorNetworkState
from ...ttno.ttno_class import TTNO
from ...time_evolution.time_evo_util.update_path import TDVPUpdatePathFinder
from ...operators.tensorproduct import TensorProduct
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...contractions.state_operator_contraction import contract_any
from .krylov import eigh_krylov

class DMRGAlgorithm():
    """
    The general abstract class of a DMRG algorithm.
    
    Attributes:
        initial_state (TreeTensorNetworkState): The initial state of the system.
        hamiltonian (TTNO): The Hamiltonian under which to time-evolve the system.
        num_sweeps (int): The number of sweeps to perform.
        max_bond_dim (int): The maximum bond dimension.
        tol (float): The tolerance for the SVD.
        max_iter (int): The maximum number of iterations.
        update_path (List[str]): The order in which the nodes are updated.
        orthogonalisation_path (List[List[str]]): The path along which the
            TTNS has to be orthogonalised between each node update.
        partial_tree_cache (PartialTreeCacheDict): A dictionary to hold
            already contracted subtrees of the TTNS.
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO,
                 num_sweeps: int,
                 max_bond_dim: int,
                 tol: float,
                 max_iter: int):
        """
        Initilises an instance of a DMRG algorithm.
        
        Args:
            initial_state (TreeTensorNetworkState): The initial state of the
                system.
            hamiltonian (TTNO): The Hamiltonian in TTNO form under which to
                time-evolve the system.
            num_sweeps (int): The number of sweeps to perform.
            max_bond_dim (int): The maximum bond dimension.
            tol (float): The tolerance for the SVD.
            max_iter (int): The maximum number of iterations.
        """
        assert len(initial_state.nodes) == len(hamiltonian.nodes)
        self.initial_state = initial_state
        self.hamiltonian = hamiltonian
        self.num_sweeps = num_sweeps
        self.max_bond_dim = max_bond_dim
        self.tol = tol
        self.max_iter = max_iter
        self.update_path = self._finds_update_path()
        self.orthogonalization_path = self._find_orthogonalization_path(self.update_path)
        self._orthogonalize_init()

        # Caching for speed up
        self.partial_tree_cache = PartialTreeCachDict()
        self._init_partial_tree_cache()

    def _orthogonalize_init(self, force_new: bool=False):
        """
        Orthogonalises the state to the start of the DMRG update path.
        
        If the state is already orthogonalised, the orthogonalisation center
        is moved to the start of the update path.

        Args:
            force_new (bool, optional): If True a complete orthogonalisation
                is always enforced, instead of moving the orthogonality center.
                Defaults to False.
        """
        if self.state.orthogonality_center_id is None or force_new:
            self.state.canonical_form(self.update_path[0],
                                      mode=SplitMode.KEEP)
        else:
            self.state.move_orthogonalization_center(self.update_path[0],
                                                     mode=SplitMode.KEEP)

    def _find_orthogonalization_path(self,
                                          update_path: List[str]) -> List[List[str]]:
        """
        Find the DMRG orthogonalisation path.

        Args:
            update_path (List[str]): The path along which dmrg updates sites.

        Returns:
            List[List[str]]: a list of paths, along which the TTNS should be
            orthogonalised between every node update.
        """
        orthogonalization_path = []
        for i in range(len(update_path)-1):
            sub_path = self.state.path_from_to(update_path[i], update_path[i+1])
            orthogonalization_path.append(sub_path[1::])
        return orthogonalization_path

    def _finds_update_path(self) -> List[str]:
        """
        Finds the update path for this DMRG Algorithm.

        Overwrite to create custom update paths for specific tree topologies.

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
        the dmrg path have the bra, ket, and hamiltonian tensor corresponding
        to this node contracted and saved into the cache dictionary.
        """
        rev_update_path, next_node_id_dict = self._find_caching_path()
        for node_id in rev_update_path[:-1]:
            next_node_id = next_node_id_dict[node_id]
            self.update_tree_cache(node_id, next_node_id)

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
        new_tensor = contract_any(node_id, next_node_id,
                                  self.state, self.hamiltonian,
                                  self.partial_tree_cache)
        self.partial_tree_cache.add_entry(node_id, next_node_id, new_tensor)

    def _find_tensor_leg_permutation(self, node_id: str) -> Tuple[int,...]:
        """
        Find the correct permutation to permute the effective hamiltonian
        tensor to fit with the state tensor legs.
        
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
        Contract bra, ket and hamiltonian for all but one node into that
        node's Hamiltonian tensor.

        Uses the cached trees to contract the bra, ket, and hamiltonian
        tensors for all nodes in the trees apart from the given target node.
        All the resulting tensors are contracted to the hamiltonian tensor
        corresponding to the target node.

        Args:
            target_node_id (str): The node which is not to be part of the
                contraction.
        
        Returns:
            np.ndarray: The tensor resulting from the contraction::
            
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
        
        ## Contract with the state tensor
        psi = self.state.tensors[target_node_id]
        tensor = np.tensordot(tensor, psi, axes=((0,1)))
        
        return tensor

    def _get_effective_site_hamiltonian(self,
                                        node_id: str) -> np.ndarray:
        """
        Obtains the effective site Hamiltonian as a matrix.

        Args:
            node_id (str): The node idea centered in the effective Hamiltonian

        Returns:
            np.ndarray: The effective site Hamiltonian
        """
        tensor = self._contract_all_except_node(node_id)
        return tensor_matricisation_half(tensor)
    
    def _min_eigh_krylov(self, node_id: str) -> np.ndarray:
        """
        Finds the lowest eigenpairs of the effective site Hamiltonian using
        a Krylov subspace method.

        Args:
            node_id (str): The node idea centered in the effective Hamiltonian

        Returns:
            np.ndarray: The lowest eigenpairs
        """
        hamiltonian_eff_site = self._get_effective_site_hamiltonian(node_id)
        eig_vals, eig_vecs = eigh_krylov(self.initial_state.tensors[node_id],
                                         hamiltonian_eff_site,
                                         self.max_iter, 1)
        return eig_vals, eig_vecs

    def sweep(self):
        """
        Performs a forward and backward sweep through the tree.
        """
        for node_id in self.update_path:
            eig_vals, eig_vecs = self._min_eigh_krylov(node_id)
            self.state.tensors[node_id] = eig_vecs
            self.state.orthogonality_center_id = node_id
            self._move_orth_and_update_cache_for_path(self.orthogonalization_path[i])

        for node_id in self.update_path[::-1]:
            eig_vals, eig_vecs = self._min_eigh_krylov(node_id)
            self.state.tensors[node_id] = eig_vecs
            self.state.orthogonality_center_id = node_id
            self._move_orth_and_update_cache_for_path(self.orthogonalization_path[i])
        return eig_vals
            
    def run(self):
        """
        Runs the DMRG algorithm.
        """
        es = None
        for i in range(self.num_sweeps):
            e = self.sweep()
            es.append(e)
        return es
    
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
                                                     mode=SplitMode.KEEP)
            previous_node_id = path[i] # +0, because path starts at 1.
            self.update_tree_cache(previous_node_id, node_id)
