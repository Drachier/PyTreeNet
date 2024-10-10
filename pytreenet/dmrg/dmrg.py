from __future__ import annotations
from typing import Union, List, Tuple, Dict

import numpy as np

from ..util.tensor_util import tensor_matricisation_half
from ..util.tensor_splitting import SplitMode, SVDParameters
from ..ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TTNO
from ..time_evolution.time_evo_util.update_path import TDVPUpdatePathFinder
from ..operators.tensorproduct import TensorProduct
from ..contractions.tree_cach_dict import PartialTreeCachDict
from ..contractions.state_operator_contraction import contract_any
from ..contractions.contraction_util import contract_all_but_one_neighbour_block_to_hamiltonian
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
                 max_iter: int,
                 svd_params: SVDParameters,
                 site: str):
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
            site (str): one site or two site dmrg.
        """
        assert len(initial_state.nodes) == len(hamiltonian.nodes)
        self.state = initial_state
        self.hamiltonian = hamiltonian
        self.num_sweeps = num_sweeps
        self.max_bond_dim = max_bond_dim
        self.tol = tol
        self.max_iter = max_iter
        self.site = site
        if svd_params is None:
            self.svd_params = SVDParameters()
        else:
            self.svd_params = svd_params
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
            orthogonalization_path.append(sub_path)
        return orthogonalization_path

    def _finds_update_path(self) -> List[str]:
        """
        Finds the update path for this DMRG Algorithm.

        Overwrite to create custom update paths for specific tree topologies.

        Returns:
            List[str]: The order in which the nodes in the TTN should be time
                evolved.
        """
        return TDVPUpdatePathFinder(self.state).find_path()

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
        
        return tensor
    
    def _find_block_leg_target_node(self,
                                    target_node_id: str,
                                    next_node_id: str,
                                    neighbour_id: str) -> int:
        """
        Determines the leg index of the input leg of the contracted subtree
        block on the effective hamiltonian tensor corresponding to a given
        neighbour of the target node.

        Args:
            target_node_id (str): The id of the target node.
            next_node_id (str): The id of the next node.
            neighbour_id (str): The id of the neighbour of the target node.
        
        Returns:
            int: The leg index of the input leg of the contracted subtree
                block on the effective hamiltonian tensor.
        """
        target_node = self.hamiltonian.nodes[target_node_id]
        index_next_node = target_node.neighbour_index(next_node_id)
        ham_neighbour_index = target_node.neighbour_index(neighbour_id)
        constant = int(ham_neighbour_index < index_next_node)
        return 2 * (ham_neighbour_index + constant)

    def _find_block_leg_next_node(self,
                                  target_node_id: str,
                                  next_node_id: str,
                                  neighbour_id: str) -> int:
        """
        Determines the leg index of the input leg of the contracted subtree
        bnlock on the effective hamiltonian tensor corresponding to a given
        eighbour of the next node.

        Args:
            target_node_id (str): The id of the target node.
            next_node_id (str): The id of the next node.
            neighbour_id (str): The id of the neighbour of the next node.
        
        Returns:
            int: The leg index of the input leg of the contracted subtree
                block on the effective hamiltonian tensor.
        """
        # Luckily the situation is pretty much the same so we can reuse most
        # of the code.
        leg_index_temp = self._find_block_leg_target_node(next_node_id,
                                                          target_node_id,
                                                          neighbour_id)
        target_node = self.hamiltonian.nodes[target_node_id]
        target_node_numn = target_node.nneighbours()
        return 2 * target_node_numn + leg_index_temp
    
    def _determine_two_site_leg_permutation(self,
                                            target_node_id: str,
                                            next_node_id: str) -> Tuple[int]:
        """
        Determine the permutation of the effective Hamiltonian tensor.
        
        This is the leg permutation required on the two-site effective
        Hamiltonian tensor to fit with the underlying TTNS, assuming
        the two sites have already been contracted in the TTNS.

        Args:
            target_node_id (str): Id of the main node on which the update is
                performed.
            next_node_id (str): The id of the second node.
        
        Returns:
            Tuple[int]: The permutation of the legs of the two-site effective
                Hamiltonian tensor.
        """
        neighbours_target = self.hamiltonian.nodes[target_node_id].neighbouring_nodes()
        neighbours_next = self.hamiltonian.nodes[next_node_id].neighbouring_nodes()
        two_site_id = self.create_two_site_id(target_node_id, next_node_id)
        neighbours_two_site = self.state.nodes[two_site_id].neighbouring_nodes()
        # Determine the permutation of the legs
        input_legs = []
        for neighbour_id in neighbours_two_site:
            if neighbour_id in neighbours_target:
                block_leg = self._find_block_leg_target_node(target_node_id,
                                                             next_node_id,
                                                             neighbour_id)
            elif neighbour_id in neighbours_next:
                block_leg = self._find_block_leg_next_node(target_node_id,
                                                           next_node_id,
                                                           neighbour_id)
            else:
                errstr = "The two-site Hamiltonian has a neighbour that is not a neighbour of the two sites."
                raise NotCompatibleException(errstr)
            input_legs.append(block_leg)
        output_legs = [leg + 1 for leg in input_legs]
        target_num_neighbours = self.hamiltonian.nodes[target_node_id].nneighbours()
        output_legs = output_legs + [0,2*target_num_neighbours] # physical legs
        input_legs = input_legs + [1,2*target_num_neighbours + 1] # physical legs
        # As in matrices, the output legs are first
        return tuple(output_legs + input_legs)

    
    def _contract_all_except_two_nodes(self,
                                       target_node_id: str,
                                       next_node_id: str) -> np.ndarray:
        """
        Contracts the nodes for all but two sites.

        Uses all cached tensors to contract bra, ket, and hamiltonian tensors
        of all nodes except for the two given nodes. Of these nodes only the
        Hamiltonian nodes are contracted.
        IMPORTANT: This function assumes that the given nodes are already
        contracted in the TTNS.

        Args:
            target_node_id (str): The id of the node that should not be
                contracted.
            next_node_id (str): The id of the second node that should not
                be contracted.

        Returns:
            np.ndarray: The resulting effective two-site Hamiltonian tensor::

                 _____                out              _____
                |     |____n-1                  0_____|     |
                |     |                               |     |
                |     |        |n           |         |     |
                |     |     ___|__        __|___      |     |
                |     |    |      |      |      |     |     |
                |     |____|      |______|      |_____|     |
                |     |    |  H_1 |      | H_2  |     |     |
                |     |    |______|      |______|     |     |
                |     |        |            |         |     |
                |     |        |2n+1        |         |     |
                |     |                               |     |
                |     |_____                     _____|     |
                |_____|  2n                       n+1 |_____|
                                      in
        
        """
        # Contract all but one neighbouring block of each node
        h_target = self.hamiltonian.tensors[target_node_id]
        target_node = self.hamiltonian.nodes[target_node_id]
        target_block = contract_all_but_one_neighbour_block_to_hamiltonian(h_target,
                                                                           target_node,
                                                                           next_node_id,
                                                                           self.partial_tree_cache)
        h_next = self.hamiltonian.tensors[next_node_id]
        next_node = self.hamiltonian.nodes[next_node_id]
        next_block = contract_all_but_one_neighbour_block_to_hamiltonian(h_next,
                                                                         next_node,
                                                                         target_node_id,
                                                                         self.partial_tree_cache)
        # Contract the two blocks
        h_eff = np.tensordot(target_block, next_block, axes=(0,0))
        # Now we need to sort the legs to fit with the underlying TTNS.
        # Note that we assume, the two tensors have already been contracted.
        leg_permutation = self._determine_two_site_leg_permutation(target_node_id,
                                                                   next_node_id)
        return h_eff.transpose(leg_permutation)

    
    def _update_one_site(self, node_id: str) -> np.ndarray:
        """
        Finds the lowest eigenpairs of the effective site Hamiltonian using
        a Krylov subspace method.

        Args:
            node_id (str): The node idea centered in the effective Hamiltonian

        Returns:
            np.ndarray: The lowest eigenvalues
        """
        hamiltonian_eff_site = tensor_matricisation_half(self._contract_all_except_node(node_id))
        Afunc = lambda x:hamiltonian_eff_site@x
        shape = self.state.nodes[node_id].shape
        eig_vals, eig_vecs = eigh_krylov(Afunc, self.state.tensors[node_id].reshape(-1),
                                         self.max_iter, 1)
        self.state.tensors[node_id] = eig_vecs.reshape(shape)
        return eig_vals
    
    def _update_two_site(self, target_node_id: str, next_node_id: str) -> np.ndarray:
        """
        Finds the lowest eigenpairs of the effective site Hamiltonian using
        a Krylov subspace method.

        Args:
            target_node_id (str): The current node in the effective Hamiltonian
            next_node_id (str): The next node in the effective Hamiltonian

        Returns:
            np.ndarray: The lowest eigenvalues   
        """
        
        u_legs, v_legs = self.state.legs_before_combination(target_node_id,
                                                            next_node_id)
        new_id = self.create_two_site_id(target_node_id, next_node_id)
        self.state.contract_nodes(target_node_id, next_node_id,
                                  new_identifier=new_id)
        shape = self.state.nodes[new_id].shape
        
        hamiltonian_eff_site = tensor_matricisation_half(self._contract_all_except_two_nodes(target_node_id, next_node_id))
        Afunc = lambda x: hamiltonian_eff_site@x
        
        eig_vals, eig_vecs = eigh_krylov(Afunc, self.state.tensors[new_id].reshape(-1),
                                         self.max_iter, 1)
        self.state.tensors[new_id] = eig_vecs.reshape(shape)
        
        self.state.split_node_svd(new_id, u_legs, v_legs,
                                  u_identifier=target_node_id,
                                  v_identifier=next_node_id,
                                  svd_params=self.svd_params)
        self.state.orthogonality_center_id = next_node_id
        self.update_tree_cache(target_node_id, next_node_id)
        
        return eig_vals
    
    @staticmethod
    def create_two_site_id(node_id: str, next_node_id: str) -> str:
        """
        Create the identifier of a two site node obtained from contracting
        the two note with the input identifiers.
        """
        return "TwoSite_" + node_id + "_contr_" + next_node_id
    
    def sweep_one_site(self):
        """
        Performs a forward and backward sweep through the tree.
        """
        node_id_i = self.update_path[0]
        self._update_one_site(node_id_i)
        
        for i,node_id in enumerate(self.update_path[1:]):
            current_orth_path = self.orthogonalization_path[i]
            self._move_orth_and_update_cache_for_path(current_orth_path)
            eig_vals = self._update_one_site(node_id)
        
        node_id_f = self.update_path[-1]
        self._update_one_site(node_id_f)
        
        for i,node_id in enumerate(self.update_path[::-1][1:]):
            current_orth_path = self.orthogonalization_path[::-1][i]
            self._move_orth_and_update_cache_for_path(current_orth_path[::-1])
            eig_vals = self._update_one_site(node_id)
        return eig_vals[0]
    
    def sweep_two_site(self):
        """
        Performs a forward and backward sweep through the tree.
        """
        
        for i,node_id in enumerate(self.update_path[:-1]):
            assert node_id == self.orthogonalization_path[i][0], 'node is wrong'
            for j,this_node_id in enumerate(self.orthogonalization_path[i][:-1]):
                next_node_id = self.orthogonalization_path[i][j+1]
                self._update_two_site(this_node_id, next_node_id)
                
        for i,node_id in enumerate(self.update_path[::-1][:-1]):
            assert node_id == self.orthogonalization_path[::-1][i][-1], 'node is wrong'
            
            for j, this_node_id in enumerate(self.orthogonalization_path[::-1][i][::-1][:-1]):
                next_node_id = self.orthogonalization_path[::-1][i][::-1][j+1]
                eig_vals = self._update_two_site(this_node_id, next_node_id)
                                
        return eig_vals[0]
    
    def sweep(self):
        if self.site == 'one-site':
            return self.sweep_one_site()
        elif self.site == 'two-site':
            return self.sweep_two_site()
            
    def run(self):
        """
        Runs the DMRG algorithm.
        """
        es = []
        for _ in range(self.num_sweeps):
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
