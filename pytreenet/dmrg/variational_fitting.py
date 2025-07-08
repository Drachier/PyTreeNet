from __future__ import annotations
from typing import List, Tuple, Dict

import numpy as np
from scipy.sparse.linalg import gmres as gmres
import scipy

from ..util.tensor_splitting import SplitMode, SVDParameters

from ..ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TTNO
from ..time_evolution.time_evo_util.update_path import TDVPUpdatePathFinder
from ..contractions.tree_cach_dict import PartialTreeCachDict

from ..contractions.sandwich_caching import SandwichCache
from ..contractions.effective_hamiltonians import get_effective_single_site_hamiltonian, contract_all_except_node

from ..operators.hamiltonian import Hamiltonian


class VariationalFitting():
    """
    The general abstract class of a Variational Fitting algorithm. It finds a TTNS y with a given rank that best approximates $y \approx \sum_i O_i x_i $
    """

    def __init__(self, O: List[TTNO],
                 x: List[TreeTensorNetworkState],
                 y: TreeTensorNetworkState,
                 num_sweeps: int,
                 max_iter: int,
                 svd_params: SVDParameters,
                 site: str,
                 residual_rank: int = 4):
        """
        Initilises an instance of a ALS algorithm.
        
        Args:
            O (List[TTNO]): The operators in TTNO form.
            x (List[TreeTensorNetworkState]): The TTNS to be approximated.
            y (TreeTensorNetworkState): The given right hand side TTNS.
            num_sweeps (int): The number of sweeps to perform.
            max_iter (int): The maximum number of iterations.
            svd_params (SVDParameters): The parameters for the SVD.
            site (str): one site or two site sweeps.
            residual_rank (int): The rank of the residual.
        """
        assert len(x) == len(O)
        for xi, oi in zip(x, O):
            assert len(xi.nodes) == len(oi.nodes) == len(y.nodes), "The nodes of the TTNS and the operators must be the same. But got %s and %s"%(len(xi.nodes), len(oi.nodes))
        self.x = x
        self.y = y
        self.O = O
        self.num_sweeps = num_sweeps
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
        self.partial_tree_cache = self._init_partial_tree_cache()
        self.partial_tree_cache_states = self._init_partial_tree_cache_states()

    def _orthogonalize_init(self, force_new: bool=False):
        """
        Orthogonalises the state to the start of the ALS update path.
        
        If the state is already orthogonalised, the orthogonalisation center
        is moved to the start of the update path.

        Args:
            force_new (bool, optional): If True a complete orthogonalisation
                is always enforced, instead of moving the orthogonality center.
                Defaults to False.
        """
        if self.y.orthogonality_center_id is None or force_new:
            self.y.canonical_form(self.update_path[0],
                                      mode=SplitMode.KEEP)
        for xi in self.x:
            xi.canonical_form(self.update_path[0],
                                      mode=SplitMode.KEEP)

    def _find_orthogonalization_path(self,
                                          update_path: List[str]) -> List[List[str]]:
        """
        Find the ALS orthogonalisation path.

        Args:
            update_path (List[str]): The path along which dmrg updates sites.

        Returns:
            List[List[str]]: a list of paths, along which the TTNS should be
            orthogonalised between every node update.
        """
        orthogonalization_path = []
        for i in range(len(update_path)-1):
            sub_path = self.y.path_from_to(update_path[i], update_path[i+1])
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
        return TDVPUpdatePathFinder(self.y).find_path()

    def _init_partial_tree_cache(self)->SandwichCache:
        """
        Initialises the caching for the partial trees. 
        
        This means all the partial trees that are not the starting node of
        the dmrg path have the bra, ket, and hamiltonian tensor corresponding
        to this node contracted and saved into the cache dictionary.
        """
        identity_ttno = TTNO.from_hamiltonian(Hamiltonian.identity_like(self.y), self.y)
        return SandwichCache.init_cache_but_one(self.y, identity_ttno,self.update_path[0])
    
    def _init_partial_tree_cache_states(self)->SandwichCache:
        """
        Initialises the caching for the partial trees. 
        
        This means all the partial trees that are not the starting node of
        the dmrg path have the bra, ket, and hamiltonian tensor corresponding
        to this node contracted and saved into the cache dictionary.
        """
        partial_tree_cache_states_list = []
        for xi, oi in zip(self.x, self.O):
            partial_tree_cache_states_list.append(SandwichCache.init_cache_but_one(xi, oi,
                                                self.update_path[0],self.y.conjugate()))
        return partial_tree_cache_states_list
        
    def _contract_two_ttns_except_node(self,
                                   node_id: str, state_idx: int, rho: bool=False) -> np.ndarray:
        """
        Contracts two TreeTensorNetworks.

        Args:
            node_id (str): The identifier of the node to exclude from the
                contraction.
            state_idx (int): The index of the state to contract.
            rho (bool, optional): If True, the rho tensor is returned, otherwise the state tensor is returned.
            
        Returns:
            np.ndarray: The resulting tensor. A and B are the tensors in ttn1 and
                ttn2, respectively, corresponding to the node with the identifier
                node_id. C aer the tensors in the dictionary corresponding to the
                subtrees going away from the node::

                 ______                 ______
                |      |____       ____|      |
                |      |               |      |  
                |      |       |       |      |
                |      |       |       |      |
                |      |     __|__     |      |
                |  C1  |____|     |____|  C2  |
                |      |    |  A  |    |      |
                |      |    |_____|    |      |
                |      |       |       |      |
                |      |       |       |      |
                |      |____      _____|      |
                |______|               |______|
        
        """
        state_node, state_tensor = self.x[state_idx][node_id]
        hamiltonian_node, hamiltonian_tensor = self.O[state_idx][node_id]
        heff = contract_all_except_node(state_node, hamiltonian_node, hamiltonian_tensor, self.partial_tree_cache_states[state_idx])
        # y = self.y.tensors[node_id]
        # print(heff.shape, state_tensor.shape)
        ax1 = range(len(state_tensor.shape),(len(heff.shape)))
        ax2 = range(len(state_tensor.shape))
        result = np.tensordot(heff, state_tensor, axes=(ax1,ax2))
        return result

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
        self.partial_tree_cache.update_tree_cache(node_id, next_node_id)

    def update_tree_cache_state(self, node_id: str, next_node_id: str):
        """
        Contracts two TreeTensorNetworks.

        Args:
            node_id (str): The identifier of the node to which this cache
                    corresponds.
            next_node_id (str): The identifier of the node to which the open
                    legs of the tensor point.   
        """
        for cache in self.partial_tree_cache_states:
            cache.bra_state = self.y.conjugate()
            cache.update_tree_cache(node_id, next_node_id)

    def _update_one_site(self, node_id: str) -> np.ndarray:
        """
        Finds the least squares solution of the effective site Hamiltonian using
        GMRES method.

        Args:
            node_id (str): The node idea centered in the effective Hamiltonian

        Returns:
            np.ndarray: The lowest eigenvalues
        """
        identity_ttno = TTNO.from_hamiltonian(Hamiltonian.identity_like(self.y), self.y)
        hamiltonian_eff_site = get_effective_single_site_hamiltonian(node_id, self.y, identity_ttno, self.partial_tree_cache)
        
        shape = self.y.tensors[node_id].shape
        
        kvec = np.zeros(shape, dtype=np.complex128)
        for state_idx in range(len(self.x)):
            kvec += self._contract_two_ttns_except_node(node_id, state_idx, rho=False)
        
        y,exit_code = scipy.sparse.linalg.gcrotmk(hamiltonian_eff_site, kvec.reshape(-1), 
                                                x0=self.y.tensors[node_id].reshape(-1),maxiter=self.max_iter)
        y_norm = np.linalg.norm(y)
        y=y/ y_norm
        self.y.replace_tensor(node_id, y.reshape(shape))
        l = np.einsum('i,ij,j->', y.conj(), hamiltonian_eff_site, y)
        residual = hamiltonian_eff_site@y - kvec.reshape(-1)/y_norm
        residual =residual.reshape(shape)
        residual_norm = np.linalg.norm(residual)
        return l
    
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
        # print("eig_vals",eig_vals)
        return eig_vals
    
    
    def sweep(self):
        if self.site == 'one-site':
            return self.sweep_one_site()
        elif self.site == 'two-site':
            raise NotImplementedError("Two site sweeps not implemented")
            
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
        # assert self.y.orthogonality_center_id == path[0]
        for i, node_id in enumerate(path[1:]):
            self.y.move_orthogonalization_center(node_id,
                                                     mode=SplitMode.KEEP)
            for xi in self.x:
                xi.move_orthogonalization_center(node_id,
                                                     mode=SplitMode.KEEP)
            previous_node_id = path[i] # +0, because path starts at 1.
            self.update_tree_cache(previous_node_id, node_id)
            self.update_tree_cache_state(previous_node_id, node_id)