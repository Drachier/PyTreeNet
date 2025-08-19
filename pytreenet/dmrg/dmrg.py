from __future__ import annotations
from typing import List, Tuple, Dict

import numpy as np

from ..util.tensor_util import tensor_matricisation_half
from ..util.tensor_splitting import SplitMode, SVDParameters
from ..util.ttn_exceptions import NotCompatibleException
from ..ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TTNO
from ..time_evolution.time_evo_util.update_path import TDVPUpdatePathFinder
from ..contractions.tree_cach_dict import PartialTreeCachDict
from ..contractions.state_operator_contraction import contract_any
from ..contractions.contraction_util import contract_all_but_one_neighbour_block_to_hamiltonian

from ..contractions.sandwich_caching import SandwichCache
from ..contractions.effective_hamiltonians import get_effective_single_site_hamiltonian, get_effective_two_site_hamiltonian_nodes
try:
    from pyscf.lib import davidson as davidson
except ImportError:
    from .dmrg_utils import davidson

class DMRGAlgorithm():
    """
    The general abstract class of a DMRG algorithm.
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO,
                 num_sweeps: int,
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
            max_iter (int): The maximum number of iterations.
            svd_params (SVDParameters): The parameters for the SVD.
            site (str): one site or two site dmrg.
        """
        assert len(initial_state.nodes) == len(hamiltonian.nodes)
        self.state = initial_state
        self.hamiltonian = hamiltonian
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

    def _init_partial_tree_cache(self)->SandwichCache:
        """
        Initialises the caching for the partial trees. 
        
        This means all the partial trees that are not the starting node of
        the dmrg path have the bra, ket, and hamiltonian tensor corresponding
        to this node contracted and saved into the cache dictionary.
        """
        return SandwichCache.init_cache_but_one(self.state, self.hamiltonian,self.update_path[0])

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

    
    def _update_one_site(self, node_id: str) -> np.ndarray:
        """
        Finds the lowest eigenpairs of the effective site Hamiltonian using
        a Krylov subspace method.

        Args:
            node_id (str): The node idea centered in the effective Hamiltonian

        Returns:
            np.ndarray: The lowest eigenvalues
        """
        hamiltonian_eff_site = get_effective_single_site_hamiltonian(node_id, self.state, self.hamiltonian, self.partial_tree_cache)
        
        Afunc = lambda x:hamiltonian_eff_site@x
        eps = 1e-8  # Small constant to avoid division by zero
        diag = hamiltonian_eff_site.diagonal()
        max_diag = np.max(np.abs(diag))
        min_denom = max_diag * eps  # Scale eps with matrix size

        precond_davidson = lambda dx, e, x0: dx/np.maximum(np.abs(diag - e), min_denom) * np.sign(diag - e)
        
        shape = self.state.tensors[node_id].shape
        # l,v = davidson(hamiltonian_eff_site, [self.state.tensors[node_id].reshape(-1)],1,max_iter=self.max_iter)
        # self.state.tensors[node_id] = v[0].reshape(shape)
        # return l[0]
        l,v = davidson(Afunc, self.state.tensors[node_id].reshape(-1), precond=precond_davidson, nroots=1,max_cycle=max(50,int(hamiltonian_eff_site.shape[0]*0.01)))
        self.state.tensors[node_id] = v.reshape(shape)
        return l
    
    
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
        
        target_node, target_tensor = self.hamiltonian[target_node_id]
        next_node, next_tensor = self.hamiltonian[next_node_id]
        
        hamiltonian_eff_site = get_effective_two_site_hamiltonian_nodes(self.state.nodes[new_id], target_node, target_tensor, next_node, next_tensor, self.partial_tree_cache)
        
        Afunc = lambda x: hamiltonian_eff_site@x
        
        eps = 1e-8  # Small constant to avoid division by zero
        diag = hamiltonian_eff_site.diagonal()
        max_diag = np.max(np.abs(diag))
        min_denom = max_diag * eps  # Scale eps with matrix size

        precond_davidson = lambda dx, e, x0: dx/np.maximum(np.abs(diag - e), min_denom) * np.sign(diag - e)
        
        eig_vals, eig_vecs = davidson(Afunc, self.state.tensors[new_id].reshape(-1), precond=precond_davidson, nroots=1,max_cycle=self.max_iter)
        self.state.tensors[new_id] = eig_vecs.reshape(shape)

        # eig_vals, eig_vecs = davidson(hamiltonian_eff_site, [self.state.tensors[new_id].reshape(-1)],1, max_iter=self.max_iter)
        # self.state.tensors[new_id] = eig_vecs[0].reshape(shape)
        
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
        return eig_vals
    
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
        return eig_vals
    
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
