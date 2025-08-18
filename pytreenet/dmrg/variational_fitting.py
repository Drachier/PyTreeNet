from __future__ import annotations
from typing import List, Tuple, Dict, Union

import numpy as np
from scipy.sparse.linalg import gmres as gmres
import scipy
from copy import deepcopy
from ..util.tensor_splitting import SplitMode, SVDParameters

from ..ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TTNO
from ..time_evolution.time_evo_util.update_path import TDVPUpdatePathFinder
from ..contractions.tree_cach_dict import PartialTreeCachDict
from ..core.truncation.svd_truncation import svd_truncation
from ..contractions.sandwich_caching import SandwichCache
from ..contractions.effective_hamiltonians import get_effective_single_site_hamiltonian, contract_all_except_node
from ..contractions.contraction_util import get_equivalent_legs

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
                 coeffs: Union[float, complex, List[float], List[complex], None] = None,
                 residual_rank: int = 4,
                 dtype: np.dtype = np.complex128):
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
        y = svd_truncation(deepcopy(y), svd_params)
        y.pad_bond_dimensions(svd_params.max_bond_dim)
        self.y = y
        self.O = O
        self.num_sweeps = num_sweeps
        self.max_iter = max_iter
        self.site = site
        self.residual_rank = residual_rank
        self.dtype = dtype
        if svd_params is None:
            self.svd_params = SVDParameters()
        else:
            self.svd_params = svd_params
            
        if coeffs is None:
            self.coeffs = [1]*len(self.x)
        elif isinstance(coeffs, float) or isinstance(coeffs, complex):
            self.coeffs = [coeffs]*len(self.x)
        elif isinstance(coeffs, list) and isinstance(coeffs[0], float) or isinstance(coeffs[0], complex):
            assert len(coeffs) == len(self.x), "The number of coefficients must be the same as the number of TTNS"
            self.coeffs = coeffs
        else:
            raise ValueError("coeffs must be a float, complex, list of floats, or list of complexes")
        
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
        self.identity_ttno = TTNO.from_hamiltonian(Hamiltonian.identity_like(self.y, dtype = self.dtype), self.y,dtype=self.dtype)
        return SandwichCache.init_cache_but_one(self.y, self.identity_ttno,self.update_path[0])
    
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
        
        ttno_point = np.prod(self.O[state_idx].nodes[node_id]._shape[:-2])
        if ttno_point > 1:
            state_node, state_tensor = self.x[state_idx][node_id]
            hamiltonian_node, hamiltonian_tensor = self.O[state_idx][node_id]
            heff = contract_all_except_node(state_node, hamiltonian_node, hamiltonian_tensor, self.partial_tree_cache_states[state_idx])
            heff = heff.astype(self.dtype)
            # y = self.y.tensors[node_id]
            # print(heff.shape, state_tensor.shape)
            ax1 = range(len(state_tensor.shape),(len(heff.shape)))
            ax2 = range(len(state_tensor.shape))
            result = np.tensordot(heff, state_tensor, axes=(ax1,ax2))
            _, legs2 = get_equivalent_legs(self.y.nodes[node_id], state_node)
            legs2.append(-1)
            result = np.transpose(result, legs2)
            return result.astype(self.dtype)
        else:
            state_node, state_tensor = self.x[state_idx][node_id]
            ketblock_tensor = state_tensor
            for neighbour_id in state_node.neighbouring_nodes():
                cached_neighbour_tensor = self.partial_tree_cache_states[state_idx].get_entry(neighbour_id,state_node.identifier)
                ketblock_tensor = np.tensordot(ketblock_tensor, cached_neighbour_tensor,
                                    axes=([0],[0])) 
                ketblock_tensor = np.squeeze(ketblock_tensor,axis=-2)
            legs_block = []
            for neighbour_id in self.y.nodes[node_id].neighbouring_nodes():
                legs_block.append(state_node.neighbour_index(neighbour_id) + 1)
            # The kets physical leg is now the first leg
            legs_block.append(0)
            ketblock_tensor = np.transpose(ketblock_tensor, axes=legs_block)
            if rho:
                rho_tensor = np.tensordot(ketblock_tensor, ketblock_tensor.conj(),0)
                return rho_tensor.astype(self.dtype)
            else:
                return ketblock_tensor.astype(self.dtype)

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

    def _update_one_site(self, node_id: str, next_node_id: str = None) -> np.ndarray:
        """
        Finds the least squares solution of the effective site Hamiltonian using
        GMRES method.

        Args:
            node_id (str): The node idea centered in the effective Hamiltonian

        Returns:
            np.ndarray: The lowest eigenvalues
        """
        # identity_ttno = TTNO.from_hamiltonian(Hamiltonian.identity_like(self.y), self.y)
        hamiltonian_eff_site = get_effective_single_site_hamiltonian(node_id, self.y, self.identity_ttno, self.partial_tree_cache)
        hamiltonian_eff_site = hamiltonian_eff_site.astype(self.dtype)
        shape = self.y.tensors[node_id].shape
        kvec = np.zeros(shape, dtype=self.dtype)
        for state_idx in range(len(self.x)):
            if self.dtype == np.float64:
                coeffs = self.coeffs[state_idx].real
            else:
                coeffs = self.coeffs[state_idx]
            kvec += self._contract_two_ttns_except_node(node_id, state_idx, rho=False)*coeffs
        if self.dtype == np.float64:
            yf,exit_code = scipy.sparse.linalg.minres(hamiltonian_eff_site, kvec.reshape(-1), 
                                                    x0=self.y.tensors[node_id].reshape(-1),maxiter=self.max_iter)
        else:
            yf,exit_code = scipy.sparse.linalg.gmres(hamiltonian_eff_site, kvec.reshape(-1), 
                                                    x0=self.y.tensors[node_id].reshape(-1),maxiter=self.max_iter)
        y_norm = np.linalg.norm(yf)
        yf=yf/ y_norm

        l = np.einsum('i,ij,j->', yf.conj(), hamiltonian_eff_site, yf)
        if self.residual_rank>0:
            residual = hamiltonian_eff_site@yf - kvec.reshape(-1)/y_norm
            residual =residual.reshape(shape)
            residual_norm = np.linalg.norm(residual)
            if residual_norm > 1e-2:
                if next_node_id is not None:
                    neighbour_id = self.y.nodes[node_id].neighbour_index(next_node_id)
                    node_old = deepcopy(self.y.nodes[node_id])
                    self.y.pad_bond_dimension(next_node_id, node_id, shape[neighbour_id]+self.residual_rank)
                    node_new = self.y.nodes[node_id]
                    _, leg2 = get_equivalent_legs(node_new, node_old)
                    leg2.append(-1)
                    
                    residual = np.moveaxis(residual, neighbour_id, -1)
                    residual = np.reshape(residual, (-1, shape[neighbour_id]))
                    q,r,p = scipy.linalg.qr(residual, pivoting=True)
                    residual = (q[:, :self.residual_rank]).T 
                    residual = np.moveaxis(residual, -1, neighbour_id)
                    residual_shape = list(shape)
                
                    residual_shape[neighbour_id] = self.residual_rank
                    residual_shape = tuple(residual_shape)
                    residual = np.reshape(residual, residual_shape)
                    
                    tensor = np.concatenate((yf.reshape(shape),residual),axis=neighbour_id)
                    tensor = np.transpose(tensor, leg2)
                    self.y.replace_tensor(node_id, tensor)
            else:
                self.y.replace_tensor(node_id, yf.reshape(shape))
        return l
    
    def sweep_one_site(self):
        """
        Performs a forward and backward sweep through the tree.
        """
        node_id_i = self.update_path[0]
        self._update_one_site(node_id_i,self.orthogonalization_path[0][1])
        
        for i,node_id in enumerate(self.update_path[1:]):
            current_orth_path = self.orthogonalization_path[i]
            self._move_orth_and_update_cache_for_path(current_orth_path)
            
            if i != len(self.update_path)-2:
                next_node_id = self.orthogonalization_path[i+1][1]
                eig_vals = self._update_one_site(node_id,next_node_id)
            else:
                eig_vals = self._update_one_site(node_id)

        
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