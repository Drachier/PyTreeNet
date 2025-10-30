from __future__ import annotations
from typing import List, Union
from copy import deepcopy
from enum import Enum

import numpy as np
import scipy.sparse.linalg as scsplinalg
import scipy.linalg as sclinalg

from ..util.tensor_splitting import SplitMode, SVDParameters
from ..ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TTNO
from ..time_evolution.time_evo_util.update_path import TDVPUpdatePathFinder
from ..core.truncation.svd_truncation import svd_truncation
from ..contractions.sandwich_caching import SandwichCache
from ..contractions.effective_hamiltonians import (get_effective_single_site_hamiltonian)
from ..contractions.contraction_util import get_equivalent_legs
from ..contractions.local_contr import (LocalContraction)
from ..operators.hamiltonian import Hamiltonian

class SiteUpdateMethod(Enum):
    """
    The available site update methods.
    """
    ONE_SITE = "one-site"
    TWO_SITE = "two-site"

class VariationalFitting():
    r"""
    The general abstract class of a Variational Fitting algorithm.

    It finds a TTNS y with a given rank that best approximates

    ..math:: 

        y \approx \sum_i O_i x_i.

    """

    def __init__(self,
                 O: List[TTNO] | TTNO,
                 x: List[TreeTensorNetworkState] | TreeTensorNetworkState,
                 y: TreeTensorNetworkState,
                 num_sweeps: int,
                 max_iter: int,
                 svd_params: SVDParameters,
                 site: str | SiteUpdateMethod = SiteUpdateMethod.ONE_SITE,
                 coeffs: Union[float, complex, List[float],
                               List[complex], None] = None,
                 residual_rank: int = 4,
                 dtype: np.dtype = np.float64):
        """
        Initilises an instance of a ALS algorithm.

        Args:
            O (List[TTNO]): The operators in TTNO form.
            x (List[TreeTensorNetworkState]): The TTNS to be approximated.
            y (TreeTensorNetworkState): The given right hand side TTNS.
            num_sweeps (int): The number of sweeps to perform.
            max_iter (int): The maximum number of iterations.
            svd_params (SVDParameters): The parameters for the SVD.
            site (str | SiteUpdateMethod): one site or two site sweeps. Defaults
                to one site.
            coeffs (Union[float, complex, List[float], List[complex], None]):
                The coefficients for the linear combination of the TTNS. If None
                all coefficients are set to 1. If a single float or complex
                number is provided, it is used for all TTNS. If a list is
                provided, it must have the same length as the TTNS.
            residual_rank (int): The rank of the residual.
            dtype (np.dtype): The dtype to use for the calculations.
        """
        if isinstance(x, TreeTensorNetworkState):
            self.x = [x]
        else:
            self.x = x
        if isinstance(O, TTNO):
            self.O = [O]
        else:
            self.O = O
        if len(self.x) != len(self.O):
            errstr = f"The number of TTNS and TTNO must be the same. But got {len(self.x)} and {len(self.O)}"
            raise ValueError(errstr)
        y = svd_truncation(deepcopy(y), svd_params)
        y.pad_bond_dimensions(svd_params.max_bond_dim)
        self.y = y
        self.num_sweeps = num_sweeps
        self.max_iter = max_iter
        if isinstance(site, str):
            self.site = SiteUpdateMethod(site)
        else:
            self.site = site
        self.residual_rank = residual_rank
        self.dtype = dtype
        if svd_params is None:
            self.svd_params = SVDParameters()
        else:
            self.svd_params = svd_params

        if coeffs is None:
            self.coeffs = [1]*len(self.x)
        elif isinstance(coeffs, list):
            if not len(coeffs) == len(self.x):
                errstr = f"The number of coefficients must be the same as the number of TTNS. But got {len(coeffs)} and {len(self.x)}"
                raise ValueError(errstr)
            self.coeffs = coeffs
        else:
            self.coeffs = [coeffs]*len(self.x)

        self.update_path = self._finds_update_path()
        self.orthogonalization_path = self._find_orthogonalization_path(
            self.update_path)
        self._orthogonalize_init()

        # Caching for speed up
        self.partial_tree_cache = self._init_partial_tree_cache()
        self.partial_tree_cache_states = self._init_partial_tree_cache_states()

    def get_tensor_cache(self,
                         indx: int | None = None
                         ) -> SandwichCache:
        """
        Returns the one specified tensor cache.

        Args:
            indx (int | None): The index of the tensor cache to return. If
                None, the y-state's cache is returned.
        
        Returns:
            SandwichCache: The requested tensor cache.
        """
        if indx is None:
            return self.partial_tree_cache
        else:
            return self.partial_tree_cache_states[indx]

    def get_result_state(self) -> TreeTensorNetworkState:
        """
        Returns the resulting TTNS after the variational fitting.

        Returns:
            TreeTensorNetworkState: The resulting TTNS.
        """
        return self.y

    def _orthogonalize_init(self):
        """
        Orthogonalises the state to the start of the ALS update path.

        If the state is already orthogonalised, the orthogonalisation center
        is moved to the start of the update path.
        """
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

    def _init_partial_tree_cache(self) -> SandwichCache:
        """
        Initialises the caching for the partial trees. 

        This means all the partial trees that are not the starting node of
        the dmrg path have the bra, ket, and hamiltonian tensor corresponding
        to this node contracted and saved into the cache dictionary.
        """
        self.identity_ttno = TTNO.from_hamiltonian(Hamiltonian.identity_like(
            self.y, dtype=self.dtype), self.y, dtype=self.dtype)
        return SandwichCache.init_cache_but_one(self.y, self.identity_ttno, self.update_path[0])

    def _init_partial_tree_cache_states(self) -> list[SandwichCache]:
        """
        Initialises the caching for the partial trees. 

        This means all the partial trees that are not the starting node of
        the dmrg path have the bra, ket, and hamiltonian tensor corresponding
        to this node contracted and saved into the cache dictionary.
        """
        partial_tree_cache_states_list = []
        for xi, oi in zip(self.x, self.O):
            new_cache = SandwichCache.init_cache_but_one(xi, oi,
                                                         self.update_path[0],
                                                         bra_state=self.y,
                                                         bra_state_conjugated=False)
            partial_tree_cache_states_list.append(new_cache)
        return partial_tree_cache_states_list

    def _obtain_effective_ket_tensor(self,
                                     node_id: str,
                                     state_idx: int
                                     ) -> np.ndarray:
        """
        Obtain the effective ket tensor for a given node and state.

        Args:
            node_id (str): The identifier of which to obtain the effective
                ket tensor.
            state_idx (int): The index of the state to consider.

        Returns:
            np.ndarray: The effective ket tensor::

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
                |      |     __|__     |      |
                |      |____|     |____|      |
                |______|    |_____|    |______|
        """
        nodes_tensors = [self.x[state_idx][node_id],
                         self.O[state_idx][node_id]]
        neighbour_order = self.y.nodes[node_id].neighbouring_nodes()
        loc_contr = LocalContraction(nodes_tensors,
                                     self.get_tensor_cache(state_idx),
                                     neighbour_order=neighbour_order)
        return loc_contr().astype(self.dtype)

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
        hamiltonian_eff_site = get_effective_single_site_hamiltonian(node_id,
                                                                     self.y,
                                                                     self.identity_ttno,
                                                                     self.get_tensor_cache())
        hamiltonian_eff_site = hamiltonian_eff_site.astype(self.dtype)
        shape = self.y.tensors[node_id].shape
        kvec = np.zeros(shape, dtype=self.dtype)
        for state_idx in range(len(self.x)):
            if self.dtype == np.float64:
                coeff = self.coeffs[state_idx].real
            else:
                coeff = self.coeffs[state_idx]
            kvec_idx = self._obtain_effective_ket_tensor(node_id,
                                                      state_idx)
            kvec_idx = coeff * kvec_idx
            kvec += kvec_idx
        if self.dtype == np.float64:
            yf, _ = scsplinalg.minres(hamiltonian_eff_site,
                                    kvec.reshape(-1),
                                    x0=self.y.tensors[node_id].reshape(-1),
                                    maxiter=self.max_iter)
        else:
            yf, _ = scsplinalg.gmres(hamiltonian_eff_site, kvec.reshape(-1),
                                   x0=self.y.tensors[node_id].reshape(-1),
                                   maxiter=self.max_iter)
        y_norm = np.linalg.norm(yf)
        yf = yf / y_norm

        l = np.einsum('i,ij,j->', yf.conj(), hamiltonian_eff_site, yf)
        if self.residual_rank > 0:
            residual = hamiltonian_eff_site@yf - kvec.reshape(-1)/y_norm
            residual = residual.reshape(shape)
            residual_norm = np.linalg.norm(residual)
            if residual_norm > 1e-2:
                if next_node_id is not None:
                    node = self.y.nodes[node_id]
                    neighbour_id = node.neighbour_index(next_node_id)
                    node_old = deepcopy(node)
                    self.y.pad_bond_dimension(next_node_id,
                                              node_id,
                                              shape[neighbour_id]+self.residual_rank)
                    node_new = self.y.nodes[node_id]
                    _, leg2 = get_equivalent_legs(node_new, node_old)
                    leg2.extend(node.open_legs)

                    residual = np.moveaxis(residual, neighbour_id, -1)
                    residual = np.reshape(residual, (-1, shape[neighbour_id]))
                    q, _, _ = sclinalg.qr(residual, pivoting=True)

                    if q.shape[-1] < self.residual_rank:
                        q = np.pad(q, ((0,0),(0,self.residual_rank - q.shape[-1])))
                    else:
                        q = q[:, :self.residual_rank]
                    residual = q.T
                    residual = np.moveaxis(residual, -1, neighbour_id)
                    residual_shape = list(shape)
                    residual_shape[neighbour_id] = self.residual_rank
                    residual_shape = tuple(residual_shape)

                    residual = np.reshape(residual, residual_shape)
                    tensor = np.concatenate((yf.reshape(shape), residual), axis=neighbour_id)
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
        self._update_one_site(node_id_i, self.orthogonalization_path[0][1])

        for i, node_id in enumerate(self.update_path[1:]):
            current_orth_path = self.orthogonalization_path[i]
            self._move_orth_and_update_cache_for_path(current_orth_path)

            if i != len(self.update_path)-2:
                next_node_id = self.orthogonalization_path[i+1][1]
                eig_vals = self._update_one_site(node_id, next_node_id)
            else:
                eig_vals = self._update_one_site(node_id)

        for i, node_id in enumerate(self.update_path[::-1][1:]):
            current_orth_path = self.orthogonalization_path[::-1][i]
            self._move_orth_and_update_cache_for_path(current_orth_path[::-1])
            eig_vals = self._update_one_site(node_id)
        return eig_vals

    def sweep(self):
        if self.site is SiteUpdateMethod.ONE_SITE:
            return self.sweep_one_site()
        if self.site is SiteUpdateMethod.TWO_SITE:
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
            previous_node_id = path[i]  # +0, because path starts at 1.
            self.update_tree_cache(previous_node_id, node_id)
            self.update_tree_cache_state(previous_node_id, node_id)
