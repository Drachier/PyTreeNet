from __future__ import annotations
from typing import List, Union, Any, Dict

from copy import deepcopy
from math import modf

import numpy as np
from tqdm import tqdm
from ..util.tensor_util import tensor_matricisation_half
from ..ttns import TreeTensorNetworkState
from ..ttno.ttno import TTNO
from ..contractions.tree_cach_dict import PartialTreeCachDict
from ..contractions.state_operator_contraction import contract_any
from .time_evo_util.update_path import TDVPUpdatePathFinder
from ..util.tensor_splitting import SVDParameters
from ..time_evolution import TDVPAlgorithm



class Krylov(TDVPAlgorithm):

    def __init__(self, initial_state, hamiltonian, time_step_size, final_time, operators):
        super().__init__(initial_state, hamiltonian, time_step_size, final_time, operators)

    def _move_orth_and_update_cache_for_path_svd(self, path: List[str], SVDParameters):
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
            self.state.move_orthogonalization_center_svd(node_id, SVDParameters)
            previous_node_id = path[i] # +0, because path starts at 1.
            self.update_tree_cache(previous_node_id, node_id)

    def _apply_ham(self, node_id: str):
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

    def run(self, num_steps: int, SVDParameters = SVDParameters()):
        """
            At each step, one site projector * hamiltonian(TTNO) is applied to the state(TTN).
            results = [initial_state, Krylov basis 1, state2, ..., Krylov basis num_steps]    
        """
        results = list(np.zeros(num_steps + 2))
        results[0] = self.initial_state
        for i in tqdm(range(num_steps+1)):
                self.apply_1sproj_H(SVDParameters)
                results[i+1] = self.state
                # orthogonalize self.results[i+1] against self.results[i+1]
        results = results[:-1] 
        return results
                
    def _assert_orth_center(self, node_id: str):
        errstr = f"Node {node_id} is not the orthogonality center! It should be!"
        assert self.state.orthogonality_center_id == node_id, errstr

    def _update_site_and_move_orth(self, node_id: str, update_index: int, SVDParameters: SVDParameters):
        assert update_index < len(self.orthogonalization_path)
        self._apply_ham(node_id)
        next_node_id = self.orthogonalization_path[update_index][0]
        self._move_orth_and_update_cache_for_path_svd([node_id, next_node_id], SVDParameters)

    def _first_update(self, node_id: str, SVDParameters):
        self._assert_orth_center(node_id)
        self._update_site_and_move_orth(node_id, 0, SVDParameters)

    def _normal_update(self, node_id: str, update_index: int , SVDParameters: SVDParameters):
        current_orth_path = self.orthogonalization_path[update_index-1]
        self._move_orth_and_update_cache_for_path_svd(current_orth_path, SVDParameters)
        self._update_site_and_move_orth(node_id, update_index, SVDParameters)

    def _reset_for_next_time_step(self):
        # Orthogonalise for next time step
        self.state.move_orthogonalization_center_svd(self.update_path[0], SVDParameters())
        # We have to recache all partial tree tensors
        self._init_partial_tree_cache()

    def _final_update(self, node_id: str, SVDParameters: SVDParameters):
        if len(self.orthogonalization_path) > 0: # Not for the special case of one node
            current_orth_path = self.orthogonalization_path[-1]
            self._move_orth_and_update_cache_for_path_svd(current_orth_path,SVDParameters)
        self._update_site(node_id)
        self._reset_for_next_time_step()

    def apply_1sproj_H(self,SVDParameters):
        for i, node_id in enumerate(self.update_path):
            if i == len(self.update_path)-1:
                self._final_update(node_id,SVDParameters)
            elif i == 0:
                self._first_update(node_id,SVDParameters)
            else:
                self._normal_update(node_id,i,SVDParameters)     
        self.state = deepcopy(self.state)
                                        
       
def apply_hamiltonian(psi: np.ndarray, hamiltonian: np.ndarray) -> np.ndarray:
    shape = psi.shape
    HC = hamiltonian @ psi.flatten()
    return np.reshape(HC, shape)



class Krylov_second_method:

    def __init__(self,initial_state,hamiltonian) -> None:
        self.results = None
        self.initial_state = initial_state
        self.hamiltonian = hamiltonian


    def run(self, num_steps: int, SVDParameters: SVDParameters()):
        self.results = [deepcopy(self.initial_state)]
        ttn = deepcopy(self.initial_state)
        for i in tqdm(range(num_steps)):
                ttn = self.apply_H(ttn)
                ttn.complete_canonical_form(SVDParameters)
                self.results.append(ttn)
                # orthogonalize self.results[i+1] against self.results[i+1]

    def apply_H(self,ttn):
        state = deepcopy(ttn)
        for node_id in state.nodes:
            n = state.nodes[node_id].nneighbours()
            tensor = np.tensordot(state.tensors[node_id], 
                                  self.hamiltonian.tensors[node_id], 
                                  axes=(-1, n+1))
            legs = []
            list1 = list(range(n))
            list2 = list(range(n, 2 * n))
            legs = list(zip(list1, list2))
            shape = []
            for index in legs:
                shape.append(tensor.shape[index[0]] * tensor.shape[index[1]])
            shape.append(tensor.shape[-1])
            transpose = []
            for legs in legs:
                transpose.append(legs[0])
                transpose.append(legs[1])
            transpose.append(tensor.ndim-1)
            tensor = np.transpose(tensor, transpose)
            state.tensors[node_id] = np.reshape(tensor, shape)
            state.nodes[node_id].link_tensor(ttn.tensors[node_id])
        return state



