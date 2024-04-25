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
from ..util.ttn_exceptions import NoConnectionException
from ..core.leg_specification import LegSpecification
from .tdvp_algorithms import OneSiteTDVP



class Krylov(TDVPAlgorithm):

    def __init__(self, initial_state, hamiltonian, time_step_size, final_time, operators):
        super().__init__(initial_state, hamiltonian, time_step_size, final_time, operators)
    def _get_effective_site_hamiltonian(self, node_id: str) -> np.ndarray:
        return super()._get_effective_site_hamiltonian(node_id)
    def _get_effective_link_hamiltonian(self, node_id: str, next_node_id: str) -> np.ndarray:
        return OneSiteTDVP._get_effective_link_hamiltonian(self,node_id, next_node_id)
    def _update_cache_after_split(self,node_id, next_node_id):
        return OneSiteTDVP._update_cache_after_split(self,node_id, next_node_id)
    def _split_updated_site(self,node_id: str, next_node_id: str): 
        return OneSiteTDVP._split_updated_site(self,node_id,next_node_id)


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
            self.state.canonical_form(self.update_path[0], SVDParameters(max_bond_dim = np.inf, rel_tol= -np.inf, total_tol=-np.inf))
        else:
            self.state.move_orthogonalization_center(self.update_path[0], SVDParameters(max_bond_dim = np.inf, rel_tol= -np.inf, total_tol=-np.inf))    



    def _apply_ham_site(self, node_id: str):

        hamiltonian_eff_site = self._get_effective_site_hamiltonian(node_id)
        psi = self.state.tensors[node_id]
        self.state.tensors[node_id] = apply_hamiltonian(psi,hamiltonian_eff_site)


    def _apply_ham_link(self, node_id: str ,next_node_id: str):

        assert self.state.orthogonality_center_id == node_id
        self._split_updated_site(node_id, next_node_id)
        self._update_link_tensor(node_id,next_node_id)
        link_id = self.create_link_id(node_id, next_node_id)
        self.state.contract_nodes(link_id, next_node_id,
                                  new_identifier=next_node_id)
        self.state.orthogonality_center_id = next_node_id


    def _update_link_tensor(self, node_id: str,
                                next_node_id: str):
        
        link_id = self.create_link_id(node_id, next_node_id)
        link_tensor = self.state.tensors[link_id]
        hamiltonian_eff_link = self._get_effective_link_hamiltonian(node_id,
                                                                      next_node_id)
        

        self.state.tensors[link_id] = self.state.tensors[link_id] - apply_hamiltonian(link_tensor,hamiltonian_eff_link)

    @staticmethod
    def create_link_id(node_id: str, next_node_id: str) -> str:
        """
        Creates the identifier of a link node after a split happened.
        """
        return "link_" + node_id + "_with_" + next_node_id
    

    def run(self, num_steps: int):
        """
            At each step, one site projector * hamiltonian(TTNO) is applied to the state(TTN).
            results = [initial_state, Krylov basis 1, state2, ..., Krylov basis num_steps]    
        """
        results = list(np.zeros(num_steps + 2))
        results[0] = self.initial_state
        for i in tqdm(range(num_steps+1)):
                self.apply_1sproj_H()
                results[i+1] = self.state
                # orthogonalize self.results[i+1] against self.results[i+1]
        results = results[:-1] 
        return results
    
    def _assert_orth_center(self, node_id: str):
        errstr = f"Node {node_id} is not the orthogonality center! It should be!"
        assert self.state.orthogonality_center_id == node_id, errstr

    def _assert_leaf_node(self, node_id: str):
        errstr = f"Node {node_id} is not a leaf! It should be!"
        assert self.state.nodes[node_id].is_leaf(), errstr

    def _update_site_and_link_2(self, node_id: str, update_index: int):
        assert update_index < len(self.orthogonalization_path)
        self._apply_ham_site(node_id)
        next_node_id = self.orthogonalization_path[update_index][0]
        self._apply_ham_link(node_id, next_node_id)

    def _first_update(self, node_id: str):
        self._assert_orth_center(node_id)
        self._assert_leaf_node(node_id)
        self._update_site_and_link_2(node_id, 0)

    def _normal_update(self, node_id: str, update_index: int):
        current_orth_path = self.orthogonalization_path[update_index-1]
        self._move_orth_and_update_cache_for_path(current_orth_path)
        self._update_site_and_link_2(node_id, update_index)

    def _reset_for_next_time_step(self):
        # Orthogonalise for next time step
        self.state.move_orthogonalization_center(self.update_path[0])
        # We have to recache all partial tree tensors
        self._init_partial_tree_cache()
        

    def apply_1sproj_H(self):
        for i, node_id in enumerate(self.update_path):
            if i == len(self.update_path)-1:
                self._apply_ham_site(node_id)
            elif i == 0:
                self.state.move_orthogonalization_center(self.update_path[0])
                self._init_partial_tree_cache()
                self._first_update(node_id)
            else:
                self._normal_update(node_id,i)  
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
