from __future__ import annotations
from typing import List, Union

from ...operators.tensorproduct import TensorProduct
from ...ttno.ttno_class import TTNO
from ...ttns import TreeTensorNetworkState
from ..ttn_time_evolution import TTNTimeEvolutionConfig
from tqdm import tqdm
from copy import deepcopy
from ...Lindblad.util import (adjust_ttn1_structure_to_ttn2,
                             normalize_ttn_Lindblad_X,
                             normalize_ttn_Lindblad_XX,
                             normalize_ttn_Lindblad_A,
                             normalize_ttn_Lindblad_11)
from .onesitetdvp import OneSiteTDVP
import copy
import numpy as np
import pytreenet as ptn

class SecondOrderOneSiteTDVP(OneSiteTDVP):
    """
    The first order one site TDVP algorithm.

    This means we have second order Trotter splitting for the time evolution:
      exp(At+Bt) approx exp(At/2)*exp(Bt/2)*exp(Bt/2)*exp(At/2)

    Has the same attributes as the TDVP-Algorithm clas with two additions.

    Attributes:
        backwards_update_path (List[str]): The update path that traverses
            backwards.
        backwards_orth_path (List[List[str]]): The orthogonalisation paths for
            the backwards run.
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO, time_step_size: float, final_time: float,
                 operators: Union[TensorProduct, List[TensorProduct]],
                 connections: List,
                 config: Union[TTNTimeEvolutionConfig,None] = None) -> None:
        """
        Initialize the second order one site TDVP algorithm.

        Args:
            initial_state (TreeTensorNetworkState): The initial state of the
                system.
            hamiltonian (TTNO): The Hamiltonian of the system.
            time_step_size (float): The time step size.
            final_time (float): The final time of the evolution.
            operators (Union[TensorProduct, List[TensorProduct]]): The operators
                for which the expectation values are calculated.
            config (Union[TTNTimeEvolutionConfig,None], optional): The time
                evolution configuration. Defaults to None.
        """
        super().__init__(initial_state, hamiltonian,
                         time_step_size, final_time, operators, connections,
                         config)
        self.backwards_update_path = self._init_second_order_update_path()
        self.backwards_orth_path = self._init_second_order_orth_path()

    def _init_second_order_update_path(self) -> List[str]:
        """
        Find the update path that traverses backwards.
        """
        return list(reversed(self.update_path))

    def _init_second_order_orth_path(self) -> List[List[str]]:
        """
        Find the orthogonalisation paths for the backwards run.
        """
        back_orthogonalization_path = []
        for i, node_id in enumerate(self.backwards_update_path[1:-1]):
            current_path = self.state.path_from_to(node_id,
                                                   self.backwards_update_path[i+2])
            current_path = current_path[:-1]
            back_orthogonalization_path.append(current_path)
        back_orthogonalization_path.append([self.backwards_update_path[-1]])
        return back_orthogonalization_path

    def _update_forward_site_and_link(self, node_id: str,
                                      next_node_id: str):
        """
        Run the forward update with half time step.

        First the site tensor is updated and then the link tensor.

        Args:
            node_id (str): The identifier of the site to be updated.
            next_node_id (str): The other node of the link to be updated.
        """
        assert self.state.orthogonality_center_id == node_id
        self._update_site(node_id,
                          time_step_factor=0.5)
        self._update_link(node_id, next_node_id,
                          time_step_factor=0.5)

    def forward_sweep(self):
        """
        Perform the forward sweep through the state.
        """
        for i, node_id in enumerate(self.update_path[:-1]):
            # Orthogonalize
            if i>0:
                self._move_orth_and_update_cache_for_path(self.orthogonalization_path[i-1])
            # Select Next Node
            next_node_id = self.orthogonalization_path[i][0]
            # Update
            self._update_forward_site_and_link(node_id, next_node_id)

    def _final_forward_update(self):
        """
        Perform the final forward update. 
        
        To save some computation, the update is performed with a full time
        step. Since the first update backwards occurs on the same node. We
        also do not need to update any link tensors.
        """
        node_id = self.update_path[-1]
        assert node_id == self.backwards_update_path[0]
        assert self.state.orthogonality_center_id == node_id
        self._update_site(node_id)

    def _update_first_backward_link(self):
        """
        Update the link between the first and second node in the backwards
        update path with a half time step.
        
        We have already updated the first site on the backwards update path
        and the link will always be next to it, so the orthogonality center
        is already at the correct position.
        """
        next_node_id = self.backwards_update_path[1]
        self._update_link(self.state.orthogonality_center_id,
                          next_node_id,
                          time_step_factor=0.5)

    def _normal_backward_update(self, node_id: str,
                                update_index: int):
        """
        The normal way to make a backwards update.
        
        First the site tensor is updated. Then the orthogonality center is
        moved, if needed. Finally the link tensor between the new
        orthogonality center and the next node is updated. 
        
        Args:
            node_id (str): The identifier of the site to be updated.
            update_index (int): The index of the update in the backwards
                update path.
        """
        assert self.state.orthogonality_center_id == node_id
        self._update_site(node_id, time_step_factor=0.5)
        new_orth_center = self.backwards_orth_path[update_index-1]
        self._move_orth_and_update_cache_for_path(new_orth_center)
        next_node_id = self.backwards_update_path[update_index+1]
        self._update_link(self.state.orthogonality_center_id,
                          next_node_id,
                          time_step_factor=0.5)

    def _final_backward_update(self):
        """
        Perform the final backward update.
        
        Since this is the last node that needs updating, no link update is
        required afterwards.
        """
        node_id = self.backwards_update_path[-1]
        assert self.state.orthogonality_center_id == node_id
        self._update_site(node_id, time_step_factor=0.5)

    def backward_sweep(self):
        """
        Perform the backward sweep through the state.
        """
        self._update_first_backward_link()
        for i, node_id in enumerate(self.backwards_update_path[1:-1]):
            self._normal_backward_update(node_id, i+1)
        self._final_backward_update()

    def run_one_time_step(self):
        """
        Run a single second order time step.
        
        This mean we run a full forward and a full backward sweep through the
        tree.
        """
        self._orthogonalize_init()
        self.adjust_to_initial_structure()
        self.partial_tree_cache = self._init_partial_tree_cache() 

        self.forward_sweep()
        self._final_forward_update()
        self.backward_sweep()
        self.adjust_to_initial_structure()

        orth_center_id_1 = self.state.root_id
        orth_center_id_2 = orth_center_id_1.replace('Site', 'Node')
        #self.state = normalize_ttn_Lindblad_XX(self.state , orth_center_id_1,orth_center_id_2)
        #self.state = normalize_ttn_Lindblad_1(self.state , orth_center_id_1 , orth_center_id_2, self.connections)
        #self.state = normalize_ttn_Lindblad_2(self.state ,  self.connections)
        self.adjust_to_initial_structure()

    def create_temp_copy(self):
        """
        Creates a temporary copy of the object with deep copies of state and cache.
        """
        temp_self = copy.copy(self)
        temp_self.state = copy.deepcopy(self.state)
        return temp_self

    def run_one_time_step_copy(self):
        # remove self._orthogonalize_init() and self.partial_tree_cache in TDVPAlgorithm
        temp_self = self.create_temp_copy()

        temp_self._orthogonalize_init()
        temp_self.adjust_to_initial_structure()
        temp_self.partial_tree_cache = temp_self._init_partial_tree_cache()

        temp_self.forward_sweep()
        temp_self._final_forward_update()
        temp_self.backward_sweep()
        temp_self.adjust_to_initial_structure()

        #orth_center_id_1 = self.state.root_id
        #orth_center_id_2 = orth_center_id_1.replace('Site', 'Node')
        #temp_self.state = normalize_ttn_Lindblad_X(temp_self.state , orth_center_id_1)
        #temp_self.state = normalize_ttn_Lindblad_XX(temp_self.state , orth_center_id_1 , orth_center_id_2)
        #temp_self.state = ptn.normalize_ttn_Lindblad_2(temp_self.state , temp_self.connections)
        self.state = temp_self.state
        self.adjust_to_initial_structure()

    def run_Lindblad(self, evaluation_time: Union[int,"inf"] = 1, filepath: str = "",
            pgbar: bool = True,):
        self._orthogonalize_init()
        self.partial_tree_cache = self._init_partial_tree_cache()
        self._init_results(evaluation_time)
        assert self._results is not None
        ket , bra = ptn.devectorize_pho(self.state ,self.connections)
        #ket , bra =  ptn.devectorize_pho_1d(self.state , self.connections , 5) 

        I = ptn.TTNO.Identity(ket)        
        for i in tqdm(range(self.num_time_steps + 1), disable=not pgbar):
            if i != 0:  # We also measure the initial expectation_values 
                self.run_one_time_step_copy()
            if evaluation_time != "inf" and i % evaluation_time == 0 and len(self._results) > 0:
                index = i // evaluation_time
                bra_ket = ptn.bra_ket(self.state , self.connections)
                print("BRAKET" , bra_ket)
                current_results = self.evaluate_operators_Lindblad()
                
                print(current_results[0])
                self._results[0:-1, index] = current_results
                # Save current time
                self._results[-1, index] = i*self.time_step_size
        if evaluation_time == "inf":
            current_results = self.evaluate_operators_Lindblad()
            self._results[0:-1, 0] = current_results
            self._results[-1, 0] = i*self.time_step_size
        if filepath != "":
            self.save_results_to_file(filepath)

    def reset_to_initial_state(self):
        """
        Resets the current state to the intial state
        """
        self.state = deepcopy(self._intital_state)   