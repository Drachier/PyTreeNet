"""
Implements the class for the second order two-site TDVP algorithm.
"""
from typing import List, Union, Dict

from .twositetdvp import TwoSiteTDVP
from ..ttn_time_evolution import TTNTimeEvolutionConfig
from ...ttns.ttns import TreeTensorNetworkState
from ...ttno.ttno_class import TTNO
from ...operators.tensorproduct import TensorProduct
from ...util.tensor_splitting import SVDParameters

class SecondOrderTwoSiteTDVP(TwoSiteTDVP):

    def __init__(self, initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO,
                 time_step_size: float, final_time: float,
                 operators: Union[TensorProduct, List[TensorProduct]],
                 truncation_parameters: SVDParameters,
                 config: Union[TTNTimeEvolutionConfig,None] = None) -> None:
        """
        Initialises an instance of a second ordertwo-site TDVP algorithm.

        Args:
            initial_state (TreeTensorNetworkState): The initial state of the
                system.
            hamiltonian (TTNO): The Hamiltonian in TTNO form under which to
                time-evolve the system.
            time_step_size (float): The size of one time-step.
            final_time (float): The final time until which to run the
                evolution.
            operators (Union[TensorProduct, List[TensorProduct]]): Operators
                to be measured during the time-evolution.
            truncation_parameters (Dict): A dictionary containing the
                parameters used for truncation. The dictionary can define a
                maximum bond dimension ('maximum_bond_dim'), a relative
                tolerance ('rel_tol') and a total tolerance ('total_tol') to be
                used during the truncation. For details see the documentation
                of `tensor_util.truncate_singular_values`.
        """
        super().__init__(initial_state, hamiltonian,
                         time_step_size, final_time, operators,
                         truncation_parameters, config)
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
        for path in reversed(self.orthogonalization_path):
            back_orthogonalization_path.append(list(reversed(path)))
        return back_orthogonalization_path

    def complete_two_site_update(self,
                                 target_node_id: str,
                                 next_node_id: str):
        """
        Performs the forward evolution of both sites and the backwards
        evolution of the next node.

        Args:
            target_node_id (str): The identifier of the first site.
            next_node_id (str): The identifier of the second site.
        """
        self._update_two_site_nodes(target_node_id, next_node_id,
                                    time_step_factor=0.5)
        self._single_site_backwards_update(next_node_id,
                                           time_step_factor=0.5)

    def complete_two_site_forward_update(self,
                                         target_node_id: str,
                                         update_index: int):
        """
        Performs the forward evolution of both sites and the backwards
        evolution of the next node, while automatically determining the next
        node.

        Args:
            target_node_id (str): The identifier of the first site.
            update_index (int): The index of the update.
        """
        next_node_id = self.orthogonalization_path[update_index][0]
        assert self.state.orthogonality_center_id == target_node_id
        self.complete_two_site_update(target_node_id, next_node_id)

    def first_forward_update(self):
        """
        Perform the first forward update.
        """
        self.complete_two_site_forward_update(self.update_path[0],0)

    def normal_forward_update(self, node_id: str,
                              update_index: int):
        """
        Perform a normal forward update.

        Args:
            node_id (str): The identifier of the first site.
            next_node_id (str): The identifier of the second site.
            update_index (int): The index of the update.
        """
        current_orth_path = self.orthogonalization_path[update_index-1]
        self._move_orth_and_update_cache_for_path(current_orth_path)
        self.complete_two_site_forward_update(node_id, update_index)

    def final_forward_update(self):
        """
        Perform the final forward update.

        The final forward update is a special case, as it both last sites are
        updated at the same time.
        """
        self._update_two_site_nodes(self.update_path[-2],
                                    self.update_path[-1],
                                    time_step_factor=0.5)

    def forward_sweep(self):
        """
        Perform the full forward sweep through the state.
        """
        for i, node_id in enumerate(self.update_path[:-2]):
            if i == 0:
                self.first_forward_update()
            else:
                self.normal_forward_update(node_id, i)
        self.final_forward_update()

    def first_backwards_update(self):
        """
        Perform the first backwards update.

        Here the first two sites are updated, but no backwards time evolution
        of the second node is performed.
        """
        self._update_two_site_nodes(self.backwards_update_path[0],
                                    self.backwards_update_path[1],
                                    time_step_factor=0.5)

    def normal_backwards_update(self,
                                update_index: int):
        """
        Perform a normal backwards update.

        This means, moving the orthogonality center to the correct site,
        evolving it backwards in time, and then updating both tensors forward
        in time.

        Args:
            update_index (int): The index of the update.
        """
        current_orth_path = self.backwards_orth_path[update_index]
        self._move_orth_and_update_cache_for_path(current_orth_path)
        target_node_id = current_orth_path[-1]
        next_node_id = self.backwards_update_path[update_index+1]
        self._single_site_backwards_update(target_node_id,
                                           time_step_factor=0.5)
        self._update_two_site_nodes(target_node_id, next_node_id,
                                    time_step_factor=0.5)

    def backwards_sweep(self):
        """
        Performs the full backwards sweep through the state.
        """
        for i in range(len(self.backwards_update_path[:-1])):
            if i == 0:
                self.first_backwards_update()
            else:
                self.normal_backwards_update(i)

    def run_one_time_step(self, **kwargs):
        """
        Run one time step of the secondo order two-site TDVP algorithm.

        This means running a full forward sweep and a full backwards sweep.
        """
        self.forward_sweep()
        self.backwards_sweep()
