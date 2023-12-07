from __future__ import annotations
from typing import List, Union

from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.ttno.ttno import TTNO
from pytreenet.ttns import TreeTensorNetworkState

from ..tdvp import TDVPAlgorithm

class SecondOrderOneSiteTDVP(TDVPAlgorithm):
    """
    The first order one site TDVP algorithm.
     This means we have second order Trotter splitting for the time evolution:
      exp(At+Bt) approx exp(At/2)*exp(Bt/2)*exp(Bt/2)*exp(At/2)
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO, time_step_size: float, final_time: float,
                 operators: Union[TensorProduct, List[TensorProduct]]) -> None:
        super().__init__(initial_state, hamiltonian,
                         time_step_size, final_time, operators)
        self.backwards_update_path = self._init_second_order_update_path()
        self.backwards_orth_path = self._init_second_order_orth_path()

    def _init_second_order_update_path(self) -> List[str]:
        """
        Find the update path that traverses backwards.
        """
        return list(reversed(self.update_path))

    def _init_second_order_orth_path(self) -> List[str]:
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

    def _update_forward(self, node_id: str,
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
            if i+1 < len(self.update_path):
                next_node_id = self.orthogonalization_path[i][0]
                # Update
                self._update_forward(node_id, next_node_id)

    def _final_forward_update(self, node_id: str):
        """
        Perform the final forward update. To save some computation, the update
         is performed with a full time step. Since the first update backwards
         occurs on the same node.

        Args:
            node_id (str): The identifier of the site to be updated.
        """
        assert self.state.orthogonality_center_id == node_id
        assert self.update_path[-1] == self.backwards_update_path[0]
        self._update_site(node_id)

    def backward_sweep(self):
        """
        Perform the backward sweep through the state.
        """
        for i, node_id in enumerate(self.backwards_update_path):
            if i > 0: # We already updated the last site for second time in the forward pass.
                assert self.state.orthogonality_center_id == node_id
                self._update_site(node_id, time_step_factor=0.5)
                self._move_orth_and_update_cache_for_path(self.backwards_orth_path[i-1])
            if i < len(self.backwards_update_path) - 1:
                next_node_id = self.backwards_update_path[i+1]
                self._update_link(self.state.orthogonality_center_id,
                                  next_node_id,
                                  time_step_factor=0.5)

    def run_one_time_step(self):
        """
        Run a single second order time step. This mean we run a full forward
         and a full backward sweep through the tree.
        """
        self.forward_sweep()
        self._final_forward_update(self.update_path[-1])
        self.backward_sweep()
