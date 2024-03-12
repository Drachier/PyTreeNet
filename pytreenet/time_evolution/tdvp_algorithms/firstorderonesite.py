from typing import Union

from ...tensor_util import SplitMode
from ..tdvp import TDVPAlgorithm

class FirstOrderOneSiteTDVP(TDVPAlgorithm):
    """
    The first order one site TDVP algorithm.
     This means we have first order Trotter splitting for the time evolution:
      exp(At+Bt) approx exp(At)*exp(Bt)
    """
    def _assert_orth_center(self, node_id: str):
        errstr = f"Node {node_id} is not the orthogonality center! It should be!"
        assert self.state.orthogonality_center_id == node_id, errstr

    def _assert_leaf_node(self, node_id: str):
        errstr = f"Node {node_id} is not a leaf! It should be!"
        assert self.state.nodes[node_id].is_leaf(), errstr

    def _update_site_and_link(self, node_id: str, update_index: int):
        assert update_index < len(self.orthogonalization_path)
        self._update_site(node_id)
        next_node_id = self.orthogonalization_path[update_index][0]
        self._update_link(node_id, next_node_id)

    def _first_update(self, node_id: str):
        self._assert_orth_center(node_id)
        self._assert_leaf_node(node_id)
        self._update_site_and_link(node_id, 0)

    def _normal_update(self, node_id: str, update_index: int):
        current_orth_path = self.orthogonalization_path[update_index-1]
        self._move_orth_and_update_cache_for_path(current_orth_path)
        self._update_site_and_link(node_id, update_index)

    def _reset_for_next_time_step(self):
        # Orthogonalise for next time step
        self.state.move_orthogonalization_center(self.update_path[0],
                                                 mode = SplitMode.KEEP)
        # We have to recache all partial tree tensors
        self._init_partial_tree_cache()

    def _final_update(self, node_id: str):
        if len(self.orthogonalization_path) > 0: # Not for the special case of one node
            current_orth_path = self.orthogonalization_path[-1]
            self._move_orth_and_update_cache_for_path(current_orth_path)
        if len(self.state.nodes) > 2: # Not for the special case of two nodes
            self._assert_leaf_node(node_id) # The final site to be updated should be a leaf node
        self._update_site(node_id)
        self._reset_for_next_time_step()

    def run_one_time_step(self):
        for i, node_id in enumerate(self.update_path):
            if i == len(self.update_path)-1:
                self._final_update(node_id)
            elif i == 0:
                self._first_update(node_id)
            else:
                self._normal_update(node_id, i)
