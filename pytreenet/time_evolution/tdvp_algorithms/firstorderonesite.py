from typing import Union

from ..tdvp import TDVPAlgorithm

class FirstOrderOneSiteTDVP(TDVPAlgorithm):
    """
    The first order one site TDVP algorithm.
     This means we have first order Trotter splitting for the time evolution:
      exp(A*B) approx exp(A)*exp(B)
    """

    def _update(self, node_id: str,
                next_node_id: Union[str, None]):
        assert self.state.orthogonality_center_id == node_id
        self._update_site(node_id)
        if next_node_id is not None:
            self._update_link(node_id, next_node_id)

    def run_one_time_step(self):
        for i, node_id in enumerate(self.update_path):
            # Orthogonalize
            if i>0:
                self._move_orth_and_update_cache_for_path(self.orthogonalization_path[i-1])
            # Select Next Node
            if i+1 < len(self.update_path):
                next_node_id = self.orthogonalization_path[i][0]
            else:
                next_node_id = None
            # Update
            self._update(node_id, next_node_id)
