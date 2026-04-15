"""
Implements the class for the first order two-site TDVP algorithm.
"""
from __future__ import annotations

from .twositetdvp import TwoSiteTDVP
from ...util.tensor_splitting import SplitMode

class FirstOrderTwoSiteTDVP(TwoSiteTDVP):

    def run_one_time_step(self, **kwargs):
        """
        Runs one time step of the first order two-site TDVP algorithm.

        Args:
            **kwargs: Additional keyword arguments to be passed to the
                time-evolution function. Refer to the documentation of
                `ptn.time_evolution.time_evolution` for further information.
        """
        for ind, node in enumerate(self.update_path[:-2]):
            next_node = self.orthogonalization_path[ind+1][0]
            self._update_two_site_nodes(node, next_node)
            self._single_site_backwards_update(next_node)
            self._move_orth_and_update_cache_for_path(self.orthogonalization_path[ind])
        # The last update includes both nodes in one go
        self._update_two_site_nodes(self.update_path[-2], self.update_path[-1])
        # The orthogonality center is now at the final node
        # Thus we need to move it back to the original position for the next time step
        self.state.move_orthogonalization_center(self.update_path[0],
                                                 mode=SplitMode.KEEP)
