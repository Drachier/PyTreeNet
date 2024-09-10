"""
Implementation of the first order one site TDVP algorithm.
"""
from ...util.tensor_splitting import SplitMode
from .onesitetdvp import OneSiteTDVP

class FirstOrderOneSiteTDVP(OneSiteTDVP):
    """
    The first order one site TDVP algorithm.

    This means we have first order Trotter splitting for the time evolution:
      exp(At+Bt) approx exp(At)*exp(Bt)
    Has the same attributes as the TDVP-Algorithm class
    """
    def _assert_orth_center(self, node_id: str):
        errstr = f"Node {node_id} is not the orthogonality center! It should be!"
        assert self.state.orthogonality_center_id == node_id, errstr

    def _assert_leaf_node(self, node_id: str):
        errstr = f"Node {node_id} is not a leaf! It should be!"
        assert self.state.nodes[node_id].is_leaf(), errstr

    def _update_site_and_link(self, node_id: str, update_index: int):
        """
        Updates the site and the link to the next site in the
        orthogonalization path.

        Args:
            node_id (str): The id of the node to update.
            update_index (int): The index of the update in the
                orthogonalization
        """
        assert update_index < len(self.orthogonalization_path)
        self._update_site(node_id)
        next_node_id = self.orthogonalization_path[update_index][0]
        self._update_link(node_id, next_node_id)

    def _first_update(self, node_id: str):
        """
        Updates the first site in the orthogonalization path.

        Here we do not have to move the orthogonality center, since it is
        already at the correct place. We only have to update the site and the
        link to the next site.
        """
        self._assert_orth_center(node_id)
        self._assert_leaf_node(node_id)
        self._update_site_and_link(node_id, 0)

    def _normal_update(self, node_id: str, update_index: int):
        """
        Updates a site in the middle of the orthogonalization path.

        This means the orthogonality center is moved to the correct place and
        the site as well as the link to the next site are updated.

        Args:
            node_id (str): The id of the node to update.
            update_index (int): The index of the update in the
                orthogonalization path.
        """
        current_orth_path = self.orthogonalization_path[update_index-1]
        self._move_orth_and_update_cache_for_path(current_orth_path)
        self._update_site_and_link(node_id, update_index)

    def _reset_for_next_time_step(self):
        """
        Resets the state of the algorithm for the next time step by correctly
        orthogonalising it and updating the partial tree tensor cache.
        """
        # Orthogonalise for next time step
        self.state.move_orthogonalization_center(self.update_path[0],
                                                 mode = SplitMode.KEEP)
        # We have to recache all partial tree tensors
        self.partial_tree_cache = self._init_partial_tree_cache()

    def _final_update(self, node_id: str):
        """
        Updates the last site in the orthogonalization path.

        Here we have to move the orthogonality center to the correct place and
        update the site. We do not have to update the link to the next site.
        """
        if len(self.orthogonalization_path) > 0: # Not for the special case of one node
            current_orth_path = self.orthogonalization_path[-1]
            self._move_orth_and_update_cache_for_path(current_orth_path)
        if len(self.state.nodes) > 2: # Not for the special case of two nodes
            self._assert_leaf_node(node_id) # The final site to be updated should be a leaf node
        self._update_site(node_id)
        self._reset_for_next_time_step()

    def run_one_time_step(self):
        """
        Runs one time step of the first order one site TDVP algorithm.

        This means we do one sweep through the tree, updating each site once.
        """
        for i, node_id in enumerate(self.update_path):
            if i == len(self.update_path)-1:
                self._final_update(node_id)
            elif i == 0:
                self._first_update(node_id)
            else:
                self._normal_update(node_id, i)
