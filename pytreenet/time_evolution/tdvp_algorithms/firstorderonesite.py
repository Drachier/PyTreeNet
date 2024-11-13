"""
Implementation of the first order one site TDVP algorithm.
"""
from typing import Union
from ...util.tensor_splitting import SplitMode
from .onesitetdvp import OneSiteTDVP
from copy import deepcopy
from pytreenet.time_evolution.Subspace_expansion import expand_subspace 
from ...util.tensor_splitting import SplitMode


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
        #if len(self.state.nodes) > 2: # Not for the special case of two nodes
        #    self._assert_leaf_node(node_id) # The final site to be updated should be a leaf node
        self._update_site(node_id)
        # self._reset_for_next_time_step()

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



    def perform_expansion_ttn(self, ttn, tol):
        self.config.Expansion_params["tol"] = tol
        state_ex_ttn = expand_subspace(ttn, 
                                       self.hamiltonian, 
                                       self.config.Expansion_params)
        after_ex_total_bond_ttn = state_ex_ttn.total_bond_dim()
        expanded_dim_tot = after_ex_total_bond_ttn - ttn.total_bond_dim()
        return state_ex_ttn, after_ex_total_bond_ttn, expanded_dim_tot

    def adjust_tol_and_expand_ttn(self, tol):
        ttn = deepcopy(self.state)
        before_ex_total_bond_ttn = ttn.total_bond_dim()
        print("tol :", tol)

        # Initial Expansion Attempt
        state_ex_ttn, after_ex_total_bond_ttn, expanded_dim_tot = self.perform_expansion_ttn(ttn, tol)

        # Check if Expansion is Acceptable
        if expanded_dim_tot > self.config.Expansion_params["rel_tot_bond"]:
            print("EXPANSIONED DIM > REL:", expanded_dim_tot)
            A = True
            for _ in range(10):
                if A:
                    tol *= self.config.Expansion_params["tol_step"]
                    print("TRY: tol", tol)
                    state_ex_ttn, after_ex_total_bond_ttn, expanded_dim_tot = self.perform_expansion_ttn(ttn, tol)
                    print("TRY: EXPANSIONED DIM :", expanded_dim_tot)
                    if expanded_dim_tot < 0:
                        state_ex_ttn = ttn
                        tol /= self.config.Expansion_params["tol_step"]
                        A = False
                    elif expanded_dim_tot < self.config.Expansion_params["rel_tot_bond"]:
                        A = False

        # Check for Overgrown Bond Dimensions
        if self.config.Expansion_params["max_bond"] <= state_ex_ttn.total_bond_dim():
            print(self.config.Expansion_params["max_bond"], state_ex_ttn.total_bond_dim())
            should_expand = False
            print("REACH MAX BOND DIM")
        else:
            should_expand = True

        # Ensure Positive Expansion
        if state_ex_ttn.total_bond_dim() - before_ex_total_bond_ttn <= 0:
            state_ex_ttn = ttn
            tol /=  self.config.Expansion_params["tol_step"]
            print(state_ex_ttn.total_bond_dim(), before_ex_total_bond_ttn)
            print("EXPANSIONED DIM <= 0 ")

        after_ex_total_bond_ttn = state_ex_ttn.total_bond_dim()
        expanded_dim_total_bond_ttn = after_ex_total_bond_ttn - before_ex_total_bond_ttn

        print("expanded_dim TTN:", expanded_dim_total_bond_ttn)
        print("TTN:", before_ex_total_bond_ttn, "--->", after_ex_total_bond_ttn)

        return state_ex_ttn, tol, should_expand

    def run_ex_ttn(self, evaluation_time: Union[int, "inf"] = 1, filepath: str = "", pgbar: bool = True):
        self.init_results(evaluation_time)
        should_expand = True
        tol = self.config.Expansion_params["tol"]

        for i in self.create_run_tqdm(pgbar):
            self.evaluate_and_save_results(evaluation_time, i)
            self.run_one_time_step()

            # Expansion Step
            if (i + 1) % (self.config.Expansion_params["expansion_steps"] + 1) == 0 and should_expand:
                state_ex_ttn, tol, should_expand = self.adjust_tol_and_expand_ttn(tol)
                self.state = state_ex_ttn
                
            self._reset_for_next_time_step()
            self.record_bond_dimensions()

        self.save_results_to_file(filepath)