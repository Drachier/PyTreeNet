"""
Implementation of the first order one site TDVP algorithm.
"""
from typing import Union
from ...util.tensor_splitting import SplitMode
from .onesitetdvp import OneSiteTDVP
from copy import deepcopy
from pytreenet.time_evolution.Subspace_expansion import expand_subspace 
from ...util.tensor_splitting import SplitMode
from pytreenet.time_evolution import ExpansionMode
from ..Lattice_simulation.util import ttn_to_t3n 
from pytreenet.time_evolution.Lattice_simulation import build_leg_specs
from ...contractions.state_operator_contraction import contract_any


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
        next_node_id = self.orthogonalization_path[update_index][0]

        if str(self.state.nodes[node_id].identifier).startswith("3"):
            # remove open leg
            new_tensor = self.state.tensors[node_id].reshape(self.state.tensors[node_id].shape[:-1])
            open_leg = self.state.nodes[node_id].open_legs[0]
            self.state.nodes[node_id]._leg_permutation.pop(open_leg)
            self.state.tensors[node_id] = new_tensor
            self.state.nodes[node_id].link_tensor(new_tensor)

            self.state.contract_nodes(node_id, 
                                        next_node_id,
                                        new_identifier = next_node_id)
            next_next_neighbour_id = self.orthogonalization_path[update_index+1][0]
            main_legs, next_legs = build_leg_specs(self.state.nodes[next_node_id], next_next_neighbour_id)
            self.state.split_node_it(next_node_id , main_legs, next_legs,
                                        identifier_a= "3_" + next_node_id,
                                        identifier_b= next_node_id)  

            # add open leg
            shape = self.state.tensors[node_id].shape
            T = self.state.tensors[node_id].reshape(shape + (1,))
            self.state.nodes[node_id]._leg_permutation.append(open_leg)
            self.state.tensors[node_id] = T 
            self.state.nodes[node_id].link_tensor(T)           

            new_tensor = contract_any(node_id, next_node_id,
                                        self.state, self.hamiltonian,
                                        self.partial_tree_cache)
            self.partial_tree_cache.add_entry(node_id, next_node_id, new_tensor)
            
            self.state.orthogonality_center_id = next_node_id
        else:     
            self._update_site(node_id)
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
        #self.state.move_orthogonalization_center(self.update_path[0], mode = SplitMode.REDUCED)
        self.state.canonical_form(self.update_path[0], SplitMode.REDUCED)
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

    def RUN(self, evaluation_time: Union[int, "inf"] = 1, filepath: str = "", pgbar: bool = True):
        if   self.config.Expansion_params["ExpansionMode"] == ExpansionMode.No_expansion:
             self.run(evaluation_time, filepath, pgbar)
        elif self.config.Expansion_params["ExpansionMode"] == ExpansionMode.TTN:
             self.run_ex(evaluation_time, filepath, pgbar)
        elif self.config.Expansion_params["ExpansionMode"] == ExpansionMode.Partial_T3N:   
             self.run_ex_partial_t3n(evaluation_time, filepath , pgbar)
        elif self.config.Expansion_params["ExpansionMode"] == ExpansionMode.Full_T3N:
             self.run_ex_full_t3n(evaluation_time, filepath, pgbar) 

    def check_overgrown_bond_dimensions(self, ttn_ex, ttn):
        if ttn_ex.total_bond_dim() >= self.config.Expansion_params["max_bond"]:
            print("Exceed max bond dimension:", self.config.Expansion_params["max_bond"])
            ttn_ex = ttn
            should_expand = False
        else:
            should_expand = True
        return ttn_ex, should_expand
    
    
    # EXPANDS with no T3NS transformation

    def perform_expansion(self, state, tol):
        self.config.Expansion_params["tol"] = tol
        state_ex = expand_subspace(state, 
                                   self.hamiltonian,
                                   self.config.Expansion_params)
        after_ex_total_bond = state_ex.total_bond_dim()
        expanded_dim_tot = after_ex_total_bond - state.total_bond_dim()
        return state_ex, after_ex_total_bond, expanded_dim_tot

    def phase1_increase_tol(self, state, tol, expanded_dim_tot):
        max_trials = self.config.Expansion_params["num_second_trial"]
        num_trials = 0
        min_rel_tot_bond, max_rel_tot_bond = self.config.Expansion_params["rel_tot_bond"]
        while num_trials < max_trials:
            print(f"Phase 1 - Trial {num_trials+1}:")
            if expanded_dim_tot > max_rel_tot_bond:
                print(f"Expanded dim ({expanded_dim_tot}) > rel_tot_bond ({self.config.Expansion_params['rel_tot_bond']})")
                # Increase tol to reduce expanded_dim_tot
                tol += self.config.Expansion_params["tol_step_increase"]
                print("Increasing tol:", tol)
                state_ex, _, expanded_dim_tot = self.perform_expansion(state, tol)
                num_trials += 1
                if min_rel_tot_bond <= expanded_dim_tot <= max_rel_tot_bond:
                    # Acceptable expansion found
                    print("Acceptable expansion found in Phase 1:", expanded_dim_tot)
                    return state_ex, tol, expanded_dim_tot, False
                elif expanded_dim_tot < min_rel_tot_bond:
                    # Need to switch to Phase 2
                    print("Expanded dim:", expanded_dim_tot, "falls below min_rel_tot_bond:", min_rel_tot_bond)
                    if expanded_dim_tot <= 0:
                       state_ex = state
                    print("Switching to Phase 2")
                    return state_ex, tol, expanded_dim_tot, True  # Proceed to Phase 2                
        # Exceeded max trials
        print("Exceeded maximum trials in Phase 1 without acceptable expansion")
        state_ex = state
        tol += self.config.Expansion_params["tol_step_increase"]
        return state_ex, tol, expanded_dim_tot, False  # Proceed to Phase 2

    def phase2_decrease_tol(self, state, tol, expanded_dim_tot):
        max_trials = self.config.Expansion_params["num_second_trial"]
        num_trials = 0
        min_rel_tot_bond, max_rel_tot_bond = self.config.Expansion_params["rel_tot_bond"]
        while num_trials < max_trials:
            num_trials += 1
            print(f"Phase 2 - Trial {num_trials}:")
            if expanded_dim_tot < min_rel_tot_bond:
                # Decrease tol to increase expanded_dim_tot
                tol -= self.config.Expansion_params["tol_step_decrease"]
                print("Decreasing tol:", tol)
                state_ex, _, expanded_dim_tot = self.perform_expansion(state, tol)
                print("Expanded_dim_tot:", expanded_dim_tot)
                if min_rel_tot_bond <= expanded_dim_tot <= max_rel_tot_bond:
                    # Acceptable expansion found
                    print("Acceptable expansion found in Phase 2:", expanded_dim_tot)
                    return state_ex, tol, expanded_dim_tot
                elif expanded_dim_tot > max_rel_tot_bond:
                    # Expanded dimension exceeded rel_tot_bond again
                    print("Expanded dim exceeded rel_tot_bond again:", expanded_dim_tot)
                    # Reset state_ex to initial state
                    state_ex = state
                    return state_ex, tol, expanded_dim_tot  # Reset and exit
        # Exceeded max trials
        print("Exceeded maximum trials in Phase 2 without acceptable expansion")
        # Reset state_ex to initial state
        state_ex = state
        tol -= self.config.Expansion_params["tol_step_decrease"]
        return state_ex, tol, expanded_dim_tot  # Reset and exit

    def adjust_tol_and_expand(self, tol):
        state = deepcopy(self.state)
        before_ex_total_bond = state.total_bond_dim()

        #self.config.Expansion_params["SVDParameters"] = replace(self.config.Expansion_params["SVDParameters"],max_bond_dim=state.max_bond_dim())
        #print("SVD MAX:", state.max_bond_dim())
        print("Initial tol:", tol)

        # Initial Expansion Attempt
        state_ex, after_ex_total_bond, expanded_dim_tot = self.perform_expansion(state, tol)

        # Unpack the acceptable range
        min_rel_tot_bond, max_rel_tot_bond = self.config.Expansion_params["rel_tot_bond"]

        # Check initial expansion
        if min_rel_tot_bond <= expanded_dim_tot <= max_rel_tot_bond:
            # Acceptable expansion found in initial attempt
            print("Acceptable expansion found in initial attempt:", expanded_dim_tot)
        elif expanded_dim_tot > max_rel_tot_bond:
            # Need to adjust tol in Phase 1
            state_ex, tol, expanded_dim_tot, switch_to_phase_2 = self.phase1_increase_tol(state, tol, expanded_dim_tot)
            if switch_to_phase_2:
                # Switch to Phase 2
                state_ex, tol, expanded_dim_tot = self.phase2_decrease_tol(state, tol, expanded_dim_tot)
        elif expanded_dim_tot < min_rel_tot_bond:
            # Need to adjust tol in Phase 2
            state_ex, tol, expanded_dim_tot = self.phase2_decrease_tol(state, tol, expanded_dim_tot)

        after_ex_total_bond = state_ex.total_bond_dim()
        expanded_dim_total_bond = after_ex_total_bond - before_ex_total_bond

        state_ex, should_expand = self.check_overgrown_bond_dimensions(state_ex, state)

        print("Final expanded_dim:", expanded_dim_total_bond, ":", before_ex_total_bond, "--->", after_ex_total_bond)

        return state_ex, tol, should_expand

    def run_ex(self, evaluation_time: Union[int, "inf"] = 1, filepath: str = "", pgbar: bool = True):
        
        self.init_results(evaluation_time)
        should_expand = True
        tol = self.config.Expansion_params["tol"]

        for i in self.create_run_tqdm(pgbar):
            self.evaluate_and_save_results(evaluation_time, i)
            self.run_one_time_step()

            if (i + 1) % (self.config.Expansion_params["expansion_steps"] + 1) == 0 and should_expand:

                state_ex_ttn, tol, should_expand = self.adjust_tol_and_expand(tol)
                self.state = state_ex_ttn
                self.state.normalize_ttn()
                #self._orthogonalize_init(force_new=True)
                #self.partial_tree_cache = PartialTreeCachDicst()
                #self.partial_tree_cache = self._init_partial_tree_cache()         

            self._reset_for_next_time_step()    
            self.record_bond_dimensions()

        self.save_results_to_file(filepath) 





    # Run fully on T3N
    def run_ex_full_t3n(self, evaluation_time: Union[int, "inf"] = 1, filepath: str = "", pgbar: bool = True):
        
        self.init_results(evaluation_time)
        should_expand = True
        tol = self.config.Expansion_params["tol"]
        self.hamiltonian, _ = ttn_to_t3n(self.hamiltonian, 
                                         self.update_path,
                                         self.orthogonalization_path)
        self.operators      = [ttn_to_t3n(self.operators[i], 
                                          self.update_path,
                                          self.orthogonalization_path)[0]
                                        for i in range(len(self.operators))]
        self.state ,   _    = ttn_to_t3n(self.state, 
                                        self.update_path,
                                        self.orthogonalization_path)
                                                 

        # Initialization according to the T3NS
        self.update_path = self._finds_update_path()
        self.state.canonical_form(self.update_path[0] , SplitMode.REDUCED)
        self.orthogonalization_path = self._find_tdvp_orthogonalization_path(self.update_path)
        self._orthogonalize_init()
        self.partial_tree_cache = self._init_partial_tree_cache()

        for i in self.create_run_tqdm(pgbar):
            self.evaluate_and_save_results(evaluation_time, i)

            
            self.run_one_time_step()
            # Expansion Step
            if (i + 1) % (self.config.Expansion_params["expansion_steps"]+ 1) == 0 and should_expand:
                state_ex, tol, should_expand = self.adjust_tol_and_expand(tol)
                self.state = state_ex
                #self._orthogonalize_init(force_new=True)
                #self.partial_tree_cache = PartialTreeCachDict()
                #self.partial_tree_cache = self._init_partial_tree_cache()   
            
            self._reset_for_next_time_step() 
            self.record_bond_dimensions()

        self.save_results_to_file(filepath)  

        