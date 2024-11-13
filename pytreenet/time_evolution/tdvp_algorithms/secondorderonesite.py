from __future__ import annotations
from typing import List, Union
from copy import deepcopy , copy
from dataclasses import replace
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...operators.tensorproduct import TensorProduct
from ...ttno.ttno_class import TTNO
from ...ttns import TreeTensorNetworkState
from ..ttn_time_evolution import TTNTimeEvolutionConfig
from pytreenet.time_evolution.Subspace_expansion import expand_subspace 
from ..Lattice_simulation.util import ttn_to_t3n , t3n_to_ttn
from pytreenet.time_evolution import ExpansionMode
from .onesitetdvp import OneSiteTDVP
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

    def __init__(self, 
                 initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO, 
                 time_step_size: float, 
                 final_time: float,
                 operators: Union[TensorProduct, List[TensorProduct]],               
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
                         time_step_size, 
                         final_time, 
                         operators, 
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
        self.forward_sweep()
        self._final_forward_update()
        self.backward_sweep()

    def RUN(self, evaluation_time: Union[int, "inf"] = 1, filepath: str = "", pgbar: bool = True):
        if   self.config.Expansion_params["ExpansionMode"] == ExpansionMode.No_expansion:
             self.run(evaluation_time, filepath, pgbar)
        elif self.config.Expansion_params["ExpansionMode"] == ExpansionMode.TTN:
             self.run_ex(evaluation_time, filepath, pgbar)
        elif self.config.Expansion_params["ExpansionMode"] == ExpansionMode.Partial_T3N:   
             self.run_ex_partial_t3n(evaluation_time, filepath , pgbar)
        elif self.config.Expansion_params["ExpansionMode"] == ExpansionMode.Full_T3N:
             self.run_ex_full_t3n(evaluation_time, filepath, pgbar)
 

# TODO : transfer both adjust_tol_and_expand into OneSiteTDVP

    # EXPANDS with no T3NS transformation
    def perform_expansion(self, state, tol):
        self.config.Expansion_params["tol"] = tol
        state_ex = expand_subspace(state, 
                                   self.hamiltonian,
                                   self.config.Expansion_params)
        after_ex_total_bond = state_ex.total_bond_dim()
        expanded_dim_tot = after_ex_total_bond - state.total_bond_dim()
        return state_ex, after_ex_total_bond, expanded_dim_tot

    def adjust_tol_and_expand(self, tol):
        state = deepcopy(self.state)
        before_ex_total_bond = state.total_bond_dim()
        print("tol :", tol)

        self.config.Expansion_params["SVDParameters"] = replace(self.config.Expansion_params["SVDParameters"], max_bond_dim=state.max_bond_dim())
        print("SVD MAX:", state.max_bond_dim())
        print("tol:", tol)

        # Initial Expansion Attempt
        state_ex, after_ex_total_bond, expanded_dim_tot = self.perform_expansion(state, tol)

        # Check if Expansion is Acceptable
        if expanded_dim_tot > self.config.Expansion_params["rel_tot_bond"]:
            print("EXPANSIONED DIM > REL:", expanded_dim_tot)
            A = True
            for _ in range(self.config.Expansion_params["num_second_trial"]):
                if A:
                    tol *= self.config.Expansion_params["tol_step"]
                    print("TRY: tol", tol)
                    state_ex, after_ex_total_bond, expanded_dim_tot = self.perform_expansion(state, tol)
                    print("TRY: EXPANSIONED DIM :", expanded_dim_tot)
                    if expanded_dim_tot < 0:
                        state_ex = state
                        tol /= self.config.Expansion_params["tol_step"]
                        A = False
                    elif expanded_dim_tot < self.config.Expansion_params["rel_tot_bond"]:
                        A = False

        # Check for Overgrown Bond Dimensions
        if self.config.Expansion_params["max_bond"] <= state_ex.total_bond_dim():
            print(self.config.Expansion_params["max_bond"], state_ex.total_bond_dim())
            state_ex = state
            should_expand = False
            print("REACH MAX BOND DIM")
        else:
            should_expand = True    

        # Ensure Positive Expansion
        if state_ex.total_bond_dim() - before_ex_total_bond <= 0:
            state_ex = state
            tol /= self.config.Expansion_params["tol_step"]
            print("EXPANSIONED DIM <= 0 ")

        after_ex_total_bond = state_ex.total_bond_dim()
        expanded_dim_total_bond = after_ex_total_bond - before_ex_total_bond

        print("expanded_dim :", expanded_dim_total_bond ,":" , before_ex_total_bond, "--->", after_ex_total_bond)

        return state_ex, tol, should_expand

    def run_ex(self, evaluation_time: Union[int, "inf"] = 1, filepath: str = "", pgbar: bool = True):
        self.init_results(evaluation_time)
        should_expand = True
        tol = self.config.Expansion_params["tol"]

        for i in self.create_run_tqdm(pgbar):
            self.evaluate_and_save_results(evaluation_time, i)
            self.run_one_time_step()

            # Expansion Step
            if (i + 1) % (self.config.Expansion_params["expansion_steps"] + 1) == 0 and should_expand:
                state_ex, tol, should_expand = self.adjust_tol_and_expand(tol)
                self.state = state_ex
                self._orthogonalize_init(force_new=True)
                self.partial_tree_cache = PartialTreeCachDict()
                self.partial_tree_cache = self._init_partial_tree_cache()   

            self.record_bond_dimensions()

        self.save_results_to_file(filepath)  
    
    # EXPANDS with a predefined T3NS conf. if "T3N_dict" is not None and random T3NS otherwise
    def perform_expansion_t3n(self, t3n, t3no, tol):
        self.config.Expansion_params["tol"] = tol
        state_ex_t3n = expand_subspace(t3n, 
                                       t3no, 
                                       self.config.Expansion_params)
        after_ex_total_bond_t3ns = state_ex_t3n.total_bond_dim()
        state_ex_ttn = t3n_to_ttn(state_ex_t3n, self.config.Expansion_params["T3N_dict"])
        after_ex_total_bond_ttns = state_ex_ttn.total_bond_dim()
        expanded_dim_tot = state_ex_ttn.total_bond_dim() - self.state.total_bond_dim()
        return state_ex_ttn, expanded_dim_tot, after_ex_total_bond_ttns , after_ex_total_bond_t3ns

    def adjust_tol_and_expand_t3n(self, t3no, tol):
        ttn = deepcopy(self.state)
        t3n, _ = ttn_to_t3n(self.state, 
                            self.config.Expansion_params["T3N_dict"],
                            self.config.Expansion_params["T3NMode"],
                            self.config.Expansion_params["T3N_contr_mode"])
        before_ex_total_bond_ttn = ttn.total_bond_dim()
        before_ex_total_bond_t3ns = t3n.total_bond_dim()

        self.config.Expansion_params["SVDParameters"] = replace(self.config.Expansion_params["SVDParameters"], max_bond_dim=t3n.max_bond_dim())
        print("SVD MAX:", t3n.max_bond_dim())
        print("tol:", tol)
        state_ex_ttn, expanded_dim_tot, after_ex_total_bond_ttns, after_ex_total_bond_t3ns = self.perform_expansion_t3n(t3n, t3no, tol)

        if expanded_dim_tot > self.config.Expansion_params["rel_tot_bond"]:
            print("EXPANSIONED DIM > REL:", expanded_dim_tot)
            for _ in range(self.config.Expansion_params["num_second_trial"]):
                tol *= self.config.Expansion_params["tol_step"]
                print("TRY: tol", tol)
                state_ex_ttn, expanded_dim_tot, after_ex_total_bond_ttns, after_ex_total_bond_t3ns = self.perform_expansion_t3n(t3n, t3no, tol)
                print("TRY: EXPANSIONED DIM :", expanded_dim_tot)
                if expanded_dim_tot < 0:
                    state_ex_ttn = ttn
                    tol /= self.config.Expansion_params["tol_step"]
                    break
                elif expanded_dim_tot < self.config.Expansion_params["rel_tot_bond"]:
                    break

        # First Check: Overgrown Bond Dimensions
        if self.config.Expansion_params["max_bond"] < state_ex_ttn.total_bond_dim():
            print(self.config.Expansion_params["max_bond"], state_ex_ttn.total_bond_dim())
            state_ex_ttn = ttn
            should_expand = False
            print("REACH MAX BOND DIM")
        else:
            should_expand = True    

        # Second Check: Ensure Positive Expansion
        if state_ex_ttn.total_bond_dim() - before_ex_total_bond_ttn <= 0:
            state_ex_ttn = ttn
            tol /= self.config.Expansion_params["tol_step"]
            print("EXPANSIONED DIM <= 0 ")

        # Third Check: Overgrown Bond Dimensions After All Adjustments
        if self.config.Expansion_params["max_bond"] < state_ex_ttn.total_bond_dim():
            print("END:", state_ex_ttn.total_bond_dim())
            should_expand = False

        # Debugging Information
        print("expanded_dim T3NS:", after_ex_total_bond_t3ns - before_ex_total_bond_t3ns , ":", "T3NS:", before_ex_total_bond_t3ns, "--->", after_ex_total_bond_t3ns)
        print("expanded_dim TTN: ", after_ex_total_bond_ttns - before_ex_total_bond_ttn ,  ":", "TTN:", before_ex_total_bond_ttn,   "--->", after_ex_total_bond_ttns)

        return state_ex_ttn, tol, should_expand

    def run_ex_partial_t3n(self, evaluation_time: Union[int, "inf"] = 1, filepath: str = "", pgbar: bool = True):
        self.init_results(evaluation_time)
        should_expand = True
        tol = self.config.Expansion_params["tol"]

        if self.config.Expansion_params["T3N_dict"] is not None:
            t3no, _ = ttn_to_t3n(self.hamiltonian, 
                                self.config.Expansion_params["T3N_dict"],
                                self.config.Expansion_params["T3NMode"],
                                self.config.Expansion_params["T3N_contr_mode"])

        for i in self.create_run_tqdm(pgbar):
            self.evaluate_and_save_results(evaluation_time, i)
            self.run_one_time_step()

            if (i + 1) % (self.config.Expansion_params["expansion_steps"] + 1) == 0 and should_expand:
                if self.config.Expansion_params["T3N_dict"] is None:
                    t3no, T3N_dict = ttn_to_t3n(self.hamiltonian,
                                                T3N_dict       = None, 
                                                T3N_mode       = self.config.Expansion_params["T3NMode"],
                                                T3N_contr_mode = self.config.Expansion_params["T3N_contr_mode"])
                    self.config.Expansion_params["T3N_dict"] = T3N_dict
                       
                state_ex_ttn, tol, should_expand = self.adjust_tol_and_expand_t3n(t3no, tol)
                self.state = state_ex_ttn
                self._orthogonalize_init(force_new=True)
                self.partial_tree_cache = PartialTreeCachDict()
                self.partial_tree_cache = self._init_partial_tree_cache()         
           
            self.record_bond_dimensions()

        self.save_results_to_file(filepath)


    # RUN entirely on a predefined T3NS conf. if "T3N_dict" is not None and random T3NS otherwise
    # NOT WORKING
    def run_ex_full_t3n(self, evaluation_time: Union[int, "inf"] = 1, filepath: str = "", pgbar: bool = True):
        self.init_results(evaluation_time)
        should_expand = True
        tol = self.config.Expansion_params["tol"]
        if self.config.Expansion_params["T3N_dict"] is not None:
            self.hamiltonian, _ = ttn_to_t3n(self.hamiltonian, 
                                            self.config.Expansion_params["T3N_dict"],
                                            self.config.Expansion_params["T3NMode"],
                                            self.config.Expansion_params["T3N_contr_mode"])

            self.state ,   _    = ttn_to_t3n(self.state, 
                                            self.config.Expansion_params["T3N_dict"],
                                            self.config.Expansion_params["T3NMode"],
                                            self.config.Expansion_params["T3N_contr_mode"])
        else:
            self.hamiltonian, T3N_dict = ttn_to_t3n(self.hamiltonian, 
                                                    None, 
                                                    self.config.Expansion_params["T3NMode"],
                                                    self.config.Expansion_params["T3N_contr_mode"])
            self.state ,     _         = ttn_to_t3n(self.state, 
                                                    T3N_dict, 
                                                    self.config.Expansion_params["T3NMode"],
                                                    self.config.Expansion_params["T3N_contr_mode"])                                                   

        # Initialization according to the T3NS
        self.update_path = self._finds_update_path()
        self.orthogonalization_path = self._find_tdvp_orthogonalization_path(self.update_path)
        self._orthogonalize_init()
        self.partial_tree_cache = self._init_partial_tree_cache()
        self.backwards_update_path = self._init_second_order_update_path()
        self.backwards_orth_path = self._init_second_order_orth_path()

        for i in self.create_run_tqdm(pgbar):
            self_copy = deepcopy(self)
            #self_copy = copy(self)
            #self_copy.state = deepcopy(self.state)
            #self_copy.hamiltonian = deepcopy(self.hamiltonian)
            if self_copy.config.Expansion_params["T3N_dict"] is not None:
                self_copy.hamiltonian = t3n_to_ttn(self_copy.hamiltonian, 
                                                      self_copy.config.Expansion_params["T3N_dict"])
                self_copy.state       = t3n_to_ttn(self_copy.state, 
                                                      self_copy.config.Expansion_params["T3N_dict"])
            else:
                self_copy.hamiltonian  = t3n_to_ttn(self_copy.hamiltonian, 
                                                             T3N_dict)
                self_copy.config.Expansion_params["T3N_dict"] = T3N_dict      
                self_copy.state        = t3n_to_ttn(self_copy.state, 
                                                             T3N_dict) 
            self_copy.evaluate_and_save_results(evaluation_time, i)
            self._results = self_copy._results

            self.run_one_time_step()
            # Expansion Step
            if (i + 1) % (self.config.Expansion_params["expansion_steps"]+ 1) == 0 and should_expand:
                state_ex, tol, should_expand = self.adjust_tol_and_expand(tol)
                self.state = state_ex
                self._orthogonalize_init(force_new=True)
                self.partial_tree_cache = PartialTreeCachDict()
                self.partial_tree_cache = self._init_partial_tree_cache()   

            self.record_bond_dimensions()

        self.save_results_to_file(filepath)   

    def run_ex_full_t3n_2(self, evaluation_time: Union[int, "inf"] = 1, filepath: str = "", pgbar: bool = True):
        self.init_results(evaluation_time)
        should_expand = True
        tol = self.config.Expansion_params["tol"]
        if self.config.Expansion_params["T3N_dict"] is not None:
            self.hamiltonian, _ = ttn_to_t3n(self.hamiltonian, 
                                            self.config.Expansion_params["T3N_dict"],
                                            self.config.Expansion_params["T3NMode"],
                                            self.config.Expansion_params["T3N_contr_mode"])
            self.operators      = [ttn_to_t3n(self.operators[i], 
                                            self.config.Expansion_params["T3N_dict"],
                                            self.config.Expansion_params["T3NMode"],
                                            self.config.Expansion_params["T3N_contr_mode"])[0]
                                            for i in range(len(self.operators))]
            self.state ,   _    = ttn_to_t3n(self.state, 
                                            self.config.Expansion_params["T3N_dict"],
                                            self.config.Expansion_params["T3NMode"],
                                            self.config.Expansion_params["T3N_contr_mode"])
        else:
            self.hamiltonian, T3N_dict = ttn_to_t3n(self.hamiltonian, 
                                                    None, 
                                                    self.config.Expansion_params["T3NMode"],
                                                    self.config.Expansion_params["T3N_contr_mode"])
            self.config.Expansion_params["T3N_dict"] = T3N_dict
            self.operators               = [ttn_to_t3n(self.operators[i], 
                                                    self.config.Expansion_params["T3N_dict"],
                                                    self.config.Expansion_params["T3NMode"],
                                                    self.config.Expansion_params["T3N_contr_mode"])[0]
                                                    for i in range(len(self.operators))]            
            self.state ,     _         = ttn_to_t3n(self.state, 
                                                    self.config.Expansion_params["T3N_dict"], 
                                                    self.config.Expansion_params["T3NMode"],
                                                    self.config.Expansion_params["T3N_contr_mode"])                                                   

        # Initialization according to the T3NS
        self.update_path = self._finds_update_path()
        self.orthogonalization_path = self._find_tdvp_orthogonalization_path(self.update_path)
        self._orthogonalize_init()
        self.partial_tree_cache = self._init_partial_tree_cache()
        self.backwards_update_path = self._init_second_order_update_path()
        self.backwards_orth_path = self._init_second_order_orth_path()

        for i in self.create_run_tqdm(pgbar):
            self.evaluate_and_save_results(evaluation_time, i)

            
            self.run_one_time_step()
            # Expansion Step
            if (i + 1) % (self.config.Expansion_params["expansion_steps"]+ 1) == 0 and should_expand:
                state_ex, tol, should_expand = self.adjust_tol_and_expand(tol)
                self.state = state_ex
                self._orthogonalize_init(force_new=True)
                self.partial_tree_cache = PartialTreeCachDict()
                self.partial_tree_cache = self._init_partial_tree_cache()   

            self.record_bond_dimensions()

        self.save_results_to_file(filepath)  