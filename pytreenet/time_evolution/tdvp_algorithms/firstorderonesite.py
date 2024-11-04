"""
Implementation of the first order one site TDVP algorithm.
"""
from typing import List, Union, Any, Dict
from tqdm import tqdm
from ..ttn_time_evolution import TTNTimeEvolutionConfig
from ...util.tensor_splitting import SplitMode
from ...operators.tensorproduct import TensorProduct
from .onesitetdvp import OneSiteTDVP
from dataclasses import replace
from copy import deepcopy
from pytreenet.time_evolution.Subspace_expansion import expand_subspace , KrylovBasisMode
from ...ttns import TreeTensorNetworkState 
from ...ttno.ttno_class import TTNO
from ...util.tensor_splitting import SplitMode , SVDParameters


class FirstOrderOneSiteTDVP(OneSiteTDVP):
    """
    The first order one site TDVP algorithm.

    This means we have first order Trotter splitting for the time evolution:
      exp(At+Bt) approx exp(At)*exp(Bt)
    Has the same attributes as the TDVP-Algorithm class
    """
    def __init__(self, 
                 initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO, 
                 time_step_size: float, 
                 final_time: float,
                 operators: Union[TensorProduct, List[TensorProduct]],

                 num_vecs: int , 
                 tau: float, 
                 SVDParameters : SVDParameters,
                 expansion_steps: int = 10,
  
                 initial_tol: float = 1e-20,
                 tol_step: float = 10, 
                 rel_ttn_tot_bond : int = 30,
                 max_ttn_bond: int = 100,
                 KrylovBasisMode : KrylovBasisMode = KrylovBasisMode.apply_1st_order_expansion,                  
                 config: Union[TTNTimeEvolutionConfig,None] = None) -> None:
 
        super().__init__(initial_state, hamiltonian,
                         time_step_size, 
                         final_time, 
                         operators, 
                         num_vecs, 
                         tau,
                         SVDParameters,
                         expansion_steps,
                         initial_tol,
                         tol_step,
                         KrylovBasisMode,  
                         config)
        self.rel_ttn_tot_bond = rel_ttn_tot_bond
        self.max_ttn_bond = max_ttn_bond 
        
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
 
    def run_ex(self, evaluation_time: Union[int,"inf"] = 1, filepath: str = "",
               pgbar: bool = True,):
        """
        Runs this time evolution algorithm for the given parameters.

        The desired operator expectation values are evaluated and saved.

        Args:
            evaluation_time (int, optional): The difference in time steps after which
                to evaluate the operator expectation values, e.g. for a value of 10
                the operators are evaluated at time steps 0,10,20,... If it is set to
                "inf", the operators are only evaluated at the end of the time.
                Defaults to 1.
            filepath (str, optional): If results are to be saved in an external file,
                the path to that file can be specified here. Defaults to "".
            pgbar (bool, optional): Toggles the progress bar. Defaults to True.
        """
        self._init_results(evaluation_time)
        assert self._results is not None
        should_expand = True
        tol = self.initial_tol

        for i in tqdm(range(self.num_time_steps + 1), disable=not pgbar):

            if evaluation_time != "inf" and i % evaluation_time == 0 and len(self._results) > 0:
                index = i // evaluation_time
                current_results = self.evaluate_operators()
                self._results[0:-1, index] = current_results
                # Save current time
                self._results[-1, index] = i*self.time_step_size

            self.run_one_time_step()

            ########### EXAPNSION ###########
            if (i+1) % (self.expansion_steps+1) == 0 and should_expand:  
                ######## T3NS ########
                ttn = deepcopy(self.state)
                before_ex_total_bond_ttn = ttn.total_bond_dim()

                print("tol :" , tol)       
                state_ex_ttn = expand_subspace(ttn, 
                                                self.hamiltonian, 
                                                self.num_vecs, 
                                                self.tau, 
                                                self.SVDParameters, 
                                                tol,
                                                self.KrylovBasisMode)
                after_ex_total_bond_ttn = state_ex_ttn.total_bond_dim()
                
                expanded_dim_tot = state_ex_ttn.total_bond_dim() - ttn.total_bond_dim()
                if  expanded_dim_tot > self.rel_ttn_tot_bond:
                    print("expanded_dim_tot :" , expanded_dim_tot)
                    A = True
                    for _ in range(10):
                        if A:
                            tol *= self.tol_step
                            print("1) tol" , tol)                            
                            state_ex_ttn = expand_subspace(ttn, 
                                                                self.hamiltonian, 
                                                                self.num_vecs, 
                                                                self.tau, 
                                                                self.SVDParameters, 
                                                                tol,
                                                                self.KrylovBasisMode)
                            after_ex_total_bond_ttn = state_ex_ttn.total_bond_dim()
                            expanded_dim_tot = state_ex_ttn.total_bond_dim() - ttn.total_bond_dim()
                            print("2) expanded_dim :" , expanded_dim_tot)
                            if expanded_dim_tot < 0:
                                state_ex_ttn = ttn
                                tol /= self.tol_step
                                A = False
                            elif expanded_dim_tot < self.rel_ttn_tot_bond :  
                                A = False  
                                
                
                if self.max_ttn_bond <= state_ex_ttn.total_bond_dim():
                    print(self.max_ttn_bond , state_ex_ttn.total_bond_dim()) 
                    #state_ex_ttn = ttn
                    should_expand = False
                    print("3")

                if state_ex_ttn.total_bond_dim() - before_ex_total_bond_ttn <= 0:
                   state_ex_ttn = ttn
                   tol /= self.tol_step
                   print(state_ex_ttn.total_bond_dim() , before_ex_total_bond_ttn)
                   print("4")      

                self.state = state_ex_ttn
                after_ex_total_bond_ttn = self.state.total_bond_dim()

                expanded_dim_total_bond_ttn = after_ex_total_bond_ttn - before_ex_total_bond_ttn
   

                print("expanded_dim TTN:" , expanded_dim_total_bond_ttn)
                print("TTN:" , before_ex_total_bond_ttn , "--->" , after_ex_total_bond_ttn)    
            ################################## 
              
            self._reset_for_next_time_step()                     
            self.record_bond_dimensions()



        if evaluation_time == "inf":
            current_results = self.evaluate_operators()
            self._results[0:-1, 0] = current_results
            self._results[-1, 0] = i*self.time_step_size
        if filepath != "":
            self.save_results_to_file(filepath)                
