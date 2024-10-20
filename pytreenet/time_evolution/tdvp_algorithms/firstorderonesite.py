"""
Implementation of the first order one site TDVP algorithm.
"""
from ...util.tensor_splitting import SplitMode
from .onesitetdvp import OneSiteTDVP
from ...Lindblad.util import adjust_ttn1_structure_to_ttn2
import copy
import pytreenet as ptn
from tqdm import tqdm
from copy import deepcopy

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

    def create_temp_copy(self):
        """
        Creates a temporary copy of the object with deep copies of state and cache.
        """
        temp_self = copy.copy(self)
        temp_self.state = copy.deepcopy(self.state)
        return temp_self

    def run_one_time_step_copy(self):
        # remove self._orthogonalize_init() and self.partial_tree_cache in TDVPAlgorithm
        # remove self._reset_for_next_time_step() in self._final_update(node_id)
        # TODO : remove extra adjust_ttn1_structure_to_ttn2
        temp_self = self.create_temp_copy()

        #temp_self._orthogonalize_init()
        #temp_self.partial_tree_cache = temp_self._init_partial_tree_cache()

        for i, node_id in enumerate(self.update_path):
            if i == len(self.update_path)-1:
                temp_self._final_update(node_id)
            elif i == 0:
                temp_self._first_update(node_id)
            else:
                temp_self._normal_update(node_id, i) 
        #orth_center_id_1 = self.state.root_id
        #orth_center_id_2 = orth_center_id_1.replace('Site', 'Node')
        #temp_self.state = ptn.normalize_ttn_Lindblad_2(temp_self.state , temp_self.connections)
        
        self.state = temp_self.state


    def run_Lindblad(self, evaluation_time  = 1, filepath: str = "",
            pgbar: bool = True,):
        self._orthogonalize_init()
        self.partial_tree_cache = self._init_partial_tree_cache()
        self.adjust_to_initial_structure()
        self._init_results(evaluation_time)
        assert self._results is not None
 
        for i in tqdm(range(self.num_time_steps + 1), disable=not pgbar):
            if i != 0:  # We also measure the initial expectation_values 
                self.run_one_time_step_copy()
                self._reset_for_next_time_step() 
                self.adjust_to_initial_structure()
            if evaluation_time != "inf" and i % evaluation_time == 0 and len(self._results) > 0:
                index = i // evaluation_time
                current_results = self.evaluate_operators_Lindblad()
                print(current_results[0])
                
                self._results[0:-1, index] = current_results
                # Save current time
                self._results[-1, index] = i*self.time_step_size
   
        if evaluation_time == "inf":
            current_results = self.evaluate_operators_Lindblad()
            self._results[0:-1, 0] = current_results
            self._results[-1, 0] = i*self.time_step_size
        if filepath != "":
            self.save_results_to_file(filepath)

    def reset_to_initial_state(self):
        """
        Resets the current state to the intial state
        """
        self.state = deepcopy(self._intital_state)           