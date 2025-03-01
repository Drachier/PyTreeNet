from __future__ import annotations
from typing import List, Union
from copy import deepcopy , copy
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...operators.tensorproduct import TensorProduct
from ...util.tensor_splitting import SplitMode
from ...ttno.ttno_class import TTNO
from ...ttns import TreeTensorNetworkState
from ..ttn_time_evolution import TTNTimeEvolutionConfig
from ...time_evolution.Subspace_expansion import Krylov_basis , enlarge_ttn1_bond_with_ttn2
from ..Lattice_simulation.util import ttn_to_t3n , t3n_to_ttn
from ...time_evolution.Lattice_simulation import build_leg_specs
from .onesitetdvp_random import OneSiteTDVP_random
from ...contractions.state_operator_contraction import contract_any
from ...contractions.state_state_contraction import contract_two_ttns
import numpy as np
from scipy.linalg import expm
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import random


import queue  
from threading import Thread

def monitor_progress(progress_queue, total_steps, stop_event):
    pbar = tqdm(total=total_steps, desc="Time steps", ncols=80)
    completed = 0

    while True:
        try:
            msg = progress_queue.get(timeout=0.1)
            if msg[0] == "progress":
                increment = msg[1]
                pbar.update(increment)
                completed += increment
                if completed >= total_steps:
                    break
            elif msg[0] == "log":
                tqdm.write(str(msg[1]))
        except queue.Empty:
            # If we have no new messages AND the event is set, we can break.
            if stop_event.is_set():
                break
    pbar.close()

class NoiseModel:
    def __init__(self, processes: list[str]=[], strengths: list[float]=[]):
        assert len(processes) == len(strengths)
        self.processes = processes
        self.strengths = strengths
        self.jump_operators = []
        for process in processes:
            self.jump_operators.append(getattr(NoiseLibrary, process)().matrix)

class Excitation:
    d = 2
    matrix = np.zeros((d, d))
    for row, array in enumerate(matrix):
        for col, _ in enumerate(array):
            if row - col == 1:
                matrix[row][col] = 1

class Relaxation:
    d = 2
    matrix = np.zeros((d, d))
    for row, array in enumerate(matrix):
        for col, _ in enumerate(array):
            if col - row == 1:
                matrix[row][col] = 1

class Dephasing:
    matrix = np.array([[1, 0],
                       [0, -1]])

class NoiseLibrary:
    excitation = Excitation
    relaxation = Relaxation
    dephasing = Dephasing

class GSE_TJM_TDVP_random(OneSiteTDVP_random):
    def __init__(self, 
                 initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO, 
                 time_step_size: float, 
                 final_time: float,
                 operators: Union[TensorProduct, List[TensorProduct]],   
                 N : int,
                 noise_model : NoiseModel, 
                 max_workers: int = 7,   
                 config: Union[TTNTimeEvolutionConfig,None] = None) -> None:

        super().__init__(initial_state, hamiltonian,
                         time_step_size, 
                         final_time, 
                         operators, 
                         config)

        self.N = N
        self.noise_model = noise_model
        self.max_workers = max_workers
        self.init_trajectories()

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
                                      next_node_id: str, i):
        """
        Run the forward update with half time step.

        First the site tensor is updated and then the link tensor.

        Args:
            node_id (str): The identifier of the site to be updated.
            next_node_id (str): The other node of the link to be updated.
        """
        assert self.state.orthogonality_center_id == node_id

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
            next_next_neighbour_id = self.orthogonalization_path[i+1][0]
            print("Node ID:", node_id, "Next Node ID:", next_node_id, "Next Next Neighbour ID:", next_next_neighbour_id)
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
            self._update_forward_site_and_link(node_id, next_node_id, i)

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

# TJM
    def init_trajectories(self):
        """
        Initialises an appropriately sized zero valued numpy array for storage.

        Each row contains the results obtained for one operator, while the
        last row contains the times. Note, the the entry with index zero
        corresponds to time 0.

        Args:
            evaluation_time (int, optional): The difference in time steps after which
                to evaluate the operator expectation values, e.g. for a value of 10
                the operators are evaluated at time steps 0,10,20,... If it is set to
                "inf", the operators are only evaluated at the end of the time.
                Defaults to 1.
        """
        self.trajectories = {}
        for operator in self.operators:
            id = list(operator.keys())[0]
            self.trajectories[id] = np.empty((self.N, len(self.times())), dtype=float)

    def init_results_TJM(self, evaluation_time = 1):
        """
        Initialises an appropriately sized zero valued numpy array for storage.

        Each row contains the results obtained for one operator, while the
        last row contains the times. Note, the the entry with index zero
        corresponds to time 0.

        Args:
            evaluation_time (int, optional): The difference in time steps after which
                to evaluate the operator expectation values, e.g. for a value of 10
                the operators are evaluated at time steps 0,10,20,... If it is set to
                "inf", the operators are only evaluated at the end of the time.
                Defaults to 1.
        """
        self._results = {}
        for operator in self.operators:
            id = list(operator.keys())[0]
            self._results[id] = np.zeros(self.num_time_steps//evaluation_time + 1,
                                             dtype=complex)
            
        assert self._results is not None

    def apply_dissipation(self , dt):
        # result is not normalized

        A = sum(self.noise_model.strengths[i] * np.conj(jump_operator).T @ jump_operator
            for i, jump_operator in enumerate(self.noise_model.jump_operators))
        dissipative_operator = expm(-0.5 * dt * A)

        for node_id in list(self.state.nodes.keys()): 
            self.state.absorb_into_open_legs(node_id , dissipative_operator)
        
        self.state.canonical_form(self.update_path[0] , SplitMode.KEEP)

    def create_probability_distribution(self):
        jump_dict = {'jumps': [], 'strengths': [], 'sites': [], 'probabilities': []}
        dp_m_list = []
        
        assert self.state.is_in_canonical_form(self.update_path[0])
        
        for i,node_id in enumerate(self.update_path[:-1]): 
            assert node_id == self.state.orthogonality_center_id

            for j, jump_operator in enumerate(self.noise_model.jump_operators):

                jumped_state = deepcopy(self.state)
                jumped_state.absorb_into_open_legs(node_id, jump_operator)
                
                jumped_state_conj = jumped_state.conjugate()
                # dp_m = self.time_step_size * self.noise_model.strengths[j] * contract_two_ttns(jumped_state, jumped_state_conj) 
                dp_m = self.time_step_size * self.noise_model.strengths[j] * scalar_product(jumped_state ,use_orthogonal_center =  True) 

                dp_m_list.append(dp_m.real)
                jump_dict['jumps'].append(jump_operator)
                jump_dict['strengths'].append(self.noise_model.strengths[j])
                jump_dict['sites'].append(node_id)

            orth_node_id = self.orthogonalization_path[i][-1]
            self.state.move_orthogonalization_center(orth_node_id , SplitMode.KEEP)

        # last node 
        node_id = self.update_path[-1]
        assert node_id == self.state.orthogonality_center_id
        for j, jump_operator in enumerate(self.noise_model.jump_operators):
            jumped_state = deepcopy(self.state)
            jumped_state.absorb_into_open_legs(node_id, jump_operator)
            #jumped_state_conj = jumped_state.conjugate()
            #dp_m = self.time_step_size * self.noise_model.strengths[j] * contract_two_ttns(jumped_state, jumped_state_conj) 
            dp_m = self.time_step_size * self.noise_model.strengths[j] * scalar_product(jumped_state ,use_orthogonal_center =  True) 

            dp_m_list.append(dp_m.real)
            jump_dict['jumps'].append(jump_operator)
            jump_dict['strengths'].append(self.noise_model.strengths[j])
            jump_dict['sites'].append(node_id)

        jump_dict['probabilities'] = (dp_m_list / np.sum(dp_m_list)).astype(float)

        return jump_dict

    def stochastic_process(self):
        #state_conj = self.state.conjugate()
        #dp = 1 - contract_two_ttns(self.state , state_conj)
        dp = 1 - scalar_product(self.state ,use_orthogonal_center =  True)
        if np.random.rand() >= dp.real:
            # No jump
            self.state = normalize_ttn(self.state)
            # assert self.state.is_in_canonical_form(self.update_path[0])
        else :
            jump_dict = self.create_probability_distribution()
            choices = list(range(len(jump_dict['probabilities'])))
            choice = np.random.choice(choices, p=jump_dict['probabilities'])
            jump_operator = jump_dict['jumps'][choice]
            self.state.absorb_into_open_legs(jump_dict['sites'][choice], jump_operator)
            self.state = normalize_ttn(self.state)
            self.state.canonical_form(self.state.orthogonality_center_id, SplitMode.KEEP) 

    def initialize(self):
        self.apply_dissipation(self.time_step_size /2)
        self.stochastic_process()

    def sample(self, state, evaluation_time,  j):
        start_state = deepcopy(self.state)
        self.state = deepcopy(state)

        self._orthogonalize_init(force_new=True)
        self.partial_tree_cache = PartialTreeCachDict()
        self.partial_tree_cache = self._init_partial_tree_cache() 
        self.run_one_time_step()

        self.apply_dissipation(self.time_step_size /2)
        self.stochastic_process() 

        self.evaluate_and_save_results_TJM(evaluation_time, j)

        self.state = start_state

    def step_through(self):

        self._orthogonalize_init(force_new=True)
        self.partial_tree_cache = PartialTreeCachDict()
        self.partial_tree_cache = self._init_partial_tree_cache()  
        self.run_one_time_step()

        self.apply_dissipation(self.time_step_size)
        self.stochastic_process()

    def run_trajectory_second_order(self, args):
        _ , evaluation_time,  initial_state, hamiltonian,  progress_queue = args
        self.state = deepcopy(initial_state)
        self.hamiltonian = hamiltonian

        self.update_path = self._finds_update_path()
        self.orthogonalization_path = self._find_tdvp_orthogonalization_path(self.update_path)        
        self.backwards_update_path = self._init_second_order_update_path()
        self.backwards_orth_path = self._init_second_order_orth_path()

        tol = self.config.Expansion_params["tol"]

        # save initial result
        self._orthogonalize_init()
        self.evaluate_and_save_results_TJM(evaluation_time, 0)
        progress_queue.put(("progress", 1))

        self.initialize()
        self.sample(self.state, evaluation_time,  j = 1)
        progress_queue.put(("progress", 1))
        should_expand = True
        self.total_bond_dim = []
        
        for j, _ in enumerate(self.times()[2:], start=2):
            self.step_through()
            self.sample(self.state, evaluation_time, j)

            progress_queue.put(("progress", 1))
            self.total_bond_dim.append(self.state.total_bond_dim())

            if j % (self.config.Expansion_params["expansion_steps"]) == 0 and should_expand:
                state_ex_ttn, tol, should_expand = self.adjust_tol_and_expand(tol, progress_queue)
                self.state = state_ex_ttn 
            
        return self.total_bond_dim , self._results

    def run_TJM(self, evaluation_time=1, filepath=None):

        # 1) Initialize the arrays/dicts you need
        self.init_results_TJM(evaluation_time)
        
        normalized_states = [normalize_ttn(state) for state in self.initial_state]

        # 2) Create multiprocessing tools
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        stop_event = manager.Event()

        num_time_steps = len(self.times())
        total_steps = self.N * num_time_steps

        # 3) Start the monitor thread
        monitor_thread = Thread(
            target=monitor_progress,
            args=(progress_queue, total_steps, stop_event),
            daemon=True
        )
        monitor_thread.start()

        # Prepare argument tuples for each trajectory
        args_list = []
        for i in range(self.N):
            # Choose a random index to pick a matching state and TTNo
            idx = random.randint(0, len(normalized_states) - 1)
            chosen_ttn = normalized_states[idx]
            chosen_hamiltonian = self.hamiltonian[idx]

            args_list.append(
                (i, evaluation_time, chosen_ttn, chosen_hamiltonian, progress_queue)
            )

        # Local storage for bond-dims of each trajectory and failed trajectories
        all_bond_dims = []
        failed_trajectories = []

        # 4) Start the process pool
        #max_workers = max(1, multiprocessing.cpu_count() - 1)
        max_workers = self.max_workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map each args tuple to a future
            futures = {
                executor.submit(self.run_trajectory_second_order, arg): arg[0]
                for arg in args_list}

            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    # Get (bond_dim_list, results_dict) from each worker
                    local_bond_dims, local_results = future.result()

                    # Merge trajectory’s results into self.trajectories
                    for op_id, op_data in local_results.items():
                        # op_data is your per-time-step array for that operator
                        self.trajectories[op_id][i] = op_data

                    # Collect bond dims so we can store/analyze them
                    all_bond_dims.append((i, local_bond_dims))

                except Exception as e:
                    print(f"\nTrajectory {i} failed with exception: {e}. Retrying...")
                    # You could retry here if desired
                    failed_trajectories.append(i)

        # 5) All workers are done, stop the monitor thread
        stop_event.set()
        monitor_thread.join()


        # 6) Average results across trajectories
        for operator in self.operators:
            op_id = list(operator.keys())[0]
            n_traj = self.trajectories[op_id].shape[0]
            mask = np.ones(n_traj, dtype=bool)
            mask[failed_trajectories] = False
            self.trajectories[op_id] = self.trajectories[op_id][mask]
                
            # self.trajectories[op_id] shape is [N, number_of_time_points]
            self.results[op_id] = np.mean(self.trajectories[op_id], axis=0)

        # 7) Store the bond-dimension data from each trajectory
        #    For example, as a dict of {trajectory_index: [dims over time]}
        self.bond_dims = {traj_idx: dims for traj_idx, dims in all_bond_dims}

        # 8) Save results to file if desired
        self.save_results_to_file_TJM(filepath)  

# EXPANSION

    def perform_expansion(self, basis, tol):
        self.config.Expansion_params["tol"] = tol
        ttn_ex = basis[0]
        for i in range(len(basis)-1):
            ttn_ex = enlarge_ttn1_bond_with_ttn2(ttn_ex, basis[i+1], self.config.Expansion_params["tol"])
        state_ex = ttn_ex
        after_ex_total_bond = state_ex.total_bond_dim()
        expanded_dim_tot = after_ex_total_bond - basis[0].total_bond_dim()
        return state_ex, after_ex_total_bond, expanded_dim_tot

    def phase1_increase_tol(self, basis, tol, expanded_dim_tot, progress_queue):
        max_trials = self.config.Expansion_params["num_second_trial"]
        num_trials = 0
        min_rel_tot_bond, max_rel_tot_bond = self.config.Expansion_params["rel_tot_bond"]
        while num_trials < max_trials:
            #progress_queue.put(("log", f"Phase 1 - Trial {num_trials+1}:"))
            if expanded_dim_tot > max_rel_tot_bond:
                #progress_queue.put(("log",f"Expanded dim ({expanded_dim_tot}) > rel_tot_bond ({self.config.Expansion_params['rel_tot_bond']})"))
                # Increase tol to reduce expanded_dim_tot
                tol += self.config.Expansion_params["tol_step_increase"]
                #progress_queue.put(("log","Increasing tol:", tol))
                state_ex, _, expanded_dim_tot = self.perform_expansion(basis, tol)
                num_trials += 1
                if min_rel_tot_bond <= expanded_dim_tot <= max_rel_tot_bond:
                    # Acceptable expansion found
                    #progress_queue.put(("log", f"Acceptable expansion found in Phase 1: {expanded_dim_tot}"))
                    return state_ex, tol, expanded_dim_tot, False
                elif expanded_dim_tot < min_rel_tot_bond:
                    # Need to switch to Phase 2
                    #progress_queue.put(("log", f"Expanded dim: {expanded_dim_tot} falls below min_rel_tot_bond: {min_rel_tot_bond}"))
                    if expanded_dim_tot <= 0:
                       state_ex = basis[0]
                    #progress_queue.put(("log","Switching to Phase 2"))
                    return state_ex, tol, expanded_dim_tot, True  # Proceed to Phase 2                
        # Exceeded max trials
        #progress_queue.put(("log","Exceeded maximum trials in Phase 1 without acceptable expansion"))
        state_ex = basis[0]
        tol += self.config.Expansion_params["tol_step_increase"]
        return state_ex, tol, expanded_dim_tot, False  # Proceed to Phase 2

    def phase2_decrease_tol(self, basis, tol, expanded_dim_tot,progress_queue):
        max_trials = self.config.Expansion_params["num_second_trial"]
        num_trials = 0
        min_rel_tot_bond, max_rel_tot_bond = self.config.Expansion_params["rel_tot_bond"]
        while num_trials < max_trials:
            num_trials += 1
            #progress_queue.put(("log",f"Phase 2 - Trial {num_trials}:"))
            if expanded_dim_tot < min_rel_tot_bond:
                # Decrease tol to increase expanded_dim_tot
                tol -= self.config.Expansion_params["tol_step_decrease"]
                #progress_queue.put(("log", f"Decreasing tol: {tol}"))
                state_ex, _, expanded_dim_tot = self.perform_expansion(basis, tol)
                #progress_queue.put(("log", f"Expanded_dim_tot: {expanded_dim_tot}"))
                if min_rel_tot_bond <= expanded_dim_tot <= max_rel_tot_bond:
                    # Acceptable expansion found
                    #progress_queue.put(("log", f"Acceptable expansion found in Phase 2: {expanded_dim_tot}"))
                    return state_ex, tol, expanded_dim_tot
                elif expanded_dim_tot > max_rel_tot_bond:
                    # Expanded dimension exceeded rel_tot_bond again
                    #progress_queue.put(("log", f"Expanded dim exceeded rel_tot_bond again: {expanded_dim_tot}"))
                    # Reset state_ex to initial state
                    state_ex = basis[0]
                    return state_ex, tol, expanded_dim_tot  # Reset and exit
        # Exceeded max trials
        #progress_queue.put(("log","Exceeded maximum trials in Phase 2 without acceptable expansion"))
        # Reset state_ex to initial state
        state_ex = basis[0]
        tol -= self.config.Expansion_params["tol_step_decrease"]
        return state_ex, tol, expanded_dim_tot  # Reset and exit

    def adjust_tol_and_expand(self, tol, progress_queue):
        before_ex_total_bond = self.state.total_bond_dim()
        
        #self.config.Expansion_params["SVDParameters"] = replace(self.config.Expansion_params["SVDParameters"],max_bond_dim=state.max_bond_dim())
        #print("SVD MAX:", state.max_bond_dim())
        #progress_queue.put(("log", f"Initial tol: {tol}"))

        basis = Krylov_basis(self.state,
                             self.hamiltonian,
                             self.config.Expansion_params["num_vecs"],
                             self.config.Expansion_params["tau"],
                             self.config.Expansion_params["SVDParameters"],
                             self.config.Expansion_params["KrylovBasisMode"] )

        # Initial Expansion Attempt
        state_ex, after_ex_total_bond, expanded_dim_tot = self.perform_expansion(basis, tol)

        # Unpack the acceptable range
        min_rel_tot_bond, max_rel_tot_bond = self.config.Expansion_params["rel_tot_bond"]

        # Check initial expansion
        if min_rel_tot_bond <= expanded_dim_tot <= max_rel_tot_bond:
            # Acceptable expansion found in initial attempt
            #progress_queue.put(("log", f"Acceptable expansion found in initial attempt: {expanded_dim_tot}"))
            pass
        elif expanded_dim_tot > max_rel_tot_bond:
            # Need to adjust tol in Phase 1
            state_ex, tol, expanded_dim_tot, switch_to_phase_2 = self.phase1_increase_tol(basis, tol, expanded_dim_tot,progress_queue)
            if switch_to_phase_2:
                # Switch to Phase 2
                state_ex, tol, expanded_dim_tot = self.phase2_decrease_tol(basis, tol, expanded_dim_tot,progress_queue)
        elif expanded_dim_tot < min_rel_tot_bond:
            # Need to adjust tol in Phase 2
            state_ex, tol, expanded_dim_tot = self.phase2_decrease_tol(basis, tol, expanded_dim_tot,progress_queue)

        # state_ex, should_expand = self.check_overgrown_bond_dimensions(state_ex, state, progress_queue)
        _, should_expand = self.check_overgrown_bond_dimensions(state_ex, self.state, progress_queue)

        after_ex_total_bond = state_ex.total_bond_dim()
        expanded_dim_total_bond = after_ex_total_bond - before_ex_total_bond

        progress_queue.put(("log", f"Final expanded_dim: {expanded_dim_total_bond} : {before_ex_total_bond} ---> {after_ex_total_bond}"))

        return state_ex, tol, should_expand
    

    def check_overgrown_bond_dimensions(self, ttn_ex, ttn, progress_queue):
        if ttn_ex.total_bond_dim() >= self.config.Expansion_params["max_bond"]:
            progress_queue.put(("log", f"Exceed max bond dimension: {self.config.Expansion_params['max_bond']}"))
            ttn_ex = ttn
            should_expand = False
        else:
            should_expand = True
        return ttn_ex, should_expand


def normalize_ttn(ttn , to_copy = False):
    """
    Normalize a tree tensor network.
    Args:
        ttn : TreeTensorNetwork
        The tree tensor network to normalize.
        to_copy : bool, optional
                  If True, the input tree tensor network is not modified and a new tree tensor network is returned.
                  If False, the input tree tensor network is modified and returned.
                  Default is False.
    Returns : 
        The normalized tree tensor network.
    """
    if  to_copy:
        ttn_normalized = deepcopy(ttn)
    else:
        ttn_normalized = copy(ttn)

    # ttn_normalized_conj = ttn_normalized.conjugate()
    # norm = contract_two_ttns(ttn_normalized,ttn_normalized_conj) 
    norm = scalar_product(ttn_normalized ,use_orthogonal_center =  True)   
    if  ttn_normalized.is_in_canonical_form():
        ttn_normalized.tensors[ttn_normalized.orthogonality_center_id] = ttn_normalized.tensors[ttn_normalized.orthogonality_center_id].astype(complex) / np.sqrt(norm)
    else :
        n = len(ttn.nodes) 
        norm = np.sqrt(norm ** (1/n))
        for node_id in list(ttn_normalized.nodes.keys()):
            ttn_normalized.tensors[node_id] = ttn_normalized.tensors[node_id].astype(complex) / norm    
    return ttn_normalized 

def scalar_product(state: TreeTensorNetworkState,
                    use_orthogonal_center: bool = True
                    ) -> complex:
        """
        Computes the scalar product of this TTNS.

        Args:
            other (Union[TreeTensorNetworkState,None], optional): The other
                TTNS to compute the scalar product with. If None, the scalar
                product is computed with itself. Defaults to None.
            use_orthogonal_center (bool, optional): Whether to use the current
                orthogonalization center to compute  the norm. This usually
                speeds up the computation. Defaults to True.

        Returns:
            complex: The resulting scalar product <TTNS|Other>
        """
        if state.orthogonality_center_id is not None and use_orthogonal_center:
            tensor = state.tensors[state.orthogonality_center_id]
            tensor_conj = tensor.conj()
            legs = tuple(range(tensor.ndim))
            return complex(np.tensordot(tensor, tensor_conj, axes=(legs,legs)))
        else:
            other_conj = state.conjugate()    
            return contract_two_ttns(state, other_conj)
        
