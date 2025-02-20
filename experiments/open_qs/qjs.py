from typing import Union, List
from copy import deepcopy

from numpy.random import default_rng, Generator
from tqdm import tqdm

from pytreenet.time_evolution import TTNTimeEvolution, TTNTimeEvolutionConfig
from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.util.ttn_exceptions import non_negativity_check

class QuantumJumpMethod(TTNTimeEvolution):
    """
    A class representing the quantum jump method.

    Note that any system model given to this class must be the effective
    model of the system, i.e. the Hamiltonian including the jump operators.

    Attributes:
        time_evo_initial (TTNTimeEvolution): The intialised intial state of th
            time evolution with which the jump operator method should be
            performed.
    """

    def __init__(self,
                 time_evolution: type,
                 num_trajectories: int,
                 seed: Union[int,Generator, None] = None,
                 jump_operators: Union[None, List[TensorProduct]] = None,
                 *args, **kwargs):
        """
        Initialises an instance of the quantum jump method.

        The ``*args`` and ``**kwargs`` are the arguments of the time evolution
        method.
        """
        self.time_evo_initial: TTNTimeEvolution = time_evolution(*args, **kwargs)
        record_bond_dim = self.time_evo_initial.records_bond_dim
        config = TTNTimeEvolutionConfig(record_bond_dim=record_bond_dim)
        super().__init__(self.time_evo_initial.initial_state,
                         self.time_evo_initial.time_step_size,
                         self.time_evo_initial.final_time,
                         self.time_evo_initial.operators,
                         config=config)
        self.rng = default_rng(seed=seed)
        self.current_trajectory: Union[None, TTNTimeEvolution] = None
        if jump_operators is not None:
            self.jump_operators = jump_operators
        else:
            self.jump_operators = []
        # We need L^dagger*L a lot
        self.jump_op_herm = [TensorProduct({node_id: op.conj().T @ op
                                            for op, node_id in jump_op.items()})
                             for jump_op in self.jump_operators]
        non_negativity_check(num_trajectories, "number of trajectories")
        self.num_trajectories = num_trajectories

    def generate_uniform_random_number(self) -> float:
        """
        Generates a uniform random number in [0,1).
        """
        return self.rng.random()

    def apply_jump_operator(self, state: TreeTensorNetworkState):
        """
        Apply a jump operator to the current trajectory.
        """
        probabilities = []
        for operator in self.jump_op_herm:
            exp_value = state.operator_expectation_value(operator)
            probabilities.append(exp_value)
        sum_tot = sum(probabilities)
        probabilities = [prob / sum_tot for prob in probabilities]
        threshold = self.generate_uniform_random_number()
        for i, prob in enumerate(probabilities):
            threshold -= prob
            if threshold < 0:
                state.absorb_into_open_legs(self.jump_operators[i])
                break
        self.current_trajectory.state = state.normalise()

    def run_one_trajectory_step(self):
        """
        Run one time step of a single trajectory.
        """
        threshold = self.generate_uniform_random_number()
        self.current_trajectory.run_one_time_step()
        current_state = self.current_trajectory.state
        norm = current_state.scalar_product()
        if norm < threshold:
            self.apply_jump_operator(current_state)

    def run_one_trajectory(self,
                           evaluation_time: Union[int, str] = "inf"):
        """
        Run a single trajectory.
        """
        self.current_trajectory = deepcopy(self.time_evo_initial)
        for i in range(self.num_time_steps + 1):
            if i != 0:  # We also measure the initial expectation_values
                self.run_one_time_step()
            if evaluation_time != "inf" and i % evaluation_time == 0 and len(self._results) > 0:
                index = i // evaluation_time
                current_results = self.current_trajectory.evaluate_operators()
                self._results[0:-1, index] += current_results
                # Save current time
                self._results[-1, index] += i*self.time_step_size
        if evaluation_time == "inf":
            current_results = self.current_trajectory.evaluate_operators()
            self._results[0:-1, 0] += current_results
            self._results[-1, 0] += i*self.time_step_size

    def run(self, evaluation_time: Union[int,str] = "inf",
            pgbar: bool = True):
        """
        Run the quantum jump method.
        """
        self._init_results(evaluation_time)
        for _ in tqdm(range(self.num_trajectories), disable=not pgbar):
            self.run_one_trajectory(evaluation_time)
        self._results /= self.num_trajectories
