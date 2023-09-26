from __future__ import annotations
from typing import List, Union

from copy import deepcopy

import numpy as np
from tqdm import tqdm

from ..ttns import TreeTensorNetworkState
from ..operators.tensorproduct import TensorProduct

class TimeEvolution:
    """
    An abstract class that can be used for various time-evolution algorithms.
    """

    def __init__(self, initial_state: TreeTensorNetworkState, time_step_size: float,
                 final_time: float, operators: Union[List[TensorProduct], TensorProduct]):
        """
        A time evolution starting from an initial state and running to a final
         time with a given time step size.

        Args:
            initial_state (TreeTensorNetworkState): The initial state of our
             time-evolution
            time_step_size (float): The time step size to be used.
            final_time (float): The final time until which to run.
            operators (Union[List[TensorProduct], TensorProduct]): Operators in 
             the form of single site tensor product for which expectation values
             should be determined.
        """
        self._intital_state = initial_state
        self.state = deepcopy(initial_state)
        if time_step_size <= 0:
            errstr = "The size of one time step has to be positive!"
            raise ValueError(errstr)
        self._time_step_size = time_step_size
        if final_time <= 0:
            errstr = "The final time has to be positive!"
            raise ValueError(errstr)
        self._final_time = final_time
        self._num_time_steps = self._compute_num_time_steps()
        if isinstance(operators, TensorProduct):
            # A single operator was provided
            self.operators = [operators]
        else:
            self.operators = operators
        # Place to hold the results obtained during computation
        # Each row contains the data obtained during the run and the last row
        # contains the time_steps.
        self._results = np.zeros((len(self.operators) + 1, self.num_time_steps + 1),
                                 dtype=complex)

    def _compute_num_time_steps(self) -> int:
        """
        Compute the number of time steps from attributes.
        """
        return int(np.ceil(self._final_time / self._time_step_size))

    @property
    def initial_state(self) -> TreeTensorNetworkState:
        """
        Returns the initial state.
        """
        return self._intital_state

    @property
    def time_step_size(self) -> float:
        """
        Returns the size of one time step.
        """
        return self._time_step_size

    @property
    def results(self) -> np.ndarray:
        """
        Returns the currently obtained results
        """
        return self._results

    @property
    def final_time(self) -> float:
        """
        Returns the final time.
        """
        return self._final_time

    @property
    def num_time_steps(self) -> int:
        """
        Returns the current number of time steps.
        """
        return self._num_time_steps

    def run_one_time_step(self):
        """
        Abstract method to run one time step.
        """
        raise NotImplementedError()

    def evaluate_operators(self) -> List:
        """
        Evaluates the expectation value for all operators given in
        `self.operators` for the current TTNS.

        Returns:
            List: The expectation values with indeces corresponding to those in
             operators.
        """
        current_results = np.zeros(len(self.operators), dtype=complex)
        for i, tensor_product in enumerate(self.operators):
            exp_val = self.state.operator_expectation_value(tensor_product)
            current_results[i] = exp_val
        return current_results

    def save_results_to_file(self, filepath: str):
        """
        Saves the data of `self.results` into a .npz file.

        Args:
            filepath (str): The path of the file.
        """
        if filepath is None:
            print("No filepath given. Data wasn't saved.")
            return
        # We have to lable our data
        kwarg_dict = {}
        for i, operator in enumerate(self.operators):
            kwarg_dict["operator" + str(i)] = operator
            kwarg_dict["operator" + str(i) + "results"] = self.results[i]
        kwarg_dict["time"] = self.results[-1]
        np.savez(filepath, **kwarg_dict)

    def run(self, evaluation_time: int = 1, filepath: str = "", pgbar: bool = True):
        """
        Runs this time evolution algorithm for the given parameters and
         saves the computed expectation values.

        Args:
            evaluation_time (int, optional): The difference in time steps after which
                to evaluate the operator expectation values, e.g. for a value 0f 10
                the operators are evaluated at time steps 0,10,20,... Defaults to 1.
            filepath (str, optional): If results are to be saved in an external file,
             the path to that file can be specified here. Defaults to "".
            pgbar (bool, optional): Toggles the progress bar. Defaults to True.
        """
        # Always start from the same intial state
        self.state = deepcopy(self.intital_state)
        for i in tqdm(range(self.num_time_steps + 1), disable=not pgbar):
            if i != 0:  # We also measure the initial expectation_values
                self.run_one_time_step()
            if i % evaluation_time == 0 and len(self._results) > 0:
                current_results = self.evaluate_operators()
                self._results[0:-1, i] = current_results
                # Save current time
                self._results[-1, i] = i*self.time_step_size
        if filepath != "":
            self.save_results_to_file(filepath)

    def reset_to_initial_state(self):
        """
        Resets the current state to the intial state
        """
        self.state = deepcopy(self.intital_state)
