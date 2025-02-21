"""
This module provides the abstract TimeEvolution class
"""
from __future__ import annotations
from typing import List, Union, Any, Dict, Iterable
from enum import Enum
from copy import deepcopy
from math import modf

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from h5py import File

from ..util.ttn_exceptions import positivity_check, non_negativity_check
from ..util import fast_exp_action

class TimeEvolution:
    """
    An abstract class that can be used for various time-evolution algorithms.

    Contains all methods required to cover common discrete time evolutions and
    avoid code duplication. The algorithms runs a number of time steps of given
    size until a specified final time is reached.

    While results can be called directly as a big martix using the `results`
    attribute, there are convenience functions to easily extract the desired
    results.

    Attributes:
        initial_state (Any): The initial state of the system to be time evolved.
        time_step_size (float): The size of each discreet time step.
        final_time (float): The time at which to conclude the time-evolution.
        operators (List[Any]): A list of operators to be evaluated.
    """

    def __init__(self, initial_state: Any, time_step_size: float,
                 final_time: float, operators: Union[List[Any], Dict[str, Any], Any]):
        """
        Initialises a TimeEvoluion object.
        
        Args:
            initial_state (Any): The initial state.
            time_step_size (float): The size of one time step.
            final_time (float): The final time.
            operators (Union[List[Any], Dict[str, Any], Any]): The operators
                for which to compute the expectation values. Can be a single
                operator or a list of operators. If a dictionary is given, the
                results can be called by using the keys of the dictionary.
        """
        self._initial_state = initial_state
        self.state = deepcopy(initial_state)
        positivity_check(time_step_size, "size of one time step")
        self._time_step_size = time_step_size
        positivity_check(final_time, "final time")
        self._final_time = final_time
        self._num_time_steps = self._compute_num_time_steps()
        self._operator_index_dict = self._init_operator_index_dict(operators)
        if isinstance(operators, List):
            self.operators = operators
        elif isinstance(operators, Dict):
            self.operators = list(operators.values())
        else:
            # A single operator was provided
            self.operators = [operators]
        self._results = None

    def _compute_num_time_steps(self) -> int:
        """
        Compute the number of time steps from attributes.

        If the decimal part of the time steps is close to 0, the calculated
        number of time steps is directly returned. Otherwise, it is assumed
        to be better to run one more time step.
        """
        decimal, integer = modf(self._final_time / self._time_step_size)
        threshold = 0.1
        if decimal < threshold:
            return int(integer)
        return int(integer + 1)

    def _init_operator_index_dict(self,
                                  operators: Union[List[Any], Dict[str, Any], Any]) -> Dict[str, int]:
        """
        Initialise a dictionary mapping from operator keys to results indices.

        If the operator is given alone or as a list, an empty dictionary is
        returned. If a dictionary is given, the keys of the dictionary are used
        as keys for the operator index dictionary. This allows the access of
        results via the given operator key.
        
        Args:
            operators (Union[List[Any], Dict[str, Any], Any]): The operators
                for which to compute the expectation values during time
                evolution.
        
        Returns:
            Dict[str, int]: The operator index dictionary.
        """
        if isinstance(operators, dict):
            return {key: i for i, key in enumerate(operators.keys())}
        return {}

    @property
    def initial_state(self) -> Any:
        """
        Returns the initial state.
        """
        return self._initial_state

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
        self.check_result_exists()
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

    def set_num_time_steps(self, num_time_steps: int):
        """
        Set the number of time steps to be run.
        
        Sometimes it is more convenient to define the size of the time steps
        and the number of steps to be run, rather than using a final time.
        This method modifies the internal attributes accordingly.
        """
        non_negativity_check(num_time_steps, "number of time steps")
        self._num_time_steps = num_time_steps
        self._final_time = num_time_steps * self._time_step_size

    def set_num_time_steps_constant_final_time(self, num_time_steps: int):
        """
        Sets the number of time-steps and keeps the final time constant.
        
        The internal attributes are modified accordingly.
        """
        non_negativity_check(num_time_steps, "number of time steps")
        self._num_time_steps = num_time_steps
        self._time_step_size = self._final_time / num_time_steps

    def check_result_exists(self):
        """
        Checks if results have been obtained.
        """
        if self._results is None:
            errstr = "Currently there are no results!"
            raise AssertionError(errstr)

    def results_real(self) -> bool:
        """
        Returns if the results are real.
        """
        return np.allclose(np.imag(self.results), np.zeros_like(self._results))

    def times(self, offset: float = 0.0) -> np.ndarray:
        """
        Returns the times at which the operators were evaluated.
        """
        return np.real(self.results[-1]) + offset

    def operator_result(self, operator_id: Union[str, int],
                        realise: bool = False) -> np.ndarray:
        """
        Returns the result of a single operator.

        The operator can be a string, if the operators were originally provided
        with strings as keys. Otherwise one has to provide the correct index
        at which the operator was saved.

        Args:
            operator_id (Union[str, int]): The index or key of the operator.
            realise (bool, optional): If the imaginary part of the results
                should be discarded. Defaults to False.
        
        Returns:
            np.ndarray: The result of the operator as a vector.
        """
        self.check_result_exists()
        if isinstance(operator_id, str):
            operator_id = self._operator_index_dict[operator_id]
        if realise:
            return np.real(self.results[operator_id])
        return self.results[operator_id]

    def operator_results(self, realise: bool = False) -> np.ndarray:
        """
        Returns all of the operator results.

        Args:
            realise (bool, optional): If the imaginary part of the results
                should be discarded. Defaults to False.
        
        Returns:
            np.ndarray: The operator results in the same order as the operators
                were given.
        """
        if realise:
            return np.real(self.results[0:-1])
        return self.results[0:-1]

    def run_one_time_step(self, **kwargs):
        """
        Abstract method to run one time step.
        """
        raise NotImplementedError()

    def evaluate_operator(self, operator: Any) -> complex:
        """
        Abstract method to evaluate the expectation value of a single operator.

        Args:
            operator (Any): The operator for which to compute the expectation
                value.

        Returns:
            complex: The expectation value of the operator.
        """
        raise NotImplementedError()

    def evaluate_operators(self) -> np.ndarray:
        """
        Evaluate the expectation value for all operators for the current state.

        Returns:
            List: The expectation values with indeces corresponding to those in
                operators.
        """
        current_results = np.zeros(len(self.operators), dtype=complex)
        for i, operator in enumerate(self.operators):
            exp_val = self.evaluate_operator(operator)
            current_results[i] = exp_val
        return current_results

    def save_results_to_file(self, filepath: str | File):
        """
        Saves the data of `self.results` into a .npz file.

        Args:
            filepath (str | File): The path of the file.
        """
        if isinstance(filepath, File):
            self.save_to_h5(filepath)
            return
        if filepath == "":
            return # No filepath given, no need to save anything
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

    def save_to_h5(self, file: File) -> File:
        """
        Saves the data of `self.results` into a HDF5 file.

        Args:
            file (File): The HDF5 file to save the data to.

        Returns:
            File: The file object.

        """
        for op_id in self._operator_index_dict.keys():
            results = self.operator_result(op_id)
            file.create_dataset(op_id, data=results)
        file.create_dataset("time", data=self.times())
        return file

    def init_results(self, evaluation_time: Union[int,"inf"] = 1):
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
        if evaluation_time != "inf":
            self._results = np.zeros((len(self.operators) + 1,
                                    self.num_time_steps//evaluation_time + 1),
                                    dtype=complex)
        else:
            self._results = np.zeros((len(self.operators) + 1, 1),
                                    dtype=complex)
        assert self._results is not None

    def save_operator_results(self,
                              results: np.ndarray,
                              index: int):
        """
        Saves the results of an operator evaluation at a given time index.

        Args:
            results (np.ndarray): The results of the operator evaluation.
            index (int): The index at which to save the results.
        
        """
        self._results[0:-1, index] = results

    def save_time(self, time_step: int, index: int):
        """
        Saves the current time at a given index.

        Args:
            time_step (int): The current time step.
            index (int): The index at which to save the time.

        """
        self._results[-1, index] = time_step*self.time_step_size

    def result_index(self,
                     evalutation_time: Union[int,"inf"],
                     time_step: int) -> int:
        """
        Returns the time index at which to save the results.

        Args:
            evalutation_time (Union[int,"inf"]): The difference in time steps
                after which to evaluate the operator expectation values.
            time_step (int): The current time step.
        
        Returns:
            int: The index at which to save the results.

        """
        if evalutation_time == "inf":
            return 0
        return time_step // evalutation_time

    def should_evaluate(self,
                        evaluation_time: Union[int,"inf"],
                        time_step: int
                        ) -> bool:
        """
        Returns if the operators should be evaluated at the current time step.

        Args:
            evaluation_time (Union[int,"inf"]): The difference in time steps after which
                to evaluate the operator expectation values.
            time_step (int): The current time step.
        
        Returns:
            bool: If the operators should be evaluated at the current time step.
        
        """
        some_time_step = evaluation_time != "inf" and time_step % evaluation_time == 0
        end_time = evaluation_time == "inf" and time_step == self.num_time_steps
        return some_time_step or end_time

    def evaluate_and_save_results(self,
                                  evaluation_time: Union[int,"inf"],
                                  time_step: int):
        """
        Evaluates and saves the operator results at a given time step.

        Args:
            evaluation_time (Union[int,"inf"]): The difference in time steps after which
                to evaluate the operator expectation values, e.g. for a value of 10
                the operators are evaluated at time steps 0,10,20,... If it is set to
                "inf", the operators are only evaluated at the end of the time.
            time_step (int): The current time step.

        """
        if self.should_evaluate(evaluation_time, time_step):
            index = self.result_index(evaluation_time, time_step)
            current_results = self.evaluate_operators()
            self.save_operator_results(current_results, index)
            self.save_time(time_step, index)

    def create_run_tqdm(self, pgbar: bool = True) -> Iterable:
        """
        Creates the decorated iterator for the progress bar.

        Args:
            pgbar (bool, optional): If True, a progress bar is shown. 
                Defaults to True.
        
        Returns:
            Iterable: The decorated iterator.

        """
        return tqdm(range(self.num_time_steps + 1), disable=not pgbar)

    def run(self, evaluation_time: Union[int,"inf"] = 1, filepath: str = "",
            pgbar: bool = True):
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
        self.init_results(evaluation_time)
        for i in self.create_run_tqdm(pgbar):
            if i != 0:  # We also measure the initial expectation_values
                self.run_one_time_step()
            self.evaluate_and_save_results(evaluation_time, i)
        self.save_results_to_file(filepath)

    def reset_to_initial_state(self):
        """
        Resets the current state to the intial state
        """
        self.state = deepcopy(self._initial_state)

class TimeEvoMode(Enum):

    FASTEST = "fastest"
    EXPM = "expm"
    EIGSH = "eigsh"
    CHEBYSHEV = "chebyshev"
    SPARSE = "sparse"
    RK45 = "RK45"
    RK23 = "RK23"
    DOP853 = "DOP853"
    BDF = "BDF"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def fastest_equivalent() -> TimeEvoMode:
        """
        Selects the mode that is equivalent to the fastest.
        """
        return TimeEvoMode.CHEBYSHEV

    def is_scipy(self) -> bool:
        """
        Determines, if this mode is a scipy ODE solver.
        """
        if self == TimeEvoMode.FASTEST:
            return self.fastest_equivalent().is_scipy()
        return self in [TimeEvoMode.RK45,
                        TimeEvoMode.RK23,
                        TimeEvoMode.DOP853,
                        TimeEvoMode.BDF]

def time_evolve(psi: np.ndarray, hamiltonian: np.ndarray,
                time_difference: float,
                forward: bool = True,
                mode: TimeEvoMode = TimeEvoMode.FASTEST) -> np.ndarray:
    """
    Time evolves a state psi via a Hamiltonian.
     
    The evolution can be either forward or backward in time:
        psi(t +/- dt) = exp(-/+ i*h*dt) @ psi(t)
        -iHdt: forward = True
        +iHdt: forward = False

    Args:
        psi (np.ndarray): The initial state as a vector.
        hamiltonian (np.ndarray): The Hamiltonian determining the dynamics as
            a matrix.
        time_difference (float): The duration of the time-evolution
        forward (bool, optional): If the time evolution should be forward or
            backward in time. Defaults to True.
        mode (TimeEvoMode, optional): The mode to use for the time evolution.

    Returns:
        np.ndarray: The time evolved state
    """
    sign = -2 * forward + 1  # forward=True -> -1; forward=False -> +1
    rhs_matrix = sign * 1.0j * hamiltonian
    if mode.is_scipy():
        def ode_rhs(_, y_vec):
            return rhs_matrix @ y_vec
        t_span = (0, time_difference)
        solution = solve_ivp(ode_rhs, t_span, psi.flatten(),
                                method=mode.value,
                                t_eval=[time_difference])
        result_vector = solution.y[:,0]
    else:
        exponent = rhs_matrix * time_difference
        result_vector = fast_exp_action(exponent, psi.flatten(),
                                        mode=mode.value)
    return np.reshape(result_vector,
                      shape=psi.shape)
