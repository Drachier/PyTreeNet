from __future__ import annotations
from typing import List, Union, Any

from copy import deepcopy
from math import modf

import numpy as np
from tqdm import tqdm

from ..ttn_exceptions import positiviy_check
from ..util import fast_exp_action

class TimeEvolution:
    """
    An abstract class that can be used for various time-evolution algorithms.
    """

    def __init__(self, initial_state: Any, time_step_size: float,
                 final_time: float, operators: Union[List[Any], Any]):
        """
        A time evolution starting from and initial state and running to a
         final time with a given time step size. During the time evolution,
         expectation values of operators are computed.
        
        Args:
            initial_state (Any): The initial state.
            time_step_size (float): The size of one time step.
            final_time (float): The final time.
            operators (Union[List[Any], Any]): The operators for which to
             compute the expectation values. Can be a single operator or a
             list of operators.
        """
        self._intital_state = initial_state
        self.state = deepcopy(initial_state)
        positiviy_check(time_step_size, "size of one time step")
        self._time_step_size = time_step_size
        positiviy_check(final_time, "final time")
        self._final_time = final_time
        self._num_time_steps = self._compute_num_time_steps()
        if isinstance(operators, List):
            self.operators = operators
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
        if decimal < 0.1:
            return int(integer)
        return int(integer + 1)

    @property
    def initial_state(self) -> Any:
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

    def check_result_exists(self):
        """
        Checks if results have been obtained.
        """
        if self._results is None:
            errstr = "Currently there are no results!"
            raise AssertionError(errstr)

    def results_real(self) -> bool:
        """
        Checks if the results are real.
        """
        return np.allclose(np.imag(self.results), np.zeros_like(self._results))

    def times(self) -> np.ndarray:
        """
        Returns the times at which the operators were evaluated.
        """
        return np.real(self.results[-1])

    def operator_results(self, realise: bool = False) -> np.ndarray:
        """
        Returns the operator results.

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

    def run_one_time_step(self):
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
        Evaluates the expectation value for all operators given in
        `self.operators` for the current TTNS.

        Returns:
            List: The expectation values with indeces corresponding to those in
             operators.
        """
        current_results = np.zeros(len(self.operators), dtype=complex)
        for i, operator in enumerate(self.operators):
            exp_val = self.evaluate_operator(operator)
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

    def _init_results(self, evaluation_time: Union[int,"inf"] = 1):
        """
        Initialises an appropriately sized zero valued numpy array to save
         all aquired measurements into.
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

    def run(self, evaluation_time: Union[int,"inf"] = 1, filepath: str = "",
            pgbar: bool = True):
        """
        Runs this time evolution algorithm for the given parameters and
         saves the computed expectation values.

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
        for i in tqdm(range(self.num_time_steps + 1), disable=not pgbar):
            if i != 0:  # We also measure the initial expectation_values
                self.run_one_time_step()
            if evaluation_time != "inf" and i % evaluation_time == 0 and len(self._results) > 0:
                index = i // evaluation_time
                current_results = self.evaluate_operators()
                self._results[0:-1, index] = current_results
                # Save current time
                self._results[-1, index] = i*self.time_step_size
        if evaluation_time == "inf":
            current_results = self.evaluate_operators()
            self._results[0:-1, 0] = current_results
            self._results[-1, 0] = i*self.time_step_size
        if filepath != "":
            self.save_results_to_file(filepath)

    def reset_to_initial_state(self):
        """
        Resets the current state to the intial state
        """
        self.state = deepcopy(self._intital_state)

def time_evolve(psi: np.ndarray, hamiltonian: np.ndarray,
                time_difference: float,
                forward: bool = True) -> np.ndarray:
    """
    Time evolves a state psi via a Hamiltonian either forward or backward in
     time by a certain time difference:
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

    Returns:
        np.ndarray: The time evolved state
    """
    sign = -2 * forward + 1  # forward=True -> -1; forward=False -> +1
    exponent = sign * 1.0j * hamiltonian * time_difference
    return np.reshape(fast_exp_action(exponent, psi.flatten(), mode="fastest"),
                      newshape=psi.shape)
