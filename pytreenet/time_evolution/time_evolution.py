"""
This module provides the abstract TimeEvolution class
"""
from __future__ import annotations
from typing import List, Union, Any, Dict, Iterable, Callable
from enum import Enum
from copy import deepcopy
from math import modf
from warnings import warn
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from numpy.typing import DTypeLike

from ..util.ttn_exceptions import positivity_check, non_negativity_check
from ..util import fast_exp_action
from .results import Results

@dataclass
class TimeEvoConfig:
    """
    Abstract configuration for time evolution algorithms.
    
    This is to ensure that all time evolution algorithms can be configured.
    """
    time_dep: bool = False

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
    config_class = TimeEvoConfig

    def __init__(self,
                 initial_state: Any,
                 time_step_size: float,
                 final_time: float,
                 operators: Union[List[Any], Dict[str, Any], Any],
                 solver_options: Union[Dict[str, Any], None] = None,
                 config: Union[TimeEvoConfig, None] = None):
        """
        Initialises a TimeEvolution object.

        Args:
            initial_state (Any): The initial state.
            time_step_size (float): The size of one time step.
            final_time (float): The final time.
            operators (Union[List[Any], Dict[str, Any], Any]): The operators
                for which to compute the expectation values. Can be a single
                operator or a list of operators. If a dictionary is given, the
                results can be called by using the keys of the dictionary.
            solver_options (Union[Dict[str, Any], None], optional): Most time
                evolutions algorithms use some kind of solver to resolve a
                partial differential equation. This dictionary can be used to
                pass additional options to the solver. Refer to the
                documentation of `ptn.time_evolution.TimeEvoMode` for further
                information. Defaults to None.
            config (Union[TimeEvoConfig, None], optional): The configuration
                for the time evolution algorithm. If None, a default
                configuration is used.
        """
        self._initial_state = initial_state
        self.state = deepcopy(initial_state)
        positivity_check(time_step_size, "size of one time step")
        self._time_step_size = time_step_size
        positivity_check(final_time, "final time")
        self._final_time = final_time
        self._num_time_steps = self._compute_num_time_steps()
        self.operators = self.init_operators(operators)
        if config is None:
            self.config = self.config_class()
        else:
            self.config = config
        if solver_options is None:
            self.solver_options = {}
        else:
            self.solver_options = solver_options
        self.results = Results()

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

    def init_operators(self,
                       operators: Union[List[Any], Dict[str, Any], Any]
                       ) -> Dict[str, Any]:
        """
        Initialises the operators for which to compute the expectation values.

        Args:
            operators (Union[List[Any], Dict[str, Any], Any]): The operators
                for which to compute the expectation values. Can be a single
                operator or a list of operators. If a dictionary is given, the
                results can be called by using the keys of the dictionary
                otherwise the keys are the indices of the operators in the list.

        Returns:
            Dict[str, Any]: A dictionary mapping the operator names to the
                operators. If a list of operators is given, the keys are the
        """
        if isinstance(operators, List):
            return {str(i): op for i, op in enumerate(operators)}
        if isinstance(operators, Dict):
            return operators
        return {"0": operators}

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

    def update_hamiltonian(self):
        """
        Abstract method to update the Hamiltonian for the next time step.
        
        This is only required for time-dependent Hamiltonians.
        """
        raise NotImplementedError("This method should be implemented in a subclass!")

    def evaluate_operators(self, index: int):
        """
        Evaluate the expectation value for all operators for the current state.

        Args:
            index (int): The index at which to save the results.
        """
        for key, operator in self.operators.items():
            exp_val = self.evaluate_operator(operator)
            self.results.set_element(key, index, exp_val)

    def result_init_dictionary(self) -> Dict[str, DTypeLike]:
        """
        Returns a dictionary with the initialisation of the results.

        The keys are the operator names and the values are the data types
        of the results.
        
        Returns:
            Dict[str, DTypeLike]: The initialisation dictionary for the results.
        """
        return {key: complex for key in self.operators.keys()}

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
        if evaluation_time == "inf":
            num_results = 1
        else:
            num_results = self.num_time_steps // evaluation_time
        res_dtypes = self.result_init_dictionary()
        self.results.initialize(res_dtypes, num_results)

    def save_time(self, time_step: int, index: int):
        """
        Saves the current time at a given index.

        Args:
            time_step (int): The current time step.
            index (int): The index at which to save the time.

        """
        self.results.set_time(index, time_step*self.time_step_size)

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
            self.evaluate_operators(index)
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

    def run(self,
            evaluation_time: Union[int,"inf"] = 1,
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
            pgbar (bool, optional): Toggles the progress bar. Defaults to True.
        """
        self.init_results(evaluation_time)
        for i in self.create_run_tqdm(pgbar):
            if i != 0:  # We also measure the initial expectation_values
                self.run_one_time_step()
            self.evaluate_and_save_results(evaluation_time, i)
            if self.config.time_dep and i != 0:
                self.update_hamiltonian()

    def reset_hamiltonian(self):
        """
        Resets the Hamiltonian to its initial state.

        This is only required for time-dependent Hamiltonians.
        """
        raise NotImplementedError("This method should be implemented in a subclass!")

    def reset_to_initial_state(self):
        """
        Resets the current state to the intial state
        """
        self.state = deepcopy(self._initial_state)
        self.results = Results()
        if self.config_class.time_dep:
            self.reset_hamiltonian()

class EvoDirection(Enum):
    """
    Enum for the direction of time evolution.
    """
    FORWARD = -1
    BACKWARD = 1

    def __str__(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_bool(forward: bool) -> EvoDirection:
        """
        Converts a boolean to a EvoDirection enum.

        Args:
            forward (bool): If True, FORWARD is returned. Otherwise BACKWARD.

        Returns:
            EvoDirection: The corresponding EvoDirection enum.
        """
        return EvoDirection.FORWARD if forward else EvoDirection.BACKWARD

    def exp_sign(self) -> int:
        """
        Returns the sign of the exponent for the time evolution.

        Returns:
            int: The sign of the exponent.
        """
        return self.value

    def exp_factor(self) -> complex:
        """
        Returns the factor for the exponent in the time evolution.

        Returns:
            complex: The factor for the exponent.
        """
        return 1.0j * self.exp_sign()

class TimeEvoMode(Enum):
    """
    Mode for the time evolution of a matrix.
    """

    FASTEST = "fastest"
    EXPM = "expm"
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

    @staticmethod
    def scipy_modes() -> List[TimeEvoMode]:
        """
        Returns a list of all modes that are scipy ODE solvers.
        
        Returns:
            List[TimeEvoMode]: The list of scipy modes.
        """
        return [TimeEvoMode.RK45,
                TimeEvoMode.RK23,
                TimeEvoMode.DOP853,
                TimeEvoMode.BDF]

    def is_scipy(self) -> bool:
        """
        Determines, if this mode is a scipy ODE solver.
        """
        if self == TimeEvoMode.FASTEST:
            return self.fastest_equivalent().is_scipy()
        return self in self.scipy_modes()

    def action_evolvable(self) -> bool:
        """
        Determines, if this mode can be used for time evolution via an action.

        Returns:
            bool: True if the mode is suitable for time evolution via an action.
        """
        return self.is_scipy()

    def time_evolve_action(self,
                           psi: np.ndarray,
                           time_evo_action: Callable,
                           time_difference: float,
                           forward: EvoDirection| bool = EvoDirection.FORWARD,
                           **options
                           ) -> np.ndarray:
        """
        Performs the time evolution of a tensor from an action.

        Args:
            psi (np.ndarray): The initial state as an arbitrary tensor.
            time_evo_action (Callable): The action to be performed on the
                right hand side of the effective Schrödinger equation. Takes
                the time and the state as arguments.
            time_difference (float): The duration of the time-evolution.
            forward (EvoDirection|bool, optional): The direction of the time evolution.
            **options: Additional options for the solver
                underlying the time evolution. See the scipy documentation for
                the available options of the specific solver.

        Returns:
            np.ndarray: The time-evolved state.
        """
        if self.is_scipy():
            if isinstance(forward, bool):
                forward = EvoDirection.from_bool(forward)
            exp_factor = forward.exp_factor()
            orig_shape = psi.shape
            if not np.issubdtype(psi.dtype, np.complexfloating):
                psi = psi.astype(np.complex64)
                warnstr ="You supplied a real dtyped tensor, but the time"
                warnstr += " evolution is complex. The tensor will be cast to "
                warnstr += "complex64. This might impede performance!"
                warn(warnstr, UserWarning)
            def ode_rhs(t, y_vec):
                orig_array = y_vec.reshape(orig_shape)
                action_result = time_evo_action(t, orig_array).flatten()
                return exp_factor * action_result
            t_span = (0, time_difference)
            res = solve_ivp(ode_rhs, t_span, psi.flatten(),
                            method=self.value,
                            t_eval=[time_difference],
                            **options)
            return res.y[:, 0].reshape(orig_shape)
        raise NotImplementedError(
            "The time evolution via action is not implemented for this mode: "
            + str(self.value)
        )

    def time_evolve(self,
                    psi: np.ndarray,
                    hamiltonian: np.ndarray,
                    time_difference: float,
                    forward: EvoDirection| bool = EvoDirection.FORWARD,
                    **options: dict[str, Any]) -> np.ndarray:
        """
        Time evolves a state psi via a Hamiltonian matrix.

        Args:
            psi (np.ndarray): The initial state as a vector.
            hamiltonian (np.ndarray): The Hamiltonian determining the dynamics as
                a matrix.
            time_difference (float): The duration of the time-evolution
            forward (EvoDirection|bool, optional): The direction of the time evolution.
                    Defaults to EvoDirection.FORWARD.
            **options (dict[str, Any]): Additional options for the solver
                underlying the time evolution. See the scipy documentation for
                the available options of the specific solver.
        
        Returns:
            np.ndarray: The time evolved state
        """
        if isinstance(forward, bool):
            forward = EvoDirection.from_bool(forward)
        if self.is_scipy():
            return self.time_evolve_action(psi,
                                           lambda t, y: hamiltonian @ y.flatten(),
                                           time_difference,
                                           forward=forward,
                                           **options)
        exponent = forward.exp_sign() * 1.0j * hamiltonian * time_difference
        return fast_exp_action(exponent, psi.flatten(),
                               mode=self.value).reshape(psi.shape)

def time_evolve(psi: np.ndarray, hamiltonian: np.ndarray,
                time_difference: float,
                forward: EvoDirection| bool = EvoDirection.FORWARD,
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
        forward (EvoDirection|bool, optional): The direction of the time evolution.
                Defaults to EvoDirection.FORWARD.
        mode (TimeEvoMode, optional): The mode to use for the time evolution.

    Returns:
        np.ndarray: The time evolved state
    """
    if isinstance(forward, bool):
        forward = EvoDirection.from_bool(forward)
    sign = forward.exp_sign()
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
    return result_vector.reshape(
                      psi.shape)
