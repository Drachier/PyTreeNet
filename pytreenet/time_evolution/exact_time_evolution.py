"""
An exact time evolution.
"""
from __future__ import annotations
from typing import Any, List, Union

import numpy as np
from scipy.linalg import expm

from .time_evolution import TimeEvolution

class ExactTimeEvolution(TimeEvolution):
    """
    An exact time evolution working with state vectors and matrix operators.

    Note that this time evolution is very limited in the number of sites it
    can simulate.

    Attributes:
        hamiltonian (np.ndarray): The Hamiltonian controlling the time
            time-evolution of the system.
    """

    def __init__(self, initial_state: np.ndarray, hamiltonian: np.ndarray,
                 time_step_size: float, final_time: float,
                 operators: Union[List[np.ndarray], np.ndarray]):
        """
        An exact time evolution for a given Hamiltonian.

        Args:
            initial_state (np.ndarray): The initial state as a state vector.
            hamiltonian (np.ndarray): The Hamiltonian as a matrix.
            time_step_size (float): The size of one time step.
            final_time (float): The final time.
            operators (Union[List[np.ndarray],np.ndarray]): The operators for
                which to compute the expectation values as matrices. Can be a
                single operator or a list of operators.
        """
        super().__init__(initial_state, time_step_size,
                         final_time, operators)
        self.hamiltonian = hamiltonian
        self._time_evolution_operator = self._compute_time_evolution_operator()

    def _compute_time_evolution_operator(self) -> np.ndarray:
        """
        Compute the time evolution operator for one time step.
         
        This is achived by exponentiating the Hamiltonian.
        
        Returns:
            np.ndarray: The time evolution operator.
                e^(-itH)
        """
        return expm(-1j * self._time_step_size * self.hamiltonian)

    def evaluate_operator(self, operator: Any) -> complex:
        """
        Evaluate an operator at the current time step.
        
        Args:
            operator (Any): The operator to evaluate.
        
        Returns:
            complex: The expectation value of the operator.
        """
        return self.state.conj().T @ operator @ self.state

    def run_one_time_step(self, **kwargs):
        """
        Run one time step of the exact time evolution.

        This is achieved by multiplying the time evolution operator with the
        current state of the system.
        """
        self.state = self._time_evolution_operator @ self.state
