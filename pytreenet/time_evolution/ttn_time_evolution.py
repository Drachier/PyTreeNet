from __future__ import annotations
from typing import List, Union

from .time_evolution import TimeEvolution
from ..ttns import TreeTensorNetworkState
from ..operators.tensorproduct import TensorProduct

class TTNTimeEvolution(TimeEvolution):
    """
    A time evolution for tree tensor networks. Provides functionality to
     compute expectation values of operators during the time evolution.
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 time_step_size: float, final_time: float,
                 operators: Union[List[TensorProduct], TensorProduct]) -> None:
        """
        A time evolution for tree tensor networks starting from and initial
         state and running to a final time with a given time step size. During
         the time evolution, expectation values of operators are computed.
        """
        super().__init__(initial_state, time_step_size, final_time, operators)

        self.initial_state: TreeTensorNetworkState
        self.state: TreeTensorNetworkState

    def evaluate_operator(self, operator: TensorProduct) -> complex:
        """
        Evaluate the expectation value of a single operator.

        Args:
            operator (TensorProduct): The operator for which to compute the
             expectation value.
        
        Returns:
            np.ndarray: The expectation value of the operator.
        """
        return self.state.operator_expectation_value(operator)
        