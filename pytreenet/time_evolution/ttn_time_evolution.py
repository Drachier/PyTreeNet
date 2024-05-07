from __future__ import annotations
from typing import List, Union, Dict, Tuple
from dataclasses import dataclass

from numpy import ndarray

from .time_evolution import TimeEvolution
from ..ttns import TreeTensorNetworkState
from ..operators.tensorproduct import TensorProduct

@dataclass
class TTNTimeEvolutionConfig:
    """
    Configuration for the TTN time evolution.

    In this configuration class additional parameters for the time evolution
    of a tree tensor network can be specified and entered. This allows for the
    same extendability as `**kwargs` but with the added benefit of type hints
    and better documentation.
    """
    record_bond_dim: bool = False

class TTNTimeEvolution(TimeEvolution):
    """
    A time evolution for tree tensor networks. Provides functionality to
     compute expectation values of operators during the time evolution.
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 time_step_size: float, final_time: float,
                 operators: Union[List[TensorProduct], TensorProduct],
                 config: Union[TTNTimeEvolutionConfig,None] = None) -> None:
        """
        A time evolution for tree tensor networks starting from and initial
         state and running to a final time with a given time step size. During
         the time evolution, expectation values of operators are computed.
        """
        super().__init__(initial_state, time_step_size, final_time, operators)
        self.initial_state: TreeTensorNetworkState
        self.state: TreeTensorNetworkState

        if config is not None and config.record_bond_dim:
            self.bond_dims = {}
        else:
            self.bond_dims = None

    @property
    def records_bond_dim(self) -> bool:
        """
        Returns whether the bond dimensions are recorded during the time
         evolution.
        """
        return self.bond_dims is not None

    def obtain_bond_dims(self) -> Dict[Tuple[str,str]: int]:
        """
        Obtains a dictionary of all bond dimensions in the current state.
        """
        return self.state.bond_dims()

    def record_bond_dimensions(self):
        """
        Records the bond dimensions of the current state, if the bond
         dimensions are being recorded.
        """
        if self.records_bond_dim:
            if len(self.bond_dims) == 0:
                self.bond_dims = {key: [value] for key, value in self.obtain_bond_dims().items()}
            else:
                for key, value in self.obtain_bond_dims().items():
                    self.bond_dims[key].append(value)

    def operator_result(self,
                        operator_id: str | int,
                        realise: bool = False) -> ndarray:
        if isinstance(operator_id, str) and operator_id == "bond_dim":
            if self.records_bond_dim is not None:
                return self.bond_dims
            errstr = "Bond dimensions are not being recorded."
            raise ValueError(errstr)
        return super().operator_result(operator_id, realise)

    def evaluate_operators(self) -> ndarray:
        current_results = super().evaluate_operators()
        self.record_bond_dimensions()
        return current_results

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
