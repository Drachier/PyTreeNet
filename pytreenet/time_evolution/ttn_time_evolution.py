from __future__ import annotations
from typing import List, Union, Dict, Tuple
from dataclasses import dataclass

from numpy import ndarray, asarray, max as arrmax

from .time_evolution import TimeEvolution
from ..ttns import TreeTensorNetworkState
from ..ttno import TTNO
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
    A time evolution for tree tensor networks.
    
    Provides functionality to compute expectation values of operators during
    the time evolution and record bond dimensions of the current state.

    Attributes:
        bond_dims (Union[None,Dict[str,int]]): If a recording of the bond
            dimension is intended, they are recorded here.
    """
    bond_dim_id = "bond_dim"
    config_class = TTNTimeEvolutionConfig

    def __init__(self, initial_state: TreeTensorNetworkState,
                 time_step_size: float, final_time: float,
                 operators: Union[List[Union[TensorProduct, TTNO]],
                                  Dict[str, Union[TensorProduct, TTNO]],
                                  TensorProduct,
                                  TTNO],
                 config: Union[TTNTimeEvolutionConfig,None] = None) -> None:
        """
        A time evolution for a tree tensor network state.

        Args:
            initial_state (TreeTensorNetwork): The initial state of the time
                evolution.
            time_step_site (float): The time difference progressed by one time
                step.
            final_time (float): The final time until which the time evolution
                runs.
            operators (Union[List[Union[TensorProduct, TTNO]], TensorProduct, TTNO]):
                Operators for which the expectation value should be recorded
                during the time evolution.
            config (Union[TTNTimeEvolutionConfig,None]): The configuration of
                time evolution. Defaults to None.
        """
        super().__init__(initial_state, time_step_size, final_time, operators)
        self._initial_state: TreeTensorNetworkState
        self.state: TreeTensorNetworkState

        if config is not None and config.record_bond_dim:
            self.bond_dims = {}
        else:
            self.bond_dims = None
        if config is None:
            self.config = self.config_class()
        else:
            self.config = config

    @property
    def records_bond_dim(self) -> bool:
        """
        Are the bond dimensions recorded or not.
        """
        return self.bond_dims is not None

    def obtain_bond_dims(self) -> Dict[Tuple[str,str], int]:
        """
        Obtains a dictionary of all bond dimensions in the current state.
        """
        return self.state.bond_dims()

    def bond_dim_matrix(self) -> ndarray:
        """
        Obtain the bond dimensions as a matrix.
        """
        bond_dims = self.operator_result(self.bond_dim_id)
        matrix = asarray(list(bond_dims.values()))
        return matrix

    def max_bond_dim(self) -> ndarray:
        """
        Obtain the maximum bond dimension over time.
        """
        return arrmax(self.bond_dim_matrix(),axis=0)

    def record_bond_dimensions(self):
        """
        Records the bond dimensions of the current state, if desired to do so.
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
        """
        Includes the possibility to obtain the bond dimension from the results.

        Args:
            operator_id (Union[str,int]): The identifier or position of the
                operator, whose expectation value results should be returned.
            realise (bool, optional): Whether the results should be
                transformed into real numbers.

        Returns:
            ndarray: The expectation value results over time.
        """
        if isinstance(operator_id, str) and operator_id == self.bond_dim_id:
            if self.records_bond_dim is not None:
                return self.bond_dims
            errstr = "Bond dimensions are not being recorded."
            raise ValueError(errstr)
        return super().operator_result(operator_id, realise)

    def evaluate_operators(self) -> ndarray:
        """
        Evaluates the operator including the recording of bond dimensions.
        """
        current_results = super().evaluate_operators()
        self.record_bond_dimensions()
        return current_results

    def evaluate_operator(self, operator: Union[TensorProduct,TTNO]) -> complex:
        """
        Evaluate the expectation value of a single operator.

        Args:
            operator (TensorProduct): The operator for which to compute the
                expectation value.
        
        Returns:
            np.ndarray: The expectation value of the operator with respect to
                the current state.
        """
        return self.state.operator_expectation_value(operator)
