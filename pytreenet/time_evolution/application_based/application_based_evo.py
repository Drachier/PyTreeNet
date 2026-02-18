"""
Implements a time evolution algorithm based on the application of a TTNO to a TTNS.
"""
from __future__ import annotations
from typing import Any, Union, TYPE_CHECKING
from dataclasses import dataclass

from ..ttn_time_evolution import (TTNOBasedTimeEvolution,
                                  TTNTimeEvolutionConfig)

if TYPE_CHECKING:
    from ...operators.tensorproduct import TensorProduct
    from ...ttno.ttno_class import TTNO
    from ...ttns.ttns import TreeTensorNetworkState
    from ...core.addition.linear_combination import LinCombParams

@dataclass
class ApplicationEvolutionConfig(TTNTimeEvolutionConfig):
    """
    Configuration for the ApplicationEvolution class.
    """
    pass

class ApplicationEvolution(TTNOBasedTimeEvolution):
    """
    Implements a time evolution algorithm based on the application of a TTNO to a TTNS.
    """

    def __init__(self,
                 initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO,
                 time_step_size: float,
                 final_time: float,
                 operators: Union[list[TensorProduct | TTNO],
                                  dict[str, TensorProduct | TTNO],
                                  TensorProduct,
                                  TTNO],
                 config: Union[ApplicationEvolutionConfig, None] = None,
                 lin_comb_params: Union[LinCombParams, None] = None,
                 ) -> None:
        """
        Initializes the ApplicationEvolution class.

        Args:
            initial_state (TreeTensorNetworkState): The initial state of the
                time evolution.
            hamiltonian (TTNO): The Hamiltonian of the system.
                system.
            time_step_size (float): The time difference progressed by one time
                step.
            final_time (float): The final time until which the time evolution
                runs.
            operators (Union[List[Union[TensorProduct, TTNO]], TensorProduct, TTNO]):
                Operators for which the expectation value should be recorded
                during the time evolution.
            config (Union[ApplicationEvolutionConfig,None]): The configuration of
                time evolution. Defaults to None.
            lin_comb_params (Union[LinCombParams, None], optional): Parameters for
                the linear combination used in the application of the TTNO to the
                TTNS. Defaults to None
        """
        super().__init__(initial_state,
                         hamiltonian,
                         time_step_size,
                         final_time,
                         operators,
                         config=config,
                         solver_options=None)
        if lin_comb_params is None:
            lin_comb_params = LinCombParams.default()
        self.lin_comb_params = lin_comb_params

