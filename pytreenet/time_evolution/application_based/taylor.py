"""
Implements a time evolution algorithm based on the application of a TTNO to a TTNS
along the lines of a Taylor expansion of the time evolution operator.
"""
from __future__ import annotations
from typing import Any, Union, TYPE_CHECKING
from math import factorial

from .application_based_evo import (ApplicationEvolution,
                                  ApplicationEvolutionConfig)
from ...ttns.ttns_ttno.application import compute_power_series
from ...core.addition.linear_combination import LinearCombination

if TYPE_CHECKING:
    from ...operators.tensorproduct import TensorProduct
    from ...ttno.ttno_class import TTNO
    from ...ttns.ttns import TreeTensorNetworkState

class Taylor(ApplicationEvolution):
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
                 max_power: int = 4,
                 config: Union[ApplicationEvolutionConfig, None] = None,
                 lin_comb_params: Union[Any, None] = None
                 ) -> None:
        """
        Initializes the Taylor class.

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
            max_power (int): The maximum power of the Taylor expansion. Defaults to 4.
            config (Union[ApplicationEvolutionConfig,None]): The configuration of
                time evolution. Defaults to None.
            lin_comb_params (Union[Any,None]): Parameters for the linear
                combination of the states during the time evolution. Defaults to
                None.
        """
        super().__init__(initial_state,
                         hamiltonian,
                         time_step_size,
                         final_time,
                         operators,
                         config=config,
                         lin_comb_params=lin_comb_params)
        self.max_power = max_power
        self.coeffs = [(-1j*self.time_step_size)**n/factorial(n)
                       for n in range(0, max_power+1)]

    def run_one_time_step(self, **kwargs):
        """
        Runs one time step of the time evolution.
        """
        powers = compute_power_series(self.state,
                                      self.hamiltonian,
                                      self.max_power,
                                      self.lin_comb_params.application_method,
                                      *self.lin_comb_params.args_ap,
                                      **self.lin_comb_params.kwargs_ap)
        lin_comb = LinearCombination(powers, None, self.coeffs)
        self.state = lin_comb.compute_via_params(self.lin_comb_params)
