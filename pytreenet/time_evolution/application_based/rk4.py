"""
Implements an RK4 time evolution algorithm based on the application of a TTNO to a TTNS.
"""
from __future__ import annotations
from typing import Any, Union, TYPE_CHECKING

from .application_based_evo import (ApplicationEvolution,
                                  ApplicationEvolutionConfig)
from ...ttns.ttns_ttno.application import compute_power_series
from ...core.addition.linear_combination import LinearCombination

if TYPE_CHECKING:
    from ...operators.tensorproduct import TensorProduct
    from ...ttno.ttno_class import TTNO
    from ...ttns.ttns import TreeTensorNetworkState

class RK4(ApplicationEvolution):
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
                 lin_comb_params: Union[Any, None] = None
                 ) -> None:
        """
        Initializes the RK4 class.

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
        self.coeffs = [1.0+0.0j,
                       -1j*self.time_step_size/6,
                       -1*self.time_step_size**2/6+0.0j,
                       1j*self.time_step_size**3/12,
                       self.time_step_size**4/24+0.0j
                       ]

    def run_one_time_step(self, **kwargs):
        """
        Runs one time step of the RK4 time evolution.

        Args:
            **kwargs: Additional keyword arguments for
                the time evolution step.
        """
        max_pow = 4
        powers = compute_power_series(self.state,
                                      self.hamiltonian,
                                      max_pow,
                                      self.lin_comb_params.application_method,
                                      *self.lin_comb_params.args_ap,
                                      **self.lin_comb_params.kwargs_ap)
        lin_comb = LinearCombination(powers, None, self.coeffs)
        self.state = lin_comb.compute_via_params(self.lin_comb_params)
