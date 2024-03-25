"""
Implements the mother class for all two-site TDVP algorithms.
 This class mostly contains functions to contract the effective Hamiltonian
 and to update the sites.
"""
from typing import List, Union, Dict, Tuple
import numpy as np

from ..tdvp import TDVPAlgorithm
from ...ttns.ttns import TreeTensorNetworkState
from ...ttno.ttno import TTNO
from ...operators.tensorproduct import TensorProduct
from ...tensor_util import check_truncation_parameters

class TwoSiteTDVP(TDVPAlgorithm):

    def __init__(self, initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO,
                 time_step_size: float, final_time: float,
                 operators: Union[TensorProduct, List[TensorProduct]],
                 truncation_parameters: Dict) -> None:
        """
        Initilises an instance of a TDVP algorithm.

        Args:
            intial_state (TreeTensorNetworkState): The initial state of the
             system.
            hamiltonian (TTNO): The Hamiltonian in TTNO form under which to
             time-evolve the system.
            time_step_size (float): The size of one time-step.
            final_time (float): The final time until which to run the evolution.
            operators (Union[TensorProduct, List[TensorProduct]]): Operators
             to be measured during the time-evolution.
            truncation_parameters (Dict): A dictionary containing the
             parameters used for truncation. The dictionary can define a
             maximum bond dimension ('maximum_bond_dim'), a relative
             tolerance ('rel_tol') and a total tolerance ('total_tol') to be
             used during the truncation. For details see the documentation of
             tensor_util.truncate_singular_values.
        """
        super().__init__(initial_state, hamiltonian,
                         time_step_size, final_time,
                         operators)
        self.max_bond_dim, self.rel_tol, self.total_tol = self._init_truncation_parameters(truncation_parameters)

    def _init_truncation_parameters(self, truncation_parameters: Dict) -> Tuple[int, float, float]:
        """
        Initialises the truncation parameters for the TDVP algorithm by
         unpacking the dictionary.

        Args:
            truncation_parameters (Dict): A dictionary containing the
             parameters used for truncation. The dictionary can define a
             maximum bond dimension ('maximum_bond_dim'), a relative
             tolerance ('rel_tol') and a total tolerance ('total_tol') to be
             used during the truncation. For details see the documentation of
             tensor_util.truncate_singular_values.
        """
        if "max_bond_dim" in truncation_parameters:
            max_bond_dim = truncation_parameters["max_bond_dim"]
        else:
            max_bond_dim = 200
        if "rel_tol" in truncation_parameters:
            rel_tol = truncation_parameters["rel_tol"]
        else:
            rel_tol = 1e-10
        if "total_tol" in truncation_parameters:
            total_tol = truncation_parameters["total_tol"]
        else:
            total_tol = 1e-10
        check_truncation_parameters(max_bond_dim, rel_tol, total_tol)
        return max_bond_dim, rel_tol, total_tol
        
