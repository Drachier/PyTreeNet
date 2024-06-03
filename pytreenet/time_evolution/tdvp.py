"""
Module to create a builder function for the TDVP algorithms.
"""
from __future__ import annotations
from typing import Union, List

from .tdvp_algorithms import (TDVPAlgorithm,
                              FirstOrderOneSiteTDVP,
                              SecondOrderOneSiteTDVP,
                              SecondOrderTwoSiteTDVP)
from .ttn_time_evolution import TTNTimeEvolutionConfig
from ..operators.tensorproduct import TensorProduct
from ..ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TTNO
from ..util.tensor_splitting import SVDParameters

class TDVPConfig:
    """
    A configuration file for TDVP algorithms.

    Contains all possible parameters for the TDVP algorithms.apart from the
    time evolution configuration.

    Attributes:
        oder (int): The order of the TDVP algorithm. Can be 1 or 2.
        sites (int): The number of sites that are updated in each step. Can be
            1 or 2.
        svd_params (Union[SVDParameters,None]): The parameters for the SVD
            decomposition. If None, the default parameters are used.
        time_evo_config (Union[TTNTimeEvolutionConfig,None]): The time
            evolution configuration for the general tree tensor network time
            evolution algorithm. If None, the default configuration is used.
    """
    def __init__(self,
                 order: int = 2,
                 sites: int = 1,
                 svd_params: Union[SVDParameters,None] = None,
                 time_evo_config: Union[TTNTimeEvolutionConfig,None] = None):

        self.order = order
        self.sites = sites
        if svd_params is None:
            self.svd_params = SVDParameters()
        else:
            self.svd_params = svd_params
        if time_evo_config is None:
            self.time_evo_config = TTNTimeEvolutionConfig()
        else:
            self.time_evo_config = time_evo_config

def tdvp(intitial_state: TreeTensorNetworkState, hamiltonian: TTNO,
         time_step_size: float, final_time: float,
         operators: Union[List[TensorProduct],TensorProduct],
         config: TDVPConfig) -> TDVPAlgorithm:
    """
    Utility function to create a TDVP algorithm.

    Args:
        intitial_state (TreeTensorNetworkState): The initial state of the
            system.
        hamiltonian (TTNO): The Hamiltonian of the system.
        time_step_size (float): The time step size.
        final_time (float): The final time.
        operators (Union[List[TensorProduct],TensorProduct]): The operators
            that are used to calculate the expectation values.
        config (TDVPConfig): The configuration for the TDVP algorithm.
    """
    ordersites = (config.order,config.sites)
    if ordersites == (1,1):
        return FirstOrderOneSiteTDVP(intitial_state,
                                     hamiltonian,
                                     time_step_size,
                                     final_time,
                                     operators,
                                     config.time_evo_config)
    if ordersites == (2,1):
        return SecondOrderOneSiteTDVP(intitial_state,
                                      hamiltonian,
                                      time_step_size,
                                      final_time,
                                      operators,
                                      config.time_evo_config)
    if ordersites == (2,2):
        return SecondOrderTwoSiteTDVP(intitial_state,
                                      hamiltonian,
                                      time_step_size,
                                      final_time,
                                      operators,
                                      config.svd_params,
                                      config.time_evo_config)
    errstr = f"TDVP algorithm with order {config.order} and {config.sites} sites not implemented!"
    raise NotImplementedError(errstr)
