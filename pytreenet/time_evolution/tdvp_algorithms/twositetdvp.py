"""
Implements the mother class for all two-site TDVP algorithms.

This class mostly contains functions to contract the effective Hamiltonian
and to update the sites in the two site scheme.
"""
from typing import List, Union, Any
from dataclasses import dataclass

from .tdvp_algorithm import TDVPAlgorithm, TDVPConfig
from ..time_evolution import EvoDirection
from ..time_evo_util.effective_time_evolution import two_site_time_evolution
from ...ttns.ttns import TreeTensorNetworkState
from ...ttno.ttno_class import TTNO
from ...operators.tensorproduct import TensorProduct
from ...util.tensor_splitting import SVDParameters

@dataclass
class TwoSiteTDVPConfig(TDVPConfig, SVDParameters):
    """
    Configuration for the two-site TDVP algorithm.

    In this configuration class additional parameters for the two-site TDVP
    algorithm can be specified and entered. This allows for the same
    extendability as `**kwargs` but with the added benefit of type hints
    and better documentation.
    """

class TwoSiteTDVP(TDVPAlgorithm):
    """
    The two site TDVP algorithm.

    It contracts two sites and updates them at the same time before splitting
    them using an SVD with truncation.

    Attributes:
        svd_parameters: Contains values for the SVD truncation.
    """
    config_class = TwoSiteTDVPConfig

    def __init__(self, initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO,
                 time_step_size: float, final_time: float,
                 operators: Union[TensorProduct, List[TensorProduct]],
                 config: Union[TwoSiteTDVPConfig,None] = None,
                 solver_options: Union[dict[str, Any], None] = None
                 ) -> None:
        """
        Initialises an instance of a two-site TDVP algorithm.

        Args:
            intial_state (TreeTensorNetworkState): The initial state of the
                system.
            hamiltonian (TTNO): The Hamiltonian in TTNO form under which to
                time-evolve the system.
            time_step_size (float): The size of one time-step.
            final_time (float): The final time until which to run the evolution.
            operators (Union[TensorProduct, List[TensorProduct]]): Operators
                to be measured during the time-evolution.
            truncation_parameters (Union[SVDParameters,None]): The truncation
                parameters to use during the time-evolution. Can be used to set
                the maximum bond dimension, the absolute tolerance, and the
                relative tolerance.
            config (Union[TwoSiteTDVPConfig,None]): The configuration of
                time evolution. Defaults to None.
            solver_options (Union[Dict[str, Any], None], optional): Most time
                evolutions algorithms use some kind of solver to resolve a
                partial differential equation. This dictionary can be used to
                pass additional options to the solver. Refer to the
                documentation of `ptn.time_evolution.TimeEvoMode` for further
                information. Defaults to None.
                solver_options (Union[Dict[str, Any], None], optional): Most time
                evolutions algorithms use some kind of solver to resolve a
                partial differential equation. This dictionary can be used to
                pass additional options to the solver. Refer to the
                documentation of `ptn.time_evolution.TimeEvoMode` for further
                information. Defaults to None.
        """
        super().__init__(initial_state, hamiltonian,
                         time_step_size, final_time,
                         operators,
                         config=config,
                         solver_options=solver_options)
        self.config: TwoSiteTDVPConfig

    def _update_two_site_nodes(self,
                               target_node_id: str,
                               next_node_id: str,
                               time_step_factor: float = 1):
        """
        Perform the two-site update on the two given sites.
        
        The target node is the original orthogonalisation center. The center
        will be moved to the next node in this update.

        Args:
            target_node_id (str): The id of the target node.
            next_node_id (str): The id of the next node.
            time_step_factor (float): The factor by which to multiply the
                time step size. Default is 1.
        """
        u_legs, v_legs = self.state.legs_before_combination(target_node_id,
                                                            next_node_id)
        new_id = self.create_two_site_id(target_node_id, next_node_id)
        updated_two_sites = two_site_time_evolution(target_node_id,
                                                    next_node_id,
                                                    new_id,
                                                    self.state,
                                                    self.hamiltonian,
                                                    time_step_factor*self.time_step_size,
                                                    self.partial_tree_cache,
                                                    forward=EvoDirection.FORWARD,
                                                    mode=self.config.time_evo_mode,
                                                    solver_options=self.solver_options)
        self.state.tensors[new_id] = updated_two_sites
        # Split the two-site tensor using SVD
        self.state.split_node_svd(new_id, u_legs, v_legs,
                                  u_identifier=target_node_id,
                                  v_identifier=next_node_id,
                                  svd_params=self.config)
        self.state.orthogonality_center_id = next_node_id
        self.update_tree_cache(target_node_id, next_node_id)

    def _single_site_backwards_update(self,
                                      node_id: str,
                                      time_step_factor: float = 1):
        """
        Performs the single-site backwards update on the given node.

        Args:
            node_id (str): The id of the node to update.
            time_step_factor (float): The factor by which to multiply the
                time step size. Default is 1.
        """
        self._update_site(node_id, -1 * time_step_factor)

    @staticmethod
    def create_two_site_id(node_id: str, next_node_id: str) -> str:
        """
        Create the identifier of a two site node obtained from contracting
        the two note with the input identifiers.
        """
        return "TwoSite_" + node_id + "_contr_" + next_node_id
