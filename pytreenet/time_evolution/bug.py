from typing import Dict, List, Union
from dataclasses import dataclass

from .ttn_time_evolution import TTNTimeEvolution
from ..operators.tensorproduct import TensorProduct
from ..core.truncation.recursive_truncation import recursive_truncation
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..util.tensor_splitting import SVDParameters
from copy import copy
from .time_evo_util.common_bug import root_update, CommonBUGConfig
from ..core.truncation.recursive_truncation import truncate_node, post_truncate_node

@dataclass
class BUGConfig(CommonBUGConfig, SVDParameters):
    """
    Configuration class for the common BUG method.

    Combines the configuration options for the common BUG method with the
    configuration options for the SVD truncation.

    """

    def __post_init__(self):
        assert self.fixed_rank is False, "Fixed rank is not supported for BUG!"

class BUG(TTNTimeEvolution):
    """
    The BUG method for time evolution of tree tensor networks.

    BUG stands for Basis-Update and Galerkin. This class implements the rank-
    adaptive version introduced in https://www.doi.org/10.1137/22M1473790 .

    """
    config_class = BUGConfig

    def __init__(self,
                 initial_state: TreeTensorNetworkState,
                 hamiltonian: TreeTensorNetworkOperator,
                 time_step_size: float,
                 final_time: float,
                 operators: Union[
                     List[Union[TensorProduct, TreeTensorNetworkOperator]],
                     Dict[str, Union[TensorProduct, TreeTensorNetworkOperator]],
                     TensorProduct,
                     TreeTensorNetworkOperator
                 ],
                 config: Union[BUGConfig, None] = None,
                 ) -> None:
        super().__init__(initial_state,
                         time_step_size, final_time,
                         operators, config)
        self.hamiltonian = hamiltonian
        self.state : TreeTensorNetworkState
        self.state.ensure_root_orth_center()
        self.config: BUGConfig

    def truncation(self):
        """
        Truncates the tree after the time evolution.

        """
        recursive_truncation(self.state,
                             self.config)

    def recursive_update(self):
        """
        Recursively updates the tree tensor network.

        """
        self.state = root_update(self.state,
                                 self.hamiltonian,
                                 self.time_step_size,
                                 bug_config=self.config)

    def run_one_time_step(self, **kwargs):
        """
        Runs one time step of the BUG method.

        The method is based on the rank-adaptive BUG method introduced in
        https://www.doi.org/10.1137/22M1473790 .

        If the configured time step is >= 0.1, it performs two internal updates
        each with half the time step for potentially better stability/accuracy.

        Args:
            kwargs: Additional keyword arguments for the time step.

        """
        if self.time_step_size >= 0.1 :
            # Temporarily set the halved time step for internal updates
            self.time_step_size /= 2

            self.recursive_update()

            post_svd_params = copy(self.config)
            post_svd_params.sum_trunc = False
            post_svd_params.total_tol = float('-inf')
            post_svd_params.rel_tol = float('-inf')
            post_truncate_node(self.state.root_id, self.state, post_svd_params)
            if self.state.orthogonality_center_id != self.state.root_id:
                self.state.move_orthogonalization_center(self.state.root_id)

            self.recursive_update() # Second update with halved step size
            self.truncation() # Final truncation for the full original step

            self.time_step_size *= 2

        else:
            # Standard single update for small time steps
            self.recursive_update()
            self.truncation()
