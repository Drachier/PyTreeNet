
from typing import Dict, List, Union, Any
from dataclasses import dataclass

from .ttn_time_evolution import TTNOBasedTimeEvolution
from ..operators.tensorproduct import TensorProduct
from ..core.truncation.recursive_truncation import recursive_truncation
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..util.tensor_splitting import SVDParameters

from .time_evo_util.common_bug import root_update, CommonBUGConfig

@dataclass
class BUGConfig(CommonBUGConfig, SVDParameters):
    """
    Configuration class for the common BUG method.

    Combines the configuration options for the common BUG method with the
    configuration options for the SVD truncation.

    """

    def __post_init__(self):
        assert self.fixed_rank is False, "Fixed rank is not supported for BUG!"

class BUG(TTNOBasedTimeEvolution):
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
                 solver_options: Union[dict[str, Any], None] = None
                 ) -> None:
        """
        Initilises an instance of the BUG algorithm.

        Args:
            intial_state (TreeTensorNetworkState): The initial state of the
                system.
            hamiltonian (TTNO): The Hamiltonian in TTNO form under which to
                time-evolve the system.
            time_step_size (float): The size of one time-step.
            final_time (float): The final time until which to run the
                evolution.
            operators (Union[TensorProduct, List[TensorProduct]]): Operators
                to be measured during the time-evolution.
            config (Union[BUGConfig,None]): The configuration of
                time evolution. Defaults to None.
            solver_options (Union[Dict[str, Any], None], optional): Most time
                evolutions algorithms use some kind of solver to resolve a
                partial differential equation. This dictionary can be used to
                pass additional options to the solver. Refer to the
                documentation of `ptn.time_evolution.TimeEvoMode` for further
                information. Defaults to None.
        """
        super().__init__(initial_state,
                         hamiltonian,
                         time_step_size, final_time,
                         operators,
                         config=config,
                         solver_options=solver_options)
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
                                 bug_config=self.config,
                                 solver_options=self.solver_options)

    def run_one_time_step(self, **kwargs):
        """
        Runs one time step of the BUG method.

        The method is based on the rank-adaptive BUG method introduced in
        https://www.doi.org/10.1137/22M1473790 .

        Args:
            kwargs: Additional keyword arguments for the time step.

        """
        self.recursive_update()
        self.truncation()
