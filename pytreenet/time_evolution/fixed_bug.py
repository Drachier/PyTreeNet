from typing import Dict, List, Union, Any
from dataclasses import dataclass

from .ttn_time_evolution import TTNOBasedTimeEvolution
from ..ttns import TreeTensorNetworkState
from ..ttno import TreeTensorNetworkOperator
from ..operators.tensorproduct import TensorProduct
from ..util.tensor_splitting import SplitMode

from .time_evo_util.common_bug import root_update, CommonBUGConfig

@dataclass
class FixedBUGConfig(CommonBUGConfig):
    """
    Configuration class for the fixed rank BUG method.
    """

    def __post_init__(self):
        self.fixed_rank = True

class FixedBUG(TTNOBasedTimeEvolution):
    """
    The fixed rank Basis-Update and Galerkin (BUG) time evolution algorithm.
    """
    config_class = FixedBUGConfig

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
                 config: Union[FixedBUGConfig, None] = None,
                 solver_options: Union[dict[str, Any], None] = None
                 ) -> None:
        """
        Initilises an instance of a fixed bond dimension BUG algorithm.

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
            config (Union[FixedBUGConfig,None]): The configuration of
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

        super().__init__(initial_state,
                         hamiltonian,
                         time_step_size,
                         final_time,
                         operators,
                         config=config,
                         solver_options=solver_options)

        self.hamiltonian = hamiltonian
        self.state: TreeTensorNetworkState
        self.state.ensure_root_orth_center(mode=SplitMode.KEEP)
        self.config: FixedBUGConfig

    def recursive_update(self):
        """
        Recursively updates the state according to the fixed rank BUG.
        """
        self.state = root_update(self.state,
                                 self.hamiltonian,
                                 self.time_step_size,
                                 bug_config=self.config,
                                 solver_options=self.solver_options)

    def run_one_time_step(self, **kwargs):
        """
        Run one time step of the time evolution.
        """
        self.recursive_update()
