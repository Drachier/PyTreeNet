from typing import Dict, List, Union
from dataclasses import dataclass

from .ttn_time_evolution import TTNTimeEvolution
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

class FixedBUG(TTNTimeEvolution):
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
                 config: Union[FixedBUGConfig, None] = None
                 ) -> None:

        super().__init__(initial_state,
                         time_step_size,
                         final_time,
                         operators,
                         config=config)

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
                                 bug_config=self.config)

    def run_one_time_step(self, **kwargs):
        """
        Run one time step of the time evolution.
        """
        self.recursive_update()
