
from typing import Dict, List, Union

from .ttn_time_evolution import TTNTimeEvolution, TTNTimeEvolutionConfig
from ..operators.tensorproduct import TensorProduct
from ..core.truncation.recursive_truncation import recursive_truncation
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..util.tensor_splitting import SVDParameters

from .time_evo_util.common_bug import root_update

class BUG(TTNTimeEvolution):
    """
    The BUG method for time evolution of tree tensor networks.

    BUG stands for Basis-Update and Galerkin. This class implements the rank-
    adaptive version introduced in https://www.doi.org/10.1137/22M1473790 .

    """

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
                 config: Union[TTNTimeEvolutionConfig, None] = None,
                 svd_params: Union[SVDParameters,None] = None
                 ) -> None:
        super().__init__(initial_state,
                         time_step_size, final_time,
                         operators, config)
        self.hamiltonian = hamiltonian
        self.state : TreeTensorNetworkState
        self.state.ensure_root_orth_center()
        self.svd_parameters = svd_params

    def truncation(self):
        """
        Truncates the tree after the time evolution.

        """
        recursive_truncation(self.state,
                             self.svd_parameters)

    def recursive_update(self):
        """
        Recursively updates the tree tensor network.

        """
        self.state = root_update(self.state,
                                 self.hamiltonian,
                                 self.time_step_size,
                                 fixed_rank=False)

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
