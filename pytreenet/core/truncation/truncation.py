"""
This module defines the TruncationEngine class and manages different truncation methods for tree tensor networks.
"""
from enum import Enum

from ...ttns.ttns import TreeTensorNetworkState
from ...util.tensor_splitting import SVDParameters
from .svd_truncation import sweeping_onward_truncation, recursive_downward_truncation
from .greedy_truncation import sweeping_greedy_truncation, recursive_greedy_truncation
from .truncation_util import find_orthogonalization_path, find_greedy_trunc_path
from ...time_evolution.time_evo_util.update_path import SweepingUpdatePathFinder, PathFinderMode

class TruncationMode(Enum):
    """
    Enum for different truncation methods.
    - RECURSIVE_DOWNWARD: Recursively perform truncation, on children bonds, down to the leaf nodes.
    - RECURSIVE_GREEDY: Recursively Perform truncation on all bonds followed by greedy optimization. 
    - SWEEPING_ONWARD: Perform truncation on one bond in a sweeping manner.
    - SWEEPING_GREEDY: Perform truncation on all bonds in a sweeping manner.
    """
    RECURSIVE_DOWNWARD = "recursive_downward"
    RECURSIVE_GREEDY = "recursive_greedy"
    SWEEPING_ONWARD = "sweeping_onward"
    SWEEPING_GREEDY = "sweeping_greedy"

class TruncationEngine:
    """
    Prepares the paths for sweeping truncation modes and define the truncation process.
    Args:
        initial_ttn (TreeTensorNetworkState): The TTN state to be truncated.
        method (TruncationMode): The truncation method to be used.
    """
    def __init__(self,
                 initial_ttn: TreeTensorNetworkState, 
                 method: TruncationMode):

        self.initial_ttn = initial_ttn
        self.method = method
        
        # Initialize paths based on the method
        if self.method == TruncationMode.SWEEPING_ONWARD:
            self.forward_svd_trunc_path = SweepingUpdatePathFinder(self.initial_ttn, 
                                                                          PathFinderMode.LeafToLeaf_Forward).find_path()
            self.forward_orth_path = find_orthogonalization_path(self.initial_ttn,
                                                                 self.forward_svd_trunc_path)
            self.backward_individual_trunc_path = SweepingUpdatePathFinder(self.initial_ttn, 
                                                                           PathFinderMode.LeafToLeaf_Backward).find_path()
            self.backward_orth_path = find_orthogonalization_path(self.initial_ttn,
                                                                  self.backward_individual_trunc_path)
        elif self.method == TruncationMode.SWEEPING_GREEDY:
            self.forward_greedy_trunc_path = find_greedy_trunc_path(self.initial_ttn, True)
            self.backward_greedy_trunc_path = find_greedy_trunc_path(self.initial_ttn, False)

    def truncate(self,
                ttn: TreeTensorNetworkState,
                svd_params: SVDParameters,
                **kwargs) -> TreeTensorNetworkState:
        """
        Truncate a tree tensor network state using the specified method.
        Args:
            ttn (TreeTensorNetworkState): The tree tensor network state to truncate.
            svd_params (SVDParameters): Parameters for SVD truncation.
                - The max_product_dim parameter in svd_params is ignored for SWEEPING_ONWARD and
                  RECURSIVE_DOWNWARD truncation methods.
                - The max_bond_dim parameter in svd_params is ignored for SWEEPING_GREEDY and
                  RECURSIVE_GREEDY truncation methods.
            **kwargs: Additional keyword arguments including:
                - forward (bool, optional): Direction for SWEEPING modes (SWEEPING_ONWARD, SWEEPING_GREEDY).
                  If True (default), uses forward direction. If False, uses backward direction.
                  This parameter is only used for SWEEPING modes and ignored for RECURSIVE modes.
        Returns:
            TreeTensorNetworkState: The truncated tree tensor network state.
        """
        if self.method == TruncationMode.SWEEPING_ONWARD:
            forward = kwargs.get('forward', True)
            directed_trunc_path = self.forward_svd_trunc_path if forward else self.backward_individual_trunc_path
            directed_orth_path = self.forward_orth_path if forward else self.backward_orth_path
            sweeping_onward_truncation(ttn,
                                        directed_trunc_path,
                                        directed_orth_path,
                                        svd_params)
        elif self.method == TruncationMode.SWEEPING_GREEDY:
            forward = kwargs.get('forward', True)
            directed_trunc_path = self.forward_greedy_trunc_path if forward else self.backward_greedy_trunc_path
            sweeping_greedy_truncation(ttn,
                                       directed_trunc_path,
                                       svd_params)
        elif self.method == TruncationMode.RECURSIVE_DOWNWARD:
            recursive_downward_truncation(ttn, svd_params)
        elif self.method == TruncationMode.RECURSIVE_GREEDY:
            recursive_greedy_truncation(ttn, svd_params)

