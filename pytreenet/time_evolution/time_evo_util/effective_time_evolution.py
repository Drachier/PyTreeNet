"""
This module contains functions for time evolution using effective Hamiltonians.

The effective Hamiltonian's are usually obtained by partially contracting a
TTNO Hamiltonian with a TTNS and its conjugate.

"""

from numpy import ndarray

from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.contractions.sandwich_caching import SandwichCache
from pytreenet.contractions.effective_hamiltonians import get_effective_single_site_hamiltonian
from pytreenet.time_evolution.time_evolution import time_evolve, TimeEvoMode

def single_site_time_evolution(node_id: str,
                               state: TreeTensorNetworkState,
                               hamiltonian: TreeTensorNetworkOperator,
                               time_step_size: float,
                               tensor_cache: SandwichCache,
                               forward: bool = True,
                               mode: TimeEvoMode = TimeEvoMode.FASTEST
                               ) -> ndarray:
    """
    Perform the time evolution for a single site.

    For this the effective Hamiltonian is build and the time evolution is
    performed by exponentiating the effective Hamiltonian and applying it
    to the current node.

    Args:
        node_id (str): The id of the node to be updated.
        state (TreeTensorNetworkState): The state of the system.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian of the system.
        time_step_size (float): The time step size.
        tensor_cache (SandwichCache): The cache for the neighbour blocks.
        forward (bool): Whether to time evolve forward or backward. Defaults to
            True.
        mode (TimeEvoMode): The mode of the time evolution. Defaults to
            TimeEvoMode.FASTEST

    Returns:
        ndarray: The updated tensor.

    """
    ham_eff = get_effective_single_site_hamiltonian(node_id,
                                                    state,
                                                    hamiltonian,
                                                    tensor_cache)
    updated_tensor = time_evolve(state.tensors[node_id],
                                 ham_eff,
                                 time_step_size,
                                 forward=forward,
                                 mode=mode)
    return updated_tensor
