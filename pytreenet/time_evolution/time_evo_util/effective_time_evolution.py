"""
This module contains functions for time evolution using effective Hamiltonians.

The effective Hamiltonian's are usually obtained by partially contracting a
TTNO Hamiltonian with a TTNS and its conjugate.

"""
from __future__ import annotations
from typing import Any

from numpy import ndarray
import numpy as np
from numpy.typing import NDArray

from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.contractions.sandwich_caching import SandwichCache
from pytreenet.contractions.effective_hamiltonians import (get_effective_single_site_hamiltonian_nodes,
                                                           get_effective_bond_hamiltonian_nodes,
                                                           get_effective_single_site_hamiltonian)
from pytreenet.time_evolution.time_evolution import time_evolve, TimeEvoMode
from pytreenet.core.node import Node
from pytreenet.contractions.state_operator_contraction import (contract_ket_ham_with_envs,
                                                               contract_bond_tensor)

# TODO: Rewrite this to use the new time evolution interface
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

def effective_bond_evolution(
        state_tensor: NDArray[np.complex128],
        state_node: Node,
        time_step_size: float,
        tensor_cache: SandwichCache,
        mode: TimeEvoMode = TimeEvoMode.FASTEST,
        forward: bool = True,
        **options: dict[str, Any]
    ) -> NDArray[np.complex128]:
    """
    Perform the effective bond time evolution.

    This function computes the time evolution of a bond in a TTNS via
    a TTNO representing the Hamiltonian. The state tensor represents a tensor
    on a bond in the TTNS without an equivalent tensor in the TTNO.

    Args:
        state_tensor (NDArray[np.complex128]): The tensor of the state to be
            updated, representing a bond in the TTNS.
        state_node (Node): The node of the state tensor.
        time_step_size (float): The time step size.
        tensor_cache (SandwichCache): The cache for the neighbour blocks.
        mode (TimeEvoMode, optional): The mode of the time evolution. Defaults
            to TimeEvoMode.FASTEST. If possible, the time evolution is
            performed without constructing the full effective Hamiltonian.
        forward (bool, optional): Whether to time evolve forward or backward.
            Defaults to True.
        **options: Additional options for the time evolution. See the
            documentation of the `TimeEvoMode`-class for more
            information.

    """
    if mode.action_evolvable():
        def contraction_action(t,y):
            """Defines the contraction to be used for the time evolution."""
            return contract_bond_tensor(state_node, y, tensor_cache)
        updated_tensor = mode.time_evolve_action(state_tensor,
                                                 contraction_action,
                                                 time_step_size,
                                                 forward=forward,
                                                 **options)
        return updated_tensor
    # Fallback to the default time evolution
    ham_eff = get_effective_bond_hamiltonian_nodes(state_node, tensor_cache)
    updated_tensor = mode.time_evolve(state_tensor,
                                        ham_eff,
                                        time_difference=time_step_size,
                                        forward=forward,
                                        **options
                                        )
    return updated_tensor

def effective_single_site_evolution(
        state_tensor: NDArray[np.complex128],
        state_node: Node,
        hamiltonian_tensor: NDArray[np.complex128],
        hamiltonian_node: Node,
        time_step_size: float,
        tensor_cache: SandwichCache,
        mode: TimeEvoMode = TimeEvoMode.FASTEST,
        forward: bool = True,
        **options: dict[str, Any]
    ) -> NDArray[np.complex128]:
    """
    Perform the effective single site time evolution.

    This function computes the time evolution of a single site in a TTNS via
    a TTNO representing the Hamiltonian. The effective Hamiltonian is
    defined by the hamiltonian tensor and the environment tensors in the
    cache.

    Args:
        state_tensor (NDArray[np.complex128]): The tensor of the state to be
            updated.
        state_node (Node): The node of the state tensor.
        hamiltonian_tensor (NDArray[np.complex128]): The tensor of the
            Hamiltonian.
        hamiltonian_node (Node): The node of the Hamiltonian tensor.
        time_step_size (float): The time step size.
        tensor_cache (SandwichCache): The cache for the neighbour blocks.
        mode (TimeEvoMode, optional): The mode of the time evolution. Defaults
            to TimeEvoMode.FASTEST. If possible, the time evolution is
            performed without constructing the full effective Hamiltonian.
        forward (bool, optional): Whether to time evolve forward or backward.
            Defaults to True.
        **options: Additional options for the time evolution. See the
            documentation of the `TimeEvoMode`-class for more
            information.

    Returns:
        NDArray[np.complex128]: The updated state tensor after the time
            evolution.

    """
    if mode.action_evolvable():
        def contraction_action(t,y):
            """Defines the contraction to be used for the time evolution."""
            return contract_ket_ham_with_envs(state_node,
                                                y,
                                                hamiltonian_node,
                                                hamiltonian_tensor,
                                                tensor_cache)
        updated_tensor = mode.time_evolve_action(state_tensor,
                                                 contraction_action,
                                                 time_step_size,
                                                 forward=forward,
                                                 **options)
        return updated_tensor
    # Fallback to the default time evolution
    ham_eff = get_effective_single_site_hamiltonian_nodes(state_node,
                                                            hamiltonian_node,
                                                            hamiltonian_tensor,
                                                            tensor_cache)
    updated_tensor = mode.time_evolve(state_tensor,
                                        ham_eff,
                                        time_difference=time_step_size,
                                        forward=forward,
                                        **options
                                        )
    return updated_tensor

