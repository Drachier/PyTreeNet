"""
This module contains functions for time evolution using effective Hamiltonians.

The effective Hamiltonian's are usually obtained by partially contracting a
TTNO Hamiltonian with a TTNS and its conjugate.

"""
from __future__ import annotations
from typing import Any
from uuid import uuid4

from numpy import ndarray
import numpy as np
from numpy.typing import NDArray

from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict
from pytreenet.contractions.effective_hamiltonians import (get_effective_single_site_hamiltonian_nodes,
                                                           get_effective_bond_hamiltonian_nodes)
from pytreenet.time_evolution.time_evolution import TimeEvoMode, EvoDirection
from pytreenet.core.node import Node
from pytreenet.contractions.state_operator_contraction import (contract_ket_ham_with_envs,
                                                               contract_bond_tensor)
from pytreenet.contractions.node_contraction import contract_nodes

# Evolution functions using TTN
def bond_time_evolution(node_id: str,
                        state: TreeTensorNetworkState,
                        time_step_size: float,
                        tensor_cache: PartialTreeCachDict,
                        forward: EvoDirection = EvoDirection.FORWARD,
                        mode: TimeEvoMode = TimeEvoMode.FASTEST,
                        solver_options: dict[str, Any] | None = None
                        ) -> ndarray:
    """
    Perform the time evolution for a bond.

    Args:
        node_id (str): The id of the node to be updated.
        state (TreeTensorNetworkState): The state of the system.
        time_step_size (float): The time step size.
        tensor_cache (PartialTreeCachDict): The cache for the neighbour blocks.
        forward (EvoDirection, optional): Whether to time evolve forward or
            backward. Defaults to EvoDirection.FORWARD.
        mode (TimeEvoMode): The mode of the time evolution. Defaults to
            TimeEvoMode.FASTEST
        solver_options (dict[str, Any] | None, optional): Additional options
            for the time evolution. See the documentation of the
            `TimeEvoMode`-class for more information. Defaults to None.

    Returns:
        ndarray: The updated tensor.

    """
    if solver_options is None:
        solver_options = {}
    state_node, state_tensor = state[node_id]
    updated_tensor = effective_bond_evolution(
        state_tensor,
        state_node,
        time_step_size,
        tensor_cache,
        mode=mode,
        forward=forward,
        **solver_options
    )
    return updated_tensor

def single_site_time_evolution(node_id: str,
                               state: TreeTensorNetworkState,
                               hamiltonian: TreeTensorNetworkOperator,
                               time_step_size: float,
                               tensor_cache: PartialTreeCachDict,
                               forward: EvoDirection = EvoDirection.FORWARD,
                               mode: TimeEvoMode = TimeEvoMode.FASTEST,
                               solver_options: dict[str, Any] | None = None
                               ) -> ndarray:
    """
    Perform the time evolution for a single site.

    Args:
        node_id (str): The id of the node to be updated.
        state (TreeTensorNetworkState): The state of the system.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian of the system.
        time_step_size (float): The time step size.
        tensor_cache (SandwichCache): The cache for the neighbour blocks.
        forward (EvoDirection, optional): Whether to time evolve forward or
            backward.
        mode (TimeEvoMode): The mode of the time evolution. Defaults to
            TimeEvoMode.FASTEST
        solver_options (dict[str, Any] | None, optional): Additional options
            for the time evolution. See the documentation of the
            `TimeEvoMode`-class for more information. Defaults to None.

    Returns:
        ndarray: The updated tensor.

    """
    if solver_options is None:
        solver_options = {}
    state_node, state_tensor = state[node_id]
    ham_node, ham_tensor = hamiltonian[node_id]
    updated_tensor = effective_single_site_evolution(
        state_tensor,
        state_node,
        ham_tensor,
        ham_node,
        time_step_size,
        tensor_cache,
        mode=mode,
        forward=forward,
        **solver_options
    )
    return updated_tensor

def two_site_time_evolution(node_id: str,
                            next_node_id: str,
                            contracted_id: str,
                            state: TreeTensorNetworkState,
                            hamiltonian: TreeTensorNetworkOperator,
                            time_step_size: float,
                            tensor_cache: PartialTreeCachDict,
                            forward: EvoDirection = EvoDirection.FORWARD,
                            mode: TimeEvoMode = TimeEvoMode.FASTEST,
                            solver_options: dict[str, Any] | None = None
                            ) -> ndarray:
    """
    Perform the time evolution for two sites.

    Args:
        node_id (str): The id of the node to be updated. Should be represented
            by the first leg in the state tensor.
        next_node_id (str): The id of the other node to be updated. Should be
            represented by the second leg in the state tensor.
        contracted_id (str): The id of the node that represents the
            contraction of the two nodes.
        state (TreeTensorNetworkState): The state of the system with the two
            nodes representing the two sites not yet contracted.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian of the system.
        time_step_size (float): The time step size.
        tensor_cache (SandwichCache): The cache for the neighbour blocks.
        forward (EvoDirection, optional): Whether to time evolve forward or
            backward.
        mode (TimeEvoMode): The mode of the time evolution. Defaults to
            TimeEvoMode.FASTEST
        solver_options (dict[str, Any] | None, optional): Additional options
            for the time evolution. See the documentation of the
            `TimeEvoMode`-class for more information. Defaults to None.

    Returns:
        ndarray: The updated tensor.
    """
    if solver_options is None:
        solver_options = {}
    # Get the neighbours before the contraction.
    # Otherwise, the two nodes are gone from the state.
    main_neighbours = state.nodes[node_id].neighbouring_nodes()
    next_neighbours = state.nodes[next_node_id].neighbouring_nodes()
    # Perform contraction of the two nodes.
    state.contract_nodes(node_id, next_node_id,
                            new_identifier=contracted_id)
    # Pull out the correct tensors from the state and hamiltonian.
    state_node, state_tensor = state[contracted_id]
    ham1_node, ham1_tensor = hamiltonian[node_id]
    ham2_node, ham2_tensor = hamiltonian[next_node_id]
    # We have to ensure that the cache points to the contracted node.
    for neighbour_id in main_neighbours:
        if neighbour_id != next_node_id:
            tensor_cache.change_next_id_for_entry(neighbour_id,
                                                  node_id,
                                                  contracted_id)
    for neighbour_id in next_neighbours:
        if neighbour_id != node_id:
            tensor_cache.change_next_id_for_entry(neighbour_id,
                                                  next_node_id,
                                                  contracted_id)
    # Now we update the two sites using the effective Hamiltonian.
    updated_tensor = effective_two_site_evolution(
        state_tensor,
        state_node,
        (ham1_tensor, ham2_tensor),
        (ham1_node, ham2_node),
        time_step_size,
        tensor_cache,
        mode=mode,
        forward=forward,
        **solver_options
    )
    # We have to ensure that the cache points to the old nodes again.
    for neighbour_id in main_neighbours:
        if neighbour_id != next_node_id:
            tensor_cache.change_next_id_for_entry(neighbour_id,
                                                  contracted_id,
                                                  node_id)
    for neighbour_id in next_neighbours:
        if neighbour_id != node_id:
            tensor_cache.change_next_id_for_entry(neighbour_id,
                                                  contracted_id,
                                                  next_node_id)
    # Finally, we return the updated tensor.
    return updated_tensor

# Evolution functions using nodes and tensors
def effective_bond_evolution(
        state_tensor: NDArray[np.complex128],
        state_node: Node,
        time_step_size: float,
        tensor_cache: PartialTreeCachDict,
        mode: TimeEvoMode = TimeEvoMode.FASTEST,
        forward: EvoDirection = EvoDirection.FORWARD,
        **options
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
        tensor_cache (PartialTreeCachDict): The cache for the neighbour blocks.
        mode (TimeEvoMode, optional): The mode of the time evolution. Defaults
            to TimeEvoMode.FASTEST. If possible, the time evolution is
            performed without constructing the full effective Hamiltonian.
        forward (EvoDirection, optional): Whether to time evolve forward or backward.
            Defaults to EvoDirection.FORWARD.
        **options: Additional options for the time evolution. See the
            documentation of the `TimeEvoMode`-class for more
            information.

    """
    if mode.action_evolvable():
        def contraction_action(t,y):
            """Defines the contraction to be used for the time evolution."""
            return contract_bond_tensor(y, state_node, tensor_cache)
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
        tensor_cache: PartialTreeCachDict,
        mode: TimeEvoMode = TimeEvoMode.FASTEST,
        forward: EvoDirection = EvoDirection.FORWARD,
        **options
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
        tensor_cache (PartialTreeCachDict): The cache for the neighbour blocks.
        mode (TimeEvoMode, optional): The mode of the time evolution. Defaults
            to TimeEvoMode.FASTEST. If possible, the time evolution is
            performed without constructing the full effective Hamiltonian.
        forward (EvoDirection, optional): Whether to time evolve forward or backward.
            Defaults to EvoDirection.FORWARD.
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

def effective_two_site_evolution(
        state_tensor: NDArray[np.complex128],
        state_node: Node,
        hamiltonian_tensors: NDArray[np.complex128] | tuple[NDArray[np.complex128], NDArray[np.complex128]],
        hamiltonian_nodes: Node | tuple[Node, Node],
        time_step_size: float,
        tensor_cache: PartialTreeCachDict,
        mode: TimeEvoMode = TimeEvoMode.FASTEST,
        forward: EvoDirection = EvoDirection.FORWARD,
        **options
    ) -> NDArray[np.complex128]:
    """
    Perform the effective two site time evolution.

    This function computes the time evolution of two sites in a TTNS via
    a TTNO representing the Hamiltonian. The effective Hamiltonian is
    defined by the hamiltonian tensors and the environment tensors in the
    cache.

    Args:
        state_tensors (NDArray[np.complex128]): A tensor representing the two
            sites to be updated. It should have two open legs, one for each
            site.
        state_node (Node): The node corresponding to the state tensor and the
            two sites.
        hamiltonian_tensors (NDArray[np.complex128] | tuple[NDArray[np.complex128], NDArray[np.complex128]]):
            The tensors of the Hamiltonian. If a tuple is provided, the tensors
            will be contracted in this function. If the contraction result is
            already available, a single tensor can be provided. It should have
            four open legs in the order (h1out, h2out, h1in, h2in), where 1
            corresponds to the site represented by the first leg of the stte
            tensor and 2 corresponds to the site represented by the second
            leg of the state tensor.
        hamiltonian_nodes (Node | tuple[Node, Node]): The nodes of the
            Hamiltonian tensors. If a tuple is provided, the nodes will be
            contracted in this function. If the contraction result is already
            available, a single node can be provided.
        time_step_size (float): The time step size.
        tensor_cache (PartialTreeCachDict): The cache for the neighbour blocks.
            The relevant blocks have to already point to the contracted node,
            i.e. the state node supplied here.
        mode (TimeEvoMode, optional): The mode of the time evolution. Defaults
            to TimeEvoMode.FASTEST. If possible, the time evolution is
            performed without constructing the full effective Hamiltonian.
        forward (EvoDirection, optional): Whether to time evolve forward or backward.
            Defaults to EvoDirection.FORWARD.
        **options: Additional options for the time evolution. See the
            documentation of the `TimeEvoMode`-class for more
            information.
        
    Returns:
        NDArray[np.complex128]: The updated state tensor after the time
            evolution.

    """
    if isinstance(hamiltonian_tensors, tuple):
        if not isinstance(hamiltonian_nodes, tuple):
            errstr = "If two Hamiltonian tensors are provided, " \
                        "the corresponding nodes must also be provided as a tuple."
            raise ValueError(errstr)
        hamiltonian_nodes, hamiltonian_tensors = contract_nodes(
                                                        hamiltonian_nodes[0],
                                                        hamiltonian_tensors[0],
                                                        hamiltonian_nodes[1],
                                                        hamiltonian_tensors[1],
                                                        new_identifier=state_node.identifier)
        # Now the open legs are not in the correct order, but in (h1out, h1in, h2out, h2in).
        # We need to transpose them to (h1out, h2out, h1in, h2in).
        hamiltonian_nodes.exchange_open_leg_ranges(range(-2, -1), range(-3, -2))
        hamiltonian_tensors = hamiltonian_nodes.transpose_tensor(hamiltonian_tensors)
    elif isinstance(hamiltonian_nodes, tuple):
        errstr = "If two Hamiltonian nodes are provided, " \
                    "the corresponding tensors must also be provided as a tuple."
        raise ValueError(errstr)
    # Now the two nodes can act as a single node.
    updated_tensor = effective_single_site_evolution(state_tensor,
                                                     state_node,
                                                     hamiltonian_tensors,
                                                     hamiltonian_nodes,
                                                     time_step_size,
                                                     tensor_cache,
                                                     mode=mode,
                                                     forward=forward,
                                                     **options)
    return updated_tensor
