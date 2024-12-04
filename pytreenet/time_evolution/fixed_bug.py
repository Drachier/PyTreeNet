from typing import Dict, List, Union, Tuple
from copy import deepcopy, copy

from numpy import ndarray, tensordot

from pytreenet.time_evolution.ttn_time_evolution import TTNTimeEvolution, TTNTimeEvolutionConfig
from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.contractions.sandwich_caching import SandwichCache
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict
from pytreenet.contractions.effective_hamiltonians import get_effective_single_site_hamiltonian
from pytreenet.time_evolution.time_evolution import time_evolve
from pytreenet.util.tensor_splitting import SplitMode, tensor_qr_decomposition
from pytreenet.core.leg_specification import LegSpecification
from pytreenet.contractions.state_operator_contraction import contract_leaf, contract_any
from pytreenet.core.ttn import pull_tensor_from_different_ttn

from pytreenet.time_evolution.time_evo_util.bug_util import (basis_change_tensor_id,
                                                             reverse_basis_change_tensor_id,
                                                             compute_fixed_size_new_basis_tensor,
                                                             compute_basis_change_tensor)

class FixedBUG(TTNTimeEvolution):
    """
    The fixed rank Basis-Update and Galerkin (BUG) time evolution algorithm.
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
                 config: Union[TTNTimeEvolutionConfig, None] = None
                 ) -> None:

        super().__init__(initial_state,
                         time_step_size,
                         final_time,
                         operators,
                         config=config)

        self.hamiltonian = hamiltonian
        self.state : TreeTensorNetworkState
        self._ensure_root_orth_center()

    def _ensure_root_orth_center(self) -> None:
        """
        Ensure that the root node is the orthogonality center of the state.
        """
        root_id = self.state.root_id
        assert root_id is not None, "The state has no root node!"
        if root_id != self.state.orthogonality_center_id:
            self.state.move_orthogonalization_center(root_id, mode=SplitMode.KEEP)

    def recursive_update(self):
        """
        Recursively updates the state according to the fixed rank BUG.
        """
        self.state = root_update(self.state,
                                 self.hamiltonian,
                                 self.time_step_size)

    def run_one_time_step(self, **kwargs):
        """
        Run one time step of the time evolution.
        """
        self.recursive_update()

def root_update(current_state: TreeTensorNetworkState,
                hamiltonian: TreeTensorNetworkOperator,
                time_step_size: float) -> TreeTensorNetworkState:
    """
    Updates the root node of the state according to the fixed rank BUG.

    Args:
        current_state (TreeTensorNetworkState): The current state of the system
            with the root node as the orthogonality center.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian operator of
            model as a TTNO.
        time_step_size (float): The time step size.
    
    Returns:
        TreeTensorNetworkState: The updated state.

    """
    root_id = current_state.root_id
    assert root_id is not None, "The state has no root node!"
    assert current_state.orthogonality_center_id == root_id, \
        "The orthogonality center is not the root node!"
    new_state = deepcopy(current_state)
    current_cache = SandwichCache.init_cache_but_one(current_state,
                                                     hamiltonian,
                                                     root_id)
    child_bc_tensors = PartialTreeCachDict() # Needed to save the basis change tensors of the children
    for child_id in frozenset(current_state.nodes[root_id].children):
        new_state, child_block, child_bc_tensor = update_node(child_id,
                                                                new_state,
                                                                current_state,
                                                                current_cache,
                                                                hamiltonian,
                                                                time_step_size)
        child_bc_tensors.add_entry(child_id, root_id, child_bc_tensor)
        current_cache.add_entry(child_id, root_id, child_block)
    # In the new state the current node is not the orth center, but we
    # we need that tensor as initial condition for the update
    pull_tensor_from_different_ttn(current_state,
                                    new_state,
                                    root_id,
                                    reverse_basis_change_tensor_id)
    # Now we need to contract the basis change tensors into the tensor
    # of the current node
    new_state.contract_all_children(root_id)
    # Now we can update the tensor
    updated_tensor = single_site_time_evolution(root_id,
                                                new_state,
                                                hamiltonian,
                                                time_step_size,
                                                current_cache)
    new_state.replace_tensor(root_id, updated_tensor)
    return new_state

def update_leaf_node(node_id: str,
                current_state: TreeTensorNetworkState,
                new_state: TreeTensorNetworkState,
                parent_state: TreeTensorNetworkState,
                current_cache: SandwichCache,
                hamiltonian: TreeTensorNetworkOperator,
                time_step_size: float
                ) -> Tuple[TreeTensorNetworkState, ndarray, ndarray]:
    """
    Updates a leaf node according to the fixed rank BUG.

    Args:
        node_id (str): The id of the node to be updated.
        current_state (TreeTensorNetworkState): The state, where the current
            node is the orthogonality center.
        new_state (TreeTensorNetworkState): The state into which all the
            updated nodes are stored.
        parent_state (TreeTensorNetworkState): The state where the parent of
            the current node is the orthogonality center.
        current_cache (SandwichCache): The cache for the neighbour blocks.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian operator of
            model as a TTNO.
        time_step_size (float): The time step size.

    Returns:
        TreeTensorNetworkState: The updated new state.
        ndarray: The neighbour block towards the parent, with the updated
            tensors.
        ndarray: The basis change tensor.

    """
    updated_tensor = single_site_time_evolution(node_id,
                                                current_state,
                                                hamiltonian,
                                                time_step_size,
                                                current_cache)
    new_basis_tensor, _ = tensor_qr_decomposition(updated_tensor,
                                                (1, ),
                                                (2, ))
    old_basis_tensor = parent_state.tensors[node_id]
    basis_change_tensor = tensordot(old_basis_tensor,
                                    new_basis_tensor.conj(),
                                    axes=([1],[1]))
    assert not current_state.nodes[node_id].is_root()
    parent_id = current_state.nodes[node_id].parent
    new_state.split_node_replace(node_id,
                                new_basis_tensor,
                                basis_change_tensor,
                                node_id,
                                basis_change_tensor_id(node_id),
                                LegSpecification(None, [], [1]),
                                LegSpecification(parent_id,[],[]))
    block_tensor = contract_leaf(node_id,new_state,hamiltonian)
    return new_state, block_tensor, basis_change_tensor

def update_non_leaf_node(node_id: str,
                         current_state: TreeTensorNetworkState,
                         new_state: TreeTensorNetworkState,
                         parent_state: TreeTensorNetworkState,
                         current_cache: SandwichCache,
                         hamiltonian: TreeTensorNetworkOperator,
                         time_step_size: float
                         ) -> Tuple[TreeTensorNetworkState, ndarray, ndarray]:
    """
    Updates a non-leaf node according to the fixed rank BUG.

    Args:
        node_id (str): The id of the node to be updated.
        current_state (TreeTensorNetworkState): The state, where the current
            node is the orthogonality center.
        new_state (TreeTensorNetworkState): The state into which all the
            updated nodes are stored.
        parent_state (TreeTensorNetworkState): The state where the parent of
            the current node is the orthogonality center.
        current_cache (SandwichCache): The cache for the neighbour blocks.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian operator of
            model as a TTNO.
        time_step_size (float): The time step size.
    
    Returns:
        TreeTensorNetworkState: The updated new state.
        ndarray: The neighbour block towards the parent, with the updated
            tensors.
        ndarray: The basis change tensor.

    """
    child_bc_tensors = PartialTreeCachDict() # Needed to save the basis change tensors of the children
    for child_id in frozenset(current_state.nodes[node_id].children):
        new_state, child_block, child_bc_tensor = update_node(child_id,
                                                                new_state,
                                                                current_state,
                                                                current_cache,
                                                                hamiltonian,
                                                                time_step_size)
        child_bc_tensors.add_entry(child_id, node_id, child_bc_tensor)
        current_cache.add_entry(child_id, node_id, child_block)
    # In the new state the current node is not the orth center, but we
    # we need that tensor as initial condition for the update
    pull_tensor_from_different_ttn(current_state,
                                    new_state,
                                    node_id,
                                    reverse_basis_change_tensor_id)
    # Now we need to contract the basis change tensors into the tensor
    # of the current node
    new_state.contract_all_children(node_id)
    # Now we can update the tensor
    updated_tensor = single_site_time_evolution(node_id,
                                                new_state,
                                                hamiltonian,
                                                time_step_size,
                                                current_cache)
    new_state_node = new_state.nodes[node_id]
    new_basis_tensor = compute_fixed_size_new_basis_tensor(new_state_node,
                                                            updated_tensor)
    old_basis_node, old_basis_tensor = parent_state[node_id]
    basis_change_tensor = compute_basis_change_tensor(old_basis_node,
                                                        new_state_node,
                                                        old_basis_tensor,
                                                        new_basis_tensor,
                                                        child_bc_tensors)
    # Now we need to insert the new tensors into the new state
    new_state.split_node_replace(node_id,
                                    new_basis_tensor,
                                    basis_change_tensor,
                                    node_id,
                                    basis_change_tensor_id(node_id),
                                    LegSpecification(None, new_state_node.children, new_state_node.open_legs),
                                    LegSpecification(new_state_node.parent,[],[]))
    block_tensor = contract_any(node_id,
                                new_state_node.parent,
                                new_state,
                                hamiltonian,
                                current_cache)
    return new_state, block_tensor, basis_change_tensor

def update_node(node_id: str,
                new_state: TreeTensorNetworkState,
                parent_state: TreeTensorNetworkState,
                parent_tensor_cache: SandwichCache,
                hamiltonian: TreeTensorNetworkOperator,
                time_step_size: float
                ) -> Tuple[TreeTensorNetworkState, ndarray, ndarray]:
    """
    Updates a node according to the fixed rank BUG.

    Args:
        node_id (str): The id of the node to be updated.
        new_state (TreeTensorNetworkState): The state into which all the
            updated nodes are stored.
        parent_state (TreeTensorNetworkState): The state where the parent is
            the orthogonality center.
        parent_tensor_cache (SandwichCache): The cache for the parents
            neighbour blocks.
        hamiltonian (TreeTensorNetworkOperator): The Hamiltonian operator of
            model as a TTNO.
        time_step_size (float): The time step size.

    Returns:
        TreeTensorNetworkState: The updated new state.
        ndarray: The neighbour block towards the parent, with the updated
            tensors.
        ndarray: The basis change tensor.

    """
    current_state = deepcopy(parent_state)
    current_state.move_orthogonalization_center(node_id, mode=SplitMode.KEEP)
    current_cache = copy(parent_tensor_cache)
    current_cache.state = current_state
    # Since the orth center changed, we need to update that specific neighbour block
    parent_id = current_state.nodes[node_id].parent
    assert parent_id is not None, "There is no basis change for the root node!"
    current_cache.update_tree_cache(parent_id, node_id)
    # We can also delete the obsolete neighbour block from the child to the parent
    current_cache.delete_entry(node_id, parent_id)
    if current_state.nodes[node_id].is_leaf():
        return update_leaf_node(node_id,
                                current_state,
                                new_state,
                                parent_state,
                                current_cache,
                                hamiltonian,
                                time_step_size)
    else:
        return update_non_leaf_node(node_id,
                                    current_state,
                                    new_state,
                                    parent_state,
                                    current_cache,
                                    hamiltonian,
                                    time_step_size)

def single_site_time_evolution(node_id: str,
                               state: TreeTensorNetworkState,
                               hamiltonian: TreeTensorNetworkOperator,
                               time_step_size: float,
                               tensor_cache: SandwichCache) -> ndarray:
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

    Returns:
        ndarray: The updated tensor.

    """
    ham_eff = get_effective_single_site_hamiltonian(node_id,
                                                    state,
                                                    hamiltonian,
                                                    tensor_cache)
    updated_tensor = time_evolve(state.tensors[node_id],
                                 ham_eff,
                                 time_step_size)
    return updated_tensor
