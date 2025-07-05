from typing import Dict, List, Union, Tuple, Any
from dataclasses import dataclass
from copy import deepcopy
from numpy import ndarray, concat, tensordot, eye, allclose

from ..ttn_time_evolution import TTNTimeEvolution
from ...operators.tensorproduct import TensorProduct
from ...ttns.ttns import TreeTensorNetworkState
from ...ttno.ttno_class import TreeTensorNetworkOperator
from ...util.tensor_splitting import SVDParameters
from ...contractions.sandwich_caching import SandwichCache
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...core.ttn import pull_tensor_from_different_ttn
from . .time_evo_util.effective_time_evolution import single_site_time_evolution
from ...core.truncation.recursive_truncation import truncate_node
from ...util.tensor_splitting import tensor_qr_decomposition, SplitMode
from ...contractions.state_operator_contraction import contract_leaf
from ...core.leg_specification import LegSpecification
from ...contractions.state_operator_contraction import contract_any
from ...util.tensor_util import compute_transfer_tensor
from ..time_evo_util.bug_util import (basis_change_tensor_id,
                                    reverse_basis_change_tensor_id,
                                    compute_basis_change_tensor,
                                    compute_new_basis_tensor,
                                    compute_fixed_size_new_basis_tensor)
from ...contractions.contraction_util import get_equivalent_legs
from ...time_evolution.ttn_time_evolution import TTNTimeEvolutionConfig
from ...time_evolution import TimeEvoMode

@dataclass
class PRBUGConfig(TTNTimeEvolutionConfig, SVDParameters):
    """
    The configuration class for the PRBUG methods.
    Attributes:
        deep (bool): Whether to use deepcopies of the TTNS during the update.
            If False, only the relevant nodes are copied at each point.
        fixed_rank (bool): Whether to use the fixed rank PRBUG or the standard
            PRBUG.
    """
    deep: bool = False
    fixed_rank: bool = False
    time_evo_mode: TimeEvoMode = TimeEvoMode.RK45
    def __post_init__(self):
        assert self.fixed_rank is False, "Fixed rank is not supported for PRBUG!"

class PRBUG(TTNTimeEvolution):
    """
    The PRBUG method for time evolution of tree tensor networks.

    PRBUG stands for Parallel Recursive Basis-Update and Galerkin. This class implements the rank-
    adaptive version introduced in https://www.doi.org/10.1137/22M1473790 .

    """
    config_class = PRBUGConfig

    def __init__(self,
                 initial_state: TreeTensorNetworkState,
                 hamiltonian: TreeTensorNetworkOperator,
                 time_step_size: float,
                 final_time: float,
                 operators: Union[List[Union[TensorProduct, TreeTensorNetworkOperator]],
                            Dict[str, Union[TensorProduct, TreeTensorNetworkOperator]],
                            TensorProduct,
                            TreeTensorNetworkOperator],
                 config: Union[PRBUGConfig, None] = None,
                 solver_options: Union[dict[str, Any], None] = None) -> None:

        super().__init__(initial_state,
                         time_step_size, final_time,
                         operators, config,
                         solver_options=solver_options)

        self.hamiltonian = hamiltonian
        self.state : TreeTensorNetworkState
        self.state.ensure_root_orth_center(mode = SplitMode.KEEP)
        self.config: PRBUGConfig

        self.bc_tensor_cache = PartialTreeCachDict()
        self._initial_state = deepcopy(self.state)
        self.new_state = deepcopy(self._initial_state)
        self.current_cache_old = SandwichCache.init_cache_but_one(self.state,
                                                                    self.hamiltonian,
                                                                    self._initial_state.root_id)
        self.current_cache_new = SandwichCache(state=None, hamiltonian=None)

    def recursive_update(self):
        """
        Recursively updates the tree tensor network.
        """
        self.current_cache_new = SandwichCache(state=None, hamiltonian=None)
        self.bc_tensor_cache = PartialTreeCachDict()

        self.root_update(self.state)

        self._initial_state = self.new_state
        self.state = self.new_state

    def run_one_time_step(self, **kwargs):
        """
        Runs one time step of the PRBUG method.

        The method is based on the rank-adaptive PRBUG method introduced in
        https://www.doi.org/10.1137/22M1473790 .

        If the configured time step is >= 0.1, it performs two internal updates
        each with half the time step for potentially better stability/accuracy.

        Args:
            kwargs: Additional keyword arguments for the time step.
        """
        self.recursive_update()
        truncate_node(self.state.root_id, self.state, self.config)


    def update_leaf_node(self,
                         node_id: str,
                         current_state: TreeTensorNetworkState) -> Tuple[TreeTensorNetworkState, ndarray, ndarray]:
        """
        Updates a leaf node according to the fixed rank PRBUG.

        Args:
            node_id (str): The id of the node to be updated.
            current_state (TreeTensorNetworkState): The state, where the current
                node is the orthogonality center.
            parent_state (TreeTensorNetworkState): The state where the parent of
                the current node is the orthogonality center.
            current_cache (SandwichCache): The cache for the neighbour blocks.
            fixed_rank (bool): Whether to use the fixed rank PRBUG or the standard
                PRBUG.

        Returns:
            TreeTensorNetworkState: The updated new state.
            ndarray: The neighbour block towards the parent, with the updated
                tensors.
            ndarray: The basis change tensor.

        """
        updated_tensor = single_site_time_evolution(node_id = node_id,
                                                    state = current_state,
                                                    hamiltonian = self.hamiltonian,
                                                    time_step_size = self.time_step_size,
                                                    tensor_cache = self.current_cache_old,
                                                    mode = self.config.time_evo_mode,
                                                    solver_options = self.solver_options)
        old_basis_tensor = self._initial_state.tensors[node_id]
        assert allclose(compute_transfer_tensor(old_basis_tensor, (1,)), eye(old_basis_tensor.shape[0]))
        if self.config.fixed_rank:
            new_basis_tensor, _ = tensor_qr_decomposition(updated_tensor,
                                                        (1, ),
                                                        (0, ),
                                                        mode=SplitMode.KEEP)
        else:
            concat_tensor = concat((old_basis_tensor, updated_tensor), axis=0)
            new_basis_tensor, r = tensor_qr_decomposition(concat_tensor,
                                                        (1, ),
                                                        (0, ))
            assert new_basis_tensor.shape == (2,2), "The leaf new basis tensor shape is not (2,2) !"
            assert allclose(compute_transfer_tensor(new_basis_tensor, (1,)), eye(new_basis_tensor.shape[0]))
            assert allclose(tensordot (r, new_basis_tensor, axes=(0,1)),concat_tensor )
        new_basis_tensor = new_basis_tensor.T
        basis_change_tensor = tensordot(old_basis_tensor,
                                        new_basis_tensor.conj(),
                                        axes=([1],[1]))
        if not current_state.nodes[self.new_state.nodes[node_id].parent].is_root():
            self.bc_tensor_cache.add_entry(node_id, self._initial_state.nodes[node_id].parent, basis_change_tensor)
        assert not current_state.nodes[node_id].is_root()
        parent_id = current_state.nodes[node_id].parent
        state_node_before , _ = self.new_state[node_id]
        self.new_state.split_node_replace(node_id = node_id,
                                    tensor_a= basis_change_tensor,
                                    tensor_b = new_basis_tensor,
                                    identifier_a=basis_change_tensor_id(node_id),
                                    identifier_b=node_id,
                                    legs_a =LegSpecification(parent_id,[],[]),
                                    legs_b =LegSpecification(None, [], [1]),
                                    strict_checks = False)
        state_node, state_tensor = self.new_state[node_id]

        op_node, op_tensor = self.hamiltonian[node_id]
        legs = get_equivalent_legs(state_node_before, op_node)
        assert legs[0] == legs[1]

        assert allclose(compute_transfer_tensor(state_tensor, (1,)), eye(state_tensor.shape[0]))
        child_block = contract_leaf(state_node, state_tensor,
                                    op_node, op_tensor)
        self.current_cache_new.add_entry(node_id, parent_id, child_block)

    def update_non_leaf_node(self,
                             node_id: str,
                            current_state: TreeTensorNetworkState,
                            parent_state: TreeTensorNetworkState) -> Tuple[TreeTensorNetworkState, ndarray, ndarray]:
        """
        Updates a non-leaf node according to the fixed rank PRBUG.

        Args:
            node_id (str): The id of the node to be updated.
            current_state (TreeTensorNetworkState): The state, where the current
                node is the orthogonality center.
            parent_state (TreeTensorNetworkState): The state where the parent of
                the current node is the orthogonality center.
            current_cache (SandwichCache): The cache for the neighbour blocks.

        Returns:
            TreeTensorNetworkState: The updated new state.
            ndarray: The neighbour block towards the parent, with the updated
                tensors.
            ndarray: The basis change tensor.

        """
        #child_environment_cache = PartialTreeCachDict() # We only need to add entries, so no Sandwich needed
        parent_state_before = parent_state
        for child_id in frozenset(current_state.nodes[node_id].children):
            self.update_node(node_id = child_id,
                            parent_state = current_state)
        parent_state_after = parent_state
        assert parent_state_before == parent_state_after, "The parent state has changed!"

        pull_tensor_from_different_ttn(old_ttn = current_state,
                                        new_ttn = self.new_state,
                                        node_id = node_id,
                                        mod_fct= reverse_basis_change_tensor_id)
        # Now we need to contract the basis change tensors into the tensor
        # of the current node
        self.new_state.contract_all_children(node_id)
        # Now we can update the tensor
        # downward cache from current_cache_old
        parent_id = self.new_state.nodes[node_id].parent
        old_cache = self.current_cache_old.get_entry(parent_id, node_id)
        self.current_cache_new.add_entry(parent_id, node_id, old_cache)
        updated_tensor = single_site_time_evolution(node_id,
                                                    self.new_state,
                                                    self.hamiltonian,
                                                    self.time_step_size,
                                                    self.current_cache_new,
                                                    mode=self.config.time_evo_mode,
                                                    solver_options=self.solver_options)
        new_state_node = self.new_state.nodes[node_id]
        if self.config.fixed_rank:
            new_basis_tensor = compute_fixed_size_new_basis_tensor(new_state_node,
                                                                    updated_tensor)
        else:
            old_tensor = self.new_state.tensors[node_id]
            new_basis_tensor = compute_new_basis_tensor(node = new_state_node,
                                                        old_tensor = old_tensor,
                                                        updated_tensor = updated_tensor,
                                                        neighbour_id = new_state_node.parent)
        old_basis_node, old_basis_tensor = self._initial_state[node_id]
        basis_change_tensor = compute_basis_change_tensor(node_old = old_basis_node,
                                                        node_new = new_state_node,
                                                        tensor_old = old_basis_tensor,
                                                        tensor_new = new_basis_tensor,
                                                        basis_change_tensor_cache = self.bc_tensor_cache)
        if not current_state.nodes[old_basis_node.parent].is_root():
            self.bc_tensor_cache.add_entry(node_id, new_state_node.parent, basis_change_tensor)
        # Now we need to insert the new tensors into the new state
        state_node_before , _ = self.new_state[node_id]

        self.new_state.split_node_replace(node_id = node_id,
                                    tensor_a = basis_change_tensor,
                                    tensor_b = new_basis_tensor,
                                    identifier_a = basis_change_tensor_id(node_id),
                                    identifier_b = node_id,
                                    legs_a = LegSpecification(new_state_node.parent,[],[]),
                                    legs_b = LegSpecification(None, new_state_node.children, new_state_node.open_legs),
                                    strict_checks = False)
        op_node, _ = self.hamiltonian[node_id]


        state_node_after , _ = self.new_state[node_id]
        assert op_node.parent_leg == self.new_state.nodes[node_id].parent_leg
        legs = get_equivalent_legs(state_node_before, state_node_after, [basis_change_tensor_id(node_id), state_node_before.parent] )
        assert legs[0] == legs[1]

        child_block = contract_any(node_id = node_id,
                                    next_node_id=basis_change_tensor_id(node_id), # This is currently the parent node
                                    state = self.new_state,
                                    operator = self.hamiltonian,
                                    dictionary = self.current_cache_new)
        self.current_cache_new.add_entry(node_id, parent_id, child_block)

    def update_node(self,
                    node_id: str,
                    parent_state: TreeTensorNetworkState) -> Tuple[TreeTensorNetworkState, ndarray, ndarray]:
        """
        Updates a node according to the fixed rank PRBUG.

        Args:
            node_id (str): The id of the node to be updated.
            parent_state (TreeTensorNetworkState): The state where the parent is
                the orthogonality center.
            parent_tensor_cache (SandwichCache): The cache for the parents
                neighbour blocks.

        Returns:
            TreeTensorNetworkState: The updated new state.
            ndarray: The neighbour block towards the parent, with the updated
                tensors.
            ndarray: The basis change tensor.
        """
        parent_id = parent_state.nodes[node_id].parent
        assert parent_id is not None, "There is no basis change for the root node!"
        assert parent_id == parent_state.orthogonality_center_id, "The parent is not the orth center!"
        if self.config.deep:
            target_state = deepcopy(parent_state)
        else:
            copy_nodes = [node_id, parent_id]
            target_state = parent_state.deepcopy_parts(copy_nodes)

        if self.config.fixed_rank:
            mode = SplitMode.KEEP
        else:
            mode = SplitMode.REDUCED

        target_state.move_orthogonalization_center(node_id, mode=mode)
        self.current_cache_old.state = target_state
        self.current_cache_old.update_tree_cache(parent_id, node_id)
        if target_state.nodes[node_id].is_leaf():
            self.update_leaf_node(node_id = node_id,
                                    current_state = target_state)
        else:
            self.update_non_leaf_node(node_id = node_id,
                                        current_state = target_state,
                                        parent_state = parent_state)

    def root_update(self,
                    Initial_state: TreeTensorNetworkState,
                    ) -> TreeTensorNetworkState:
        """
        Updates the root node of the state according to the fixed rank PRBUG.

        Args:
            Initial_state (TreeTensorNetworkState): The current state of the system
                with the root node as the orthogonality center.
        
        Returns:
            TreeTensorNetworkState: The updated state.

        """

        root_id = Initial_state.root_id
        assert root_id is not None, "The state has no root node!"
        assert Initial_state.orthogonality_center_id == root_id, \
            "The orthogonality center is not the root node!"
        self.new_state = deepcopy(Initial_state)
        self.current_cache_old = SandwichCache.init_cache_but_one(Initial_state,
                                                              self.hamiltonian,
                                                              root_id)
        for child_id in frozenset(Initial_state.nodes[root_id].children):
            self.update_node(node_id = child_id,
                                parent_state = Initial_state)
        pull_tensor_from_different_ttn(old_ttn = Initial_state,
                                        new_ttn = self.new_state,
                                        node_id = root_id,
                                        mod_fct= reverse_basis_change_tensor_id)
        # Now we need to contract the basis change tensors into the tensor
        # of the current node
        self.new_state.contract_all_children(root_id)
        # Now we can update the tensor
        updated_tensor = single_site_time_evolution(root_id,
                                                    self.new_state,
                                                    self.hamiltonian,
                                                    self.time_step_size,
                                                    self.current_cache_new,
                                                    mode=self.config.time_evo_mode,
                                                    solver_options=self.solver_options)
        self.new_state.replace_tensor(root_id, updated_tensor)