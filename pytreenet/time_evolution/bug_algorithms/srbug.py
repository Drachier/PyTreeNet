from typing import Dict, List, Union, Any
from dataclasses import dataclass
from copy import deepcopy

from numpy import tensordot
from ..ttn_time_evolution import TTNTimeEvolution
from ...operators.tensorproduct import TensorProduct
from ...core.truncation.recursive_truncation import truncate_node
from ...ttns.ttns import TreeTensorNetworkState
from ...ttns.ttndo import SymmetricTTNDO, BINARYTTNDO
from ...ttno.ttno_class import TreeTensorNetworkOperator
from .. import TimeEvoMode
from ...contractions.sandwich_caching import SandwichCache
from ..time_evo_util.effective_time_evolution import single_site_time_evolution
from ...util.tensor_splitting import tensor_qr_decomposition, SplitMode, SVDParameters
from ...contractions.state_operator_contraction import contract_leaf
from ...core.leg_specification import LegSpecification
from ...contractions.state_operator_contraction import contract_any
from ..time_evo_util.bug_util import (basis_change_tensor_id,
                                    compute_new_basis_tensor,
                                    compute_fixed_size_new_basis_tensor)
from ..ttn_time_evolution import TTNTimeEvolutionConfig

@dataclass
class SRBUGConfig(TTNTimeEvolutionConfig, SVDParameters):
    """
    The configuration class for the Sequential Recursive BUG methods.

    Attributes:
        deep (bool): Whether to use deepcopies of the TTNS during the update.
            If False, only the relevant nodes are copied at each point.
        fixed_rank (bool): Whether to use the fixed rank RBBUG or the standard SRBUG.
    """
    deep: bool = False
    fixed_rank: bool = False
    time_evo_mode: TimeEvoMode = TimeEvoMode.RK45


class SRBUG(TTNTimeEvolution):
    """
    The SRBUG method for time evolution of tree tensor networks.
    SRBUG stands for Sequential Recursive Basis-Update and Galerkin.

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
        config (Union[SRBUGConfig,None]): The configuration of
            time evolution. Defaults to None.
        solver_options (Union[Dict[str, Any], None], optional): Most time
            evolutions algorithms use some kind of solver to resolve a
            partial differential equation. This dictionary can be used to
            pass additional options to the solver. Refer to the
            documentation of `ptn.time_evolution.TimeEvoMode` for further
            information. Defaults to None.

    """
    config_class = SRBUGConfig

    def __init__(self,
                 initial_state: Union[SymmetricTTNDO, BINARYTTNDO],
                 hamiltonian: TreeTensorNetworkOperator,
                 time_step_size: float,
                 final_time: float,
                 operators: Union[List[Union[TensorProduct, TreeTensorNetworkOperator]],
                                  Dict[str, Union[TensorProduct, TreeTensorNetworkOperator]],
                                  TensorProduct,
                                  TreeTensorNetworkOperator],
                 config: Union[SRBUGConfig, None] = None,
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
            config (Union[SRBUGConfig,None]): The configuration of
                time evolution. Defaults to None.
            solver_options (Union[Dict[str, Any], None], optional): Most time
                evolutions algorithms use some kind of solver to resolve a
                partial differential equation. This dictionary can be used to
                pass additional options to the solver. Refer to the
                documentation of `ptn.time_evolution.TimeEvoMode` for further
                information. Defaults to None.
        """
        super().__init__(initial_state,
                         time_step_size, final_time,
                         operators,
                         config=config,
                         solver_options=solver_options)

        self.hamiltonian = hamiltonian
        self.state : TreeTensorNetworkState
        self.state.ensure_root_orth_center(mode = SplitMode.KEEP)
        self.config: SRBUGConfig
        self.new_state = None
        self.cache = SandwichCache(state=None,hamiltonian=None)

    def recursive_update(self):
        """
        Recursively updates the tree tensor network.
        """
        self.root_update(self.state)
        self.state = self.new_state


    def run_one_time_step(self, **kwargs):
        """
        Runs one time step of the SRBUG method.

        Args:
            kwargs: Additional keyword arguments for the time step.

        """
        self.recursive_update()

        truncate_node(self.state.root_id, self.state, self.config)

        # 2: Post-truncation with max_effective_ham_dim
        #post_svd_params = copy(self.config)
        #post_svd_params.sum_trunc = False
        #post_svd_params.rel_tol = float('-inf')
        #post_svd_params.total_tol = float('-inf')
        #post_truncate_node(self.state.root_id, self.state, post_svd_params)

        #self.state.canonical_form(self.state.root_id, mode=SplitMode.REDUCED)

    def update_leaf_node(self,
                         node_id: str,
                         parent_id: str):
        """
        Updates a leaf node according to the fixed rank SRBUG.

        Args:
            node_id (str): The id of the node to be updated.
            parent_id (str): The id of the parent of the node to be updated.
        Returns:
            TreeTensorNetworkState: The updated new state.
            ndarray: The neighbour block towards the parent, with the updated
                tensors.
            ndarray: The basis change tensor.
        """
        if self.config.deep:
            target_state = deepcopy(self.new_state)
        else:
            copy_nodes = [node_id, parent_id]
            target_state = self.new_state.deepcopy_parts(copy_nodes)

        target_state.move_orthogonalization_center(node_id, mode = SplitMode.REDUCED)
        self.cache.state = target_state
        self.cache.update_tree_cache(parent_id, node_id)

        self.cache.state = target_state
        updated_tensor = single_site_time_evolution(node_id = node_id,
                                                    state = target_state,
                                                    hamiltonian = self.hamiltonian,
                                                    time_step_size = self.time_step_size,
                                                    tensor_cache = self.cache,
                                                    mode = self.config.time_evo_mode,
                                                    solver_options=self.solver_options)
        old_basis_tensor = self.new_state.tensors[node_id]

        new_basis_tensor, _ = tensor_qr_decomposition(updated_tensor,
                                                    (1, ),
                                                    (0, ),
                                                    mode = SplitMode.REDUCED)

        new_basis_tensor = new_basis_tensor.T

        basis_change_tensor = tensordot(old_basis_tensor,
                                        new_basis_tensor.conj(),
                                        axes=([1],[1]))
        parent_id = target_state.nodes[node_id].parent
        self.new_state.split_node_replace(node_id = node_id,
                                    tensor_a= basis_change_tensor,
                                    tensor_b = new_basis_tensor,
                                    identifier_a=basis_change_tensor_id(node_id),
                                    identifier_b=node_id,
                                    legs_a =LegSpecification(parent_id,[],[]),
                                    legs_b =LegSpecification(None, [], [1]))
        state_node, state_tensor = self.new_state[node_id]

        op_node, op_tensor = self.hamiltonian[node_id]
        child_block = contract_leaf(state_node, state_tensor,
                                    op_node, op_tensor)
        self.cache.add_entry(node_id, parent_id, child_block)
        self.new_state.contract_to_parent(node_id = basis_change_tensor_id(node_id))

    def update_non_leaf_node(self,
                             node_id: str,
                             parent_id: str):
        """
        Updates a non-leaf node according to the fixed rank SRBUG.

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
        for child_id in frozenset(self.new_state.nodes[node_id].children):
            self.update_node(child_id, node_id)

        if self.config.deep:
            target_state = deepcopy(self.new_state)
        else:
            copy_nodes = [node_id, parent_id]
            target_state = self.new_state.deepcopy_parts(copy_nodes)

        self.new_state.move_orthogonalization_center(parent_id, mode = SplitMode.REDUCED)
        self.cache.state = target_state
        updated_tensor = single_site_time_evolution(node_id,
                                                    target_state,
                                                    self.hamiltonian,
                                                    self.time_step_size,
                                                    self.cache,
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
                                                        neighbour_id = parent_id)
        old_basis_node, old_basis_tensor = self.new_state[node_id]

        parent_leg = old_basis_node.neighbour_index(parent_id)
        cont_legs = list(range(new_basis_tensor.ndim))
        cont_legs.pop(parent_leg)
        basis_change_tensor = tensordot(old_basis_tensor,
                                        new_basis_tensor.conj(),
                                        axes=(cont_legs,cont_legs))

        self.new_state.split_node_replace(
            node_id = node_id,
            tensor_a = basis_change_tensor,
            tensor_b = new_basis_tensor,
            identifier_a = basis_change_tensor_id(node_id),
            identifier_b = node_id,
            legs_a = LegSpecification(new_state_node.parent,[],[]),
            legs_b = LegSpecification(None, new_state_node.children, new_state_node.open_legs))

        child_block = contract_any(
            node_id = node_id,
            next_node_id=basis_change_tensor_id(node_id), # This is currently the parent node
            state = self.new_state,
            operator = self.hamiltonian,
            dictionary = self.cache)

        self.cache.add_entry(node_id, parent_id, child_block)
        self.new_state.contract_to_parent(node_id = basis_change_tensor_id(node_id))

    def update_node(self,
                    node_id: str,
                    parent_id: str):
        """
        Updates a node according to the fixed rank SRBUG.

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
        if  self.new_state.nodes[node_id].is_leaf():
            self.update_leaf_node(node_id, parent_id)
        else:
            self.new_state.move_orthogonalization_center(node_id, mode = SplitMode.REDUCED)
            self.cache.state = self.new_state
            self.cache.update_tree_cache(parent_id, node_id)

            self.update_non_leaf_node(node_id, parent_id)

    def root_update(self,
                    initial_state: Union[SymmetricTTNDO, BINARYTTNDO]):
        """
        Updates the root node of the state according to the fixed rank SRBUG.

        Args:
            initial_state (Union[SymmetricTTNDO, BINARYTTNDO]): The current state of the system
                with the root node as the orthogonality center.

        """
        root_id = initial_state.root_id
        self.new_state = deepcopy(initial_state)
        self.cache = SandwichCache.init_cache_but_one(initial_state,
                                                      self.hamiltonian,
                                                      root_id)
        for child_id in frozenset(self.new_state.nodes[root_id].children):
            self.update_node(child_id, root_id)

        self.cache.state = self.new_state
        assert self.new_state.is_in_canonical_form(root_id)
        updated_tensor = single_site_time_evolution(root_id,
                                                    self.new_state,
                                                    self.hamiltonian,
                                                    self.time_step_size,
                                                    self.cache,
                                                    mode=self.config.time_evo_mode,
                                                    solver_options=self.solver_options)
        self.new_state.replace_tensor(root_id, updated_tensor)
