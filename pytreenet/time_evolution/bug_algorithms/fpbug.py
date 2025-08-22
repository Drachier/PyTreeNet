from copy import deepcopy
from typing import Dict, List, Union
from numpy import allclose, concat, eye, tensordot
from pytreenet.util.tensor_util import compute_transfer_tensor
from ..ttn_time_evolution import TTNTimeEvolution
from ...operators.tensorproduct import TensorProduct
from ...ttns.ttns import TreeTensorNetworkState
from ...ttno.ttno_class import TreeTensorNetworkOperator
from ...core.leg_specification import LegSpecification
from ..time_evo_util.update_path import SweepingUpdatePathFinder, PathFinderMode
from ...contractions.sandwich_caching import SandwichCache
from  ..time_evo_util.effective_time_evolution import single_site_time_evolution
from ...util.tensor_splitting import tensor_qr_decomposition, SplitMode
from ...contractions.state_operator_contraction import contract_leaf, contract_any
from ..time_evo_util.bug_util import (compute_new_basis_tensor,
                                      BUGConfig,
                                      adjust_ttn1_structure_to_ttn2,
                                      adjust_node1_structure_to_node2)
from ...core.truncation import TruncationEngine, TruncationMode
from ..time_evo_util.bug_util import basis_change_tensor_id
from ...contractions.state_state_contraction import get_equivalent_legs

class FPBUG(TTNTimeEvolution):
    """
    - FPBUG (First-order Path Basis-Update and Galerkin) algorithm for time evolution of tree tensor network state.
    - This algorithm performs time evolution by sweeping through the network.
    - Sweeping truncation is performed after each time step to keep the bond dimensions manageable.
    """
    config_class = BUGConfig
    def __init__(self,
                 initial_state: TreeTensorNetworkState,
                 hamiltonian: TreeTensorNetworkOperator,
                 time_step_size: float,
                 final_time: float,
                 operators: Union[List[Union[TensorProduct, TreeTensorNetworkOperator]],
                            Dict[str, Union[TensorProduct, TreeTensorNetworkOperator]],
                            TensorProduct,
                            TreeTensorNetworkOperator],
                 config: Union[BUGConfig, None] = None) -> None:

        super().__init__(initial_state,
                         time_step_size, final_time,
                         operators, config)

        self.hamiltonian = hamiltonian
        self.state : TreeTensorNetworkState
        self.config: BUGConfig

        self.update_path = self._finds_update_path(forward = True)
        self.orthogonalization_path = self._find_orthogonalization_path(self.update_path)
        self._orthogonalize_init(mode = SplitMode.KEEP)

        self.partial_tree_cache = self._init_partial_tree_cache()
        if self.config.truncation_mode is None:
            self.config.truncation_mode = TruncationMode.SWEEPING_GREEDY
        self.trunc_engine = TruncationEngine(self.initial_state,
                                             self.config.truncation_mode)

        # Using the initial state as a reference to adjust the structure throughout the time evolution.
        adjust_ttn1_structure_to_ttn2(self.initial_state, self.hamiltonian)

    def _finds_update_path(self, forward: bool = True) -> List[str]:
        """
        Finds the update path for this TDVP Algorithm.

        Overwrite to create custom update paths for specific tree topologies.

        Returns:
            List[str]: The order in which the nodes in the TTN should be time
                evolved.
        """
        if forward:
            return SweepingUpdatePathFinder(self.initial_state,
                                    PathFinderMode.LeafToLeaf_Forward).find_path()
        else:
            return SweepingUpdatePathFinder(self.initial_state,
                                    PathFinderMode.LeafToLeaf_Backward).find_path()

    def _orthogonalize_init(self, force_new: bool=False,
                            mode: SplitMode = SplitMode.KEEP):
        """
        Orthogonalises the state to the start of the update path.
        
        If the state is already orthogonalised, the orthogonalisation center
        is moved to the start of the update path.

        Args:
            force_new (bool, optional): If True a complete orthogonalisation
                is always enforced, instead of moving the orthogonality center.
                Defaults to False.
            mode: The mode to be used for the QR decomposition. For details refer to
                `tensor_util.tensor_qr_decomposition`.
        """
        if self.state.orthogonality_center_id is None or force_new:
            self.state.canonical_form(self.update_path[0],
                                      mode=mode,preserve_legs_order=True)
        else:
            self.state.move_orthogonalization_center(self.update_path[0],
                                                     mode=mode, preserve_legs_order=True)

    def _move_orth_and_update_cache_for_path(self, path: List[str]):
        """
        Moves the orthogonalisation center and updates all required caches
        along a given path.

        Args:
            path (List[str]): The path to move from. Should start with the
                orth center and end at with the final node. If the path is empty
                or only the orth center nothing happens.
        """
        if len(path) == 0:
            return
        assert self.state.orthogonality_center_id == path[0]
        for i, node_id in enumerate(path[1:]):
            self.state.move_orthogonalization_center(node_id,
                                                     mode = SplitMode.KEEP,
                                                     preserve_legs_order=True)
            previous_node_id = path[i] # +0, because path starts at 1.
            self.partial_tree_cache.update_tree_cache(previous_node_id, node_id)

    def _find_orthogonalization_path(self,
                                     update_path: List[str]) -> List[List[str]]:
        """
        Find orthogonalisation path.

        Args:
            update_path (List[str]): The path along which updates sites.

        Returns:
            List[List[str]]: a list of paths, along which the TTNS should be
            orthogonalised between every node update.
        """
        orthogonalization_path = []
        for i in range(len(update_path)-1):
            sub_path = self.state.path_from_to(update_path[i], update_path[i+1])
            orthogonalization_path.append(sub_path[1::])
        return orthogonalization_path    

    def _init_partial_tree_cache(self) -> SandwichCache:
        """
        Initialises the partial tree cache such that for all sites the bra,
        ket, and Hamiltonian tensor are contracted and saved in the cache,
        except for the first site to be updated.
        """
        return SandwichCache.init_cache_but_one(self.state,
                                                self.hamiltonian,
                                                self.update_path[0])

    def update_first_leaf(self, node_id: str, next_node_id: str):
        if self.config.deep:
            target_state = deepcopy(self.state)
        else:
            copy_nodes = [node_id, next_node_id]
            target_state = self.state.deepcopy_parts(copy_nodes)

        updated_tensor = single_site_time_evolution(node_id,
                                                    target_state,
                                                    self.hamiltonian,
                                                    self.time_step_size,
                                                    self.partial_tree_cache,
                                                    mode=self.config.time_evo_mode)
        self.state.move_orthogonalization_center(next_node_id, mode = SplitMode.KEEP, preserve_legs_order=True)
        old_basis_tensor = self.state.tensors[node_id]
        concat_tensor = concat((old_basis_tensor, updated_tensor), axis=0)
        new_basis_tensor, _ = tensor_qr_decomposition(concat_tensor,
                                                    (1, ),
                                                    (0, ))
        #assert allclose(tensordot(new_basis_tensor.T.conj(), new_basis_tensor, axes=([1],[0])), eye(new_basis_tensor.shape[1]))
        new_basis_tensor = new_basis_tensor.T
        basis_change_tensor = tensordot(old_basis_tensor,
                                        new_basis_tensor.conj(),
                                        axes=([1],[1]))
        # contract bc tensor with parent
        self.state.absorb_matrix_into_neighbour_leg(next_node_id, node_id, 
                                                    basis_change_tensor , 0)
        self.state.replace_tensor(node_id, new_basis_tensor,
                                new_shape = True)
        adjust_ttn1_structure_to_ttn2(self.state, self.initial_state)

        new_tensor = self.state.tensors[node_id]
        new_node = self.state.nodes[node_id]
        assert new_node.open_legs == [1]
        #assert allclose(compute_transfer_tensor(new_tensor, (1)), eye(new_tensor.shape[0]))

        state_node, state_tensor = self.state[node_id]
        op_node, op_tensor = self.hamiltonian[node_id]
        child_block = contract_leaf(state_node, state_tensor,
                                    op_node, op_tensor)
        self.partial_tree_cache.add_entry(node_id, next_node_id, child_block)


    def update_leaf(self, node_id: str, next_node_id: str, update_index: int):

        current_orth_path = self.orthogonalization_path[update_index-1]
        self._move_orth_and_update_cache_for_path(current_orth_path)

        if self.config.deep:
            target_state = deepcopy(self.state)
        else:
            copy_nodes = [node_id, next_node_id]
            target_state = self.state.deepcopy_parts(copy_nodes)
        updated_tensor = single_site_time_evolution(node_id,
                                                    target_state,
                                                    self.hamiltonian,
                                                    self.time_step_size,
                                                    self.partial_tree_cache,
                                                    mode=self.config.time_evo_mode)
        self.state.move_orthogonalization_center(next_node_id, mode = SplitMode.KEEP, preserve_legs_order=True)
        old_basis_tensor = self.state.tensors[node_id]

        concat_tensor = concat((old_basis_tensor, updated_tensor), axis=0)
        new_basis_tensor, _ = tensor_qr_decomposition(concat_tensor,
                                                    (1, ),
                                                    (0, ))
        new_basis_tensor = new_basis_tensor.T
        basis_change_tensor = tensordot(old_basis_tensor,
                                        new_basis_tensor.conj(),
                                        axes=([1],[1]))
        parent_id = self.state.nodes[node_id].parent

        # contract bc tensor with parent
        self.state.split_node_replace(node_id = node_id,
                                    tensor_a= basis_change_tensor,
                                    tensor_b = new_basis_tensor,
                                    identifier_a=basis_change_tensor_id(node_id),
                                    identifier_b=node_id,
                                    legs_a =LegSpecification(parent_id,[],[]),
                                    legs_b =LegSpecification(None, [], [1]),
                                    strict_checks = False)
        self.state.contract_to_parent(basis_change_tensor_id(node_id), parent_id,)
        adjust_ttn1_structure_to_ttn2(self.state, self.initial_state)
        
        new_node = self.state.nodes[node_id]
        assert new_node.open_legs == [1]
        #new_tensor = self.state.tensors[node_id]
        #assert allclose(compute_transfer_tensor(new_tensor, (1)), eye(new_tensor.shape[0]))

        state_node, state_tensor = self.state[node_id]
        op_node, op_tensor = self.hamiltonian[node_id]
        child_block = contract_leaf(state_node, state_tensor,
                                    op_node, op_tensor)
        self.partial_tree_cache.add_entry(node_id, next_node_id, child_block)

    def update_non_leaf(self, node_id: str, next_node_id: str, update_index: int):

        current_orth_path = self.orthogonalization_path[update_index-1]
        self._move_orth_and_update_cache_for_path(current_orth_path)

        if self.config.deep:
            target_state = deepcopy(self.state)
        else:
            copy_nodes = [node_id, next_node_id]
            target_state = self.state.deepcopy_parts(copy_nodes)
        self.state.move_orthogonalization_center(next_node_id, mode = SplitMode.KEEP, preserve_legs_order=True)

        updated_tensor = single_site_time_evolution(node_id,
                                                    target_state,
                                                    self.hamiltonian,
                                                    self.time_step_size,
                                                    self.partial_tree_cache,
                                                    mode=self.config.time_evo_mode)
        old_basis_node, old_basis_tensor = self.state[node_id]
        new_basis_tensor = compute_new_basis_tensor(node = old_basis_node,
                                                    old_tensor = old_basis_tensor,
                                                    updated_tensor = updated_tensor,
                                                    neighbour_id = next_node_id)

        neighbour_leg = old_basis_node.neighbour_index(next_node_id)
        cont_legs = list(range(new_basis_tensor.ndim))
        cont_legs.pop(neighbour_leg)
        #eye_shape = new_basis_tensor.shape[neighbour_leg]
        #assert allclose(compute_transfer_tensor(new_basis_tensor, cont_legs), eye(eye_shape))

        old_basis_tensor = self.state.tensors[node_id]
        basis_change_tensor = tensordot(old_basis_tensor, new_basis_tensor.conj(), axes=(cont_legs,cont_legs))

        # contract bc tensor with next_node
        self.state.absorb_matrix_into_neighbour_leg(next_node_id, node_id,
                                                    basis_change_tensor, 0)

        self.state.replace_tensor(node_id, new_basis_tensor,
                                  new_shape = True)

        #assert allclose(compute_transfer_tensor(self.state.tensors[node_id], cont_legs), eye(eye_shape))

        child_block = contract_any(node_id = node_id,
                                    next_node_id = next_node_id,
                                    state = self.state,
                                    operator = self.hamiltonian,
                                    dictionary = self.partial_tree_cache)
        self.partial_tree_cache.add_entry(node_id, next_node_id, child_block)

    def update_last_leaf(self, node_id: str):
        updated_tensor = single_site_time_evolution(node_id,
                                                    self.state,
                                                    self.hamiltonian,
                                                    self.time_step_size,
                                                    self.partial_tree_cache,
                                                    mode=self.config.time_evo_mode)
        self.state.replace_tensor(node_id, updated_tensor,
                                    new_shape = True)

    def sweeping_update(self) -> None:
        """
        Runs one time step of the FPBUG algorithm.
        This method sweeps through the update path, updating each node
        according to the FPBUG algorithm.
        """
        for update_index, node_id in enumerate(self.update_path):
            if update_index == 0:
                next_node_id = self.orthogonalization_path[update_index][0]
                self.update_first_leaf(node_id, next_node_id)
                adjust_node1_structure_to_node2(self.state, self.initial_state, node_id)
                check_two_ttn_compatibility(self.state, self.initial_state)
            elif update_index < len(self.orthogonalization_path):
                next_node_id = self.orthogonalization_path[update_index][0]
                if self.state.nodes[node_id].is_leaf():
                    self.update_leaf(node_id, next_node_id, update_index)
                    adjust_node1_structure_to_node2(self.state, self.initial_state, node_id)
                    check_two_ttn_compatibility(self.state, self.initial_state)
                else:
                    self.update_non_leaf(node_id, next_node_id, update_index)
                    adjust_node1_structure_to_node2(self.state, self.initial_state, node_id)
                    check_two_ttn_compatibility(self.state, self.initial_state)

            elif update_index == len(self.update_path)-1:
                self.update_last_leaf(node_id)
                check_two_ttn_compatibility(self.state, self.initial_state)

    def initialize_ttn(self):
        """
        Initializes TTN by orthogonalizing it to the first 
        node in the update path.
        And update the partial_tree_cache.
        So that it is ready to be used in the next time step.
        """
        if self.state.orthogonality_center_id:
            self.state.move_orthogonalization_center(self.update_path[0], 
                                                     mode=SplitMode.REDUCED,
                                                     preserve_legs_order=True)
        else:
            self.state.canonical_form(self.update_path[0], mode=SplitMode.REDUCED)
        self.partial_tree_cache = SandwichCache(self.state, self.hamiltonian)
        self.partial_tree_cache = SandwichCache.init_cache_but_one(self.state,
                                                                   self.hamiltonian,
                                                                   self.update_path[0])

    def run_one_time_step(self):
        """
        Runs one time step of the FPBUG algorithm.
        This method sweeps through the update path,
        updating each node according to the FPBUG algorithm.
        It also performs truncation and orthogonalze and cache 
        update partial_tree_cache for the next time step
        so that it is ready to be used again.
        """
        # Ensure the leg permutations of all nodes have not been changed
        adjust_ttn1_structure_to_ttn2(self.state, self.initial_state)

        # sweeping on update_path
        self.sweeping_update()

        # Truncation
        self.trunc_engine.truncate(self.state, self.config, forward = False)

        # Orthogonalie state and initialize the partial tree cache
        self.initialize_ttn()

def check_two_ttn_compatibility(ttn1, ttn2):
    for nodes in ttn1.nodes:
        legs = get_equivalent_legs(ttn1.nodes[nodes], ttn2.nodes[nodes])
        assert legs[0] == legs[1], (
            f"Node {nodes} : {ttn1.nodes[nodes].neighbouring_nodes()} vs {ttn2.nodes[nodes].neighbouring_nodes()}"
        )