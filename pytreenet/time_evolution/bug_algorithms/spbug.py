from typing import Dict, List, Union
from numpy import concat, tensordot

from ..ttn_time_evolution import TTNTimeEvolution
from ...operators.tensorproduct import TensorProduct
from ...ttns.ttns import TreeTensorNetworkState
from ...ttno.ttno_class import TreeTensorNetworkOperator
from ..time_evo_util.update_path import SweepingUpdatePathFinder, PathFinderMode
from ...contractions.sandwich_caching import SandwichCache
from ..time_evo_util.effective_time_evolution import single_site_time_evolution
from ...util.tensor_splitting import tensor_qr_decomposition, SplitMode
from ...contractions.state_operator_contraction import contract_leaf, contract_any
from ..time_evo_util.bug_util import compute_new_basis_tensor, BUGConfig
from ...core.truncation import TruncationEngine, TruncationMode

class SPBUG(TTNTimeEvolution):
    """
    - SPBUG (Second-order Path Basis-Update and Galerkin) algorithm for time evolution of tree tensor network state.
    - This algorithm performs time evolution by sweeping through the network in forward and backward directions with second-order accuracy.
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

        # Initialize the update paths and orthogonalization paths
        self.forward_update_path = self._finds_update_path(forward = True)
        self.forward_orth_path = self._find_orthogonalization_path(self.forward_update_path)

        self.backwards_update_path = self._finds_update_path(forward = False)
        self.backwards_orth_path = self._find_orthogonalization_path(self.backwards_update_path)

        # Initialize the orthogonalization center and the partial tree cache
        self._orthogonalize_init(force_new=True, mode=SplitMode.KEEP)
        self.partial_tree_cache = self._init_partial_tree_cache()
        if self.config.truncation_mode is None:
           self.config.truncation_mode = TruncationMode.SWEEPING_GREEDY
        self.trunc_engine = TruncationEngine(self.initial_state,
                                            self.config.truncation_mode)

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
            self.state.canonical_form(self.forward_update_path[0],
                                      mode=mode)
        else:
            self.state.move_orthogonalization_center(self.forward_update_path[0],
                                                     mode=mode)

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
                                                     mode=SplitMode.KEEP)
            previous_node_id = path[i]
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
        orth_path = []
        for i in range(len(update_path)-1):
            sub_path = self.state.path_from_to(update_path[i], update_path[i+1])
            orth_path.append(sub_path[1::])
        return orth_path    

    def _init_partial_tree_cache(self) -> SandwichCache:
        """
        Initialises the partial tree cache such that for all sites the bra,
        ket, and Hamiltonian tensor are contracted and saved in the cache,
        except for the first site to be updated.
        """
        return SandwichCache.init_cache_but_one(self.state,
                                                self.hamiltonian,
                                                self.forward_update_path[0])

    def update_first_leaf(self, node_id: str, next_node_id: str, time_step: float):
        """
        Updates the first leaf node in the update path.
        This method is called for the first leaf node in the update path.
        """
        target_state = self.state.deepcopy_parts([node_id , next_node_id])
        self.state.move_orthogonalization_center(next_node_id, mode=SplitMode.KEEP)

        updated_tensor = single_site_time_evolution(node_id,
                                                    target_state,
                                                    self.hamiltonian,
                                                    time_step,
                                                    self.partial_tree_cache,
                                                    mode=self.config.time_evo_mode)
        old_basis_tensor = self.state.tensors[node_id]
        concat_tensor = concat((old_basis_tensor, updated_tensor), axis=0)
        new_basis_tensor, _ = tensor_qr_decomposition(concat_tensor,
                                                    (1, ),
                                                    (0, ))

        new_basis_tensor = new_basis_tensor.T
        basis_change_tensor = tensordot(old_basis_tensor,
                                        new_basis_tensor.conj(),
                                        axes=([1],[1]))
        # contract bc tensor with next_node
        self.state.absorb_matrix_into_neighbour_leg(next_node_id, node_id,
                                                    basis_change_tensor , 0)
        self.state.replace_tensor(node_id, new_basis_tensor, 
                                new_shape = True)
        state_node, state_tensor = self.state[node_id]
        op_node, op_tensor = self.hamiltonian[node_id]
        child_block = contract_leaf(state_node, state_tensor,
                                    op_node, op_tensor)
        self.partial_tree_cache.add_entry(node_id, next_node_id, child_block)

    def update_leaf(self, node_id: str, next_node_id: str, time_step: float):
        """
        Updates a leaf node in the update path.
        This method is called for all leaf nodes in the update path,
        except for the first one.
        """
        target_state = self.state.deepcopy_parts([node_id , next_node_id])
        self.state.move_orthogonalization_center(next_node_id, mode=SplitMode.REDUCED)

        updated_tensor = single_site_time_evolution(node_id,
                                                    target_state,
                                                    self.hamiltonian,
                                                    time_step,
                                                    self.partial_tree_cache,
                                                    mode=self.config.time_evo_mode)

        old_basis_tensor = self.state.tensors[node_id]
        concat_tensor = concat((old_basis_tensor, updated_tensor), axis=0)
        new_basis_tensor, _ = tensor_qr_decomposition(concat_tensor,
                                                    (1, ),
                                                    (0, ))

        new_basis_tensor = new_basis_tensor.T
        basis_change_tensor = tensordot(old_basis_tensor,
                                        new_basis_tensor.conj(),
                                        axes=([1],[1]))
        # contract bc tensor with next_node
        self.state.absorb_matrix_into_neighbour_leg(next_node_id, node_id, 
                                                    basis_change_tensor , 0)
        self.state.replace_tensor(node_id, new_basis_tensor, 
                                new_shape = True)
        state_node, state_tensor = self.state[node_id]
        op_node, op_tensor = self.hamiltonian[node_id]
        child_block = contract_leaf(state_node, state_tensor,
                                    op_node, op_tensor)
        self.partial_tree_cache.add_entry(node_id, next_node_id, child_block)  

    def update_non_leaf(self, node_id: str, next_node_id: str, time_step: float):
        """
        Updates a non-leaf node in the update path.
        """
        target_state = self.state.deepcopy_parts([node_id , next_node_id])
        self.state.move_orthogonalization_center(next_node_id, mode=SplitMode.REDUCED)
        updated_tensor = single_site_time_evolution(node_id,
                                                    target_state,
                                                    self.hamiltonian,
                                                    time_step,
                                                    self.partial_tree_cache,
                                                    mode=self.config.time_evo_mode)
        new_state_node = self.state.nodes[node_id]

        old_tensor = self.state.tensors[node_id]
        new_basis_tensor = compute_new_basis_tensor(node = new_state_node,
                                                    old_tensor = old_tensor,
                                                    updated_tensor = updated_tensor,
                                                    neighbour_id = next_node_id)
        old_basis_node, old_basis_tensor = self.state[node_id]

        neighbour_leg = old_basis_node.neighbour_index(next_node_id)
        cont_legs = list(range(new_basis_tensor.ndim))
        cont_legs.pop(neighbour_leg)
        basis_change_tensor = tensordot(old_basis_tensor, new_basis_tensor.conj(), axes=(cont_legs,cont_legs))

        # contract bc tensor with next_node
        self.state.absorb_matrix_into_neighbour_leg(next_node_id, node_id, 
                                                    basis_change_tensor , 0)

        self.state.replace_tensor(node_id, new_basis_tensor, 
                                new_shape = True)


        child_block = contract_any(node_id = node_id,
                                    next_node_id = next_node_id,
                                    state = self.state,
                                    operator = self.hamiltonian,
                                    dictionary = self.partial_tree_cache)

        self.partial_tree_cache.add_entry(node_id, next_node_id, child_block)

    def update_last_leaf(self, node_id: str, time_step: float):
        """
        Updates the last leaf node in the update path.
        This method is called after all other nodes have been updated.
        """
        updated_tensor = single_site_time_evolution(node_id,
                                                    self.state,
                                                    self.hamiltonian,
                                                    time_step,
                                                    self.partial_tree_cache,
                                                    mode=self.config.time_evo_mode)
        self.state.replace_tensor(node_id, updated_tensor,
                                    new_shape = True) 

    def second_order_sweeping_update(self, forward: bool = True):
        """
        Runs one time step of the SPBUG algorithm with time_step_size/2
        by sweeps through :
        - forward_update_path if forward is True
        - backwards_update_path if forward is False.
        """
        if forward:
            for update_index, node_id in enumerate(self.forward_update_path):
                if update_index == 0:
                    next_node_id = self.forward_orth_path[update_index][0]
                    self.update_first_leaf(node_id, next_node_id, self.time_step_size/2)

                elif update_index < len(self.forward_orth_path):
                    next_node_id = self.forward_orth_path[update_index][0]
                    current_orth_path = self.forward_orth_path[update_index-1]
                    self._move_orth_and_update_cache_for_path(current_orth_path)
                    if self.state.nodes[node_id].is_leaf():
                        self.update_leaf(node_id, next_node_id, self.time_step_size/2)
                    else:
                        self.update_non_leaf(node_id, next_node_id, self.time_step_size/2)
                elif update_index == len(self.forward_update_path)-1:
                    self.update_last_leaf(node_id, self.time_step_size/2)
        else:
            for update_index, node_id in enumerate(self.backwards_update_path):
                if update_index == 0:
                    next_node_id = self.backwards_orth_path[update_index][0]
                    self.update_first_leaf(node_id, next_node_id, self.time_step_size/2)
                elif update_index < len(self.backwards_orth_path):
                    next_node_id = self.backwards_orth_path[update_index][0]
                    current_orth_path = self.backwards_orth_path[update_index-1]
                    self._move_orth_and_update_cache_for_path(current_orth_path)
                    if self.state.nodes[node_id].is_leaf():
                        self.update_leaf(node_id, next_node_id, self.time_step_size/2)
                    else:
                        self.update_non_leaf(node_id, next_node_id, self.time_step_size/2)
                elif update_index == len(self.backwards_update_path)-1:
                    self._move_orth_and_update_cache_for_path([node_id])
                    self.update_last_leaf(node_id, self.time_step_size/2)

    def initialize_ttn(self, orthogonality_center_id):
        """
        Initializes TTN by for the run_one_time_step method by 
        orthogonalizing the TTN and updating the partial_tree_cache.
        """
        if self.state.orthogonality_center_id:
            self.state.move_orthogonalization_center(orthogonality_center_id, mode=SplitMode.REDUCED)
        else:
            self.state.canonical_form(orthogonality_center_id, mode=SplitMode.REDUCED)
        self.partial_tree_cache = SandwichCache(self.state, self.hamiltonian)
        self.partial_tree_cache = SandwichCache.init_cache_but_one(self.state,
                                                                   self.hamiltonian,
                                                                   orthogonality_center_id)

    def run_one_time_step(self):
        """
        Runs one time step of the SPBUG algorithm.
        This method sweeps through the update path,
        updating each node according to the SPBUG algorithm.
        It also performs truncation and orthogonalze and cache 
        update partial_tree_cache for the next time step
        so that it is ready to be used again.
        """
        # Forward sweep with time_step_size/2 on forward_update_path
        self.second_order_sweeping_update(forward=True)

        # Backward truncation on trunc_path_backward
        self.trunc_engine.truncate(self.state, self.config, forward=False)

        # Initialize/Orthogonalize for the backward sweep
        self.initialize_ttn(self.backwards_update_path[0])

        # backward sweep on backwards_update_path
        self.second_order_sweeping_update(forward=False)

        # Forward truncation on trunc_path_forward
        self.trunc_engine.truncate(self.state, self.config, forward=True)

        # Initialize/Orthogonalize for next forward sweep
        self.initialize_ttn(self.forward_update_path[0])

        # TODO: Normalize ttn
