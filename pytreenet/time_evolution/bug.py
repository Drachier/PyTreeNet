
from typing import Dict, List, Union, Tuple, FrozenSet, Self
from copy import deepcopy, copy

from numpy import ndarray

from .ttn_time_evolution import TTNTimeEvolution, TTNTimeEvolutionConfig
from ..operators.tensorproduct import TensorProduct
from ..core.ttn import (pull_tensor_from_different_ttn,
                        get_tensor_from_different_ttn)
from ..core.graph_node import GraphNode
from ..core.node import Node
from ..core.truncation.recursive_truncation import recursive_truncation
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..contractions.effective_hamiltonians import get_effective_single_site_hamiltonian_nodes
from ..contractions.sandwich_caching import SandwichCache
from ..contractions.tree_cach_dict import PartialTreeCachDict
from ..util.tensor_splitting import SVDParameters
from .time_evolution import time_evolve
from .time_evo_util.bug_util import (basis_change_tensor_id,
                                     reverse_basis_change_tensor_id,
                                    compute_basis_change_tensor,
                                    compute_new_basis_tensor,
                                    find_new_basis_replacement_leg_specs)

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
        self.ttns_dict: Dict[str,TreeTensorNetworkState] = {}
        self.cache_dict: Dict[str,SandwichCache] = {}
        self.basis_change_cache = PartialTreeCachDict()
        self._ensure_root_orth_center()
        self.tensor_cache = SandwichCache.init_cache_but_one(self.state,
                                                             self.hamiltonian,
                                                             self.state.root_id)
        if svd_params is None:
            self.svd_parameters = SVDParameters()
        else:
            self.svd_parameters = svd_params

    def close_to(self, other: Self) -> bool:
        """
        Compares two BUG objects for closeness.

        Args:
            other (BUG): The other BUG object to compare to.

        Returns:
            bool: True if the two BUG objects are close, False otherwise.

        """
        if self.state != other.state:
            return False
        if self.hamiltonian != other.hamiltonian:
            return False
        if self.time_step_size != other.time_step_size:
            return False
        if self.final_time != other.final_time:
            return False
        if self.operators != other.operators:
            return False
        if self.svd_parameters != other.svd_parameters:
            return False
        if self.ttns_dict != other.ttns_dict:
            return False
        if len(self.cache_dict) != len(other.cache_dict):
            return False
        for key, cache in self.cache_dict.items():
            if key not in other.cache_dict:
                return False
            if not cache.close_to(other.cache_dict[key]):
                return False
        if len(self.basis_change_cache) != len(other.basis_change_cache):
            return False
        for key, cache in self.basis_change_cache.items():
            if key not in other.basis_change_cache:
                return False
            if not cache.close_to(other.basis_change_cache[key]):
                return False
        return True

    def _ensure_root_orth_center(self):
        """
        Ensures that the root of the initial state is the ortogonality center.

        """
        root_id = self.state.root_id
        assert root_id is not None, "Root id is None!"
        ort_center = self.state.orthogonality_center_id
        if ort_center is None:
            self.state.canonical_form(root_id)
        else:
            self.state.move_orthogonalization_center(root_id)
        # Otherwise an external reset might take the uncanonicalised state
        self._initial_state = deepcopy(self.state)

    def time_evolve_node(self,
                         initial: Tuple[GraphNode,ndarray],
                         hamiltonian: Tuple[GraphNode,ndarray],
                         current_cache: PartialTreeCachDict) -> ndarray:
        """
        Time evolves the node with the given id.

        While the new state is used for the neighbour ordering, the cache used
        corresponds to the specific nodes. It contains the old sandwich tensors
        of any parent tensor and the updated sandwich tensors of the children.

        Args:
            initial (Tuple[GraphNode,ndarray]): The initial node and tensor.
            hamiltonian (Tuple[GraphNode,ndarray]): The Hamiltonian node and
                tensor.
            current_cache (PartialTreeCachDict): The cache with all the
                necessary envrionment tensors.
        
        Returns:
            ndarray: The updated tensor
        """
        initial_node, initial_tensor = initial
        hamiltonian_node, hamiltonian_tensor = hamiltonian
        h_eff = get_effective_single_site_hamiltonian_nodes(initial_node,
                                                            hamiltonian_node,
                                                            hamiltonian_tensor,
                                                            current_cache)
        updated_tensor = time_evolve(initial_tensor,
                                     h_eff,
                                     self.time_step_size,
                                     forward=True)
        return updated_tensor

    def replace_node_with_updated_basis(self, node_id: str,
                                        new_basis_tensor: ndarray,
                                        basis_change_tensor: ndarray
                                        ):
        """
        Replaces a node with the tensors of the updated basis.

        Args:
            node_id (str): The id of the node to replace.
            new_basis_tensor (ndarray): The new basis tensor.
            basis_change_tensor (ndarray): The basis change tensor.
        
        """
        node = self.state.nodes[node_id]
        leg_specs = find_new_basis_replacement_leg_specs(node)
        self.state.split_node_replace(node_id,
                                      basis_change_tensor,
                                      new_basis_tensor,
                                      basis_change_tensor_id(node_id),
                                      node_id,
                                      leg_specs[0],
                                      leg_specs[1])

    def prepare_state_for_update(self,
                                    node_id: str
                                    ) -> TreeTensorNetworkState:
        """
        Prepares the new state for the update.

        This means that the connectivity tensor with updated basis legs
        is created in the state. (Corresponds to the creation of C_hat in the
        reference and just takes the old tensor into the new state for leafs.)

        Args:
            node_id (str): The id of the node to prepare.

        Returns:
            TreeTensorNetworkState: The state with the new tensor.

        """
        current_ttns = self.ttns_dict[node_id]
        pull_tensor_from_different_ttn(current_ttns, self.state,
                                       node_id,
                                       mod_fct=reverse_basis_change_tensor_id)
        self.state.contract_all_children(node_id)
        return self.state

    def get_current_node_and_tensor(self, node_id: str) -> Tuple[GraphNode,ndarray]:
        """
        Returns the current node and tensor for the given node.

        Args:
            node_id (str): The id of the node to get the current node and tensor
                for.

        Returns:
            Tuple[GraphNode,ndarray]: The current node and tensor.

        """
        current_node, current_tensor = self.state[node_id]
        return current_node, current_tensor

    def get_updated_cache_tensor(self,
                                 child_id: str,
                                 node_id: str
                                 ) -> ndarray:
        """
        Obtains the updated cached environment tensor.

        The tensor is the contracted subtree starting at the child and its
        open legs point towards the node. Notably this is after the child
        node was updated.

        Args:
            child_id (str): The identifier of the node that is the root of the
                subtree that has been contracted as the environment tensot.
                Should be a child of the main node.
            node_id (str): The identifier of the node to which the open legs
                should point.
        
        Returns:
            ndarray: The environment/sandwich tensor of the updated subtree.

        """
        try:
            return self.tensor_cache.get_entry(child_id,
                                                node_id)
        except KeyError as e:
            errstr = "The updated subtree tensor is not available!\n"
            errstr += f"Nodes: {child_id} -> {node_id}"
            raise KeyError(errstr) from e

    def prepare_cache_for_update(self,
                                    node_id: str
                                    ) -> Tuple[TreeTensorNetworkState,SandwichCache]:
        """
        Prepares the current cache for the update.

        To find the correct effective Hamiltonian, the cache has to include the
        updated sandwich tensors of the children.

        Args:
            node_id (str): The id of the node to prepare.

        Returns:
            TreeTensorNetworkState: The state with the tensor to be updated.
            SandwichCache: The updated cache.
        """
        state = self.prepare_state_for_update(node_id)
        current_cache = self.cache_dict[node_id]
        for child_id in self.get_children_ids_to_update(node_id):
            tensor = self.get_updated_cache_tensor(child_id,
                                                    node_id)
            current_cache.add_entry(child_id, node_id,
                                    tensor)
        return state, current_cache

    def prepare_for_node_update(self,
                                node_id: str
                                ) -> Tuple[TreeTensorNetworkState,SandwichCache]:
        """
        Prepares everything to update a node.

        This means, that the main state now has the tensor to be updated
        associated to it and that the cache of the node to be updated contains
        the contracted subtrees of the node's children.

        Args:
            node_id (str): The identifier of the node that is to be updated.
            
        Returns:
            TreeTensorNetworkState: The state containing the node and tensor to
                be updated.
            SandwichCache: The cache associated to the node containing the
                updated children environments.

        """
        return self.prepare_cache_for_update(node_id)

    def prepare_root_state(self) -> TreeTensorNetworkState:
        """
        Prepare the root state, which is just the old state.
        """
        assert len(self.ttns_dict) == 0, "TTNS dict not empty!"
        root_ttns = deepcopy(self.state)
        self.ttns_dict[root_ttns.root_id] = root_ttns
        return root_ttns

    def prepare_root_cache(self) -> SandwichCache:
        """
        Prepares the cache for the root node.
        """
        # We must first prepare the root state
        root_ttns = self.prepare_root_state()
        root_id = self.state.root_id
        assert len(self.cache_dict) == 0, "Cache dict not empty!"
        # Due to truncation, the cache has to be renewed
        current_cache =  SandwichCache.init_cache_but_one(root_ttns,
                                                            self.hamiltonian,
                                                            self.state.root_id)
        self.cache_dict[root_id] = current_cache
        return current_cache

    def prepare_root_for_child_update(self):
        """
        Does all the necessary preparation for the root before updating its
        children.
        """
        root_id = self.state.root_id
        self.state.assert_orth_center(root_id)
        self.prepare_root_cache()

    def get_children_ids_to_update(self, node_id: str) -> FrozenSet[str]:
        """
        Returns the children ids of the given node to update.

        Args:
            node_id (str): The id of the node to get the children of.

        Returns:
            Tuple[str]: The identifiers of the children to be updated.

        """
        node = self.state.nodes[node_id]
        return frozenset(node.children)

    def update_children(self, node_id: str):
        """
        Update the children of this node.
        """
        for child_id in self.get_children_ids_to_update(node_id):
            self.subtree_update(child_id)

    def reset_main_cache(self) -> SandwichCache:
        """
        Resets and empties the main cache and uses the main state.
        """
        self.tensor_cache = SandwichCache(self.state,
                                          self.hamiltonian)
        return self.tensor_cache

    def reset_basis_change_cache(self) -> PartialTreeCachDict:
        """
        Resets the basis change cache.
        """
        self.basis_change_cache = PartialTreeCachDict()
        return self.basis_change_cache

    def tree_update(self):
        """
        Starts the recursive update of the tree.
        
        """
        self.prepare_root_for_child_update()
        self.update_children(self.state.root_id)
        # Now the children in the state are the basis change tensors
        # and the children sandwich tensors and the children basis change
        # tensors are in their resepctive caches.
        root_id = self.state.root_id
        _, current_cache = self.prepare_for_node_update(root_id)
        current = self.get_current_node_and_tensor(root_id)
        ham_data = self.hamiltonian[root_id]
        updated_tensor = self.time_evolve_node(current,
                                               ham_data,
                                               current_cache)
        self.state.replace_tensor(root_id,updated_tensor)
        self.clean_up(root_id)
        self.reset_main_cache()
        self.reset_basis_change_cache()

    def prepare_node_state(self,
                           node_id: str
                           ) -> TreeTensorNetworkState:
        """
        Prepares the state for the given node.

        This is the parent state with the current node being the orthogonality
        center. The state is added to the TTNS dict.

        Args:
            node_id (str): The id of the node to prepare the state for.

        Returns:
            TreeTensorNetworkState: The prepared state.

        """
        parent_id = self.state.nodes[node_id].parent
        current_ttns = deepcopy(self.ttns_dict[parent_id])
        current_ttns.assert_orth_center(parent_id)
        current_ttns.move_orthogonalization_center(node_id)
        self.ttns_dict[node_id] = current_ttns
        return current_ttns

    def prepare_node_cache(self,
                            node_id: str
                            ) -> SandwichCache:
        """
        Prepares the cache for the given node.

        The cache contains all Sandwich nodes pointing towards the current node.
        The cache is added to the cache dict.

        Args:
            node_id (str): The id of the node to prepare the cache for.

        Returns:
            SandwichCache: The prepared cache.

        """
        current_state = self.prepare_node_state(node_id)
        parent_id = self.state.nodes[node_id].parent
        current_cache = copy(self.cache_dict[parent_id])
        current_cache.state = current_state
        current_cache.update_tree_cache(parent_id,node_id)
        current_cache.delete_entry(node_id, parent_id)
        self.cache_dict[node_id] = current_cache
        return current_cache

    def create_ttns_and_cache(self,
                               node_id: str
                               ) -> Tuple[TreeTensorNetworkState,SandwichCache]:
        """
        Creates the TTNS and cache for the given node.

        Uses the parent nodes TTNS and cache for the preparation.

        After running this method, a TTNS and cache corresponding to this node
        being the orthogonality center are stored in the respective
        dictionaries.
        """
        current_cache = self.prepare_node_cache(node_id)
        current_ttns = self.ttns_dict[node_id]
        return current_ttns, current_cache

    def _get_old_tensor_for_new_basis_tensor(self,
                                             node_id: str
                                                ) -> Tuple[GraphNode,ndarray]:
        """
        Finds the old tensor and node required to compute the new basis tensor.

        Args:
            node_id (str): The id of the node to find the old tensor for.
        
        Returns:
            GraphNode: The node in the state corresponding to the old tensor.
            ndarray: The old tensor.
        
        """
        new_node = self.state.nodes[node_id]
        if new_node.is_leaf():
            # For a leaf the old tensor is the tensor with which the node
            # is in canonical form towards its parent.
            parent_id = new_node.parent
            ttns = self.ttns_dict[parent_id]
            old_tensor = get_tensor_from_different_ttn(ttns,
                                                        self.state,
                                                        node_id)
        else:
            # Otherwise we just use the tensor with updated basis legs
            # This is C_hat in the reference
            old_tensor = self.state.tensors[node_id]
        return new_node, old_tensor

    def compute_new_basis_tensor(self,
                                 node_id: str,
                                 updated_tensor: ndarray
                                 ) -> ndarray:
        """
        Computes the new basis tensor for the given node.

        Args:
            node_id (str): The id of the node to compute the new basis tensor
                for.
            updated_tensor (ndarray): The time evolved tensor. The legs need to
                fit with the new node leg order.

        Returns:
            ndarray: The new basis tensor.
        """
        new_node, old_tensor = self._get_old_tensor_for_new_basis_tensor(node_id)
        return compute_new_basis_tensor(new_node,
                                        old_tensor,
                                        updated_tensor)

    def _get_node_and_tensor_for_basis_change_comp(self,
                                                   node_id: str
                                                   ) -> Tuple[Node,ndarray]:
        """
        Get the node and tensor of the current node required to compute the
        basis change tensor.

        Args:
            node_id: The identifier of the ucrrent node to be updated.

        Return:
            Node: The node in the new state corresponding to the identifier.
            ndarray: The isometric tensor that was associated to the node_id
                while one of the ancestor nodes was the orthogonality center.

        """
        node_new = self.state.nodes[node_id]
        # These have to be taken from the state, where the node is in caonical
        # form towards its parent
        parent_id = node_new.parent
        parent_ttns = self.ttns_dict[parent_id]
        old_tensor = get_tensor_from_different_ttn(parent_ttns,
                                                    self.state,
                                                    node_id)
        return node_new, old_tensor

    def compute_basis_change_tensor(self,
                                    node_id: str,
                                    new_basis_tensor: ndarray
                                    ) -> ndarray:
        """
        Compute the basis change tensor for the given node.

        This corrresponds to the M_hat in the reference.

        Args:
            node_id (str): The id of the node to compute the basis change
                tensor for.
            new_basis_tensor (ndarray): The new local basis tensor.
        
        Returns:
            ndarray: The basis change tensor.

        """
        node_new = self.state.nodes[node_id]
        # These have to be taken from the state, where the node is in caonical
        # form towards its parent
        parent_id = node_new.parent
        parent_ttns = self.ttns_dict[parent_id]
        old_tensor = get_tensor_from_different_ttn(parent_ttns,
                                                    self.state,
                                                    node_id)
        return compute_basis_change_tensor(node_new,
                                            node_new,
                                            old_tensor,
                                            new_basis_tensor,
                                            self.basis_change_cache)

    def find_new_tensors(self, node_id: str) -> Tuple[ndarray,ndarray]:
        """
        Finds the new basis tensor and the basis change tensor for the node.

        This will also change the state and cache to be ready to update this node.

        Args:
            node_id (str): The id of the node to find the new tensors for.

        Returns:
            Tuple[ndarray,ndarray]: The new basis tensor and the basis change
                tensor.

        """
        _, current_cache = self.prepare_for_node_update(node_id)
        current = self.get_current_node_and_tensor(node_id)
        ham_data = self.hamiltonian[node_id]
        updated_tensor = self.time_evolve_node(current,
                                               ham_data,
                                               current_cache)
        new_basis_tensor = self.compute_new_basis_tensor(node_id,
                                                         updated_tensor)
        basis_change_tensor = self.compute_basis_change_tensor(node_id,
                                                               new_basis_tensor)
        return new_basis_tensor, basis_change_tensor

    def update_tensor_cache(self, node_id: str) -> SandwichCache:
        """
        After successfully updating the node, the tensor cache must be updated.

        Args:
            node_id (str): The id of the node to update the cache for.

        Returns:
            SandwichCache: The updated total cache.

        """
        parent_id = self.ttns_dict[node_id].nodes[node_id].parent
        basis_change_id = basis_change_tensor_id(node_id)
        self.tensor_cache.update_tree_cache(node_id,
                                            basis_change_id)
        # Currently the basis change tensors are the parent nodes.
        # They will be contracted in the next step, so we can already
        # set the next id to the parent id preemtively.
        self.tensor_cache.change_next_id_for_entry(node_id,
                                                   basis_change_id,
                                                   parent_id)
        return self.tensor_cache

    def update_basis_change_cache(self,
                                  node_id: str,
                                  basis_change_tensor: ndarray
                                  ) -> PartialTreeCachDict:
        """
        After successfully updating the node, the basis change cache must be updated.

        Args:
            node_id (str): The id of the node to update the cache for.
            basis_change_tensor (ndarray): The basis change tensor.

        Returns:
            PartialTreeCachDict: The updated basis change cache.

        """
        # In the main state the parent is currently the basis change node
        parent_id = self.ttns_dict[node_id].nodes[node_id].parent
        self.basis_change_cache.add_entry(node_id, parent_id,
                                          basis_change_tensor)

    def clean_up(self, node_id: str):
        """
        After each node update, we can delete the old TTNS and cache.

        Args:
            node_id (str): The id of the that was updated.

        """
        del self.ttns_dict[node_id]
        del self.cache_dict[node_id]

    def subtree_update(self, node_id: str):
        """
        Updates the subtree rooted at the given node.

        Args:
            node_id (str): The id of the node to update.

        """
        self.create_ttns_and_cache(node_id)
        self.update_children(node_id)
        # Now the children in the state are the basis change tensors
        # and the children sandwich tensors and the children basis change
        # tensors are in their resepective caches.
        new_basis_tensor, basis_change_tensor = self.find_new_tensors(node_id)
        # Adds the new basis tensor and the basis change tensor to the new
        # state instead of the old node
        self.replace_node_with_updated_basis(node_id,
                                             new_basis_tensor,
                                             basis_change_tensor)
        # Update the caches
        self.update_tensor_cache(node_id)
        self.update_basis_change_cache(node_id,
                                       basis_change_tensor)
        # Remove the old TTNS and cache
        self.clean_up(node_id)

    def truncation(self):
        """
        Truncates the tree after the time evolution.

        """
        recursive_truncation(self.state,
                             self.svd_parameters)

    def run_one_time_step(self, **kwargs):
        """
        Runs one time step of the BUG method.

        The method is based on the rank-adaptive BUG method introduced in
        https://www.doi.org/10.1137/22M1473790 .

        Args:
            kwargs: Additional keyword arguments for the time step.

        """
        self.tree_update()
        self.truncation()
