
from typing import Dict, List, Union, Tuple
from copy import deepcopy, copy

from numpy import ndarray

from .ttn_time_evolution import TTNTimeEvolution, TTNTimeEvolutionConfig
from ..operators.tensorproduct import TensorProduct
from ..core.ttn import (pull_tensor_from_different_ttn,
                        get_tensor_from_different_ttn)
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..contractions.effective_hamiltonians import get_effective_single_site_hamiltonian
from ..contractions.sandwich_caching import SandwichCache
from ..contractions.tree_cach_dict import PartialTreeCachDict
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
                 config: Union[TTNTimeEvolutionConfig, None] = None) -> None:
        super().__init__(initial_state,
                         time_step_size, final_time,
                         operators, config)
        self.hamiltonian = hamiltonian
        self.ttns_dict: Dict[str,TreeTensorNetworkState] = {}
        self.cache_dict: Dict[str,SandwichCache] = {}
        self.basis_change_cache = PartialTreeCachDict()
        self.tensor_cache = SandwichCache(self.state, self.hamiltonian)

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

    def time_evolve_node(self, node_id: str) -> ndarray:
        """
        Time evolves the node with the given id.

        While the new state is used for the neighbour ordering, the cache used
        corresponds to the specific nodes. It contains the old sandwich tensors
        of any parent tensor and the updated sandwich tensors of the children.

        Args:
            node_id (str): The id of the node to time evolve.
        
        Returns:
            ndarray: The updated tensor
        """
        current_cache = self.cache_dict[node_id]
        h_eff = get_effective_single_site_hamiltonian(node_id,
                                                      self.state,
                                                      self.hamiltonian,
                                                      current_cache)
        updated_tensor = time_evolve(self.state.tensors[node_id],
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

    def prepare_state_for_update(self, node_id: str):
        """
        Prepares the new state for the update.

        This means that the connectivity tensor with updated basis legs
        is created in the state. (Corresponds to the creation of C_hat in the
        reference and just takes the old tensor into the new state for leafs.)

        Args:
            node_id (str): The id of the node to prepare.

        """
        current_ttns = self.ttns_dict[node_id]
        pull_tensor_from_different_ttn(current_ttns, self.state,
                                       node_id,
                                       mod_fct=reverse_basis_change_tensor_id)
        self.state.contract_all_children(node_id)

    def prepare_current_cache_for_update(self,
                                         node_id: str
                                         ) -> SandwichCache:
        """
        Prepares the current cache for the update.

        To find the correct effective Hamiltonian, the cache has to include the
        updated sandwich tensors of the children.

        Args:
            node_id (str): The id of the node to prepare.

        Returns:
            SandwichCache: The updated cache.
        """
        node = self.state.nodes[node_id]
        current_cache = self.cache_dict[node_id]
        for child_id in copy(node.children):
            tensor = self.tensor_cache.get_entry(child_id, node_id)
            current_cache.add_entry(child_id, node_id,
                                    tensor)
        return current_cache

    def tree_update(self):
        """
        Starts the recursive update of the tree.
        
        """
        root_id = self.state.root_id
        self.state.assert_orth_center(root_id)
        root_ttns = deepcopy(self.state)
        self.ttns_dict[root_id] = root_ttns
        # TODO: Maybe we don't need to build a new one at every time step
        # Maybe we can reuse the cache from the last time step
        current_cache = SandwichCache.init_cache_but_one(root_ttns,
                                                         self.hamiltonian,
                                                         root_id)
        self.cache_dict[root_id] = current_cache
        for child_id in root_ttns.nodes[root_id].children:
            self.subtree_update(child_id)
        # Now the children in the state are the basis change tensors
        # and the children sandwich tensors and the children basis change
        # tensors are in their resepctive caches.
        self.prepare_state_for_update(root_id)
        self.prepare_current_cache_for_update(root_id)
        updated_tensor = self.time_evolve_node(root_id)
        self.state.replace_tensor(root_id,updated_tensor)
        del self.ttns_dict[root_id]
        del self.cache_dict[root_id]
        self.basis_change_cache = PartialTreeCachDict()

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
        parent_id = self.state.nodes[node_id].parent
        current_ttns = deepcopy(self.ttns_dict[parent_id])
        current_ttns.assert_orth_center(parent_id)
        current_ttns.move_orthogonalization_center(node_id)
        # TODO: We might not need to copy the entire cache
        current_cache = copy(self.cache_dict[parent_id])
        current_cache.state = current_ttns
        current_cache.update_tree_cache(parent_id,node_id)
        self.ttns_dict[node_id] = current_ttns
        self.cache_dict[node_id] = current_cache
        return current_ttns, current_cache

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
        return compute_new_basis_tensor(new_node,
                                        old_tensor,
                                        updated_tensor)

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
        parent_id = node_new.parent
        # These have to be taken from the state, where the node is in caonical
        # form towards its parent
        parent_ttns = self.ttns_dict[parent_id]
        old_node, old_tensor = parent_ttns[node_id]
        return compute_basis_change_tensor(old_node,
                                            node_new,
                                            old_tensor,
                                            new_basis_tensor,
                                            self.basis_change_cache)

    def update_tensor_cache(self, node_id: str):
        """
        After successfully updating the node, the tensor cache must be updated.

        Args:
            node_id (str): The id of the node to update the cache for.

        """
        parent_id = self.state.nodes[node_id].parent
        basis_change_id = basis_change_tensor_id(node_id)
        self.tensor_cache.update_tree_cache(node_id,
                                            basis_change_id)
        # Currently the basis change tensors are the parent nodes.
        # They will be contracted in the next step, so we can already
        # set the next id to the parent id preemtively.
        self.tensor_cache.change_next_id_for_entry(node_id,
                                                   basis_change_id,
                                                   parent_id)

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
        node = self.state.nodes[node_id]
        for child_id in copy(node.children):
            self.subtree_update(child_id)
        # Now the children in the state are the basis change tensors
        # and the children sandwich tensors and the children basis change
        # tensors are in their resepctive caches.
        self.prepare_state_for_update(node_id)
        self.prepare_current_cache_for_update(node_id)
        updated_tensor = self.time_evolve_node(node_id)
        new_basis_tensor = self.compute_new_basis_tensor(node_id,
                                                         updated_tensor)
        # All the children's basis change tensors are already contracted
        basis_change_tensor = self.compute_basis_change_tensor(node_id,
                                                               new_basis_tensor)
        parent_id = node.parent
        self.basis_change_cache.add_entry(node_id, parent_id,
                                          basis_change_tensor)
        # Adds the new basis tensor and the basis change tensor to the new
        # state instead of the old node
        self.replace_node_with_updated_basis(node_id,
                                             new_basis_tensor,
                                             basis_change_tensor)
        # Update the cache with the new sandwich tensors
        self.update_tensor_cache(node_id)
        # Remove the old TTNS and cache
        self.clean_up(node_id)

    def run_one_time_step(self, **kwargs):
        """
        Runs one time step of the BUG method.

        The method is based on the rank-adaptive BUG method introduced in
        https://www.doi.org/10.1137/22M1473790 .

        Args:
            kwargs: Additional keyword arguments for the time step.

        """
        raise NotImplementedError("Not yet implemented!")
