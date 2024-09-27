
from typing import Dict, List, Union
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

        """
        h_eff = get_effective_single_site_hamiltonian(node_id,
                                                      self.state,
                                                      self.hamiltonian,
                                                      self.tensor_cache)
        # The effective Hamiltonian is created, such that the legs fit to the
        # node tensor's legs.
        # We take the tensor from the new state, where the basis change tensors
        # of the children are already contracted
        current_tensor = self.state.tensors[node_id]
        updated_tensor = time_evolve(current_tensor,
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

    def update_root(self):
        """
        Updates the root node, by time evolving the tensor.
        """
        root_id = self.state.root_id
        updated_tensor = self.time_evolve_node(root_id)
        self.state.replace_tensor(root_id, updated_tensor)

    def tree_update(self):
        """
        Starts the recursive update of the tree.
        
        """
        root_id = self.state.root_id
        self.state.assert_orth_center(root_id)
        root_ttns = deepcopy(self.state)
        self.ttns_dict[root_id] = root_ttns
        current_cache = SandwichCache.init_cache_but_one(root_ttns,
                                                         self.hamiltonian,
                                                         root_id)
        self.cache_dict[root_id] = current_cache
        for child_id in self.state.nodes[root_id].children:
            self.subtree_update(child_id)
        # Now the children in the state are the basis change tensors
        pull_tensor_from_different_ttn(root_ttns, self.state,
                                       root_id,
                                       mod_fct=reverse_basis_change_tensor_id)
        self.state.contract_all_children(root_id)
        for child_id in self.state.nodes[root_id].children:
            cache_tensor = self.tensor_cache.get_entry(child_id, root_id)
            current_cache.add_entry(child_id, root_id,
                                    cache_tensor)
        h_eff = get_effective_single_site_hamiltonian(root_id,
                                                      self.state,
                                                      self.hamiltonian,
                                                      current_cache)
        updated_tensor = time_evolve(self.state.tensors[root_id],
                                     h_eff,
                                     self.time_step_size,
                                     forward=True)
        self.state.replace_tensor(root_id,updated_tensor)
        del self.ttns_dict[root_id]
        del self.cache_dict[root_id]
        self.basis_change_cache = PartialTreeCachDict()

    def subtree_update(self, node_id: str):
        """
        Updates the subtree rooted at the given node.

        Args:
            node_id (str): The id of the node to update.

        """
        node = self.state.nodes[node_id]
        parent_id = node.parent
        current_ttns = deepcopy(self.ttns_dict[parent_id])
        current_ttns.assert_orth_center(parent_id)
        current_ttns.move_orthogonalization_center(node_id)
        current_cache = copy(self.cache_dict[parent_id])
        current_cache.state = current_ttns
        current_cache.update_tree_cache(parent_id,node_id)
        self.ttns_dict[node_id] = current_ttns
        self.cache_dict[node_id] = current_cache
        for child_id in node.children:
            self.subtree_update(child_id)
        # Now the children in the state are the basis change tensors
        # and the children sandwich tensors and the children basis change
        # tensors are in their resepctive caches.
        for child_id in node.children:
            tensor = self.tensor_cache.get_entry(child_id, node_id)
            current_cache.add_entry(child_id, node_id,
                                    tensor)
        pull_tensor_from_different_ttn(current_ttns, self.state,
                                       node_id,
                                       mod_fct=reverse_basis_change_tensor_id)
        current_ttns.contract_all_children(node_id)
        h_eff = get_effective_single_site_hamiltonian(node_id,
                                                      self.state,
                                                      self.hamiltonian,
                                                      current_cache)
        updated_tensor = time_evolve(self.state.tensors[node_id],
                                     h_eff,
                                     self.time_step_size,
                                     forward=True)
        new_basis_tensor = compute_new_basis_tensor(self.state.nodes[node_id],
                                                    self.state.tensors[node_id],
                                                    updated_tensor)
        # All the children's basis change tensors are already contracted
        basis_change_tensor = compute_basis_change_tensor(self.ttns_dict[parent_id].nodes[node_id],
                                                          self.state.nodes[node_id],
                                                          self.ttns_dict[parent_id].tensors[node_id],
                                                          new_basis_tensor,
                                                          self.basis_change_cache)
        self.replace_node_with_updated_basis(node_id,
                                             new_basis_tensor,
                                             basis_change_tensor)
        self.basis_change_cache.add_entry(node_id, parent_id,
                                          basis_change_tensor)
        self.tensor_cache.update_tree_cache(node_id,
                                            basis_change_tensor_id(node_id))
        self.tensor_cache.change_next_id_for_entry(node_id,
                                                   basis_change_tensor_id(node_id),
                                                   parent_id)

    def leaf_subtree_update(self, node_id: str):
        """
        Updates the subtree rooted at the given node.

        Args:
            node_id (str): The id of the node to update.

        """
        node = self.state.nodes[node_id]
        parent_id = node.parent
        current_ttns = deepcopy(self.ttns_dict[parent_id])
        current_ttns.assert_orth_center(parent_id)
        current_ttns.move_orthogonalization_center(node_id)
        current_cache = copy(self.cache_dict[parent_id])
        current_cache.state = current_ttns
        current_cache.update_tree_cache(parent_id,node_id)
        self.ttns_dict[node_id] = current_ttns
        self.cache_dict[node_id] = current_cache
        # Now the children in the state are the basis change tensors
        # and the children sandwich tensors and the children basis change
        # tensors are in their resepctive caches.
        pull_tensor_from_different_ttn(current_ttns, self.state,
                                       node_id,
                                       mod_fct=reverse_basis_change_tensor_id)
        h_eff = get_effective_single_site_hamiltonian(node_id,
                                                      self.state,
                                                      self.hamiltonian,
                                                      current_cache)
        updated_tensor = time_evolve(self.state.tensors[node_id],
                                     h_eff,
                                     self.time_step_size,
                                     forward=True)
        new_basis_tensor = compute_new_basis_tensor(self.state.nodes[node_id],
                                                    get_tensor_from_different_ttn(self.ttns_dict[parent_id],
                                                                                  self.state,
                                                                                  node_id),
                                                    updated_tensor)
        # All the children's basis change tensors are already contracted
        basis_change_tensor = compute_basis_change_tensor(self.ttns_dict[parent_id].nodes[node_id],
                                                          self.state.nodes[node_id],
                                                          self.ttns_dict[parent_id].tensors[node_id],
                                                          new_basis_tensor,
                                                          self.basis_change_cache)
        self.replace_node_with_updated_basis(node_id,
                                             new_basis_tensor,
                                             basis_change_tensor)
        self.basis_change_cache.add_entry(node_id, parent_id,
                                          basis_change_tensor)
        self.tensor_cache.update_tree_cache(node_id,
                                            basis_change_tensor_id(node_id))
        self.tensor_cache.change_next_id_for_entry(node_id,
                                                   basis_change_tensor_id(node_id),
                                                   parent_id)

    def run_one_time_step(self, **kwargs):
        """
        Runs one time step of the BUG method.

        The method is based on the rank-adaptive BUG method introduced in
        https://www.doi.org/10.1137/22M1473790 .

        Args:
            kwargs: Additional keyword arguments for the time step.

        """
        raise NotImplementedError("Not yet implemented!")
