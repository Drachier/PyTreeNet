
from typing import Dict, List, Union, Tuple
from copy import deepcopy, copy

from numpy import ndarray, tensordot, vstack, concatenate
from scipy.linalg import rq

from .ttn_time_evolution import TTNTimeEvolution, TTNTimeEvolutionConfig
from ..operators.tensorproduct import TensorProduct
from ..core.graph_node import find_children_permutation
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..core.leg_specification import LegSpecification
from ..core.node import Node
from ..contractions.sandwich_caching import SandwichCache, update_tree_cache
from ..contractions.effective_hamiltonians import get_effective_single_site_hamiltonian
from ..contractions.state_operator_contraction import (env_tensor_bra_leg_index,
                                                       env_tensor_ham_leg_index,
                                                       env_tensor_ket_leg_index)
from ..util.tensor_splitting import tensor_qr_decomposition
from .time_evolution import time_evolve

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
        self._ensure_root_orth_center()
        self.old_state = deepcopy(self.state)
        # Deal with cache of tensor contractions
        self.tensor_cache = self._init_tensor_cache()

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

    def _init_tensor_cache(self) -> SandwichCache:
        """
        Initializes the sandwich cache, where only the root is not contracted.

        """
        assert self.old_state.root_id is not None, "Root id is None!"
        return SandwichCache.init_cache_but_one(self.old_state, # Importantly the old state
                                                self.hamiltonian,
                                                self.state.root_id)

    def build_effective_leaf_hamiltonian(self, node_id: str) -> ndarray:
        """
        Builds the effective Hamiltonian for the leaf node with the given id.

                _________        
            |2    2|     |
          __|__    |     |
         |     |___|     |
         |     |0 1|     |
         |_____|   |     |
            |      |     |
            |1     |     |
                ___|_____|
                  0

        Args:
            node_id (str): The id of the leaf node.

        Returns:
            ndarray: The effective Hamiltonian as a matrix.
            Leg order = ((ket_env,out_node),(bra_env,in_node))

        """
        node = self.state.nodes[node_id]
        ham_tensor = self.hamiltonian.tensors[node_id]
        assert node.is_leaf(), "Node is not a leaf!"
        if node.is_root():
            return ham_tensor
        parent_id = node.parent
        environment_tensor = self.tensor_cache.get_entry(parent_id,
                                                         node_id)
        axes = [env_tensor_ham_leg_index(),node.parent_leg]
        h_eff = tensordot(environment_tensor,
                            ham_tensor,
                            axes=axes)
        # make the input and output legs to neighbours and
        # fit the usual matrix convention
        h_eff = h_eff.transpose([1,2,0,3])
        shape = h_eff.shape
        h_eff = h_eff.reshape((shape[0]*shape[1],
                               shape[2]*shape[3]))
        return h_eff

    def update_leaf(self, node_id: str):
        """
        Updates the leaf node with the given id.

        Args:
            node_id (str): The id of the leaf node to update.

        """
        assert node_id == self.old_state.orthogonality_center_id, "Node is not the orthogonality center!"
        h_eff = self.build_effective_leaf_hamiltonian(node_id)
        # We always use the data from the old state
        site_tensor = self.old_state.nodes[node_id].tensor # Leg order (parent,physical)
        # The tensor already fits with the hamiltonian leg order
        updated_tensor = time_evolve(site_tensor, h_eff,
                                     self.time_step_size) # Leg order (parent,physical)
        # We must ensure that in the old state, we have the original isometric tensor
        parent_id = self.old_state.nodes[node_id].parent
        if not parent_id == self.old_state.orthogonality_center_id:
            self.old_state.move_orthogonalization_center(parent_id)
        old_basis_tensor = self.old_state.tensors[node_id] # Leg order (parent,physical)
        combined_tensor = vstack((old_basis_tensor, updated_tensor)) # Shape (2*parent_dim,physical_dim)
        _, new_basis_tensor = rq(combined_tensor) # Leg order (new,physical)
        basis_change_tensor = old_basis_tensor @ new_basis_tensor.conj().T # Leg order (parent,new)
        legs_basis_change = LegSpecification(parent_id,[],[])
        legs_new_basis = LegSpecification(None,[],[1])
        # We use the new state to store the new tensors in
        self.state.split_node_replace(node_id,
                                      basis_change_tensor,
                                      new_basis_tensor,
                                      basis_change_tensor_id(node_id),
                                      node_id,
                                      legs_basis_change,
                                      legs_new_basis)
        # We update the cache later, to keep the old cache for sibling nodes

    def pull_tensor_from_old_state(self, node_id: str):
        """
        Pulls a tensor from the old state to the new state.

        Args:
            node_id (str): The id of the node to pull.

        """
        old_node = self.old_state.nodes[node_id]
        new_node = self.state.nodes[node_id]
        # Find a potential permutation of the children
        # The children in the new state are the basis change tensor
        perm = find_children_permutation(old_node, new_node,
                                         modify_function=reverse_basis_change_tensor_id)
        # Permute the odl tensor
        old_tensor = self.old_state.tensors[node_id]
        old_tensor = old_tensor.transpose(perm)
        # Add the tensor to the new state
        self.state.tensors[node_id] = old_tensor

    def evolve_non_leaf(self, node_id: str):
        """
        Performs the update for a non-leaf node.

        """
        h_eff = get_effective_single_site_hamiltonian(node_id,
                                                      self.state,
                                                      self.hamiltonian,
                                                      self.tensor_cache)
        # The effective Hamiltonian is created, such that the legs fit to the
        # node tensor's legs.
        old_tensor = self.state.tensors[node_id]
        updated_tensor = time_evolve(old_tensor,
                                        h_eff,
                                        self.time_step_size,
                                        forward=True)
        new_basis_tensor = compute_new_basis_tensor(self.state.nodes[node_id],
                                                    old_tensor,
                                                    updated_tensor)
        # Now we need the old orthogonal tensor
        parent_id = self.state.nodes[node_id].parent
        if not parent_id == self.old_state.orthogonality_center_id:
            self.old_state.move_orthogonalization_center(parent_id)
        old_basis_tensor = self.old_state.tensors[node_id]
        # The leg order should be the same so we contract
        # to get the basis change tensor
        legs = list(range(1,old_basis_tensor.ndim))
        basis_change_tensor = tensordot(old_basis_tensor,
                                        new_basis_tensor.conj(),
                                        axes=(legs,legs))
        # Now we replace the node in the new state
        node = self.state.nodes[node_id]
        legs_basis_change = LegSpecification(parent_id,[],[])
        leg_new_basis = LegSpecification(None,[node.children],node.open_legs)
        self.state.split_node_replace(node_id,
                                      basis_change_tensor,
                                      new_basis_tensor,
                                      basis_change_tensor_id(node_id),
                                      node_id,
                                      legs_basis_change,
                                      leg_new_basis)
        # We update the cache later, to keep the old cache for sibling nodes

    def subtree_update(self, node_id: str):
        """
        Updates the subtree rooted at the given node.

        Args:
            node_id (str): The id of the node to update.

        """
        node = self.state.nodes[node_id] # TODO: Check if these have the same python id
        assert self.old_state.orthogonality_center_id == node.parent, "Parent is not the orthogonality center!"
        self.old_state.move_orthogonalization_center(node_id)
        self.tensor_cache.update_tree_cache(node_id, node.parent)
        node = self.state.nodes[node_id] # TODO: Check if these have the same python id
        self.tensor_cache.update_tree_cache(node.parent, node_id)
        if node.is_leaf():
            self.update_leaf(node_id)
            return
        # Not a leaf
        for child_id in node.children:
            self.subtree_update(child_id)
        # Now the new state contains the updated children tensors and basis changes
        assert node_id == self.old_state.orthogonality_center_id, "Node is not the orthogonality center!"
        # We need the tensor of the old state as the orth center
        self.pull_tensor_from_old_state(node_id)
        # Now we contract the basis change tensors of the children
        for child_id in copy(node.children):
            self.state.contract_nodes(child_id,node_id,
                                      new_identifier=node_id)
        # Now the children have the original children identifiers
        node = self.state.nodes[node_id] # TODO: Check if these have the same python id
        for child_id in node.children:
            # We need to update the cache with respect to the new state
            update_tree_cache(self.tensor_cache,
                              self.state,
                              self.hamiltonian,
                              child_id,
                              node_id)
        # Now we can update the node
        self.evolve_non_leaf(node_id)


    def run_one_time_step(self, **kwargs):
        """
        Runs one time step of the BUG method.

        The method is based on the rank-adaptive BUG method introduced in
        https://www.doi.org/10.1137/22M1473790 .

        Args:
            kwargs: Additional keyword arguments for the time step.

        """
        pass

def basis_change_tensor_id(node_id: str) -> str:
    """
    The identifier for a basis change tensor.

    """
    return node_id + "basis_change_tensor"

def reverse_basis_change_tensor_id(node_id: str) -> str:
    """
    Returns the original node identifier from a basis change tensor identifier.

    """
    return node_id[-len("basis_change_tensor"):]

def new_basis_tensor_qr_legs(node: Node) -> Tuple[List[int],List[int]]:
    """
    Returns the leg indices for the QR decomposition to obtain the new basis
    tensor.

    Args:
        node (GraphNode): The node for which to perform the QR decomposition.

    Returns:
        Tuple([List[int],List[int]]): (q_legs, r_legs) The legs for the QR
            decomposition.

    """
    assert not node.is_root(), "Root node has no parent leg!"
    q_legs = node.children_legs + node.open_legs
    r_legs = [node.parent_leg]
    return q_legs, r_legs

def find_new_basis_tensor_leg_permutation(node: Node) -> List[int]:
    """
    Finds the permutation of the legs of the new basis tensor.

    We want the parent leg to be the first leg. However, the QR decomposition
    will have the parent leg as the last leg.

    Args:
        node (GraphNode): The node for which to find the permutation.
    
    Returns:
        List[int]: The permutation to apply to the new basis tensor.
    """
    nlegs = node.nlegs()
    return [nlegs-1] + list(range(nlegs-1))

def compute_new_basis_tensor(node: Node,
                             old_tensor: ndarray,
                             updated_tensor: ndarray) -> ndarray:
    """
    Compute the updated basis tensor.

    The updated basis tensor is found by concatinating the old and updated
    tensor and finding a basis of their combined range.
    
    Args:
        node (GraphNode): The node which is updated.
        old_tensor (ndarray): The old basis tensor. The tensor/node has to be
            the orthogonality center of the old state. The leg order is the
            usual.
        updated_tensor (ndarray): The updated basis tensor obtained by time
            evolving the old tensor. The leg order is the usual.

    Returns:
        ndarray: The new basis tensor. The leg order is the usual and equal to
            the leg order of the old and updated tensor, but with a new parent
            dimension.
    
    """
    # We know that both tensors have the same leg order
    # We stack them along the parent axis
    combined_tensor = concatenate((old_tensor, updated_tensor),
                                    axis=node.parent_leg)
    # We perform a QR decomposition
    q_legs, r_legs = new_basis_tensor_qr_legs(node)
    new_basis_tensor, _ = tensor_qr_decomposition(combined_tensor,
                                                q_legs,
                                                r_legs)
    # We will have to transpose this anyway later on
    # Currently the parent leg is the last leg
    transpose_legs = find_new_basis_tensor_leg_permutation(node)
    new_basis_tensor = new_basis_tensor.transpose(transpose_legs)
    return new_basis_tensor
