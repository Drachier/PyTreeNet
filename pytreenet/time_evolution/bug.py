
from typing import Dict, List, Union, Tuple
from copy import deepcopy, copy

from numpy import ndarray, tensordot, concatenate

from .ttn_time_evolution import TTNTimeEvolution, TTNTimeEvolutionConfig
from ..operators.tensorproduct import TensorProduct
from ..core.graph_node import find_child_permutation_neighbour_index
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..core.leg_specification import LegSpecification
from ..core.node import Node
from ..contractions.sandwich_caching import SandwichCache, update_tree_cache
from ..contractions.effective_hamiltonians import get_effective_single_site_hamiltonian
from ..util.tensor_util import make_last_leg_first
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

    def _assert_orth_center(self, *args, **kwargs):
        """
        Asserts that the node is the orthogonality center.

        Args:
            node_id (str): The id of the node to check.
            object_name (str): The name of the object to check.

        Raises:
            AssertionError: If the node is not the orthogonality center.

        """
        # We basically wrap the assertion of the old state.
        self.old_state.assert_orth_center(*args, **kwargs)

    def pull_tensor_from_old_state(self, node_id: str):
        """
        Pulls a tensor from the old state to the new state.

        Args:
            node_id (str): The id of the node to pull.

        """
        old_node = self.old_state.nodes[node_id]
        new_node = self.state.nodes[node_id]
        # Find a potential permutation of the children
        # The children in the new state are the basis change tensord
        child_perm = find_child_permutation_neighbour_index(old_node, new_node,
                                         modify_function=reverse_basis_change_tensor_id)
        if new_node.is_root():
            # The root has no parent leg
            perm = []
        else:
            perm = [new_node.parent_leg]
        perm = perm + child_perm + new_node.open_legs
        old_tensor = self.old_state.tensors[node_id]
        self.state.replace_tensor(node_id, deepcopy(old_tensor), perm)

    def time_evolve_node(self, node_id: str) -> ndarray:
        """
        Time evolves the node with the given id.

        """
        self._assert_orth_center(node_id)
        h_eff = get_effective_single_site_hamiltonian(node_id,
                                                      self.state,
                                                      self.hamiltonian,
                                                      self.tensor_cache)
        # The effective Hamiltonian is created, such that the legs fit to the
        # node tensor's legs.
        # We take the tensor from the new state, where the basis change tensors
        # of the children are already contracted
        old_tensor = self.state.tensors[node_id]
        updated_tensor = time_evolve(old_tensor,
                                        h_eff,
                                        self.time_step_size,
                                        forward=True)
        return updated_tensor

    def replace_node_with_updated_basis(self, node_id: str,
                                        new_basis_tensor: ndarray,
                                        basis_change_tensor: ndarray
                                        ):
        """
        Replaces a node with the tenors of the updated basis.

        Args:
            node_id (str): The id of the node to replace.
            new_basis_tensor (ndarray): The new basis tensor.
            basis_change_tensor (ndarray): The basis change tensor.
        
        """
        node = self.state.nodes[node_id]
        leg_specs = _find_new_basis_replacement_leg_specs(node)
        self.state.split_node_replace(node_id,
                                      basis_change_tensor,
                                      new_basis_tensor,
                                      basis_change_tensor_id(node_id),
                                      node_id,
                                      leg_specs[0],
                                      leg_specs[1])

    def update_non_root(self, node_id: str):
        """
        Performs the update for a non-root node.

        """
        # We combine the updated tensor and the tensor with the basis change
        # tensors of the children contracted to find the new basis tensor for
        # this node
        updated_tensor = self.time_evolve_node(node_id)
        node = self.state.nodes[node_id]
        if node.is_leaf():
            # For a leaf the orthogonalised old node is already needed for
            # the computation of the new basis tensor.
            parent_id = node.parent
            self.old_state.move_orthogonalization_center(parent_id)
        old_tensor = self.state.tensors[node_id]
        new_basis_tensor = compute_new_basis_tensor(self.state.nodes[node_id],
                                                    old_tensor,
                                                    updated_tensor)
        if not node.is_leaf():
            # For an non-leaf the orthogonalised old node is only needed for
            # the computation of the basis change tensor.
            parent_id = node.parent
            self.old_state.move_orthogonalization_center(parent_id)
        old_basis_tensor = self.old_state.tensors[node_id]
        # The leg order of both tensors is the same
        basis_change_tensor = compute_basis_change_tensor(old_basis_tensor,
                                                            new_basis_tensor)
        # Now we replace the node in the new state
        self.replace_node_with_updated_basis(node_id,
                                            new_basis_tensor,
                                            basis_change_tensor)
        # We update the cache later, to keep the old cache for sibling nodes

    def subtree_update(self, node_id: str):
        """
        Updates the subtree rooted at the given node.

        Args:
            node_id (str): The id of the node to update.

        """
        node = self.state.nodes[node_id] # TODO: Check if these have the same python id
        self._assert_orth_center(node.parent, object_name="parent node")
        self.old_state.move_orthogonalization_center(node_id)
        self.tensor_cache.update_tree_cache(node.parent, node_id)
        node = self.state.nodes[node_id] # TODO: Check if these have the same python id
        if not node.is_leaf():
            # Not a leaf
            for child_id in node.children:
                self.subtree_update(child_id)
            # Now the new state contains the updated children tensors and basis changes
            self._assert_orth_center(node_id)
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
        self.update_non_root(node_id)


    def run_one_time_step(self, **kwargs):
        """
        Runs one time step of the BUG method.

        The method is based on the rank-adaptive BUG method introduced in
        https://www.doi.org/10.1137/22M1473790 .

        Args:
            kwargs: Additional keyword arguments for the time step.

        """
        raise NotImplementedError("Not yet implemented!")

def basis_change_tensor_id(node_id: str) -> str:
    """
    The identifier for a basis change tensor.

    """
    return node_id + "_basis_change_tensor"

def reverse_basis_change_tensor_id(node_id: str) -> str:
    """
    Returns the original node identifier from a basis change tensor identifier.

    """
    return node_id[:-len("_basis_change_tensor")]

def concat_along_parent_leg(node: Node,
                            old_tensor: ndarray,
                            updated_tensor: ndarray) -> ndarray:
    """
    Concatenates two tensors along the parent leg.

    The old tensor will take the first parent dimension and the updated tensor
    the second parent dimension.

    Args:
        node (GraphNode): The node for which to concatenate the tensors.
        old_tensor (ndarray): The old tensor to concatenate.
        updated_tensor (ndarray): The updated tensor to concatenate.

    Returns:
        ndarray: The concatenated tensor with doubled parent leg dimension,
            but with the same leg order as the input tensors.
    """
    return concatenate((old_tensor, updated_tensor),
                        axis=node.parent_leg)

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

def _compute_new_basis_tensor_qr(node: Node,
                                 combined_tensor: ndarray) -> ndarray:
    """
    Computes the new basis tensor from a concatenated tensor.

    Args:
        node (GraphNode): The node for which to compute the new basis tensor.
        combined_tensor (ndarray): The concatenated tensor of the old tensor
            and the updated tensor. While the parent leg dimension is doubled,
            the leg order is the same as for the node.
        
    Returns:
        ndarray: The new basis tensor. Note that the leg order is incorrect.

    """
    q_legs, r_legs = new_basis_tensor_qr_legs(node)
    new_basis_tensor, _ = tensor_qr_decomposition(combined_tensor,
                                            q_legs,
                                            r_legs)
    return new_basis_tensor

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
    combined_tensor = concat_along_parent_leg(node,
                                              old_tensor,
                                              updated_tensor)
    # We perform a QR decomposition
    new_basis_tensor = _compute_new_basis_tensor_qr(node,
                                                    combined_tensor)
    # We will have to transpose this anyway later on
    # Currently the parent leg is the last leg
    new_basis_tensor = make_last_leg_first(new_basis_tensor)
    return new_basis_tensor

def compute_basis_change_tensor(old_basis_tensor: ndarray,
                                new_basis_tensor: ndarray
                                ) -> ndarray:
    """
    Compute the new basis change tensor M_hat.

          ____             ____  other ____
       0 |    | 1       0 |    |______|    | 0
      ___|Mhat|___  =  ___| U  |______|Uhat|___
     rold|____|rnew   rold|____|______|____|rnew
                                 legs
                                (1:n-1)
                                 
    Args:
        old_basis_tensor (ndarray): The old basis tensor.
        new_basis_tensor (ndarray): The new basis tensor.

    Returns:
        ndarray: The basis change tensor M_hat. This is a matrix with the
            leg order (parent,new), i.e. mapping the new rank to the old rank.
    
    """
    errstr = "The basis tensors have different shapes!"
    assert old_basis_tensor.shape[1:] == new_basis_tensor.shape[1:], errstr
    legs = list(range(1,old_basis_tensor.ndim))
    basis_change_tensor = tensordot(old_basis_tensor,
                                    new_basis_tensor.conj(),
                                    axes=(legs,legs))
    return basis_change_tensor

def _find_new_basis_replacement_leg_specs(node: Node
                                          ) -> Tuple[LegSpecification,LegSpecification]:
    """
    Find the leg specifications to replace a node by the new basis tensors.

    Args:
        node (GraphNode): The node to replace.
    
    Returns:
        Tuple[LegSpecification,LegSpecification]: (legs_basis_change,
            legs_new_basis) The leg specifications associated to the basis
            change tensor and the new basis tensor respectively.
    """
    legs_basis_change = LegSpecification(node.parent,[],[])
    leg_new_basis = LegSpecification(None,node.children,node.open_legs)
    return legs_basis_change, leg_new_basis
