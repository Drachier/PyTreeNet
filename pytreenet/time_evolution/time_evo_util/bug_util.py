
from __future__ import annotations
from typing import Tuple, Optional
from dataclasses import dataclass, field

from numpy import ndarray, concatenate, eye, allclose, transpose

from ...core.node import Node
from ...core.leg_specification import LegSpecification
from ...util.tensor_util import make_last_leg_first
from ...util.tensor_splitting import tensor_qr_decomposition, SplitMode
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...contractions.state_state_contraction import contract_any_nodes
from ...core.canonical_form import _build_leg_specs
from ...util.tensor_util import compute_transfer_tensor
from .. import TimeEvoMode, TimeEvoMethod
from ...util.tensor_splitting import tensor_qr_decomposition, SplitMode, SVDParameters
from ..ttn_time_evolution import TTNTimeEvolutionConfig
from ...core.truncation import TruncationMode
from ...contractions.contraction_util import get_equivalent_legs

@dataclass
class BUGConfig(TTNTimeEvolutionConfig, SVDParameters):
    """
    Configuration class for the BUG (Basis-Update and Galerkin) algorithms.

    Attributes:
        deep (bool): If True, creates deep copies of the entire TTNS during each update step. 
            If False, only copies the relevant nodes at each point, which is more memory 
            efficient. Defaults to False.
        time_evo_mode (TimeEvoMode): Specifies the local time evolution algorithm and its 
            configuration. Defaults to RK45 method with standard tolerances. The algorithm 
            and options can be accessed as:
            - Method name: `FPBUG.config.time_evo_mode.method.value` (returns string)
            - Solver options: `FPBUG.config.time_evo_mode.solver_options` (returns dict)
            See TimeEvoMode documentation for complete method descriptions and options.
        truncation_mode (TruncationMode): Specifies the truncation method used during
            tensor decompositions. Defaults to RECURSIVE_GREEDY, which is suitable for
            TTN structures. The other recommended method for this algorithm is
            RECURSIVE_DOWNWARD, which performs recursive truncation on children bonds, down
            to the leaf nodes.

    Notes:
        - SVDParameters control the truncation behavior during tensor decompositions 

    Examples:
        >>> config = PRBUGConfig()  # Uses defaults
        >>> print(config.time_evo_mode.method.value)  # "RK45"
        >>> print(config.time_evo_mode.solver_options)  # {'atol': 1e-06, 'rtol': 1e-06}

        >>> # Custom configuration
        >>> config = PRBUGConfig(
        ...     deep=True,
        ...     time_evo_mode=TimeEvoMode(TimeEvoMethod.RK23, {'atol': 1e-8}),
        ...     rel_tol=1e-10,)
    """
    deep: bool = False
    time_evo_mode: TimeEvoMode = field(default_factory=lambda: TimeEvoMode(TimeEvoMethod.RK45))
    truncation_mode: TruncationMode = None


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
    return concat_along_specific_leg(old_tensor,
                                     updated_tensor,
                                     node.parent_leg)

def concat_along_specific_leg(old_tensor: ndarray,
                             updated_tensor: ndarray,
                             leg_index: int) -> ndarray:
    """
    Concatenates two tensors along the specific leg.

    The old tensor will take the first parent dimension and the updated tensor
    the second parent dimension.

    Args:
        old_tensor (ndarray): The old tensor to concatenate.
        updated_tensor (ndarray): The updated tensor to concatenate.
        leg_index (int): The leg index to concatenate along.

    Returns:
        ndarray: The concatenated tensor with doubled parent leg dimension,
            but with the same leg order as the input tensors.
    """
    return concatenate((old_tensor, updated_tensor),
                        axis=leg_index,)


def new_basis_tensor_qr_legs(node: Node) -> Tuple[Tuple[int, ...],Tuple[int, ...]]:
    """
    Returns the leg indices for the QR decomposition to obtain the new basis
    tensor.

    Args:
        node (GraphNode): The node for which to perform the QR decomposition.

    Returns:
        Tuple([Tuple[int],Tuple[int]]): (q_legs, r_legs) The legs for the QR
            decomposition.

    """
    assert not node.is_root(), "Root node has no parent leg!"
    q_legs = node.children_legs + node.open_legs
    r_legs = [node.parent_leg]
    return tuple(q_legs), tuple(r_legs)

def _compute_new_basis_tensor_qr(node: Node,
                                 combined_tensor: ndarray,
                                 mode: SplitMode = SplitMode.REDUCED,
                                 neighbour_id: Optional[str] = None) -> ndarray:
    """
    Computes the new basis tensor from a concatenated tensor.

    Args:
        node (GraphNode): The node for which to compute the new basis tensor.
        combined_tensor (ndarray): The concatenated tensor of the old tensor
            and the updated tensor. While the parent leg dimension is doubled,
            the leg order is the same as for the node.
        neighbour_id: the neighbour of the node which is the orthogonalization center.
    Returns:
        ndarray: The new basis tensor. Note that the leg order is incorrect initially,
                 and is transposed to match the original node's leg order.
    """
    if neighbour_id is None:
        neighbour_id = node.parent
    out_legs, in_legs = _build_leg_specs(node, neighbour_id)
    out_legs.node = node
    in_legs.node = node
    out_legs_int = out_legs.find_leg_values()
    new_basis_tensor, _ = tensor_qr_decomposition(combined_tensor,
                                                out_legs_int,
                                                in_legs.find_leg_values(),
                                                mode=mode)

    qr_bond_idx = len(out_legs_int)

    transfer_tensor_check = compute_transfer_tensor(new_basis_tensor, list(range(len(out_legs_int))))
    expected_shape = (new_basis_tensor.shape[qr_bond_idx], new_basis_tensor.shape[qr_bond_idx])
    assert transfer_tensor_check.shape == expected_shape and allclose(transfer_tensor_check, eye(new_basis_tensor.shape[qr_bond_idx])), \
        f"Transfer tensor check failed for {node.identifier}. Max diff: {max(abs(transfer_tensor_check - eye(new_basis_tensor.shape[qr_bond_idx])))}"
    leg_map_orig_to_qr = {}
    for i, original_leg_index in enumerate(out_legs_int):
        leg_map_orig_to_qr[original_leg_index] = i

    neighbour_leg_idx = node.neighbour_index(neighbour_id)
    leg_map_orig_to_qr[neighbour_leg_idx] = qr_bond_idx

    orig_leg_indices = []
    if not node.is_root():
        orig_leg_indices.append(node.parent_leg)
    for child_id in node.children:
        orig_leg_indices.append(node.neighbour_index(child_id))
    orig_leg_indices.extend(node.open_legs)

    to_original_perm = []
    for original_leg_index in orig_leg_indices:
        to_original_perm.append(leg_map_orig_to_qr[original_leg_index])

    new_basis_tensor = transpose(new_basis_tensor, to_original_perm)

    return new_basis_tensor

def compute_fixed_size_new_basis_tensor(node: Node,
                                        updated_tensor: ndarray) -> ndarray:
    """
    Computes the updated basis tensor with a fixed size.

    The updated basis tensor is found by performing a QR decomposition of the
    updated tensor.

    Args:
        node (GraphNode): The node which is updated.
        updated_tensor (ndarray): The updated basis tensor obtained by time
            evolving the old tensor. The leg order is the usual.

    Returns:
        ndarray: The new basis tensor. The leg order is the usual and equal to
            the leg order of the updated tensor. Should have the same shape
            as the updated tensor.

    """
    # We perform a QR decomposition
    new_basis_tensor = _compute_new_basis_tensor_qr(node,
                                                    updated_tensor,
                                                    mode=SplitMode.KEEP)
    # We will have to transpose this anyway later on
    # Currently the parent leg is the last leg
    new_basis_tensor = make_last_leg_first(new_basis_tensor)
    assert new_basis_tensor.shape == updated_tensor.shape, \
        "The new basis tensor has the wrong shape!"
    return new_basis_tensor

def compute_new_basis_tensor(node: Node,
                             old_tensor: ndarray,
                             updated_tensor: ndarray,
                             neighbour_id: Optional[str] = None) -> ndarray:
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
        neighbour_id: the neighbour of the node which is the orthogonalization center.
    Returns:
        ndarray: The new basis tensor. The leg order is the usual and equal to
            the leg order of the old and updated tensor, but with a new neighbouring
            dimension.
    
    """
    if neighbour_id is None:
        neighbour_id = node.parent
    # We know that both tensors have the same leg order
    # We stack them along the parent axis
    neighbour_index = node.neighbour_index(neighbour_id)
    combined_tensor = concat_along_specific_leg(old_tensor,
                                              updated_tensor,
                                              neighbour_index)
    # We perform a QR decomposition
    new_basis_tensor = _compute_new_basis_tensor_qr(node,
                                                    combined_tensor,
                                                    SplitMode.REDUCED,
                                                    neighbour_id)
    return new_basis_tensor

def compute_basis_change_tensor(node_old: Node,
                                node_new: Node,
                                tensor_old: ndarray,
                                tensor_new: ndarray,
                                basis_change_tensor_cache: PartialTreeCachDict
                                ) -> ndarray:
    """
    Computes the basis change tensor.

    The basis change tensor is found by contracting the old and new basis
    tensors via their phyical leg and contract the virtual legs with the
    already computed basis change tensors.

    Args:
        node_old (GraphNode): The node of the old basis tensor.
        node_new (GraphNode): The node of the new basis tensor.
        tensor_old (ndarray): The old basis tensor.
        tensor_new (ndarray): The new basis tensor.
        basis_change_tensor_cache (PartialTreeCachDict): The cache for the
            basis change tensors.
    
    Returns:
        ndarray: The basis change tensor, which is a matrix. The first leg will
            have the dimension of the old basis tensor and the second leg the
            dimension of the new basis tensor.

    """
    parent_id = node_old.parent
    assert parent_id is not None, "There is no basis change for the root node!"
    basis_change_tensor = contract_any_nodes(parent_id,
                                             node_old, node_new,
                                             tensor_old, tensor_new.conj(),
                                             basis_change_tensor_cache)
    return basis_change_tensor

def find_new_basis_replacement_leg_specs(node: Node
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



def adjust_node1_structure_to_node2(start_ttn, tartget_ttn, node_id):
    """
    Update specific node of start_ttn leg perputation and children to match the 
    the structure of this node in tartget_ttn.
    Args:
        start_ttn (TreeTensorNetworkState): The starting TTN state.
        tartget_ttn (TreeTensorNetworkState): The target TTN state.
        node_id (str): The identifier of the node to adjust.
    """
    start_node = start_ttn.nodes[node_id]
    target_node = tartget_ttn.nodes[node_id]
    legs = get_equivalent_legs(start_node,
                               target_node)
    if legs[0] != legs[1]:
        start_node_neighbours = start_node.neighbouring_nodes()
        element_map = {elem: i for i, elem in enumerate(start_node_neighbours)}
        target_node_neighbours = target_node.neighbouring_nodes()
        permutation = tuple(element_map[elem] for elem in target_node_neighbours)
        nneighbours = target_node.nneighbours()
        if len(start_node.open_legs) == 1:
            permutation = permutation + (nneighbours,)
        elif len(start_node.open_legs) == 2:
            permutation = permutation + (nneighbours,nneighbours+1)
        else:
            raise NotImplementedError()
        start_ttn.nodes[node_id].update_leg_permutation(permutation , start_ttn.tensors[node_id].shape)
        assert start_node.parent_leg == target_node.parent_leg
        start_node.children = target_node.children.copy()

def adjust_ttn1_structure_to_ttn2(start_ttn, tartget_ttn):
    """
    Adjusts the structure of start_ttn to match tartget_ttn by updating the leg permutations
    and children of nodes in start_ttn based on the corresponding nodes in tartget_ttn.
    Args:
        start_ttn (TreeTensorNetworkState): The starting TTN state.
        tartget_ttn (TreeTensorNetworkState): The target TTN state.
    """
    for node_id in start_ttn.nodes.keys():
        try:
            legs = get_equivalent_legs(start_ttn.nodes[node_id], tartget_ttn.nodes[node_id])
            assert legs[0] == legs[1]
        except AssertionError:
            adjust_node1_structure_to_node2(start_ttn, tartget_ttn, node_id)