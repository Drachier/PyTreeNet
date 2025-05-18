
from __future__ import annotations
from typing import Tuple, Dict

from numpy import ndarray, concatenate, transpose, allclose, eye

from ...core.node import Node
from ...core.leg_specification import LegSpecification
from ...util.tensor_util import make_last_leg_first
from ...util.tensor_splitting import tensor_qr_decomposition, SplitMode
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...contractions.state_state_contraction import contract_any_nodes
from ...core.canonical_form import _build_qr_leg_specs
from ...util.tensor_util import compute_transfer_tensor
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
                                 neighbour_id: str = None) -> ndarray:
    """
    Computes the new basis tensor from a concatenated tensor.

    Args:
        node (GraphNode): The node for which to compute the new basis tensor.
        combined_tensor (ndarray): The concatenated tensor of the old tensor
            and the updated tensor. While the parent leg dimension is doubled,
            the leg order is the same as for the node.
        neighbour_id: the neighbour of the node which is the orthogonalization center. 
    Returns:
        ndarray: The new basis tensor. Note that the leg order is incorrect.

    """
    out_legs, in_legs = _build_qr_leg_specs(node, neighbour_id)
    out_legs.node = node
    in_legs.node = node
    out_legs_int = out_legs.find_leg_values()
    in_legs_int = in_legs.find_leg_values()
    new_basis_tensor, _ = tensor_qr_decomposition(combined_tensor,
                                            out_legs_int,
                                            in_legs_int, 
                                            mode=mode)
    out_identifier = node.identifier
    in_identifier = neighbour_id
    out_node = Node(tensor=new_basis_tensor, identifier=out_identifier)
    if out_legs.parent_leg is not None:
        out_node.open_leg_to_parent(out_legs.parent_leg,0)
    elif not out_legs.is_root:
        parent_leg = out_node.nlegs() - 1
        out_node.open_leg_to_parent(in_identifier,parent_leg)
    out_children = _find_out_children(out_node,
                                      out_legs,
                                      in_legs,
                                      in_identifier)
    out_node.open_legs_to_children(out_children)
    perm = out_node.leg_permutation
    new_basis_tensor = transpose(new_basis_tensor, perm)
    
    # Sanity check for the leg ordering to make sure new_basis_tensor has 
    # the same leg ordering as node
    assert node.neighbouring_nodes() == out_node.neighbouring_nodes()
    neighbour_index = node.neighbour_index(neighbour_id)
    assert allclose(compute_transfer_tensor(new_basis_tensor, out_legs_int), eye(new_basis_tensor.shape[neighbour_index]))
    
    return new_basis_tensor 
                                     
def _find_out_children(out_node: Node,
                       out_legs: LegSpecification,
                       in_legs: LegSpecification,
                       in_identifier: str) -> Dict[str, int]:
    """
    Finds the indices that correspond to the children of the out tensor.
    """
    out_children = {}
    if in_legs.is_root or in_legs.parent_leg is not None:
        assert out_legs.parent_leg is None
        out_setoff = 1
    elif out_legs.is_root:
        out_setoff = 0
        out_children[in_identifier] = out_node.nlegs() - 1
    else:
        assert out_legs.parent_leg is not None
        out_setoff = 1
        out_children[in_identifier] = out_node.nlegs() - 1
    out_children.update({child_id: leg_value + out_setoff
                        for leg_value, child_id in enumerate(out_legs.child_legs)})
    return out_children


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
                                                    SplitMode.KEEP,
                                                    node.parent)
    # We will have to transpose this anyway later on
    # Currently the parent leg is the last leg
    new_basis_tensor = make_last_leg_first(new_basis_tensor)
    assert new_basis_tensor.shape == updated_tensor.shape, \
        "The new basis tensor has the wrong shape!"
    return new_basis_tensor

def compute_new_basis_tensor(node: Node,
                             old_tensor: ndarray,
                             updated_tensor: ndarray,
                             neighbour_id: str) -> ndarray:
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
            the leg order of the old and updated tensor, but with a new parent
            dimension.
    
    """
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
