
from typing import List, Tuple

from numpy import ndarray, concatenate, tensordot

from ...core.node import Node
from ...core.leg_specification import LegSpecification
from ...util.tensor_util import make_last_leg_first
from ...util.tensor_splitting import tensor_qr_decomposition


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
