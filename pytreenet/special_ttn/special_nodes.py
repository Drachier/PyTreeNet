"""
This module implements some commonly used nodes/tensors.
"""

from numpy import zeros, ndarray, pad

def trivial_virtual_node(shape: tuple) -> ndarray:
    """
    Generates a trivial virtual tensor of a given shape.

    A trivial virtual tensor has a last leg of dimension 1 and the first
    element is the only non-zero element and equal to 1.

    Args:
        shape (tuple): The shape of the tensor.
    
    Returns:
        ndarray: The trivial virtual tensor.
    
    """
    tensor = zeros([dim for dim in shape] + [1],
                   dtype=complex)
    zeros_indices = tuple([0 for _ in range(len(shape) + 1)])
    tensor[zeros_indices] = 1
    return tensor

def constant_bd_trivial_node(bond_dim: int,
                             num_legs: int) -> ndarray:
    """
    Creates a trivial virtual node where all virtual legs have the same bond
    dimension.

    A trivial virtual tensor has a last leg of dimension 1 and the first
    element is the only non-zero element and equal to 1.

    Args:
        bond_dim (int): The bond dimension of the virtual legs.
        num_legs (int): The number of virtual legs.

    Return:
        ndarray: The trivial virtual tensor.
    
    """
    shape = tuple([bond_dim for _ in range(num_legs)])
    return trivial_virtual_node(shape)
