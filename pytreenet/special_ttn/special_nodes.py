"""
This module implements some commonly used nodes/tensors.
"""

from numpy import zeros, ndarray

from ..util.ttn_exceptions import non_negativity_check

def trivial_virtual_node(shape: tuple,
                         num_triv_legs: int = 1
                         ) -> ndarray:
    """
    Generates a trivial virtual tensor of a given shape.

    A trivial virtual tensor has trivial last legs of dimension 1, and the
    first element is the only non-zero element and equal to 1.

    Args:
        shape (tuple): The shape of the tensor.
        num_triv_legs (int): The number of trivial legs to add as last legs.
            Defaults to 1.
    
    Returns:
        ndarray: The trivial virtual tensor.
    
    """
    non_negativity_check(num_triv_legs, "number of trivial legs")
    tensor = zeros(list(shape) + [1],
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
