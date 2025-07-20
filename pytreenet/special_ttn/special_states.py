"""
This module implements functions that can generate special quantum states for
different special tensor network structures.
"""
from enum import Enum
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from ..ttns.ttns import TreeTensorNetworkState
from ..operators.common_operators import ket_i
from ..operators.models.abc_model import generate_t_topology_indices
from ..operators.exact_operators import exact_constant_product_state
from .binary import generate_binary_ttns
from .mps import MatrixProductState
from .star import StarTreeTensorState

class TTNStructure(Enum):
    """
    Enumeration for different types of tensor network structures.
    """
    MPS = "mps"
    BINARY = "binary"
    TSTAR = "tstar"
    EXACT = "exact"

STANDARD_NODE_PREFIX = "qubit"

def generate_constant_product_state(value: int,
                                    system_size: int,
                                    structure: TTNStructure,
                                    phys_dim: int = 2,
                                    node_prefix: str = STANDARD_NODE_PREFIX,
                                    bond_dim: int = 1
                                    ) -> TreeTensorNetworkState | NDArray[np.complex64]:
    """
    Generates a constant product state as a TTNS.

    Args:
        value (int): The value to fill the state with.
        system_size (int): The parameter determining the system size. In the
            case of MPS and binary TTN this is the number of physical sites.
            In the case of TSTAR this is the number of qubits on one chain.
        structure (TTNStructure): The type of tensor network structure.
        phys_dim (int): The physical dimension of the state. Default is 2.
        node_prefix (str): The prefix for the nodes in the tensor network.
            They will be enumerated through.
        bond_dim (int): The bond dimension for the state. Default is 1.

    Returns:
        TreeTensorNetworkState | NDArray[np.complex64]: The generated constant
            product state as a TTNS or a numpy array.
    """
    if structure == TTNStructure.MPS:
        # It is generally more efficient to have the middle site as the root
        middle_site = system_size // 2
        bond_dims = [bond_dim] * (system_size - 1)
        state = MatrixProductState.constant_product_state(value,
                                                          phys_dim,
                                                          system_size,
                                                          bond_dimensions=bond_dims,
                                                          node_prefix=node_prefix,
                                                          root_site=middle_site)
        return state
    if structure == TTNStructure.BINARY:
        phys_tensor = np.zeros((bond_dim, phys_dim),
                               dtype=complex)
        phys_tensor[0, value] = 1.0
        state = generate_binary_ttns(system_size,
                                     bond_dim,
                                     phys_tensor,
                                     phys_prefix=node_prefix)
        return state
    if structure == TTNStructure.TSTAR:
        num_chains = 3
        centre_shape = [1] * (num_chains + 1)
        central_tensor = np.asarray([1],
                                   dtype=np.complex64
                                   ).reshape(centre_shape)
        phys_tensors = [ket_i(value, phys_dim).reshape((1,1, phys_dim))
                        for _ in range(system_size-1)]
        # Final chain tensor has only one virtual leg.
        phys_tensors.append(ket_i(value, phys_dim).reshape((1, phys_dim)))
        chain_tensors = [deepcopy(phys_tensors) for _ in range(num_chains)]
        node_ids = generate_t_topology_indices(system_size,
                                               site_ids=node_prefix)
        state = StarTreeTensorState.from_tensor_lists(central_tensor,
                                                      chain_tensors,
                                                      identifiers=list(node_ids))
        state.pad_bond_dimensions(bond_dim)
        return state
    if structure == TTNStructure.EXACT:
        return exact_constant_product_state(value,
                                            system_size,
                                            local_dimension=phys_dim)
    errstr = f"Unsupported TTN structure: {structure}."
    errstr += "Cannot generate constant product state!"
    raise ValueError(errstr)

def generate_zero_state(system_size: int,
                        structure: TTNStructure,
                        phys_dim: int = 2,
                        node_prefix: str = STANDARD_NODE_PREFIX,
                        bond_dim: int = 1
                        ) -> TreeTensorNetworkState | NDArray[np.complex64]:
    """
    Generates a zero state as a TTNS.

    Args:
        system_size (int): The number of physical sites.
        structure (TTNStructure): The type of tensor network structure.
        phys_dim (int): The physical dimension of the state. Default is 2.
        node_prefix (str): The prefix for the nodes in the tensor network.
            They will be enumerated through.
        bond_dim (int): The bond dimension for the state. Default is 1.

    Returns:
        TreeTensorNetworkState | NDArray[np.complex64]: The generated zero
            state as a TTNS or a numpy array.
    """
    return generate_constant_product_state(0, system_size, structure,
                                           phys_dim, node_prefix, bond_dim)
