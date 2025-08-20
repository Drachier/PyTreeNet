"""
This module contains functions generating random Tree Tensor Networks.

There are a variety of given tree topologies, which can be filled with random
tensors.
"""
from __future__ import annotations
from typing import Union, List, Tuple
from numpy import random, exp, arange, maximum, ones, sqrt, diag, linalg
from numpy.linalg import svd
from ..core.ttn import TreeTensorNetwork
from .random_node import random_tensor_node
from .random_ttns import RandomTTNSMode


def generate_random_ttn(ttn,
                        large_scale=1,
                        small_scale=1e-8,
                        decay_type='power',
                        decay_rate=1):
    """
    Converts a given TTN to a random TreeTensorNetworkState with 
    structured random singular values on each node tensor. Looks at the
    generate_random_tensor function for more details.
    """
    for ndoes_id, tensors in ttn.tensors.items():
        shape = tensors.shape
        random_tensor = generate_random_tensor(shape, 
                                                large_scale=large_scale,
                                                small_scale=small_scale,
                                                decay_type=decay_type,
                                                decay_rate=decay_rate)
        ttn.tensors[ndoes_id] = random_tensor
        ttn.nodes[ndoes_id].link_tensor(random_tensor)
    norm = sqrt(abs(ttn.scalar_product()))
    ttn.normalise(norm)
    return ttn



def generate_random_tensor(shape,
                           large_scale,
                           small_scale,
                           decay_type,
                           decay_rate):
    """
    Generate a complex tensor with structured singular values.
    
    Args:
        shape: Shape of the tensor
        large_scale: Scale for large values
        small_scale: Scale for smallest values
        decay_type: 'exponential', 'power', or 'constant' decay
        decay_rate: Rate of decay (higher = faster decay)
    """
    if len(shape) == 2:
        # Your existing matrix code here
        min_dim = min(shape)

        # Generate complex unitary matrices
        U_real, _ = linalg.qr(random.randn(shape[0], min_dim))
        U_imag, _ = linalg.qr(random.randn(shape[0], min_dim))
        U = U_real + 1j * U_imag

        V_real, _ = linalg.qr(random.randn(shape[1], min_dim))
        V_imag, _ = linalg.qr(random.randn(shape[1], min_dim))
        V = V_real + 1j * V_imag

        # Create singular values based on decay type
        singular_values = _generate_singular_values(min_dim, large_scale, 
                                                   small_scale, decay_type, decay_rate)

        # Create diagonal matrix with correct dimensions
        S = diag(singular_values)
        tensor = U @ S @ V.T.conj()

    else:
        # Higher-order tensor generation
        tensor = _generate_hosvd_tensor(shape, large_scale, 
                                          small_scale, decay_type, decay_rate)

    return tensor

def _generate_singular_values(dim, large_scale, small_scale, decay_type, decay_rate):
    """Generate singular values with specified decay pattern."""
    if decay_type == 'exponential':
        singular_values = large_scale * exp(-decay_rate * arange(dim))
        singular_values = maximum(singular_values, small_scale)
    elif decay_type == 'power':
        singular_values = large_scale / ((arange(dim) + 1) ** decay_rate)
        singular_values = maximum(singular_values, small_scale)
    else:
        raise ValueError("Unsupported decay type. Use 'exponential' or 'power'.")

    # Add some randomness
    singular_values *= (1 + 0.2 * random.randn(dim))
    singular_values = abs(singular_values)
    singular_values = singular_values / linalg.norm(singular_values)

    return singular_values

def _generate_hosvd_tensor(shape, large_scale, small_scale, decay_type, decay_rate):
    """Generate tensor using Higher-Order SVD with controlled mode singular values."""
    # Start with a random tensor
    tensor_real = random.randn(*shape)
    tensor_imag = random.randn(*shape)
    tensor = tensor_real + 1j * tensor_imag

    # Apply HOSVD and reconstruct with controlled singular values
    for mode in range(len(shape)):
        # Unfold tensor along current mode
        unfolded = _unfold_tensor(tensor, mode)

        # Compute SVD
        U, s, Vt = svd(unfolded, full_matrices=False)
        
        # Replace singular values with controlled ones
        controlled_s = _generate_singular_values(len(s), large_scale, 
                                                small_scale, decay_type, decay_rate)

        # Reconstruct with controlled singular values
        reconstructed = U @ diag(controlled_s) @ Vt

        # Fold back to tensor
        tensor = _fold_tensor(reconstructed, mode, shape)

    return tensor

def _unfold_tensor(tensor, mode):
    """Unfold tensor along specified mode."""
    shape = tensor.shape
    # Move the mode dimension to the front
    dims = list(range(len(shape)))
    dims[0], dims[mode] = dims[mode], dims[0]

    # Transpose and reshape
    tensor_transposed = tensor.transpose(dims)
    unfolded = tensor_transposed.reshape(shape[mode], -1)

    return unfolded

def _fold_tensor(matrix, mode, original_shape):
    """Fold matrix back to tensor of original shape."""
    # Reshape to transposed tensor shape
    dims = list(range(len(original_shape)))
    dims[0], dims[mode] = dims[mode], dims[0]

    transposed_shape = [original_shape[i] for i in dims]
    tensor_transposed = matrix.reshape(transposed_shape)

    # Transpose back to original mode order
    inverse_dims = [0] * len(dims)
    for i, d in enumerate(dims):
        inverse_dims[d] = i

    tensor = tensor_transposed.transpose(inverse_dims)

    return tensor



def random_small_ttns(mode: RandomTTNSMode = RandomTTNSMode.DIFFVIRT) -> TreeTensorNetwork:
    """
    Generates a small TreeTensorNetworkState of three nodes:
    The root (`"root"`) and its two children (`"c1"` and `"c2"`). The associated 
    tensors are random, but their dimensions are set.

    Args:
        mode (RandomTTNSMode): The mode of random generation of the TTNS. If mode
            is DIFFVIRT, the virtual bond dimensions are as follows::

                        |2
                        |
                        r
                       / \\
                 3|  5/  6\\   |4
                  |  /     \\  |
                   c1        c2

            Otherwise all virtual bond dimensions default to 2. If the mode is
            SAMEPHYS all phyiscal dimensions will default to 2.

    Returns:
        TreeTensorNetwork: A tree tensor network with the above topology and
            randomly filled tensors.
    """
    random_ttns = TreeTensorNetwork()
    if mode == RandomTTNSMode.DIFFVIRT:
        root_node, root_tensor = random_tensor_node((5,6,2),"root")
        random_ttns.add_root(root_node, root_tensor)
        c1_node, c1_tensor = random_tensor_node((5,3),"c1")
        random_ttns.add_child_to_parent(c1_node, c1_tensor, 0, "root", 0)
        c2_node, c2_tensor = random_tensor_node((6,4),"c2")
        random_ttns.add_child_to_parent(c2_node, c2_tensor, 0, "root", 1)
    elif mode == RandomTTNSMode.SAMEPHYS:
        root_node, root_tensor = random_tensor_node((5,6,2),"root")
        random_ttns.add_root(root_node, root_tensor)
        c1_node, c1_tensor = random_tensor_node((5,2),"c1")
        random_ttns.add_child_to_parent(c1_node, c1_tensor, 0, "root", 0)
        c2_node, c2_tensor = random_tensor_node((6,2),"c2")
        random_ttns.add_child_to_parent(c2_node, c2_tensor, 0, "root", 1)
    else:
        root_node, root_tensor = random_tensor_node((2,2,2),"root")
        random_ttns.add_root(root_node, root_tensor)
        c1_node, c1_tensor = random_tensor_node((2,3),"c1")
        random_ttns.add_child_to_parent(c1_node, c1_tensor, 0, "root", 0)
        c2_node, c2_tensor = random_tensor_node((2,4),"c2")
        random_ttns.add_child_to_parent(c2_node, c2_tensor, 0, "root", 1)
    return random_ttns

def random_big_ttns(mode: RandomTTNSMode = RandomTTNSMode.SAME) -> TreeTensorNetwork:
    """
    Generates a big TTNS
    
    The node identifiers of the form `"site" + int`. The identifiers and
    dimensions are set, but the associated tensors are random.

    Args:
        mode (RandomTTNSMode): The mode of random generation of the TTNS.
            Currently the only mode supported is SAME.

    Returns:
        TreeTensorNetwork: A random TTNS with the following topology::

                1------6-----7
               / \\     \\
              /   \\     \\
             /     \\     8 
            2       4
            |       |
            |       |
            3       5
        
    """
    if mode == RandomTTNSMode.SAME:
        # All dimensions virtual and physical are initially the same
        # We need a ttn to work on.
        node1, tensor1 = random_tensor_node((2,2,2,2), identifier="site1")
        node2, tensor2 = random_tensor_node((2,2,2), identifier="site2")
        node3, tensor3 = random_tensor_node((2,2), identifier="site3")
        node4, tensor4 = random_tensor_node((2,2,2), identifier="site4")
        node5, tensor5 = random_tensor_node((2,2), identifier="site5")
        node6, tensor6 = random_tensor_node((2,2,2,2), identifier="site6")
        node7, tensor7 = random_tensor_node((2,2), identifier="site7")
        node8, tensor8 = random_tensor_node((2,2), identifier="site8")

        random_ttns = TreeTensorNetwork()
        random_ttns.add_root(node1, tensor1)
        random_ttns.add_child_to_parent(node2, tensor2, 0, "site1", 0)
        random_ttns.add_child_to_parent(node3, tensor3, 0, "site2", 1)
        random_ttns.add_child_to_parent(node4, tensor4, 0, "site1", 1)
        random_ttns.add_child_to_parent(node5, tensor5, 0, "site4", 1)
        random_ttns.add_child_to_parent(node6, tensor6, 0, "site1", 2)
        random_ttns.add_child_to_parent(node7, tensor7, 0, "site6", 1)
        random_ttns.add_child_to_parent(node8, tensor8, 0, "site6", 2)
        return random_ttns
    errstr = "The only supported mode is RandomTTNSMode.SAME"
    raise NotImplementedError(errstr)

def random_big_ttns_two_root_children(mode: Union[RandomTTNSMode,List[Tuple[int]]] = RandomTTNSMode.SAME
                                      ) -> TreeTensorNetwork:
    """
    Returns a random big TTNS where the root has only two children.

    For testing it is important to know that the children of 1 will be in a
    different order, if the TTNS is orthogonalised.

    Args:
        mode (RandomTTNSMode): The mode of random generation of the TTNS. If it
            is SAME all legs will be chosen as 2. For DIFFVIRT the virtual
            bond dimensons are all different. Alternatively, a list of tuples
            representing the shapes of the tensors can be passed.
    
    Returns:
        TreeTensorNetwork: A random TTNS with the topology::

                0
               / \\
              /   \\
             1     6
            / \\    \\
           /   \\    \\
          2     3     7
               / \\
              /   \\
             4     5
    
    """
    if mode == RandomTTNSMode.SAME:
        shapes = [(2,2,2),(2,2,2,2),(2,2),(2,2,2,2),
                  (2,2),(2,2),(2,2,2),(2,2)]
    elif mode == RandomTTNSMode.DIFFVIRT:
        shapes = [(7,6,2),(7,4,5,2),(4,2),(5,2,3,2),
                  (2,2),(3,2),(6,3,2),(3,2)]
    elif mode == RandomTTNSMode.TRIVIALVIRTUAL:
        shapes = [(1,1,2),(1,1,1,2),(1,2),(1,1,1,2),
                  (1,2),(1,2),(1,1,2),(1,2)]
    elif isinstance(mode, list):
        assert len(mode) == 8, "The list must have 8 elements!"
        shapes = mode
    else:
        errstr = "Only RandomTTNSMode.SAME, RandomTTNSMode.DIFFVIRT or a list of shapes is supported!"
        raise NotImplementedError(errstr)

    nodes = [random_tensor_node(shape, identifier="site"+str(i))
             for i, shape in enumerate(shapes)]
    random_ttns = TreeTensorNetwork()
    random_ttns.add_root(nodes[0][0], nodes[0][1])
    random_ttns.add_child_to_parent(nodes[1][0],nodes[1][1],0,"site0",0)
    random_ttns.add_child_to_parent(nodes[2][0],nodes[2][1],0,"site1",1)
    random_ttns.add_child_to_parent(nodes[3][0],nodes[3][1],0,"site1",2)
    random_ttns.add_child_to_parent(nodes[4][0],nodes[4][1],0,"site3",1)
    random_ttns.add_child_to_parent(nodes[5][0],nodes[5][1],0,"site3",2)
    random_ttns.add_child_to_parent(nodes[6][0],nodes[6][1],0,"site0",1)
    random_ttns.add_child_to_parent(nodes[7][0],nodes[7][1],0,"site6",1)
    return random_ttns
