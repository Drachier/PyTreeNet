"""
Implements a generation function to generate binary Tree Tensor Network States (TTNS).
"""
import math
import numpy as np

from numpy import ndarray
from ..ttns import TreeTensorNetworkState
from ..core.node import Node
from .special_nodes import constant_bd_trivial_node

__all__ = ["generate_binary_ttns"]


PHYS_PREFIX = "qubit"     # Prefix for physical nodes identifiers
VIRTUAL_PREFIX = "node"   # Prefix for virtual nodes identifiers

def _create_trivial_tensor_node(node_id: str,
                                bond_dim: int,
                                num_legs: int,
                                dtype=None) -> tuple[Node, ndarray]:
    """
    Args:
        node_id: Identifier for the new node
        bond_dim: Bond dimension for all legs
        num_legs: Number of legs for the tensor
        dtype: Data type for the tensor (defaults to the dtype of constant_bd_trivial_node)
        
    Returns:
        A tuple containing the created Node and its associated tensor
    """
    node = Node(identifier=node_id)
    tensor = constant_bd_trivial_node(bond_dim, num_legs)

    if dtype is not None:
        tensor = tensor.astype(dtype)

    return node, tensor


def generate_binary_ttns(num_phys: int,
                         bond_dim: int,
                         phys_tensor: ndarray,
                         depth: int | None = None) -> TreeTensorNetworkState:

    """Generate a balanced binary tree tensor network state.

    This function creates a binary Tree Tensor Network State (TTNS) with physical
    sites at the leaves and virtual nodes arranged in a binary tree structure.
    If depth is specified, it controls the maximum depth of the binary tree.
    
    Args:
        num_phys: Number of physical sites
        bond_dim: Bond dimension of the tree tensor network
        phys_tensor: Tensor for physical sites
        depth: Maximum depth of binary tree (if None, uses max possible depth)

    Returns:
        A TreeTensorNetworkState object representing the generated TTNS
    """
    # Enforce complex128 dtype for physical tensor
    phys_tensor = phys_tensor.astype(np.complex128)

    # Special case for single node
    if num_phys == 1:
        ttns = TreeTensorNetworkState()
        node_id = f"{PHYS_PREFIX}0"
        node = Node(identifier=node_id)
        ttns.add_root(node, phys_tensor.copy())
        return ttns

    # Special case: depth=0 -> generate a chain (MPS) with no virtual nodes
    if depth == 0:
        return _generate_mps_chain(
            num_phys, phys_tensor, bond_dim
        )

    # Calculate required depth if not provided
    if depth is None:
        # For a binary tree, we need depth = ceil(log2(num_phys))
        depth = max(1, math.ceil(math.log2(num_phys)))

    # Create an empty TTNS
    ttns = TreeTensorNetworkState()

    # Initialize the root node
    root_id = f"{VIRTUAL_PREFIX}0_0"

    # Calculate number of physical sites a binary tree of this depth can hold
    max_phys_in_binary_tree = 2 ** depth

    # Determine if we need a chain structure at level 0 (for many physical sites)
    need_chain = num_phys > max_phys_in_binary_tree

    if need_chain:
        # Calculate number of level-0 chain nodes needed
        num_chain_nodes = math.ceil(num_phys / max_phys_in_binary_tree)

        # Root node connects to level 1 nodes and the next chain node
        root_node, root_tensor = _create_trivial_tensor_node(
            root_id, 1, 3,  # 2 for binary branch + 1 for chain
            dtype=phys_tensor.dtype
        )
        ttns.add_root(root_node, root_tensor)

        # Create chain nodes at level 0
        current_chain_id = root_id
        phys_per_branch = [min(max_phys_in_binary_tree, num_phys - i * max_phys_in_binary_tree)
                          for i in range(num_chain_nodes)]

        for i in range(1, num_chain_nodes):
            chain_id = f"{VIRTUAL_PREFIX}0_{i}"

            # Last chain node has 2 legs (prev + branch),
            # middle chain nodes have 3 legs (prev + next + branch)
            is_last = i == num_chain_nodes - 1
            num_legs = 2 if is_last else 3

            chain_node, chain_tensor = _create_trivial_tensor_node(
                chain_id, 1, num_legs,
                dtype=phys_tensor.dtype
            )

            # Connect to previous chain node - using compatible=False
            # to ignore dimension checks
            ttns.add_child_to_parent(
                chain_node,
                chain_tensor,
                0,  # Chain node's parent leg
                current_chain_id,
                2 if i == 1 else 1,  # Previous node's chain leg
                compatible=False
            )

            current_chain_id = chain_id
    else:
        # No chain needed, just use the root node for a simple binary tree
        num_legs = min(2, num_phys)  # At most 2 legs for children
        root_node, root_tensor = _create_trivial_tensor_node(
            root_id, 1, num_legs,
            dtype=phys_tensor.dtype
        )
        ttns.add_root(root_node, root_tensor)

        # Simple case - one branch from root
        phys_per_branch = [num_phys]

    # Now build balanced binary subtrees from each chain node or root
    current_phys_idx = 0
    chain_nodes = [f"{VIRTUAL_PREFIX}0_{i}" for i in range(len(phys_per_branch))]

    for i, chain_id in enumerate(chain_nodes):
        num_phys_this_branch = phys_per_branch[i]

        if num_phys_this_branch == 0:
            continue

        # Build a balanced binary subtree using bond_dim=1 for initial construction
        ttns = build_balanced_binary_subtree(
            ttns,
            chain_id,
            num_phys_this_branch,
            depth,
            1,  # Use bond_dim=1 for initial construction
            phys_tensor,
            current_phys_idx)

        current_phys_idx += num_phys_this_branch

    # Clean up inefficient nodes (particularly virtual nodes with only one child)
    ttns = clean_inefficient_paths(ttns)

    # Final step: pad all bonds to the desired bond dimension
    ttns.pad_bond_dimensions(bond_dim)

    return ttns

def _generate_mps_chain(num_phys: int,
                        phys_tensor: ndarray,
                        bond_dim: int) -> TreeTensorNetworkState:
    """Generate a simple Matrix Product State (MPS) chain with no virtual nodes."""
    ttns = TreeTensorNetworkState()

    # One-site case: just the physical tensor as a single node
    if num_phys == 1:
        single_id = f"{PHYS_PREFIX}0"
        single_node = Node(identifier=single_id)
        # Use phys_tensor directly if 1D or squeeze last dim if needed
        if phys_tensor.ndim == 1:
            single_tensor = phys_tensor.copy()
        else:
            single_tensor = phys_tensor.squeeze(0)
        ttns.add_root(single_node, single_tensor)
        return ttns

    # First boundary node: shape (1, phys_dim)
    phys_dim = phys_tensor.size if phys_tensor.ndim == 1 else phys_tensor.shape[-1]
    first_id = f"{PHYS_PREFIX}0"
    first_node = Node(identifier=first_id)
    first_tensor = np.zeros((1, phys_dim), dtype=phys_tensor.dtype)

    # Embed physical tensor at trivial virtual index
    if phys_tensor.ndim == 1:
        first_tensor[0, :] = phys_tensor
    else:
        first_tensor[0, :] = phys_tensor[0, :]
    ttns.add_root(first_node, first_tensor)

    # Middle nodes (1 to num_phys-2)
    for i in range(1, num_phys - 1):
        node_id = f"{PHYS_PREFIX}{i}"
        node = Node(identifier=node_id)
        mid_tensor = np.zeros((1, 1, phys_dim), dtype=phys_tensor.dtype)
        # Embed physical tensor at trivial virtual indices
        if phys_tensor.ndim == 1:
            mid_tensor[0, 0, :] = phys_tensor
        else:
            mid_tensor[0, 0, :] = phys_tensor[0, :]
        # Connect to previous
        prev_id = f"{PHYS_PREFIX}{i-1}"
        # prev is boundary for i==1, interior otherwise
        parent_leg = 0 if i == 1 else 1
        ttns.add_child_to_parent(node, mid_tensor, 0, prev_id, parent_leg)

    # Last boundary node: shape (1, phys_dim)
    last_id = f"{PHYS_PREFIX}{num_phys-1}"
    last_node = Node(identifier=last_id)
    last_tensor = np.zeros((1, phys_dim), dtype=phys_tensor.dtype)
    if phys_tensor.ndim == 1:
        last_tensor[0, :] = phys_tensor
    else:
        last_tensor[0, :] = phys_tensor[0, :]
    # Connect to previous node
    prev_id = f"{PHYS_PREFIX}{num_phys-2}"
    # prev is interior only if num_phys > 2
    parent_leg = 1 if num_phys > 2 else 0
    ttns.add_child_to_parent(last_node, last_tensor, 0, prev_id, parent_leg)

    # Pad all bond dimensions to the requested size
    ttns.pad_bond_dimensions(bond_dim)
    return ttns

def build_balanced_binary_subtree(ttns: TreeTensorNetworkState,
                               parent_id: str,
                               num_phys: int,
                               max_depth: int,
                               bond_dim: int,
                               phys_tensor: ndarray,
                               phys_start_idx: int) -> TreeTensorNetworkState:
    """Build a balanced binary subtree from a parent node.
    
    This recursive function builds a balanced binary tree structure by distributing
    physical nodes evenly across the tree branches.
    
    Args:
        ttns: The tree tensor network state to modify
        parent_id: ID of the parent node
        num_phys: Number of physical nodes to distribute in this subtree
        max_depth: Maximum depth allowed for the subtree
        bond_dim: Bond dimension (should be 1 for initial construction)
        phys_tensor: Tensor for physical nodes
        phys_start_idx: Starting index for physical node numbering
        
    Returns:
        Updated TTNS with subtree added
    """
    # Early exit if no physical nodes or parent doesn't exist
    if num_phys == 0 or parent_id not in ttns.nodes:
        return ttns

    # Extract parent information
    parent_node = ttns.nodes[parent_id]
    parent_open_legs = parent_node.open_legs

    if not parent_open_legs:
        # Parent has no open legs for children
        return ttns

    # Parse parent level and position
    parent_level = int(parent_id.split('_')[0].replace(VIRTUAL_PREFIX, ''))
    parent_pos = int(parent_id.split('_')[1])
    next_level = parent_level + 1

    # Case 1: Bottom of tree or few physical nodes - connect physical nodes directly
    if max_depth == 1 or num_phys <= 2:
        return _connect_physical_nodes_to_parent(
            ttns, parent_id, num_phys, bond_dim, phys_tensor, phys_start_idx, max_depth)

    # Case 2: For deeper trees, build a balanced binary structure
    # Distribute physical nodes between left and right subtrees
    num_children = min(2, len(parent_open_legs))

    if num_children == 1:
        phys_per_child = [num_phys]
    else:
        # Distribute evenly between the two subtrees
        left_phys = num_phys // 2
        right_phys = num_phys - left_phys
        phys_per_child = [left_phys, right_phys]

    # Create virtual child nodes
    current_phys_idx = phys_start_idx

    for i in range(num_children):
        if phys_per_child[i] == 0:
            continue

        # Calculate position for this child
        child_pos = 2 * parent_pos + i

        # Create virtual node ID
        child_id = f"{VIRTUAL_PREFIX}{next_level}_{child_pos}"

        # Calculate how many legs the child tensor needs
        if max_depth == 2:
            # Leaf virtual node - parent + physical children
            child_legs = 1 + min(2, phys_per_child[i])
        else:
            # Internal node - parent + virtual children
            child_legs = 1 + min(2, math.ceil(phys_per_child[i] / 2))

        # Create node with bond_dim for all dimensions
        child_node, child_tensor = _create_trivial_tensor_node(
            child_id, bond_dim, child_legs,
            dtype=phys_tensor.dtype
        )

        # Get available parent leg
        parent_leg = parent_open_legs[i if i < len(parent_open_legs) else -1]

        # Connect child to parent using compatible=False to ignore dimension checks
        ttns.add_child_to_parent(
            child_node,
            child_tensor,
            0,  # Child's parent leg
            parent_id,
            parent_leg,
            compatible=False
        )

        # Recursively build subtree from this child
        ttns = build_balanced_binary_subtree(
            ttns,
            child_id,
            phys_per_child[i],
            max_depth - 1,
            bond_dim,
            phys_tensor,
            current_phys_idx)

        current_phys_idx += phys_per_child[i]

    return ttns

def _connect_physical_nodes_to_parent(
    ttns: TreeTensorNetworkState,
    parent_id: str,
    num_phys: int,
    bond_dim: int,
    phys_tensor: ndarray,
    phys_start_idx: int,
    max_depth: int
) -> TreeTensorNetworkState:
    """Helper function to connect physical nodes directly to a parent node.
    
    If there are more physical nodes than available parent legs, creates an 
    intermediate node to handle the overflow.
    """
    parent_node = ttns.nodes[parent_id]
    parent_open_legs = parent_node.open_legs

    # Connect physical nodes directly to parent if possible
    phys_to_connect = min(len(parent_open_legs), num_phys)

    for i in range(phys_to_connect):
        phys_id = f"{PHYS_PREFIX}{phys_start_idx + i}"
        phys_node = Node(identifier=phys_id)

        # Get available leg from parent
        parent_leg = parent_open_legs[i]

        # Create physical tensor with appropriate dimensions
        if phys_tensor.ndim == 1:
            # For 1D tensors
            phys_dim = phys_tensor.size
            new_tensor = np.zeros((bond_dim, phys_dim), dtype=phys_tensor.dtype)
            new_tensor[0, :] = phys_tensor
        else:
            # For 2D+ tensors
            phys_dim = phys_tensor.shape[-1]
            new_tensor = np.zeros((bond_dim, phys_dim), dtype=phys_tensor.dtype)
            new_tensor[0, :] = phys_tensor[0, :]

        # Connect to parent using compatible=False
        ttns.add_child_to_parent(
            phys_node,
            new_tensor,
            0,  # Physical node's parent leg
            parent_id,
            parent_leg,
            compatible=False
        )

    # Return if we connected all physical nodes
    if phys_to_connect == num_phys:
        return ttns

    # Handle remaining physical nodes that couldn't be connected directly
    remaining_phys = num_phys - phys_to_connect
    start_idx = phys_start_idx + phys_to_connect

    # Parse parent level and position
    parent_level = int(parent_id.split('_')[0].replace(VIRTUAL_PREFIX, ''))
    parent_pos = int(parent_id.split('_')[1])
    next_level = parent_level + 1

    # Create an intermediate node to handle the remaining physical nodes
    intermediate_id = f"{VIRTUAL_PREFIX}{next_level}_{2 * parent_pos}"

    # Create tensor with enough legs for the remaining physical nodes
    intermediate_legs = 1 + min(2, remaining_phys)  # 1 for parent + up to 2 for children

    # Create intermediate node and connect to parent
    intermediate_node, intermediate_tensor = _create_trivial_tensor_node(
        intermediate_id, bond_dim, intermediate_legs,
        dtype=phys_tensor.dtype
    )

    # Connect to parent - use first available open leg
    ttns.add_child_to_parent(
        intermediate_node,
        intermediate_tensor,
        0,  # Intermediate's parent leg
        parent_id,
        parent_open_legs[0] if parent_open_legs else 0,  # Use first available open leg
        compatible=False
    )

    # Connect physical nodes to intermediate node
    for i in range(min(intermediate_legs - 1, remaining_phys)):
        phys_id = f"{PHYS_PREFIX}{start_idx + i}"
        phys_node = Node(identifier=phys_id)

        # Create physical tensor
        if phys_tensor.ndim == 1:
            phys_dim = phys_tensor.size
            phys_tensor_node = np.zeros((bond_dim, phys_dim), dtype=phys_tensor.dtype)
            phys_tensor_node[0, :] = phys_tensor
        else:
            phys_dim = phys_tensor.shape[-1]
            phys_tensor_node = np.zeros((bond_dim, phys_dim), dtype=phys_tensor.dtype)
            phys_tensor_node[0, :] = phys_tensor[0, :]

        # Connect to intermediate node
        ttns.add_child_to_parent(
            phys_node,
            phys_tensor_node,
            0,  # Physical node's parent leg
            intermediate_id,
            i + 1,  # Intermediate node's leg (skip parent leg)
            compatible=False
        )

    # Recursively handle any additional physical nodes if needed
    if remaining_phys > intermediate_legs - 1:
        return build_balanced_binary_subtree(
            ttns,
            intermediate_id,
            remaining_phys - (intermediate_legs - 1),
            max_depth - 1,
            bond_dim,
            phys_tensor,
            start_idx + (intermediate_legs - 1)
        )

    return ttns

def clean_inefficient_paths(ttns: TreeTensorNetworkState) -> TreeTensorNetworkState:
    """Clean up inefficient paths in the TTN structure.
    
    This function:
    1. Ensures all virtual nodes have at least one open leg
    2. Removes inefficient virtual nodes with only one child (bypassing them)
    
    Args:
        ttns: The Tree Tensor Network State to clean up
        
    Returns:
        The cleaned up TreeTensorNetworkState
    """
    # Pass 1: Ensure all virtual nodes have at least one open leg
    for node_id in list(ttns.nodes.keys()):
        if node_id.startswith(PHYS_PREFIX):
            continue  # Skip physical nodes

        node = ttns.nodes[node_id]

        # Check if node has no open legs
        if not node.open_legs and node_id in ttns.tensors:
            # Add an open leg to the tensor
            tensor = ttns.tensors[node_id]
            new_shape = list(tensor.shape) + [1]  # Add a dimension of size 1
            new_tensor = tensor.reshape(new_shape)
            ttns.tensors[node_id] = new_tensor

            # Update the node
            new_node = Node(identifier=node_id)
            new_node.link_tensor(new_tensor)
            new_node.parent = node.parent
            new_node.children = node.children.copy()
            ttns.nodes[node_id] = new_node

    # Pass 2: Identify inefficient nodes (virtual nodes with exactly one child and one parent)
    inefficient_nodes = []

    for node_id in list(ttns.nodes.keys()):
        if node_id.startswith(PHYS_PREFIX):
            continue  # Skip physical nodes

        node = ttns.nodes[node_id]

        # Check if node has exactly one child and is not the root
        if len(node.children) == 1 and node.parent is not None:
            child_id = node.children[0]
            inefficient_nodes.append((node_id, child_id))

    # Pass 3: Bypass inefficient nodes
    for inefficient_id, child_id in inefficient_nodes:
        # Skip if node has already been removed
        if inefficient_id not in ttns.nodes:
            continue

        parent_id = ttns.nodes[inefficient_id].parent
        if parent_id is None:
            continue  # Skip if node is root

        # Get the parent and child nodes
        parent_node = ttns.nodes[parent_id]
        child_node = ttns.nodes.get(child_id)

        if child_node is None:
            continue

        # Remove inefficient node from parent's children
        if inefficient_id in parent_node.children:
            parent_node.children.remove(inefficient_id)

        # Set child's parent to parent_id
        child_node.parent = parent_id

        # Add child to parent's children
        if child_id not in parent_node.children:
            parent_node.children.append(child_id)

        # Clean up the inefficient node
        inefficient_node = ttns.nodes[inefficient_id]
        inefficient_node.parent = None
        inefficient_node.children = []

        # Remove the inefficient node from the TTN
        if inefficient_id in ttns.tensors:
            del ttns.tensors[inefficient_id]
        if inefficient_id in ttns.nodes:
            del ttns.nodes[inefficient_id]

    return ttns
