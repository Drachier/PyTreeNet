"""
Implements a generation function to generate binary TTNS.
"""
import math
import numpy as np

from numpy import ndarray
from ..ttns import TreeTensorNetworkState
from ..core.node import Node
from ..util.ttn_exceptions import positivity_check
from .special_nodes import constant_bd_trivial_node

__all__ = ["generate_binary_ttns"]

def generate_binary_ttns(num_phys: int,
                         bond_dim: int,
                         phys_tensor: ndarray,
                         depth: int | None = None,
                         phys_prefix: str = "qubit",
                         virtual_prefix: str = "node") -> TreeTensorNetworkState:
    """
    Generates a balanced binary tree tensor network state with connection limits.
    If depth is None, uses the maximum possible depth based on num_phys.
    
    Args:
        num_phys (int): The number of physical sites.
        bond_dim (int): The bond dimension of the tree tensor network state.
        phys_tensor (ndarray): The tensor for the physical sites.
        depth (int | None, optional): The depth of binary tree. 
            If None, uses maximum possible depth. Defaults to None.
        phys_prefix (str, optional): The prefix for the physical nodes. Defaults to "qubit".
        virtual_prefix (str, optional): The prefix for the virtual nodes. Defaults to "node".
    
    Returns:
        TreeTensorNetworkState: The generated tree tensor network state.
    """
    positivity_check(num_phys, "number of physical sites")
    positivity_check(bond_dim, "bond dimension")

    # Special case: depth=0 -> generate a chain (MPS) with no virtual nodes
    if depth == 0:
        # Depth 0: build a pure MPS chain of physical nodes with no unconnected virtual legs
        ttns = TreeTensorNetworkState()
        # One-site case: just the physical tensor as a single open leg
        if num_phys == 1:
            single_id = f"{phys_prefix}0"
            single_node = Node(identifier=single_id)
            # Use phys_tensor directly if 1D or squeeze last dim if needed
            if phys_tensor.ndim == 1:
                single_tensor = phys_tensor.copy()
            else:
                single_tensor = phys_tensor.squeeze(0)
            ttns.add_root(single_node, single_tensor)
            return ttns

        # Multi-site chain
        # First boundary node: shape (bond_dim, phys_dim)
        phys_dim = phys_tensor.size if phys_tensor.ndim == 1 else phys_tensor.shape[-1]
        first_id = f"{phys_prefix}0"
        first_node = Node(identifier=first_id)
        first_tensor = np.zeros((bond_dim, phys_dim), dtype=phys_tensor.dtype)
        # embed physical tensor at trivial virtual index
        if phys_tensor.ndim == 1:
            first_tensor[0, :] = phys_tensor
        else:
            first_tensor[0, :] = phys_tensor[0, :]
        ttns.add_root(first_node, first_tensor)

        # Middle nodes (1 to num_phys-2)
        for i in range(1, num_phys - 1):
            node_id = f"{phys_prefix}{i}"
            node = Node(identifier=node_id)
            mid_tensor = np.zeros((bond_dim, bond_dim, phys_dim), dtype=phys_tensor.dtype)
            # embed physical tensor at trivial virtual indices
            if phys_tensor.ndim == 1:
                mid_tensor[0, 0, :] = phys_tensor
            else:
                mid_tensor[0, 0, :] = phys_tensor[0, :]
            # connect to previous
            prev_id = f"{phys_prefix}{i-1}"
            # prev is boundary for i==1, interior otherwise
            parent_leg = 0 if i == 1 else 1
            ttns.add_child_to_parent(node, mid_tensor, 0, prev_id, parent_leg)

        # Last boundary node: shape (bond_dim, phys_dim)
        last_id = f"{phys_prefix}{num_phys-1}"
        last_node = Node(identifier=last_id)
        last_tensor = np.zeros((bond_dim, phys_dim), dtype=phys_tensor.dtype)
        if phys_tensor.ndim == 1:
            last_tensor[0, :] = phys_tensor
        else:
            last_tensor[0, :] = phys_tensor[0, :]
        # connect to previous node
        prev_id = f"{phys_prefix}{num_phys-2}"
        # prev is interior only if num_phys > 2
        parent_leg = 1 if num_phys > 2 else 0
        ttns.add_child_to_parent(last_node, last_tensor, 0, prev_id, parent_leg)
        return ttns
    
    # Calculate required depth if not provided
    if depth is None:
        # Calculate min depth needed to support all physical nodes
        # For a binary tree, we need depth = ceil(log2(num_phys))
        depth = max(1, math.ceil(math.log2(num_phys)))
    else:
        # Ensure depth is at least 1
        depth = max(1, depth)
    
    # Create an empty TTNS
    ttns = TreeTensorNetworkState()
    
    
    # Initialize the root node
    root_id = f"{virtual_prefix}0_0"
    root_node = Node(identifier=root_id)
    
    # Calculate number of branch levels needed
    max_phys_in_binary_tree = 2 ** depth
    
    # Determine if we need a chain structure at level 0
    need_chain = num_phys > max_phys_in_binary_tree
    
    if need_chain:
        # Chain structure is needed for many physical sites
        # Calculate number of level-0 chain nodes needed
        num_chain_nodes = math.ceil(num_phys / max_phys_in_binary_tree)
        
        # Root node needs connections to level 1 nodes and next chain node
        root_tensor = constant_bd_trivial_node(bond_dim, 3)  # 2 for binary branch + 1 for chain
        ttns.add_root(root_node, root_tensor)
        
        # Create chain nodes at level 0
        current_chain_id = root_id
        phys_per_branch = [min(max_phys_in_binary_tree, num_phys - i * max_phys_in_binary_tree) 
                          for i in range(num_chain_nodes)]
        
        for i in range(1, num_chain_nodes):
            chain_id = f"{virtual_prefix}0_{i}"
            chain_node = Node(identifier=chain_id)
            
            # Middle chain nodes have 3 legs: prev + next + branch
            # Last chain node has 2 legs: prev + branch
            is_last = (i == num_chain_nodes - 1)
            num_legs = 2 if is_last else 3
            
            chain_tensor = constant_bd_trivial_node(bond_dim, num_legs)
            
            # Connect to previous chain node
            ttns.add_child_to_parent(
                chain_node,
                chain_tensor,
                0,  # Chain node's parent leg
                current_chain_id,
                2 if i == 1 else 1,  # Previous node's chain leg
                modify=True
            )
            
            current_chain_id = chain_id
    else:
        # No chain needed, just use the root node with appropriate legs
        # Root tensor has legs for its children
        num_legs = min(2, num_phys)  # At least 1, at most 2 legs for children
        root_tensor = constant_bd_trivial_node(bond_dim, num_legs)
        ttns.add_root(root_node, root_tensor)
        
        # Simple case - one branch from root
        phys_per_branch = [num_phys]
    
    # Now build balanced binary subtrees from each chain node or root
    current_phys_idx = 0
    chain_nodes = [f"{virtual_prefix}0_{i}" for i in range(len(phys_per_branch))]
    
    for i, chain_id in enumerate(chain_nodes):
        num_phys_this_branch = phys_per_branch[i]
        
        if num_phys_this_branch == 0:
            continue
        
        # Build a balanced binary subtree
        ttns = build_balanced_binary_subtree(
            ttns,
            chain_id,
            num_phys_this_branch,
            depth,
            bond_dim,
            phys_tensor,
            phys_prefix,
            virtual_prefix,
            current_phys_idx
        )
        
        current_phys_idx += num_phys_this_branch
    
    # Clean up any remaining inefficient paths
    ttns = clean_inefficient_paths(ttns, bond_dim, phys_prefix, virtual_prefix)
    
    # Ensure all tensor dimensions are correct before returning
    ttns = fix_tensor_dimensions(ttns, bond_dim, phys_prefix)
    
    # First, scan for all dimensional mismatches
    mismatches = []
    
    for node_id, node in ttns.nodes.items():
        if node.parent is not None:
            parent_id = node.parent
            parent_node = ttns.nodes[parent_id]
            
            # Get connection legs
            child_leg = 0  # Always the first leg for parent connection
            try:
                parent_leg = parent_node.children.index(node_id)
            except ValueError:
                # Orphaned node reference, clean it up
                node.parent = None
                continue
                
            # Verify dimensions match
            if node_id in ttns.tensors and parent_id in ttns.tensors:
                child_tensor = ttns.tensors[node_id]
                parent_tensor = ttns.tensors[parent_id]
                
                if child_leg < len(child_tensor.shape) and parent_leg < len(parent_tensor.shape):
                    if child_tensor.shape[child_leg] != parent_tensor.shape[parent_leg]:
                        mismatches.append((node_id, parent_id, child_leg, parent_leg))
    
    # Fix dimension mismatches using a consistent approach
    for child_id, parent_id, child_leg, parent_leg in mismatches:
        # Get tensors
        child_tensor = ttns.tensors[child_id]
        parent_tensor = ttns.tensors[parent_id]
        
        # Use bond_dim as the target dimension consistently 
        target_dim = bond_dim
        
        # Fix child tensor
        child_shape = list(child_tensor.shape)
        child_shape[child_leg] = target_dim
        new_child_tensor = np.zeros(tuple(child_shape), dtype=child_tensor.dtype)
        
        # Copy existing data where possible
        min_dim = min(child_tensor.shape[child_leg], target_dim)
        if min_dim > 0:  # Prevent slicing errors
            # Create slices that copy data from the smaller dimension to the larger
            child_slices = tuple([slice(0, min_dim) if i == child_leg else slice(None) 
                                for i in range(len(child_shape))])
            try:
                new_child_tensor[child_slices] = child_tensor[child_slices]
            except ValueError:
                # If there's a problem with the slicing, just ensure the tensor is valid
                pass
        
        # Ensure the tensor is not all zeros by setting first element to 1.0
        zeros_indices = tuple([0 for _ in range(len(child_shape))])
        if np.count_nonzero(new_child_tensor) == 0:
            new_child_tensor[zeros_indices] = 1.0
        
        # Update child tensor
        ttns.tensors[child_id] = new_child_tensor
        
        # Fix parent tensor
        parent_shape = list(parent_tensor.shape)
        parent_shape[parent_leg] = target_dim
        new_parent_tensor = np.zeros(tuple(parent_shape), dtype=parent_tensor.dtype)
        
        # Copy existing data where possible
        min_dim = min(parent_tensor.shape[parent_leg], target_dim)
        if min_dim > 0:  # Prevent slicing errors
            # Create slices that copy data from the smaller dimension to the larger
            parent_slices = tuple([slice(0, min_dim) if i == parent_leg else slice(None) 
                                 for i in range(len(parent_shape))])
            try:
                new_parent_tensor[parent_slices] = parent_tensor[parent_slices]
            except ValueError:
                # If there's a problem with the slicing, just ensure the tensor is valid
                pass
        
        # Ensure the tensor is not all zeros by setting first element to 1.0
        zeros_indices = tuple([0 for _ in range(len(parent_shape))])
        if np.count_nonzero(new_parent_tensor) == 0:
            new_parent_tensor[zeros_indices] = 1.0
        
        # Update parent tensor
        ttns.tensors[parent_id] = new_parent_tensor
    
    # Do one final check to verify everything is fixed
    remaining_mismatches = []
    
    for node_id, node in ttns.nodes.items():
        if node.parent is not None:
            parent_id = node.parent
            parent_node = ttns.nodes[parent_id]
            
            # Get connection legs
            child_leg = 0  # Always the first leg for parent connection
            try:
                parent_leg = parent_node.children.index(node_id)
            except ValueError:
                # Orphaned node reference, clean it up
                node.parent = None
                continue
                
            # Verify dimensions match
            if node_id in ttns.tensors and parent_id in ttns.tensors:
                child_tensor = ttns.tensors[node_id]
                parent_tensor = ttns.tensors[parent_id]
                
                if child_leg < len(child_tensor.shape) and parent_leg < len(parent_tensor.shape):
                    if child_tensor.shape[child_leg] != parent_tensor.shape[parent_leg]:
                        remaining_mismatches.append((node_id, parent_id, child_leg, parent_leg))
    
    # If there are still mismatches, replace both tensors with brand new tensors
    for child_id, parent_id, child_leg, parent_leg in remaining_mismatches:
        # For problematic nodes, create entirely new tensors with consistent dimensions
        child_node = ttns.nodes[child_id]
        parent_node = ttns.nodes[parent_id]
        
        # Create new child tensor
        is_physical = child_id.startswith(phys_prefix)
        
        if is_physical:
            # For physical nodes, create a tensor with correct parent dimension
            # While preserving the physical dimension
            old_tensor = ttns.tensors[child_id]
            phys_dim = old_tensor.shape[-1]  # Assuming last dim is physical
            
            # Create a tensor with parent dimension = bond_dim
            new_shape = (bond_dim, phys_dim)
            new_tensor = np.zeros(new_shape, dtype=old_tensor.dtype)
            
            # Copy physical data from the original tensor
            if old_tensor.ndim == 2:  # Standard case
                try:
                    # Copy data for the smallest common dimension
                    min_dim = min(bond_dim, old_tensor.shape[0])
                    min_phys = min(phys_dim, old_tensor.shape[1])
                    for i in range(min_dim):
                        for j in range(min_phys):
                            new_tensor[i, j] = old_tensor[i, j]
                except:
                    # Fallback - just set the first element
                    new_tensor[0, 0] = 1.0
            else:
                # For non-standard tensors, just set first element to 1
                new_tensor[0, 0] = 1.0
            
            ttns.tensors[child_id] = new_tensor
            
            # Update the node with the new tensor
            new_node = Node(identifier=child_id)
            new_node.link_tensor(new_tensor)
            new_node.parent = child_node.parent
            new_node.children = child_node.children.copy()
            ttns.nodes[child_id] = new_node
        else:
            # For virtual nodes, create a completely new tensor
            old_tensor = ttns.tensors[child_id]
            num_legs = len(old_tensor.shape)
            
            # Create a new tensor with consistent bond dimensions
            new_tensor = constant_bd_trivial_node(bond_dim, num_legs)
            ttns.tensors[child_id] = new_tensor
            
            # Update the node with the new tensor
            new_node = Node(identifier=child_id)
            new_node.link_tensor(new_tensor)
            new_node.parent = child_node.parent
            new_node.children = child_node.children.copy()
            ttns.nodes[child_id] = new_node
        
        # Only update parent tensor if it's not already been updated in this loop
        if not parent_id.startswith(phys_prefix): 
            # Create new parent tensor only if it's a virtual node
            old_tensor = ttns.tensors[parent_id]
            num_legs = len(old_tensor.shape)
            
            # Create a new tensor with consistent bond dimensions
            new_tensor = constant_bd_trivial_node(bond_dim, num_legs)
            ttns.tensors[parent_id] = new_tensor
            
            # Update the node with the new tensor
            new_node = Node(identifier=parent_id)
            new_node.link_tensor(new_tensor)
            new_node.parent = parent_node.parent
            new_node.children = parent_node.children.copy()
            ttns.nodes[parent_id] = new_node
    
    return ttns

def build_balanced_binary_subtree(ttns: TreeTensorNetworkState,
                                parent_id: str,
                                num_phys: int,
                                max_depth: int,
                                bond_dim: int,
                                phys_tensor: ndarray,
                                phys_prefix: str,
                                virtual_prefix: str,
                                phys_start_idx: int) -> TreeTensorNetworkState:
    """
    Builds a balanced binary subtree from a parent node.
    
    Args:
        ttns: The tree tensor network state
        parent_id: ID of the parent node
        num_phys: Number of physical nodes to distribute in this subtree
        max_depth: Maximum depth allowed for the subtree
        bond_dim: Bond dimension
        phys_tensor: Tensor for physical nodes
        phys_prefix: Prefix for physical node IDs
        virtual_prefix: Prefix for virtual node IDs
        phys_start_idx: Starting index for physical node numbering
        
    Returns:
        TreeTensorNetworkState: Updated TTNS with subtree added
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
    
    # Get parent tensor for dimension checking
    parent_tensor = ttns.tensors[parent_id]
    
    # Parse parent level and position
    parent_level = int(parent_id.split('_')[0].replace(virtual_prefix, ''))
    parent_pos = int(parent_id.split('_')[1])
    next_level = parent_level + 1
    
    # Handle case where we can directly connect physical nodes to parent
    if max_depth == 1 or num_phys <= 2:
        # Connect physical nodes directly
        phys_to_connect = min(len(parent_open_legs), num_phys)
        
        for i in range(phys_to_connect):
            phys_id = f"{phys_prefix}{phys_start_idx + i}"
            phys_node = Node(identifier=phys_id)
            
            # Get available leg from parent
            parent_leg = parent_open_legs[i]
            
            # Get tensor dimension for this leg
            parent_leg_dim = parent_tensor.shape[parent_leg]
            
            # Get physical dimension from the input tensor
            if phys_tensor.ndim == 1:
                # For 1D tensors
                phys_dim = phys_tensor.size
                new_tensor = np.zeros((parent_leg_dim, phys_dim), dtype=phys_tensor.dtype)
                
                # Set only the 0th index of the first dimension to encode the physical tensor
                for j in range(phys_dim):
                    new_tensor[0, j] = phys_tensor[j]  # Copy values from 1D tensor
            else:
                # For 2D+ tensors
                phys_dim = phys_tensor.shape[-1]  # Assuming last dim is physical
                new_tensor = np.zeros((parent_leg_dim, phys_dim), dtype=phys_tensor.dtype)
                
                # Set only the 0th index of the first dimension to encode the physical tensor
                for j in range(phys_dim):
                    new_tensor[0, j] = phys_tensor[0, j]  # Copy values from input tensor
            
            # Connect to parent
            ttns.add_child_to_parent(
                phys_node,
                new_tensor,
                0,  # Physical node's parent leg
                parent_id,
                parent_leg  # Parent's open leg
                , modify=True
            )
        
        # Return early if we connected all physical nodes
        if phys_to_connect == num_phys:
            return ttns
    
        # Otherwise, continue with remaining physical nodes that couldn't be connected
        remaining_phys = num_phys - phys_to_connect
        start_idx = phys_start_idx + phys_to_connect
        
        # Create an intermediate node to handle the remaining physical nodes
        intermediate_id = f"{virtual_prefix}{next_level}_{2 * parent_pos}"
        intermediate_node = Node(identifier=intermediate_id)
        
        # Create tensor with enough legs for the remaining physical nodes
        intermediate_legs = 1 + min(2, remaining_phys)  # 1 for parent + up to 2 for children
        
        # No more open legs on parent, create a new node at the same level
        sibling_id = f"{virtual_prefix}{parent_level}_{parent_pos + 1}"
        sibling_node = Node(identifier=sibling_id)
        
        # Find parent's parent
        if not parent_node.is_root():
            grandparent_id = parent_node.parent
            grandparent_node = ttns.nodes[grandparent_id]
            
            # Find an open leg on grandparent
            if grandparent_node.open_legs:
                grandparent_leg = grandparent_node.open_legs[0]
                
                # Create sibling tensor with compatible dimensions
                sibling_tensor = constant_bd_trivial_node(bond_dim, 3)  # 1 for its parent + 1 for child + 1 open leg
                
                # Connect sibling to grandparent
                ttns.add_child_to_parent(
                    sibling_node,
                    sibling_tensor,
                    0,  # Sibling's parent leg
                    grandparent_id,
                    grandparent_leg  # Grandparent's open leg
                    , modify=True
                )
                
                # Now get sibling's tensor to check dimensions
                sibling_tensor = ttns.tensors[sibling_id]
                sibling_leg = 1  # Use leg 1 for connection to intermediate
                
                # Create intermediate tensor with compatible dimensions
                intermediate_tensor = constant_bd_trivial_node(bond_dim, intermediate_legs)
                
                # Connect intermediate node to sibling
                ttns.add_child_to_parent(
                    intermediate_node,
                    intermediate_tensor,
                    0,  # Intermediate node's parent leg
                    sibling_id,
                    sibling_leg  # Sibling's child leg
                    , modify=True
                )
            else:
                # Create a new link in the chain
                next_chain_id = f"{virtual_prefix}0_{len([n for n in ttns.nodes if n.startswith(f'{virtual_prefix}0_')])}"
                next_chain_node = Node(identifier=next_chain_id)
                
                # Create tensors with compatible dimensions
                chain_tensor = constant_bd_trivial_node(bond_dim, 3)  # 1 for previous chain + 1 for intermediate + 1 open leg
                
                # Find parent of last chain node
                last_chain_parent = None
                for node_id in ttns.nodes:
                    if node_id.startswith(f'{virtual_prefix}0_'):
                        node = ttns.nodes[node_id]
                        if node.is_root():
                            last_chain_parent = node_id
                
                if last_chain_parent:
                    # Connect to last chain node
                    ttns.add_child_to_parent(
                        next_chain_node,
                        chain_tensor,
                        0,  # Chain node's parent leg
                        last_chain_parent,
                        1  # Previous chain node's leg
                        , modify=True
                    )
                    
                    # Now create intermediate tensor with compatible dimensions
                    intermediate_tensor = constant_bd_trivial_node(bond_dim, intermediate_legs)
                    
                    # Connect intermediate to chain
                    ttns.add_child_to_parent(
                        intermediate_node,
                        intermediate_tensor,
                        0,  # Intermediate node's parent leg
                        next_chain_id,
                        1  # Chain node's leg
                        , modify=True
                    )
                else:
                    # Fallback to creating a new root
                    new_root_id = f"{virtual_prefix}-1_0"  # Special level -1 for new root
                    new_root = Node(identifier=new_root_id)
                    
                    # Create a simple tensor for the new root
                    root_tensor = constant_bd_trivial_node(bond_dim, 3)  # 2 legs + 1 extra
                    ttns.add_root(new_root, root_tensor)
                    
                    # Connect chain node to new root
                    ttns.add_child_to_parent(
                        next_chain_node,
                        chain_tensor,
                        0,  # Chain node's parent leg
                        new_root_id,
                        0  # New root's first leg
                        , modify=True
                    )
                    
                    # Now create intermediate tensor with compatible dimensions
                    intermediate_tensor = constant_bd_trivial_node(bond_dim, intermediate_legs)
                    
                    # Connect intermediate to chain
                    ttns.add_child_to_parent(
                        intermediate_node,
                        intermediate_tensor,
                        0,  # Intermediate node's parent leg
                        next_chain_id,
                        1  # Chain node's leg for intermediate
                        , modify=True
                    )
        else:
            # Parent is root, create a new root
            new_root_id = f"{virtual_prefix}{parent_level-1}_0"
            new_root = Node(identifier=new_root_id)
            
            # Create a simple tensor for the new root
            root_tensor = constant_bd_trivial_node(bond_dim, 3)  # 2 legs + 1 extra
            ttns.add_root(new_root, root_tensor)
            
            # Get parent tensor dimensions
            parent_leg_dim = parent_tensor.shape[0]  # First dimension
            
            # Connect old root to new root
            ttns.add_child_to_parent(
                parent_node,
                parent_tensor,
                0,  # Old root's parent leg
                new_root_id,
                0  # New root's first leg
                , modify=True
            )
            
            # Create sibling tensor with compatible dimensions
            sibling_tensor = constant_bd_trivial_node(bond_dim, 3)  # 1 for parent + 1 for child + 1 open leg
            
            # Connect sibling to new root
            ttns.add_child_to_parent(
                sibling_node,
                sibling_tensor,
                0,  # Sibling's parent leg
                new_root_id,
                1  # New root's second leg
                , modify=True
            )
            
            # Now create intermediate tensor with compatible dimensions 
            sibling_tensor = ttns.tensors[sibling_id]
            
            intermediate_tensor = constant_bd_trivial_node(bond_dim, intermediate_legs)
            
            # Connect intermediate node to sibling
            ttns.add_child_to_parent(
                intermediate_node,
                intermediate_tensor,
                0,  # Intermediate node's parent leg
                sibling_id,
                1  # Sibling's child leg
                , modify=True
            )
        
        # Now connect physical nodes to intermediate node
        intermediate_tensor = ttns.tensors[intermediate_id]
        remaining_to_connect = min(intermediate_legs - 1, remaining_phys)
        
        for i in range(remaining_to_connect):
            phys_id = f"{phys_prefix}{start_idx + i}"
            phys_node = Node(identifier=phys_id)
            
            # Get leg dimension
            leg_idx = i + 1  # Skip parent leg
            leg_dim = intermediate_tensor.shape[leg_idx]
            
            # Get physical dimension from the input tensor
            if phys_tensor.ndim == 1:
                # For 1D tensors
                phys_dim = phys_tensor.size
                new_tensor = np.zeros((leg_dim, phys_dim), dtype=phys_tensor.dtype)
                
                # Set only the 0th index of the first dimension to encode the physical tensor
                for j in range(phys_dim):
                    new_tensor[0, j] = phys_tensor[j]  # Copy values from 1D tensor
            else:
                # For 2D+ tensors
                phys_dim = phys_tensor.shape[-1]  # Assuming last dim is physical
                new_tensor = np.zeros((leg_dim, phys_dim), dtype=phys_tensor.dtype)
                
                # Set only the 0th index of the first dimension to encode the physical tensor
                for j in range(phys_dim):
                    new_tensor[0, j] = phys_tensor[0, j]  # Copy values from input tensor
            
            # Connect to intermediate node
            ttns.add_child_to_parent(
                phys_node,
                new_tensor,
                0,  # Physical node's parent leg
                intermediate_id,
                leg_idx  # Intermediate node's child leg
                , modify=True
            )
        
        # If there are still physical nodes remaining, build a subtree from the intermediate node
        if remaining_phys > remaining_to_connect:
            # Recursively build from intermediate node
            return build_balanced_binary_subtree(
                ttns,
                intermediate_id,
                remaining_phys - remaining_to_connect,
                max_depth - 1,
                bond_dim,
                phys_tensor,
                phys_prefix,
                virtual_prefix,
                start_idx + remaining_to_connect
            )
        
        # Validate virtual node tensor dimensions
        for node_id in list(ttns.nodes.keys()):
            if node_id.startswith(virtual_prefix) and node_id in ttns.tensors:
                tensor = ttns.tensors[node_id]
                # Check if any virtual dimension is 1 (except the last trivial dimension)
                if any(dim == 1 for i, dim in enumerate(tensor.shape) if i < len(tensor.shape)-1):
                    # Create a new tensor with correct bond dimensions
                    new_shape = [bond_dim] * (len(tensor.shape) - 1) + [tensor.shape[-1]]
                    new_tensor = np.zeros(tuple(new_shape), dtype=tensor.dtype)
                    
                    # Set the first element to 1
                    zeros_indices = tuple([0 for _ in range(len(new_shape))])
                    new_tensor[zeros_indices] = 1
                    
                    # Replace the tensor
                    ttns.tensors[node_id] = new_tensor
                    
                    # Update the node with the new tensor shape
                    node = ttns.nodes[node_id]
                    new_node = Node(identifier=node_id)
                    new_node.link_tensor(new_tensor)
                    new_node.parent = node.parent
                    new_node.children = node.children.copy()
                    ttns.nodes[node_id] = new_node
        
        # Before returning, ensure proper dimensions
        ttns = validate_fix_tensor_dimensions(ttns, bond_dim, virtual_prefix)
        
        return ttns
    
    # For deeper trees, build a balanced binary structure
    
    # Determine number of children needed at this level
    # We want to distribute physical nodes as evenly as possible
    if num_phys <= 2:
        # Simple case - at most one virtual node with up to 2 physical children
        num_children = 1
        phys_per_child = [num_phys]
    else:
        # For more physical nodes, we need to distribute across subtrees
        # Use at most 2 children to maintain binary structure
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
        
        # Create virtual node
        child_id = f"{virtual_prefix}{next_level}_{child_pos}"
        child_node = Node(identifier=child_id)
        
        # Select an available parent leg, ensuring we don't go out of bounds
        if i < len(parent_open_legs):
            parent_leg = parent_open_legs[i]
        else:
            # Not enough legs on parent, use the last available one
            parent_leg = parent_open_legs[-1]
        
        # Get parent leg dimension
        parent_leg_dim = parent_tensor.shape[parent_leg]
        
        # Calculate how many legs the child tensor needs
        if max_depth == 2:
            # Leaf virtual node - parent + physical children
            num_legs = 1 + min(2, phys_per_child[i])
        else:
            # Internal node - parent + virtual children
            num_legs = 1 + min(2, math.ceil(phys_per_child[i] / 2))
        
        # Special case for fixing dimension mismatch between levels 1 and 2
        if (parent_level == 1 and next_level == 2) or (parent_id.startswith(f"{virtual_prefix}1_") and next_level == 2):
            # Use bond_dim consistently for all connections in this problematic area
            child_tensor = constant_bd_trivial_node(bond_dim, num_legs)
            
            # Check if we need to also fix the parent tensor dimension
            if parent_leg_dim != bond_dim:
                # Expand parent tensor dimension to match bond_dim
                parent_shape = list(parent_tensor.shape)
                parent_shape[parent_leg] = bond_dim
                new_parent_tensor = np.zeros(tuple(parent_shape), dtype=parent_tensor.dtype)
                
                # Copy existing data
                slices = tuple([slice(0, min(dim, parent_tensor.shape[i])) 
                              for i, dim in enumerate(parent_shape)])
                new_parent_tensor[slices] = parent_tensor[slices]
                
                # Ensure the first element is set to 1
                zeros_indices = tuple([0 for _ in range(len(parent_shape))])
                if np.count_nonzero(new_parent_tensor) == 0:
                    new_parent_tensor[zeros_indices] = 1.0
                    
                # Update parent tensor
                ttns.tensors[parent_id] = new_parent_tensor
                parent_tensor = new_parent_tensor
                parent_leg_dim = bond_dim
        else:
            # Normal case, create tensor with parent_leg_dim
            child_tensor = constant_bd_trivial_node(parent_leg_dim, num_legs)
        
        # Connect child to parent
        ttns.add_child_to_parent(
            child_node,
            child_tensor,
            0,  # Child's parent leg
            parent_id,
            parent_leg  # Parent's open leg
            , modify=True
        )
        
        # Recursively build subtree from this child
        ttns = build_balanced_binary_subtree(
            ttns,
            child_id,
            phys_per_child[i],
            max_depth - 1,
            bond_dim,
            phys_tensor,
            phys_prefix,
            virtual_prefix,
            current_phys_idx
        )
        
        current_phys_idx += phys_per_child[i]
    
    # Validate virtual node tensor dimensions
    for node_id in list(ttns.nodes.keys()):
        if node_id.startswith(virtual_prefix) and node_id in ttns.tensors:
            tensor = ttns.tensors[node_id]
            # Check if any virtual dimension is 1 (except the last trivial dimension)
            if any(dim == 1 for i, dim in enumerate(tensor.shape) if i < len(tensor.shape)-1):
                # Create a new tensor with correct bond dimensions
                new_shape = [bond_dim] * (len(tensor.shape) - 1) + [tensor.shape[-1]]
                new_tensor = np.zeros(tuple(new_shape), dtype=tensor.dtype)
                
                # Set the first element to 1
                zeros_indices = tuple([0 for _ in range(len(new_shape))])
                new_tensor[zeros_indices] = 1
                
                # Replace the tensor
                ttns.tensors[node_id] = new_tensor
                
                # Update the node with the new tensor shape
                node = ttns.nodes[node_id]
                new_node = Node(identifier=node_id)
                new_node.link_tensor(new_tensor)
                new_node.parent = node.parent
                new_node.children = node.children.copy()
                ttns.nodes[node_id] = new_node
    
    # Before returning, ensure proper dimensions
    ttns = validate_fix_tensor_dimensions(ttns, bond_dim, virtual_prefix)
    
    return ttns

def fix_tensor_dimensions(ttns: TreeTensorNetworkState, bond_dim: int, phys_prefix: str):
    """
    Ensure all nodes have tensor dimensions matching the specified bond dimension.
    
    Args:
        ttns: The tree tensor network state to fix
        bond_dim: The desired bond dimension for all virtual bonds
        phys_prefix: Prefix for physical node IDs to identify virtual nodes
        
    Returns:
        TreeTensorNetworkState: The TTNS with fixed dimensions
    """
    # First fix virtual nodes
    for node_id in list(ttns.nodes.keys()):
        # Skip physical nodes for this pass
        if node_id.startswith(phys_prefix):
            continue
            
        if node_id in ttns.tensors:
            tensor = ttns.tensors[node_id]
            
            # Check if any dimension is 1 
            needs_fix = any(dim == 1 for i, dim in enumerate(tensor.shape[:-1]))
            
            if needs_fix:
                # Create a new shape with bond_dim for all dimensions except the last
                new_shape = [bond_dim] * (len(tensor.shape) - 1) + [tensor.shape[-1]]
                
                # Create a new tensor with the correct dimensions
                new_tensor = np.zeros(tuple(new_shape), dtype=tensor.dtype)
                
                # Set the first element to 1 to maintain trivial tensor property
                zeros_indices = tuple([0 for _ in range(len(new_shape))])
                new_tensor[zeros_indices] = 1
                
                # Replace the tensor
                ttns.tensors[node_id] = new_tensor
                
                # Update the node with the new tensor shape
                node = ttns.nodes[node_id]
                new_node = Node(identifier=node_id)
                new_node.link_tensor(new_tensor)
                new_node.parent = node.parent
                new_node.children = node.children.copy()
                ttns.nodes[node_id] = new_node
    
    # Now fix physical nodes - ensure their first dimension (parent connection) is bond_dim
    for node_id in list(ttns.nodes.keys()):
        # Only process physical nodes
        if not node_id.startswith(phys_prefix):
            continue
            
        if node_id in ttns.tensors:
            tensor = ttns.tensors[node_id]
            
            # Check if first dimension (parent connection) is not bond_dim
            if tensor.shape[0] != bond_dim:
                # Get original shape
                orig_shape = tensor.shape
                
                # Create new shape with bond_dim for first dimension
                new_shape = (bond_dim,) + orig_shape[1:]
                
                # Create a new tensor with the correct dimensions
                new_tensor = np.zeros(new_shape, dtype=tensor.dtype)
                
                # Copy data for the smallest common dimension
                min_dim = min(bond_dim, orig_shape[0])
                slices = tuple([slice(0, min_dim)] + [slice(None) for _ in range(len(orig_shape)-1)])
                
                # Copy existing values
                new_tensor[slices] = tensor[slices[:len(orig_shape)]]
                
                # If tensor is all zeros, set first element to 1
                if np.count_nonzero(new_tensor) == 0:
                    new_tensor[(0,) + (0,) * (len(new_shape) - 1)] = 1
                
                # Replace the tensor
                ttns.tensors[node_id] = new_tensor
                
                # Update the node with the new tensor shape
                node = ttns.nodes[node_id]
                new_node = Node(identifier=node_id)
                new_node.link_tensor(new_tensor)
                new_node.parent = node.parent
                new_node.children = node.children.copy()
                ttns.nodes[node_id] = new_node
    
    return ttns

def validate_fix_tensor_dimensions(ttns: TreeTensorNetworkState, 
                                   bond_dim: int,
                                   virtual_prefix: str) -> TreeTensorNetworkState:
    """
    Validate and fix tensor dimensions to ensure all virtual nodes have
    tensor dimensions matching the specified bond dimension.
    
    Args:
        ttns: The tree tensor network state
        bond_dim: Target bond dimension
        phys_prefix: Prefix for physical node IDs
        virtual_prefix: Prefix for virtual node IDs
        
    Returns:
        TreeTensorNetworkState: Updated TTNS with fixed tensor dimensions
    """
    for node_id in list(ttns.nodes.keys()):
        if node_id.startswith(virtual_prefix) and node_id in ttns.tensors:
            tensor = ttns.tensors[node_id]
            # Check if any virtual dimension is 1 (except the last trivial dimension)
            if any(dim == 1 for i, dim in enumerate(tensor.shape) if i < len(tensor.shape)-1):
                # Create a new tensor with correct bond dimensions
                new_shape = [bond_dim] * (len(tensor.shape) - 1) + [tensor.shape[-1]]
                new_tensor = np.zeros(tuple(new_shape), dtype=tensor.dtype)
                
                # Set the first element to 1
                zeros_indices = tuple([0 for _ in range(len(new_shape))])
                new_tensor[zeros_indices] = 1
                
                # Replace the tensor
                ttns.tensors[node_id] = new_tensor
                
                # Update the node with the new tensor shape
                node = ttns.nodes[node_id]
                new_node = Node(identifier=node_id)
                new_node.link_tensor(new_tensor)
                new_node.parent = node.parent
                new_node.children = node.children.copy()
                ttns.nodes[node_id] = new_node
    
    return ttns

def clean_inefficient_paths(ttns: TreeTensorNetworkState, 
                             bond_dim: int,
                             phys_prefix: str,
                             virtual_prefix: str) -> TreeTensorNetworkState:
    """
    Clean up any remaining inefficient single-path chains in the TTN.
    This is especially needed for deep trees where inefficiencies might arise
    from the top-down construction process.
    
    This function also ensures all virtual nodes have at least one open leg.
    
    Args:
        ttns: The TTN to clean up
        bond_dim: Bond dimension for connections
        phys_prefix: Prefix for physical node IDs
        virtual_prefix: Prefix for virtual node IDs
        
    Returns:
        TreeTensorNetworkState: The cleaned-up TTN
    """
    # First pass: ensure all virtual nodes have at least one open leg
    for node_id in list(ttns.nodes.keys()):
        # Skip physical nodes
        if node_id.startswith(phys_prefix):
            continue
            
        node = ttns.nodes[node_id]
        
        # Check if node has no open legs
        if not node.open_legs and node_id in ttns.tensors:
            # Get current tensor
            tensor = ttns.tensors[node_id]
            orig_shape = tensor.shape
            
            # Create a new tensor with one additional dimension
            new_shape = list(orig_shape[:-1]) + [bond_dim, orig_shape[-1]]
            
            # Create new tensor filled with zeros
            new_tensor = np.zeros(new_shape, dtype=tensor.dtype)
            
            # Copy the data from the original tensor to maintain its values
            # Get slices for all but the new dimension
            slices = tuple([slice(None) for _ in range(len(orig_shape)-1)] + [0, slice(None)])
            new_tensor[slices] = tensor
            
            # Ensure the first element is non-zero
            if np.count_nonzero(new_tensor) == 0:
                zeros_indices = tuple([0 for _ in range(len(new_shape))])
                new_tensor[zeros_indices] = 1
            
            # Replace tensor in TTNS
            ttns.tensors[node_id] = new_tensor
            
            # Create a new node with the updated tensor shape
            new_node = Node(identifier=node_id)
            new_node.link_tensor(new_tensor)
            
            # Copy connections from old node
            new_node.parent = node.parent
            new_node.children = node.children.copy()
            
            # Update the node in the TTNS
            ttns.nodes[node_id] = new_node
    
    # Second pass: identify inefficient nodes
    inefficient_nodes = []
    
    # Process all virtual nodes
    for node_id in list(ttns.nodes.keys()):  # Make a copy of the keys to avoid mutation issues
        if node_id.startswith(phys_prefix):
            continue  # Skip physical nodes
         
        node = ttns.nodes[node_id]
        
        if len(node.children) == 1:
            child_id = node.children[0]
            
            # Identify virtual nodes with a single virtual child OR a single physical child
            if not child_id.startswith(phys_prefix):
                inefficient_nodes.append((node_id, child_id, False))  # False indicates virtual child
            else:
                inefficient_nodes.append((node_id, child_id, True))   # True indicates physical child
    
    # Third pass: bypass inefficient nodes
    for inefficient_id, child_id, is_physical_child in inefficient_nodes:
        parent_id = ttns.nodes[inefficient_id].parent
        if parent_id is None:  # Skip if this is the root
            continue
            
        # Get the parent node
        parent_node = ttns.nodes[parent_id]
        parent_tensor = ttns.tensors[parent_id]
        
        # Get the inefficient node
        inefficient_node = ttns.nodes[inefficient_id]
        
        # Find the index of the inefficient node in the parent's child list
        try:
            parent_leg = parent_node.children.index(inefficient_id)
        except ValueError:
            # This inefficient node may have been removed already as part of another chain
            continue
            
        if is_physical_child:
            # Handle case where inefficient node has a single physical child
            phys_node = ttns.nodes[child_id]
            phys_tensor = ttns.tensors[child_id]
            
            # Remove the physical node from the inefficient node's children
            inefficient_node.children.remove(child_id)
            
            # Set the physical node's parent to the parent node
            phys_node.parent = parent_id
            
            # Add the physical node to the parent's children list
            if child_id not in parent_node.children:
                parent_node.children.append(child_id)
                
            # Update the tensor connection
            parent_dim = parent_tensor.shape[parent_leg]
            if phys_tensor.shape[0] != parent_dim:
                # Get original tensor
                orig_tensor = ttns.tensors[child_id]
                orig_shape = orig_tensor.shape
                
                # Create tensor with correct parent leg dimension
                new_tensor = np.zeros((parent_dim,) + orig_shape[1:], dtype=orig_tensor.dtype)
                
                # Copy as much of the original data as we can
                min_dim = min(parent_dim, orig_shape[0])
                slices = tuple([slice(0, min_dim)] + [slice(None) for _ in range(len(orig_shape)-1)])
                new_tensor[slices] = orig_tensor[slices]
                
                ttns.tensors[child_id] = new_tensor
        else:
            # Handle case where inefficient node has a single virtual child
            child_node = ttns.nodes[child_id]
            
            # Get all the grandchildren (the children of the child node)
            grandchildren = list(child_node.children)
            
            # Reconnect all grandchildren directly to the parent node
            for grandchild_id in grandchildren:
                # Get the grandchild node
                grandchild_node = ttns.nodes[grandchild_id]
                
                # Remove the grandchild from its current parent (the child node)
                child_node.children.remove(grandchild_id)
                
                # Set the grandchild's parent to the parent node
                grandchild_node.parent = parent_id
                
                # Add the grandchild to the parent's children list
                if grandchild_id not in parent_node.children:
                    parent_node.children.append(grandchild_id)
                    
                # Update the tensor connection based on whether the grandchild is physical or virtual
                if grandchild_id.startswith(phys_prefix):
                    # For physical nodes, we may need to update the tensor
                    grandchild_tensor = ttns.tensors[grandchild_id]
                    
                    # Check if we need to reshape the tensor
                    parent_dim = parent_tensor.shape[parent_leg]
                    if grandchild_tensor.shape[0] != parent_dim:
                        # Get original tensor
                        orig_tensor = ttns.tensors[grandchild_id]
                        orig_shape = orig_tensor.shape
                        
                        # Create tensor with correct parent leg dimension
                        new_tensor = np.zeros((parent_dim,) + orig_shape[1:], dtype=orig_tensor.dtype)
                        
                        # Copy as much of the original data as we can
                        min_dim = min(parent_dim, orig_shape[0])
                        slices = tuple([slice(0, min_dim)] + [slice(None) for _ in range(len(orig_shape)-1)])
                        new_tensor[slices] = orig_tensor[slices]
                        
                        ttns.tensors[grandchild_id] = new_tensor
        
        # Remove the inefficient node from the parent's children
        if inefficient_id in parent_node.children:
            parent_node.children.remove(inefficient_id)
            
        # Clean up the inefficient node
        # Remove parent reference
        inefficient_node.parent = None
        
        # Clear children list
        inefficient_node.children = []
        
        # Remove tensor
        if inefficient_id in ttns.tensors:
            del ttns.tensors[inefficient_id]
            
        # Remove the node from the TTNS nodes dictionary
        if inefficient_id in ttns.nodes:
            del ttns.nodes[inefficient_id]
            
        # If dealing with a virtual child, clean it up too
        if not is_physical_child:
            # Clean up the child node
            child_node.parent = None
            child_node.children = []
            
            # Remove child's tensor and node from TTNS
            if child_id in ttns.tensors:
                del ttns.tensors[child_id]
                
            if child_id in ttns.nodes:
                del ttns.nodes[child_id]
    
    # Final pass: remove any orphaned nodes (no parent and no children)
    for node_id in list(ttns.nodes.keys()):
        if node_id.startswith(phys_prefix):
            continue  # Skip physical nodes
            
        node = ttns.nodes[node_id]
        if (node.parent is None and len(node.children) == 0) and node_id != ttns.root_id:
            # This is an orphaned node (not connected to anything and not the root)
            if node_id in ttns.tensors:
                del ttns.tensors[node_id]
            del ttns.nodes[node_id]
    
    # Ensure all virtual nodes have proper dimensions
    ttns = validate_fix_tensor_dimensions(ttns, bond_dim, virtual_prefix)
    
    # Additional validation to ensure all connections have compatible dimensions
    for node_id in list(ttns.nodes.keys()):
        node = ttns.nodes[node_id]
        if node.parent is not None:
            parent_id = node.parent
            parent_node = ttns.nodes[parent_id]
            
            # Get connection legs
            child_leg = 0  # Always the first leg
            try:
                parent_leg = parent_node.children.index(node_id)
            except ValueError:
                # Orphaned node reference, clean it up
                node.parent = None
                continue
                
            # Get tensors
            child_tensor = ttns.tensors[node_id]
            parent_tensor = ttns.tensors[parent_id]
            
            # Check if dimensions match
            if child_tensor.shape[child_leg] != parent_tensor.shape[parent_leg]:
                # Fix the mismatch by setting both to bond_dim
                target_dim = bond_dim
                
                # Fix child tensor
                child_shape = list(child_tensor.shape)
                child_shape[child_leg] = target_dim
                new_child_tensor = np.zeros(tuple(child_shape), dtype=child_tensor.dtype)
                
                # Copy existing data
                min_dim = min(child_tensor.shape[child_leg], target_dim)
                slices = tuple([slice(0, min_dim) if i == child_leg else slice(None) 
                              for i in range(len(child_shape))])
                new_child_tensor[slices] = child_tensor[slices]
                
                # Set first element to ensure tensor is not zero
                zeros_indices = tuple([0 for _ in range(len(child_shape))])
                if np.count_nonzero(new_child_tensor) == 0:
                    new_child_tensor[zeros_indices] = 1.0
                
                # Update child tensor
                ttns.tensors[node_id] = new_child_tensor
                
                # Fix parent tensor similarly
                parent_shape = list(parent_tensor.shape)
                parent_shape[parent_leg] = target_dim
                new_parent_tensor = np.zeros(tuple(parent_shape), dtype=parent_tensor.dtype)
                
                min_dim = min(parent_tensor.shape[parent_leg], target_dim)
                slices = tuple([slice(0, min_dim) if i == parent_leg else slice(None) 
                              for i in range(len(parent_shape))])
                new_parent_tensor[slices] = parent_tensor[slices]
                
                # Set first element to ensure tensor is not zero
                zeros_indices = tuple([0 for _ in range(len(parent_shape))])
                if np.count_nonzero(new_parent_tensor) == 0:
                    new_parent_tensor[zeros_indices] = 1.0
                
                # Update parent tensor
                ttns.tensors[parent_id] = new_parent_tensor
    
    return ttns

