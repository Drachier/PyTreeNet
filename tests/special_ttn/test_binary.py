import unittest
from numpy import allclose, count_nonzero, isclose, array, ndarray, nonzero, testing
from pytreenet.special_ttn.binary import generate_binary_ttns
from pytreenet.operators.common_operators import ket_i

def is_virtual_node(node_id: str) -> bool:
    """Check if a node is a virtual node (not a physical node)."""
    return not node_id.startswith("qubit")


def has_sparse_identity(tensor: ndarray) -> bool:
    """
    Virtual nodes should contain only a single non-zero element equal to 1.0 at index [0,...,0].
    """
    non_zero = count_nonzero(tensor)
    if non_zero != 1:
        return False
    idx = tuple(0 for _ in range(tensor.ndim))
    return isclose(tensor[idx], 1.0)


class TestGenerateBinaryTTNS(unittest.TestCase):
    """
    Tests for generate_binary_ttns function, verifying structure of the generated TTNS:
      - One open leg per node
      - Physical tensors correctly embedded
      - Virtual nodes have sparse identity structure
      - Non-open legs have the correct bond dimension    """
    def setUp(self):
        # test configurations: (num_phys, bond_dim, depth)
        self.configs = [
            (100, 7, 2),  # Deeper binary tree
            (50, 4, 2),   # Moderate bond dimension and depth.
            (23, 5, 3),  # Shallow tree with minimal bond dimension
            (12, 4, 2),   # Shallow tree with larger bond dimension
        ]
        # base physical tensor (simple 1D vector)
        self.base_phys = array([1.0-2j, -1.0j +2], dtype=complex)

    def print_ttns_info(self, ttns):
        """Print general information about the TTNS structure."""
        num_nodes = len(ttns.nodes)
        physical_nodes = [n for n in ttns.nodes.keys() if not is_virtual_node(n)]
        virtual_nodes = [n for n in ttns.nodes.keys() if is_virtual_node(n)]
        
        print(f"\nTTNS Structure Summary:")
        print(f"  - Total nodes: {num_nodes}")
        print(f"  - Physical nodes: {len(physical_nodes)}")
        print(f"  - Virtual nodes: {len(virtual_nodes)}")
        print(f"  - Root node: {ttns.root_id}")
        
        # Print all node connections
        print("\nNode Connections:")
        for node_id, node in ttns.nodes.items():
            node_type = "Physical" if not is_virtual_node(node_id) else "Virtual"
            parent = node.parent if node.parent else "None (Root)"
            children = node.children if node.children else "None (Leaf)"
            tensor_shape = ttns.tensors[node_id].shape if node_id in ttns.tensors else "No tensor"
            open_legs = node.open_legs if node.open_legs else "None"
            print(f"  {node_id} ({node_type}):")
            print(f"    - Parent: {parent}")
            print(f"    - Children: {children}")
            print(f"    - Tensor shape: {tensor_shape}")
            print(f"    - Open legs: {open_legs}")

    def test_trivial_generation(self):
        """
        Tests the generation of a binary TTNS with a single node.
        """
        ttns = generate_binary_ttns(1,
                                    2,
                                    ket_i(0,3)
                                    )
        self.assertEqual(1, len(ttns.nodes))
        self.assertEqual(1, len(ttns.tensors))
        self.assertTrue(allclose(ttns.tensors[ttns.root_id],
                                 ket_i(0,3)))

    def test_binary_ttns_structure(self):
        for num_phys, bond_dim, depth in self.configs:
            with self.subTest(num_phys=num_phys, bond_dim=bond_dim, depth=depth):
                print(f"\n\n======== Testing TTNS with {num_phys} physical nodes, bond_dim={bond_dim}, depth={depth} ========\n")
                
                # generate TTNS
                ttns = generate_binary_ttns(
                    num_phys=num_phys,
                    bond_dim=bond_dim,
                    phys_tensor=self.base_phys,
                    depth=depth
                )
                
                # Print general TTNS information
                self.print_ttns_info(ttns)
                
                # 1. Every node has exactly one open leg
                print("\n\n----- CHECK 1: Verifying every node has exactly one open leg -----")
                open_legs_status = {}
                for node_id, node in ttns.nodes.items():
                    open_legs_count = len(node.open_legs)
                    open_legs_status[node_id] = open_legs_count
                    
                    # Print information about each node's open legs
                    node_type = "Physical" if not is_virtual_node(node_id) else "Virtual"
                    status = "✓" if open_legs_count == 1 else "✗"
                    print(f"  {status} {node_id} ({node_type}): {open_legs_count} open leg(s)")
                    
                    # Assert and provide detailed error message if needed
                    self.assertEqual(
                        open_legs_count,
                        1,
                        f"Node {node_id} has {open_legs_count} open legs, expected exactly 1"
                    )
                    
                # Additional summary
                failed_nodes = [n for n, count in open_legs_status.items() if count != 1]
                if failed_nodes:
                    print(f"\n  Failed nodes: {failed_nodes}")
                else:
                    print("\n  All nodes have exactly one open leg ✓")
                
                # 1a. Open-leg dimension: virtual=1, physical=len(base_phys)
                print("\n\n----- CHECK 1a: Verifying open leg dimensions -----")
                for node_id, node in ttns.nodes.items():
                    tensor = ttns.tensors[node_id]
                    leg = node.open_legs[0]
                    expected_dim = 1 if is_virtual_node(node_id) else self.base_phys.size
                    actual_dim = tensor.shape[leg]
                    
                    node_type = "Physical" if not is_virtual_node(node_id) else "Virtual"
                    status = "✓" if actual_dim == expected_dim else "✗"
                    print(f"  {status} {node_id} ({node_type}): Open leg dimension {actual_dim}, expected {expected_dim}")
                    
                    self.assertEqual(
                        actual_dim,
                        expected_dim,
                        f"Node {node_id} open leg dimension {actual_dim} != {expected_dim}"
                    )
                
                # 2. Physical tensor values preserved
                print("\n\n----- CHECK 2: Verifying physical tensor values preserved -----")
                for i in range(num_phys):
                    phys_id = f"qubit{i}"
                    
                    # Check if physical node exists
                    exists = phys_id in ttns.tensors
                    status = "✓" if exists else "✗"
                    print(f"  {status} Physical node {phys_id}: {'Present' if exists else 'Missing'}")
                    
                    self.assertIn(
                        phys_id,
                        ttns.tensors,
                        f"Physical node {phys_id} missing"
                    )
                    
                    tensor = ttns.tensors[phys_id]
                    # physical index is the last axis
                    # verify the first slice matches base_phys
                    last_axis = tensor.ndim - 1
                    # take slice at zeros for all other axes
                    idx = [0] * last_axis + [slice(None)]
                    slice_vals = tensor[tuple(idx)]
                    # compare to base_phys (truncated or padded)
                    size = min(slice_vals.shape[0], self.base_phys.size)
                    
                    # Check if values match
                    match = allclose(slice_vals[:size], self.base_phys[:size])
                    status = "✓" if match else "✗"
                    print(f"  {status} {phys_id} tensor values at first slice: {slice_vals}")
                    
                    testing.assert_allclose(
                        slice_vals[:size],
                        self.base_phys[:size],
                        err_msg=f"Physical tensor mismatch at {phys_id}"
                    )
                
                # 2a. Physical node padding: non-physical axes beyond index 0 are zero
                print("\n\n----- CHECK 2a: Verifying physical node padding -----")
                padding_issues = []
                for i in range(num_phys):
                    phys_id = f"qubit{i}"
                    tensor = ttns.tensors[phys_id]
                    
                    # Check each axis except the last (physical dimension)
                    for axis in range(tensor.ndim - 1):
                        slicer = [slice(None)] * tensor.ndim
                        slicer[axis] = slice(1, None)
                        sub = tensor[tuple(slicer)]
                        
                        # Check if all values are zero
                        is_zero = allclose(sub, 0)
                        status = "✓" if is_zero else "✗"
                        
                        if not is_zero:
                            padding_issues.append(f"{phys_id} axis {axis}")
                            print(f"  {status} {phys_id} axis {axis} has non-zero values beyond index 0:")
                            print(f"    Values: {sub}")
                        else:
                            print(f"  {status} {phys_id} axis {axis} padding verified")
                        
                        self.assertTrue(
                            is_zero,
                            f"Physical node {phys_id} has non-zero values beyond index 0 on axis {axis}"
                        )
                
                # Summary of padding issues
                if padding_issues:
                    print(f"\n  Nodes with padding issues: {padding_issues}")
                else:
                    print("\n  All physical nodes have correct zero padding ✓")
                
                # 3. Virtual nodes have sparse identity structure
                print("\n\n----- CHECK 3: Verifying virtual nodes have sparse identity structure -----")
                non_sparse_nodes = []
                for node_id, tensor in ttns.tensors.items():
                    if is_virtual_node(node_id):
                        is_sparse = has_sparse_identity(tensor)
                        status = "✓" if is_sparse else "✗"
                        
                        if not is_sparse:
                            non_sparse_nodes.append(node_id)
                            print(f"  {status} {node_id} is not a sparse identity tensor")
                            
                            # Print more details to aid debugging
                            non_zero_count = count_nonzero(tensor)
                            idx = tuple(0 for _ in range(tensor.ndim))
                            print(f"    - Non-zero elements: {non_zero_count}")
                            print(f"    - Value at [0,...,0]: {tensor[idx]}")
                            
                            # Find positions of non-zero elements
                            non_zero_positions = nonzero(tensor)
                            if len(non_zero_positions[0]) < 10:  # Only show if not too many
                                positions = list(zip(*non_zero_positions))
                                values = [tensor[pos] for pos in positions]
                                print(f"    - Non-zero positions and values: {list(zip(positions, values))}")
                        else:
                            print(f"  {status} {node_id} has correct sparse identity structure")
                        
                        self.assertTrue(
                            is_sparse,
                            f"Virtual node {node_id} tensor is not sparse identity"
                        )
                
                # Summary of non-sparse nodes
                if non_sparse_nodes:
                    print(f"\n  Nodes without sparse identity: {non_sparse_nodes}")
                else:
                    print("\n  All virtual nodes have correct sparse identity structure ✓")
                
                # 4. Non-open legs have correct bond dimension
                print("\n\n----- CHECK 4: Verifying non-open legs have correct bond dimension -----")
                dimension_issues = []
                for node_id, node in ttns.nodes.items():
                    tensor = ttns.tensors[node_id]
                    node_type = "Physical" if not is_virtual_node(node_id) else "Virtual"
                    
                    print(f"  {node_id} ({node_type}): Checking dimensions")
                    for axis, dim in enumerate(tensor.shape):
                        if axis in node.open_legs:
                            # Skip open legs as they're checked separately
                            continue
                        
                        # Check if dimension matches bond_dim
                        matches = dim == bond_dim
                        status = "✓" if matches else "✗"
                        
                        print(f"    - Axis {axis}: dimension {dim}, expected {bond_dim} - {status}")
                        
                        if not matches:
                            dimension_issues.append(f"{node_id} axis {axis}")
                        
                        self.assertEqual(
                            dim,
                            bond_dim,
                            f"Node {node_id} axis {axis} has dimension {dim}, expected {bond_dim}"
                        )
                
                # Summary of dimension issues
                if dimension_issues:
                    print(f"\n  Nodes with incorrect dimensions: {dimension_issues}")
                else:
                    print("\n  All non-open legs have correct bond dimension ✓")
                
                # Overall test summary
                print("\n\n=========== TTNS Structure Test Summary ===========")
                print(f"  - Configuration: {num_phys} physical nodes, bond_dim={bond_dim}, depth={depth}")
                print(f"  - All checks passed: ✓")


if __name__ == '__main__':
    unittest.main()
