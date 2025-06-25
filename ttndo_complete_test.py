"""
Comprehensive test script for TTNDO structures.
Tests different qubit counts and depths while verifying:
1. Each node has exactly one open leg
2. Physical tensor values are preserved correctly
3. Virtual nodes have the correct sparse identity structure
4. Bond dimensions are correct: virtual open legs = 1, physical open legs = 2, all other virtual bonds = bond_dim
5. The trace of the symmetric TTNDO equals 1 (proper normalization)
6. Contracted TTNDO has correct open legs:
   - For fully binary: All nodes have exactly two open legs (representing density matrix indices)
   - For physically binary: Physical nodes have two open legs, virtual nodes maintain one open leg
7. Bond dimensions in contracted TTNDO match the expected bond dimension
8. The trace of contracted TTNDO equals 1 (proper normalization)
9. Expectation values for symmetric and binary TTNDOs match for both TTNO and tensor product operators

The test works with several different network structures:
- Binary TTNS: Generated using generate_binary_ttns with specified qubit counts and depths
- Symmetric TTNDO: Created from the TTNS using from_symmetric_ttns, which builds a symmetric structure with a central root
- Fully binary TTNDO: Created from the TTNS using from_ttns_fully_binary, where all nodes (virtual and physical) have bra/ket pairs
- Physically binary TTNDO: Created using from_ttns_physically_binary, which uses dual representation only for physical nodes
"""

import numpy as np
from pytreenet.special_ttn.binary import generate_binary_ttns
from pytreenet.ttns.ttndo import from_ttns_fully_binary, from_symmetric_ttns, from_ttns_physically_binary, contract_physical_nodes, SymmetricTTNDO
from pytreenet.contractions.ttndo_contractions import (trace_symmetric_ttndo, 
                                                        trace_contracted_fully_binary_ttndo,
                                                        fully_binary_ttndo_ttno_expectation_value,
                                                        trace_contracted_physically_binary_ttndo,
                                                        physically_binary_ttndo_ttno_expectation_value)
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator

BOND_DIM = 5

def is_virtual_node(node_id):
    """Check if a node is a virtual node (not a physical node)."""
    return not (node_id.startswith("qubit") or node_id.startswith("site"))

def has_correct_sparse_structure(tensor):
    """
    Check if tensor has the correct sparse structure with only first element = 1.0.
    This is the expected structure for virtual nodes in the identity network.
    """
    # Count non-zero elements
    non_zero_count = np.count_nonzero(tensor)
    
    if non_zero_count != 1:
        return False
    
    # Check if only the first element [0,0,...,0] is non-zero and equals 1
    first_element_index = tuple([0] * len(tensor.shape))
    first_element = tensor[first_element_index]
    
    return np.abs(first_element - 1.0) < 1e-10

def is_identity_matrix(tensor):
    """
    Check if the given tensor is an identity matrix when sliced properly.
    Used specifically for checking ttndo_root node.
    """
    # For 3D tensors like (dim, dim, 1)
    if tensor.ndim == 3 and tensor.shape[2] == 1:
        matrix = tensor[:,:,0]
        identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
        return np.allclose(matrix, identity)
    return False

def check_bond_dimensions(ttns, structure_name):
    """
    Check if the bond dimensions are correct:
    - Virtual open legs = 1
    - Physical open legs = 2
    - All other virtual bonds = BOND_DIM
    
    Special cases:
    - ttndo_root in symmetric TTNDO has fixed dimensions of 2
    
    Returns a tuple of (success, error_message)
    """
    # Get all tensors and nodes
    tensors = ttns.tensors
    nodes = ttns.nodes
    
    for node_id, node in nodes.items():
        tensor = tensors[node_id]
        
        # Check open leg dimension
        for open_leg_idx in node.open_legs:
            if is_virtual_node(node_id):
                # Virtual node open leg should be dimension 1
                if tensor.shape[open_leg_idx] != 1:
                    return False, f"{structure_name} node {node_id}: open leg {open_leg_idx} has dimension {tensor.shape[open_leg_idx]} (expected 1)"
            else:
                # Physical node open leg should be dimension 2
                if tensor.shape[open_leg_idx] != 2:
                    return False, f"{structure_name} node {node_id}: physical open leg {open_leg_idx} has dimension {tensor.shape[open_leg_idx]} (expected 2)"
        
        # Check non-open leg dimensions for virtual nodes
        if is_virtual_node(node_id):
            # Special case for ttndo_root node in symmetric TTNDO
            if node_id == "ttndo_root" and structure_name == "Symmetric TTNDO":
                # ttndo_root has fixed dimensions and does not need to follow the BOND_DIM rule
                continue
                
            for i in range(len(tensor.shape)):
                # Skip open legs, which we already checked
                if i in node.open_legs:
                    continue
                    
                # All other dimensions for virtual nodes should be BOND_DIM
                if tensor.shape[i] != BOND_DIM:
                    return False, f"{structure_name} node {node_id}: non-open leg {i} has dimension {tensor.shape[i]} (expected {BOND_DIM})"
    
    return True, ""

def check_contracted_ttndo_bond_dimensions(ttndo, bond_dim):
    """
    Check if the bond dimensions in a contracted binary TTNDO match the expected bond dimension.
    
    Args:
        ttndo: The contracted binary TTNDO
        bond_dim: The expected bond dimension
        
    Returns:
        tuple: (success, error_message)
    """
    tensors = ttndo.tensors
    nodes = ttndo.nodes
    
    for node_id, node in nodes.items():
        tensor = tensors[node_id]
        
        # Virtual node connections should have bond_dim
        if is_virtual_node(node_id):
            for i in range(len(tensor.shape)):
                # Skip the last two dimensions (physical legs)
                if i >= len(tensor.shape) - 2:
                    continue
                
                # Connection legs should match bond_dim
                if tensor.shape[i] != bond_dim:
                    return False, f"Contracted TTNDO node {node_id}: leg {i} has dimension {tensor.shape[i]} (expected {bond_dim})"
    
    return True, ""

def check_contracted_ttndo_open_legs(ttndo):
    """
    Check if nodes in a contracted TTNDO have the correct number of open legs.
    
    For fully binary TTNDOs:
    - All nodes should have exactly two open legs
    - Open legs should be the last two indices
    
    For physically binary TTNDOs (with form = "physical"):
    - Physical nodes should have exactly two open legs
    - Virtual nodes should maintain their single open leg
    - Open legs should be the last indices of each node
    
    Args:
        ttndo: The contracted TTNDO
        
    Returns:
        tuple: (success, error_message)
    """
    nodes = ttndo.nodes
    is_physical_form = hasattr(ttndo, 'form') and ttndo.form == "physical"
    
    for node_id, node in nodes.items():
        # For physically binary TTNDOs, virtual nodes have one open leg
        if is_physical_form and is_virtual_node(node_id):
            # Virtual nodes should have exactly one open leg
            if len(node.open_legs) != 1:
                return False, f"Contracted physical form TTNDO virtual node {node_id} has {len(node.open_legs)} open legs (expected 1)"
            
            # The open leg should be the last index
            expected_open_leg = [len(ttndo.tensors[node_id].shape) - 1]
            if node.open_legs != expected_open_leg:
                return False, f"Contracted physical form TTNDO virtual node {node_id} has open leg {node.open_legs} (expected {expected_open_leg})"
        else:
            # Physical nodes (or all nodes in fully binary form) should have exactly two open legs
            if len(node.open_legs) != 2:
                return False, f"Contracted TTNDO node {node_id} has {len(node.open_legs)} open legs (expected 2)"
            
            # The open legs should be the last two indices
            expected_open_legs = [len(ttndo.tensors[node_id].shape) - 2, len(ttndo.tensors[node_id].shape) - 1]
            if node.open_legs != expected_open_legs:
                return False, f"Contracted TTNDO node {node_id} has open legs {node.open_legs} (expected {expected_open_legs})"
    
    return True, ""

def print_node_info(ttn, node_id, prefix=""):
    """Helper function to print node information for debugging."""
    node, tensor = ttn[node_id]
    print(f"{prefix}Node: {node_id}")
    print(f"{prefix}  Shape: {tensor.shape}")
    print(f"{prefix}  Parent: {node.parent}")
    print(f"{prefix}  Children: {node.children}")
    print(f"{prefix}  Open legs: {node.open_legs}")

def test_network_structure(ttns, ttndo_binary, ttndo_symmetric, phys_tensor, num_phys, depth):
    """    
    This function tests three different network structures:
    
    1. Binary TTNS (generated with generate_binary_ttns):
       - Each node has exactly one open leg
       - Physical tensor values are correctly preserved from input tensor
       - Virtual nodes have the correct sparse identity structure (only [0,0,...,0] = 1.0)
       - Bond dimensions are correct (virtual open legs = 1, physical open legs = 2, 
         other virtual bonds = bond_dim)
    
    2. Symmetric TTNDO (created from TTNS using from_symmetric_ttns):
       - Each node has exactly one open leg
       - Physical tensor values are correctly preserved from input tensor
       - Virtual nodes have the correct sparse identity structure
       - The root node (ttndo_root) has the identity matrix structure
       - Bond dimensions are correct (with special case for ttndo_root)
       - The trace equals 1.0 (proper normalization)
    
    3. binary TTNDO (created from TTNS using from_ttns_fully_binary):
       - Each node has exactly one open leg in original structure
       - Physical tensor values are correctly preserved (for both bra and ket parts)
       - Virtual nodes have the correct sparse identity structure
       - Bond dimensions are correct
       - When contracted:
           - Each node has exactly two open legs (representing density matrix indices)
           - Bond dimensions match expected values
           - The trace equals 1.0 (proper normalization)
    
    4. Physically binary TTNDO (created from TTNS using from_ttns_physically_binary):
       - Each node has exactly one open leg in original structure
       - Only physical nodes have dual representation (bra/ket), virtual nodes have single tensors
       - Physical tensors are correctly preserved, and bra tensors are complex conjugate of ket tensors
       - Virtual nodes have the correct sparse identity structure
       - Bond dimensions are correctly padded to have right bond_dim
       - When contracted:
           - Physical nodes have exactly two open legs (representing density matrix indices)
           - Virtual nodes maintain their single open leg
           - Bond dimensions match expected values
           - The trace equals 1.0 (proper normalization)
           - Both direct trace_contracted_physically_binary_ttndo function and class trace() method work correctly
           - TTNO expectation values match symmetric and fully binary TTNDOs
           - Tensor product expectation values are consistent across all TTNDO types
    
    Args:
        ttns: The tree tensor network state
        ttndo_binary: The binary TTNDO structure
        ttndo_symmetric: The symmetric TTNDO structure
        phys_tensor: The physical tensor used to initialize the networks
        num_phys: Number of physical qubits/sites
        depth: Depth of the tree structure
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"TESTING WITH {num_phys} QUBITS AT DEPTH {depth}")
    print(f"{'='*80}")
    
    # Create physically binary TTNDO
    print("\nCreating physically binary TTNDO...")
    ttndo_physically_binary = from_ttns_physically_binary(ttns, BOND_DIM, phys_tensor)
    print(f"Physically binary TTNDO created with {len(ttndo_physically_binary.nodes)} nodes")
    
    # Count nodes in each network
    ttns_nodes = len(ttns.nodes)
    ttndo_binary_nodes = len(ttndo_binary.nodes)
    ttndo_symmetric_nodes = len(ttndo_symmetric.nodes)
    ttndo_physically_binary_nodes = len(ttndo_physically_binary.nodes)
    
    print(f"TTNS has {ttns_nodes} nodes")
    print(f"binary TTNDO has {ttndo_binary_nodes} nodes")
    print(f"Symmetric TTNDO has {ttndo_symmetric_nodes} nodes")
    print(f"Physically binary TTNDO has {ttndo_physically_binary_nodes} nodes")
    
    # 1. Check that all nodes have exactly one open leg
    print("\nTesting open legs...")
    open_legs_correct = True
    
    try:
        for node in ttns.nodes:
            assert len(ttns.nodes[node].open_legs) == 1, f"TTNS node {node} has {len(ttns.nodes[node].open_legs)} open legs (should be 1)"
        print("✓ All TTNS nodes have exactly 1 open leg")
    except AssertionError as e:
        print(f"✗ {e}")
        open_legs_correct = False
    
    try:
        for node in ttndo_symmetric.nodes:
            assert len(ttndo_symmetric.nodes[node].open_legs) == 1, f"Symmetric TTNDO node {node} has {len(ttndo_symmetric.nodes[node].open_legs)} open legs (should be 1)"
        print("✓ All symmetric TTNDO nodes have exactly 1 open leg")
    except AssertionError as e:
        print(f"✗ {e}")
        open_legs_correct = False
    
    try:
        for node in ttndo_binary.nodes:
            assert len(ttndo_binary.nodes[node].open_legs) == 1, f"binary TTNDO node {node} has {len(ttndo_binary.nodes[node].open_legs)} open legs (should be 1)"
        print("✓ All binary TTNDO nodes have exactly 1 open leg")
    except AssertionError as e:
        print(f"✗ {e}")
        open_legs_correct = False
    
    try:
        for node in ttndo_physically_binary.nodes:
            assert len(ttndo_physically_binary.nodes[node].open_legs) == 1, f"Physically binary TTNDO node {node} has {len(ttndo_physically_binary.nodes[node].open_legs)} open legs (should be 1)"
        print("✓ All physically binary TTNDO nodes have exactly 1 open leg")
    except AssertionError as e:
        print(f"✗ {e}")
        open_legs_correct = False
    
    # 2. Check if physical tensor values are preserved
    print("\nTesting physical tensor preservation...")
    phys_tensor_preserved = True
    
    try:
        # Check TTNS physical tensors
        for node_id in ttns.nodes:
            if node_id.startswith('qubit'):
                tensor = ttns.tensors[node_id]
                
                # Check if the values match at index 0
                if phys_tensor.ndim == 1:
                    # For 1D input tensor
                    for j in range(min(tensor.shape[1], phys_tensor.size)):
                        assert tensor[0, j] == phys_tensor[j], f"TTNS node {node_id}: mismatch at [0, {j}]: {tensor[0, j]} != {phys_tensor[j]}"
                else:
                    # For 2D input tensor
                    for j in range(min(tensor.shape[1], phys_tensor.shape[1])):
                        assert tensor[0, j] == phys_tensor[0, j], f"TTNS node {node_id}: mismatch at [0, {j}]: {tensor[0, j]} != {phys_tensor[0, j]}"
                
                # Verify zeros in other indices
                for i in range(1, tensor.shape[0]):
                    assert np.all(tensor[i] == 0), f"TTNS node {node_id}: non-zero values found at index {i}"
        
        print("✓ All TTNS physical tensors preserve input values correctly")
        
        # Check binary TTNDO physical tensors
        print("\nChecking binary TTNDO physical tensors...")
        for node_id in ttndo_binary.nodes:
            # Check ket physical nodes
            if node_id.startswith('qubit') and node_id.endswith('_ket'):
                tensor = ttndo_binary.tensors[node_id]
                
                # The tensor structure is different in binary TTNDO:
                # Physical ket tensors usually have 3 legs: parent, lateral, physical
                # Only check the physical leg values at index [0,0,:] which should match input tensor
                if tensor.ndim == 3 and tensor.shape[2] >= 1:
                    if phys_tensor.ndim == 1:
                        # For 1D input tensor
                        for j in range(min(tensor.shape[2], phys_tensor.size)):
                            assert tensor[0, 0, j] == phys_tensor[j], f"binary TTNDO ket node {node_id}: mismatch at [0,0,{j}]: {tensor[0,0,j]} != {phys_tensor[j]}"
                    else:
                        # For 2D input tensor
                        for j in range(min(tensor.shape[2], phys_tensor.shape[1])):
                            assert tensor[0, 0, j] == phys_tensor[0, j], f"binary TTNDO ket node {node_id}: mismatch at [0,0,{j}]: {tensor[0,0,j]} != {phys_tensor[0,j]}"
            
            # Check bra physical nodes
            elif node_id.startswith('qubit') and node_id.endswith('_bra'):
                tensor = ttndo_binary.tensors[node_id]
                
                # Physical bra tensors usually have 2 legs: parent, physical
                # Values should match conjugate of input tensor
                if tensor.ndim == 2:
                    if phys_tensor.ndim == 1:
                        # For 1D input tensor
                        for j in range(min(tensor.shape[1], phys_tensor.size)):
                            assert tensor[0, j] == np.conj(phys_tensor[j]), f"binary TTNDO bra node {node_id}: mismatch at [0,{j}]: {tensor[0,j]} != {np.conj(phys_tensor[j])}"
                    else:
                        # For 2D input tensor
                        for j in range(min(tensor.shape[1], phys_tensor.shape[1])):
                            assert tensor[0, j] == np.conj(phys_tensor[0, j]), f"binary TTNDO bra node {node_id}: mismatch at [0,{j}]: {tensor[0,j]} != {np.conj(phys_tensor[0,j])}"
        
        print("✓ All binary TTNDO physical tensors preserve input values correctly")
        
        # Check symmetric TTNDO physical tensors
        print("\nChecking symmetric TTNDO physical tensors...")
        # Identify physical nodes in symmetric TTNDO
        for node_id in ttndo_symmetric.nodes:
            if node_id.startswith('qubit'):
                tensor = ttndo_symmetric.tensors[node_id]
                
                # The symmetric TTNDO might have different tensor structure
                # Typical symmetric physical tensor has 3 legs, with the physical tensor at index 0
                # This checks if the physical values match the input tensor (without conjugation in symmetric structure)
                if tensor.ndim >= 2:
                    if phys_tensor.ndim == 1:
                        # For 1D input tensors
                        if tensor.shape[-1] >= phys_tensor.size:
                            match_found = False
                            # Try to find where the physical data matches in the tensor
                            for i in range(min(2, tensor.shape[0])):
                                if np.allclose(tensor[i, :phys_tensor.size], phys_tensor):
                                    match_found = True
                                    break
                                elif np.allclose(tensor[i, :phys_tensor.size], np.conj(phys_tensor)):
                                    match_found = True
                                    break
                            assert match_found, f"Symmetric TTNDO node {node_id}: could not find matching physical values"
                    else:
                        # For 2D input tensors
                        if tensor.shape[-1] >= phys_tensor.shape[1]:
                            match_found = False
                            # Try to find where the physical data matches in the tensor
                            for i in range(min(2, tensor.shape[0])):
                                if np.allclose(tensor[i, :phys_tensor.shape[1]], phys_tensor[0]):
                                    match_found = True
                                    break
                                elif np.allclose(tensor[i, :phys_tensor.shape[1]], np.conj(phys_tensor[0])):
                                    match_found = True
                                    break
                            assert match_found, f"Symmetric TTNDO node {node_id}: could not find matching physical values"
        
        print("✓ All symmetric TTNDO physical tensors preserve input values correctly")
    except AssertionError as e:
        print(f"✗ {e}")
        phys_tensor_preserved = False
    
    # 3. Check virtual node sparse structure
    print("\nTesting virtual node sparse structure...")
    sparse_structure_correct = True
    
    try:
        # Test TTNS virtual nodes
        for node_id in ttns.nodes:
            if is_virtual_node(node_id):
                tensor = ttns.tensors[node_id]
                assert has_correct_sparse_structure(tensor), f"TTNS virtual node {node_id} doesn't have correct sparse structure"
        print("✓ All TTNS virtual nodes have correct sparse structure")
    except AssertionError as e:
        print(f"✗ {e}")
        sparse_structure_correct = False
    
    try:
        # Test binary TTNDO virtual nodes
        for node_id in ttndo_binary.nodes:
            if is_virtual_node(node_id) and (node_id.endswith("_ket") or node_id.endswith("_bra")):
                tensor = ttndo_binary.tensors[node_id]
                assert has_correct_sparse_structure(tensor), f"binary TTNDO virtual node {node_id} doesn't have correct sparse structure"
        print("✓ All binary TTNDO virtual nodes have correct sparse structure")
    except AssertionError as e:
        print(f"✗ {e}")
        sparse_structure_correct = False
    
    try:
        # Test symmetric TTNDO virtual nodes
        for node_id in ttndo_symmetric.nodes:
            if is_virtual_node(node_id):
                tensor = ttndo_symmetric.tensors[node_id]
                
                # Special case for ttndo_root
                if node_id == "ttndo_root":
                    assert is_identity_matrix(tensor), f"Symmetric TTNDO root node {node_id} is not an identity matrix at [:,:,0]"
                else:
                    assert has_correct_sparse_structure(tensor), f"Symmetric TTNDO virtual node {node_id} doesn't have correct sparse structure"
                    
        print("✓ All symmetric TTNDO virtual nodes have correct structure")
    except AssertionError as e:
        print(f"✗ {e}")
        sparse_structure_correct = False
    
    # 4. Check bond dimensions
    print("\nTesting bond dimensions...")
    bond_dims_correct = True
    
    try:
        # Check TTNS bond dimensions
        success, error_msg = check_bond_dimensions(ttns, "TTNS")
        assert success, error_msg
        print("✓ All TTNS bond dimensions are correct")
    except AssertionError as e:
        print(f"✗ {e}")
        bond_dims_correct = False
        
    try:
        # Check binary TTNDO bond dimensions
        success, error_msg = check_bond_dimensions(ttndo_binary, "binary TTNDO")
        assert success, error_msg
        print("✓ All binary TTNDO bond dimensions are correct")
    except AssertionError as e:
        print(f"✗ {e}")
        bond_dims_correct = False
        
    try:
        # Check symmetric TTNDO bond dimensions
        success, error_msg = check_bond_dimensions(ttndo_symmetric, "Symmetric TTNDO")
        assert success, error_msg
        print("✓ All symmetric TTNDO bond dimensions are correct")
    except AssertionError as e:
        print(f"✗ {e}")
        bond_dims_correct = False
    
    # 5. Check that the trace of the symmetric TTNDO equals 1
    print("\nTesting symmetric TTNDO trace normalization...")
    trace_correct = True
    
    try:
        # Compute the trace of the symmetric TTNDO
        trace_value = trace_symmetric_ttndo(ttndo_symmetric)
        
        # Check if the trace is close to 1.0
        assert np.abs(trace_value - 1.0) < 1e-10, f"Symmetric TTNDO trace is {trace_value}, expected 1.0"
        print(f"✓ Symmetric TTNDO trace is {trace_value:.10f}, correctly normalized to 1.0")
    except AssertionError as e:
        print(f"✗ {e}")
        trace_correct = False
    except Exception as e:
        print(f"✗ Error computing trace: {e}")
        trace_correct = False
    
    # 6. Test contracted binary TTNDO
    print("\nTesting contracted binary TTNDO...")
    contracted_ttndo_correct = True
    
    try:
        # Contract the binary TTNDO
        print("Contracting binary TTNDO...")
        # For fully binary form, all nodes should have two open legs
        ttndo_binary_contracted = contract_physical_nodes(ttndo_binary)
        print(f"Contracted TTNDO created with {len(ttndo_binary_contracted.nodes)} nodes")
        
        # Print information about some nodes
        print("\nSample of contracted TTNDO nodes:")
        sample_nodes = list(ttndo_binary_contracted.nodes.keys())[:min(3, len(ttndo_binary_contracted.nodes))]
        for node_id in sample_nodes:
            print_node_info(ttndo_binary_contracted, node_id, prefix="  ")
        
        # 6a. Check if nodes have exactly two open legs
        print("\nChecking that all nodes have two open legs...")
        success, error_msg = check_contracted_ttndo_open_legs(ttndo_binary_contracted)
        assert success, error_msg
        print("✓ All contracted TTNDO nodes have exactly 2 open legs")
        
        # 6b. Check bond dimensions
        print("\nChecking contracted TTNDO bond dimensions...")
        success, error_msg = check_contracted_ttndo_bond_dimensions(ttndo_binary_contracted, BOND_DIM)
        assert success, error_msg
        print("✓ All contracted TTNDO bond dimensions match expected value")
        
        # 6c. Calculate trace of contracted TTNDO
        print("\nCalculating trace of contracted TTNDO...")
        trace_value = trace_contracted_fully_binary_ttndo(ttndo_binary_contracted)
        print(f"Trace result: {trace_value}")
        
        # Check if trace is close to 1.0 (for pure states)
        assert np.isclose(trace_value, 1.0), f"Expected trace 1.0 for pure state, got {trace_value}"
        print("✓ Contracted TTNDO trace is correct (1.0) for pure state")
        
    except AssertionError as e:
        print(f"✗ {e}")
        contracted_ttndo_correct = False
    except Exception as e:
        print(f"✗ Error testing contracted TTNDO: {e}")
        import traceback
        traceback.print_exc()
        contracted_ttndo_correct = False
    
    # 7. Test expectation values consistency between symmetric and binary TTNDOs
    print("\nTesting expectation value consistency between TTNDO types...")
    expectation_values_correct = True
    try:
        # Import necessary modules
        try:
            from pytreenet.operators.models import ising_model
            from pytreenet.operators.sim_operators import single_site_operators
        except ImportError as e:
            print(f"✗ Error importing necessary modules: {e}")
            expectation_values_correct = False
            
        if expectation_values_correct:
            # 7a. Generate Ising Hamiltonian and test TTNO expectation values
            print("\nTesting TTNO expectation values with Ising Hamiltonian...")
            
            # Helper function to generate physical node identifiers
            def phys_node_identifiers(length: int) -> list[str]:
                """Generates the node identifiers for the physical nodes."""
                return [f"qubit{i}" for i in range(length)]
            
            # Create Ising model Hamiltonian
            length = num_phys
            ext_magn = 1.0
            coupling = 1.0
            node_identifiers = phys_node_identifiers(length)
            
            # Define nearest-neighbor pairs
            nn_pairs = [(node_identifiers[i], node_identifiers[i+1]) 
                        for i in range(length-1)]
            
            # Create Hamiltonian and TTNO
            ham = ising_model(nn_pairs, ext_magn, factor=coupling)
            ising_ttno = TreeTensorNetworkOperator.from_hamiltonian(ham, ttns)
            
            # Calculate expectation values for both TTNDO types
            sym_expectation = ttndo_symmetric.ttno_expectation_value(ising_ttno)
            
            # For binary TTNDO, we need to ensure it's contracted first
            binary_contracted = contract_physical_nodes(ttndo_binary)
            int_expectation = fully_binary_ttndo_ttno_expectation_value(ising_ttno, binary_contracted)
            
            # Compare the two expectation values (allowing for numerical precision differences)
            ttno_difference = abs(sym_expectation - int_expectation)
            assert ttno_difference < 1e-10, f"TTNO expectation values differ: symmetric={sym_expectation}, binary={int_expectation}, diff={ttno_difference}"
            print(f"✓ TTNO expectation values match between TTNDO types: {sym_expectation:.10f}")
            
            # 7b. Test tensor product expectation values
            print("\nTesting tensor product expectation values...")
            
            # Get Pauli Z operators for each site
            sigma_z = pauli_matrices()[2]
        
            # Create operators dictionary
            site_operators = {}
            for i in range(length):
                node_id = f"qubit{i}"
                site_operators[node_id] = sigma_z
            
            # Compare expectations for each site individually
            for node_id, operator in site_operators.items():
                tensor_product = TensorProduct({node_id: operator})
                
                # Calculate expectation values
                sym_tp_expectation = ttndo_symmetric.tensor_product_expectation_value(tensor_product)
                
                # For binary TTNDO
                int_tp_expectation = ttndo_binary.tensor_product_expectation_value(tensor_product)
                
                # Compare results
                tp_difference = abs(sym_tp_expectation - int_tp_expectation)
                assert tp_difference < 1e-10, f"Tensor product expectation values for {node_id} differ: symmetric={sym_tp_expectation}, binary={int_tp_expectation}, diff={tp_difference}"
            
            print(f"✓ Tensor product expectation values match between TTNDO types for all sites")
            
    except AssertionError as e:
        print(f"✗ {e}")
        expectation_values_correct = False
    except Exception as e:
        print(f"✗ Error testing expectation values: {e}")
        import traceback
        traceback.print_exc()
        expectation_values_correct = False
    
    # 8. Check if physical tensor values are preserved and bra tensors are complex conjugate of ket tensors
    print("\nTesting physically binary TTNDO physical tensor preservation...")
    phys_tensor_correct = True
    
    try:
        # Check physical nodes in physically binary TTNDO
        for node_id in ttndo_physically_binary.nodes:
            # Check ket physical nodes
            if node_id.startswith('qubit') and node_id.endswith('_ket'):
                tensor = ttndo_physically_binary.tensors[node_id]
                # Ket tensor should preserve original values
                if tensor.ndim == 3 and tensor.shape[2] >= 1:
                    # Physical ket tensors have 3 legs: parent, lateral, physical
                    orig_node_id = node_id[:-4]  # Remove _ket suffix
                    original_tensor = ttns.tensors[orig_node_id]
                    
                    # Check values at index [i,0,j]
                    for i in range(min(tensor.shape[0], original_tensor.shape[0])):
                        for j in range(min(tensor.shape[2], original_tensor.shape[1])):
                            assert np.isclose(tensor[i, 0, j], original_tensor[i, j]), \
                                f"Physically binary TTNDO ket node {node_id}: mismatch at [{i},0,{j}]: {tensor[i,0,j]} != {original_tensor[i,j]}"
            
            # Check bra physical nodes
            elif node_id.startswith('qubit') and node_id.endswith('_bra'):
                tensor = ttndo_physically_binary.tensors[node_id]
                # Bra tensors should be complex conjugate of original
                if tensor.ndim == 2:
                    # Physical bra tensors have 2 legs: parent, physical
                    orig_node_id = node_id[:-4]  # Remove _bra suffix
                    original_tensor = ttns.tensors[orig_node_id]
                    
                    # Check if values match the conjugate of original tensor at [0,j]
                    for j in range(min(tensor.shape[1], original_tensor.shape[1])):
                        assert np.isclose(tensor[0, j], np.conj(original_tensor[0, j])), \
                            f"Physically binary TTNDO bra node {node_id}: mismatch at [0,{j}]: {tensor[0,j]} != {np.conj(original_tensor[0,j])}"
                            
        print("✓ All physically binary TTNDO physical tensors preserve values correctly and bra tensors are complex conjugate of ket tensors")
    except AssertionError as e:
        print(f"✗ {e}")
        phys_tensor_correct = False
    
    # 9. Check virtual node sparse structure in physically binary TTNDO
    print("\nTesting physically binary TTNDO virtual node identity structure...")
    virtual_structure_correct = True
    
    try:
        for node_id in ttndo_physically_binary.nodes:
            if is_virtual_node(node_id) and not (node_id.endswith("_ket") or node_id.endswith("_bra")):
                # Virtual nodes in physically binary TTNDO don't have bra/ket suffixes
                tensor = ttndo_physically_binary.tensors[node_id]
                assert has_correct_sparse_structure(tensor), \
                    f"Physically binary TTNDO virtual node {node_id} doesn't have correct sparse structure"
        print("✓ All physically binary TTNDO virtual nodes have correct sparse identity structure")
    except AssertionError as e:
        print(f"✗ {e}")
        virtual_structure_correct = False
    
    # 10. Check bond dimensions in physically binary TTNDO
    print("\nTesting physically binary TTNDO bond dimensions...")
    physical_bond_dims_correct = True
    
    try:
        # For each virtual node, check that all non-open legs have bond_dim
        for node_id in ttndo_physically_binary.nodes:
            if is_virtual_node(node_id) and not (node_id.endswith("_ket") or node_id.endswith("_bra")):
                node = ttndo_physically_binary.nodes[node_id]
                tensor = ttndo_physically_binary.tensors[node_id]
                
                # Check all non-open legs
                for i in range(len(tensor.shape)):
                    if i in node.open_legs:
                        # Open leg should be dimension 1
                        assert tensor.shape[i] == 1, \
                            f"Physically binary TTNDO virtual node {node_id}: open leg {i} has dimension {tensor.shape[i]} (expected 1)"
                    else:
                        # All other legs should have bond_dim
                        assert tensor.shape[i] == BOND_DIM, \
                            f"Physically binary TTNDO virtual node {node_id}: non-open leg {i} has dimension {tensor.shape[i]} (expected {BOND_DIM})"
                            
        # For physical nodes with _ket suffix, check bond dimensions
        for node_id in ttndo_physically_binary.nodes:
            if not is_virtual_node(node_id) and node_id.endswith("_ket"):
                tensor = ttndo_physically_binary.tensors[node_id]
                
                # Physical ket nodes should have shape (parent_dim, bond_dim, phys_dim)
                assert tensor.shape[1] == BOND_DIM, \
                    f"Physically binary TTNDO ket physical node {node_id}: lateral leg has dimension {tensor.shape[1]} (expected {BOND_DIM})"
                
                # Physical dimensions should match original
                assert tensor.shape[2] == 2, \
                    f"Physically binary TTNDO ket physical node {node_id}: physical leg has dimension {tensor.shape[2]} (expected 2)"
        
        print("✓ All physically binary TTNDO bond dimensions are correct")
    except AssertionError as e:
        print(f"✗ {e}")
        physical_bond_dims_correct = False
    
    # 11. Test contracted physically binary TTNDO
    print("\nTesting contracted physically binary TTNDO...")
    contracted_physical_correct = True
    
    try:
        # Contract the physically binary TTNDO
        print("Contracting physically binary TTNDO...")
        ttndo_physical_contracted = contract_physical_nodes(ttndo_physically_binary)
        print(f"Contracted physically binary TTNDO created with {len(ttndo_physical_contracted.nodes)} nodes")
        
        # 11a. Check if nodes have the correct number of open legs
        # Physical nodes should have two open legs, virtual nodes should have one open leg
        print("\nChecking that nodes have the correct number of open legs...")
        success, error_msg = check_contracted_ttndo_open_legs(ttndo_physical_contracted)
        assert success, error_msg
        print("✓ All contracted physically binary TTNDO nodes have the correct number of open legs")
        
        # 11b. Calculate trace of contracted physically binary TTNDO
        print("\nCalculating trace of contracted physically binary TTNDO...")
        trace_value = trace_contracted_physically_binary_ttndo(ttndo_physical_contracted)
        print(f"Trace result: {trace_value}")
        
        # Check if trace is close to 1.0 (for pure states)
        assert np.isclose(trace_value, 1.0), f"Expected trace 1.0 for pure state, got {trace_value}"
        print("✓ Contracted physically binary TTNDO trace is correct (1.0) for pure state")
        
        # 11c. Test the trace method of the BINARYTTNDO class directly
        print("\nTesting ttndo_physically_binary.trace() method...")
        class_trace_value = ttndo_physically_binary.trace()
        print(f"Class trace method result: {class_trace_value}")
        
        # Check if the class trace method result is also close to 1.0
        assert np.isclose(class_trace_value, 1.0), f"Expected class trace 1.0 for pure state, got {class_trace_value}"
        print("✓ BINARYTTNDO.trace() method returns correct value (1.0) for physically binary TTNDO")
        
    except AssertionError as e:
        print(f"✗ {e}")
        contracted_physical_correct = False
    except Exception as e:
        print(f"✗ Error testing contracted physically binary TTNDO: {e}")
        import traceback
        traceback.print_exc()
        contracted_physical_correct = False
    
    # 12. Test expectation values for physically binary TTNDO
    print("\nTesting expectation values for physically binary TTNDO...")
    physical_expectation_values_correct = True
    

    try:
            # 12a. Generate Ising Hamiltonian and test TTNO expectation values
            print("\nTesting physically binary TTNO expectation values with Ising Hamiltonian...")
            
            # Create Ising model Hamiltonian (reuse the same Hamiltonian from earlier tests)
            length = num_phys
            ext_magn = 1.0
            coupling = 1.0
            node_identifiers = [f"qubit{i}" for i in range(length)]
            
            # Define nearest-neighbor pairs
            nn_pairs = [(node_identifiers[i], node_identifiers[i+1]) 
                        for i in range(length-1)]
            
            # Create Hamiltonian and TTNO
            ham = ising_model(nn_pairs, ext_magn, factor=coupling)
            ising_ttno = TreeTensorNetworkOperator.from_hamiltonian(ham, ttns)
            
            # Calculate expectation value for physically binary TTNDO
            physically_contracted = contract_physical_nodes(ttndo_physically_binary)
            phys_expectation = physically_binary_ttndo_ttno_expectation_value(ising_ttno, physically_contracted)
            
            # Compare with symmetric TTNDO expectation (already calculated)
            phys_ttno_difference = abs(sym_expectation - phys_expectation)
            assert phys_ttno_difference < 1e-10, f"TTNO expectation values differ: symmetric={sym_expectation}, physically binary={phys_expectation}, diff={phys_ttno_difference}"
            print(f"✓ Physically binary TTNO expectation value matches symmetric TTNDO: {phys_expectation:.10f}")
            
            # Test the direct class method
            class_expectation = ttndo_physically_binary.ttno_expectation_value(ising_ttno)
            class_diff = abs(class_expectation - sym_expectation)
            assert class_diff < 1e-10, f"Class method TTNO expectation values differ: symmetric={sym_expectation}, class method={class_expectation}, diff={class_diff}"
            print(f"✓ BINARYTTNDO.ttno_expectation_value() method returns correct value for physically binary TTNDO")
            
            # 12b. Test tensor product expectation values
            print("\nTesting physically binary tensor product expectation values...")
            
            # Get Pauli Z operators for each site
            sigma_z = pauli_matrices()[2]
            
            # Test for a few sites
            test_sites = node_identifiers[:min(3, len(node_identifiers))]
            for node_id in test_sites:
                tensor_product = TensorProduct({node_id: sigma_z})
                
                # Calculate expectation values
                sym_tp_expectation = ttndo_symmetric.tensor_product_expectation_value(tensor_product)
                phys_tp_expectation = ttndo_physically_binary.tensor_product_expectation_value(tensor_product)
                
                # Compare results
                tp_difference = abs(sym_tp_expectation - phys_tp_expectation)
                assert tp_difference < 1e-10, f"Tensor product expectation values for {node_id} differ: symmetric={sym_tp_expectation}, physically binary={phys_tp_expectation}, diff={tp_difference}"
            
            print(f"✓ Tensor product expectation values match between TTNDO types for tested sites")
            
    except AssertionError as e:
        print(f"✗ {e}")
        physical_expectation_values_correct = False
    except Exception as e:
        print(f"✗ Error testing physically binary expectation values: {e}")
        import traceback
        traceback.print_exc()
        physical_expectation_values_correct = False
    
    # Return overall test result
    return (open_legs_correct and phys_tensor_preserved and sparse_structure_correct 
            and bond_dims_correct and trace_correct and contracted_ttndo_correct
            and expectation_values_correct and phys_tensor_correct and virtual_structure_correct
            and physical_bond_dims_correct and contracted_physical_correct and physical_expectation_values_correct)


def run_tests():
    """Run tests with different qubit counts and depths."""
    # Create physical tensors with distinctive values
    phys_tensors = [
        np.array([1/np.sqrt(2), -1j/np.sqrt(2)], dtype=complex),  # Type 1
        np.array([1j/np.sqrt(2), -1/np.sqrt(2)], dtype=complex),  # Type 2
        np.array([-1j/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex)  # Type 3
    ]
    
    # Test configurations for number of qubits and depth
    test_configs = [
        # (qubits, depths)
        (45,5),
        (53,6),
        (54,2),
        (11, 3),
        (7,2),
        (30, 4),
        (31,3),
        (25,2),
        (35,4),

    ]
    
    results = {}
    
    # Test each physical tensor type
    for tensor_idx, phys_tensor in enumerate(phys_tensors):
        print(f"\n{'='*80}")
        print(f"TESTING WITH PHYSICAL TENSOR TYPE {tensor_idx+1}")
        print(f"{'='*80}")
        
        # Test each configuration with the current physical tensor
        for num_phys, depth in test_configs:
            try:
                print(f"\nGenerating networks for {num_phys} qubits at depth {depth}...")
                
                # Generate TTNS
                ttns = generate_binary_ttns(
                    num_phys=num_phys,
                    bond_dim=BOND_DIM,
                    phys_tensor=phys_tensor,
                    depth=depth
                )
                
                # Generate TTNDOs
                ttndo_binary = from_ttns_fully_binary(ttns, bond_dim=BOND_DIM, phys_tensor=phys_tensor)
                ttndo_symmetric = from_symmetric_ttns(ttns, bond_dim=BOND_DIM)
                
                # Run tests
                result = test_network_structure(ttns, ttndo_binary, ttndo_symmetric, phys_tensor, num_phys, depth)
                results[(tensor_idx+1, num_phys, depth)] = result
                
            except Exception as e:
                print(f"\nERROR testing tensor type {tensor_idx+1}, {num_phys} qubits at depth {depth}: {e}")
                results[(tensor_idx+1, num_phys, depth)] = False
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for (tensor_type, num_phys, depth), result in results.items():
        status = "PASSED" if result else "FAILED"
        all_passed = all_passed and result
        print(f"Tensor Type {tensor_type}, {num_phys} qubits, depth {depth}: {status}")
    
    print("\nOVERALL RESULT:", "PASSED" if all_passed else "FAILED")

if __name__ == "__main__":
    run_tests() 