"""
Provides a class that represents dennsity operators as a tree tensor network.

This leads to the concept of a tree tensor network density operator (TTNDO).

"""
from __future__ import annotations
from re import match
from copy import deepcopy

from numpy import ndarray, pad, eye, tensordot, zeros, conj

from ..core.node import Node
from .ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..util.ttn_exceptions import positivity_check
from ..operators.tensorproduct import TensorProduct
from ..contractions.ttndo_contractions import (trace_symmetric_ttndo, 
                                              symmetric_ttndo_ttno_expectation_value,
                                              contract_physical_nodes, 
                                              trace_contracted_binary_ttndo,
                                              binary_ttndo_ttno_expectation_value)


class SymmetricTTNDO(TreeTensorNetworkState):
    """
    A class representing a tree tensor network density operator (TTNDO).

    The TTNDO is a assumed to be symmetric around a root.
    """

    def __init__(self, bra_suffix = "_bra", ket_suffix = "_ket"):
        super().__init__()
        self.bra_suffix = bra_suffix
        self.ket_suffix = ket_suffix

    def add_trivial_root(self, identifier: str,
                         dimension: int = 2) -> None:
        """
        Adds a trivial root to the TTNDO.

        This root is an identity matrix (i.e. two legs) with a trivial physical
        leg.

        Args:
            identifier (str): The identifier of the new root.
            dimension (int, optional): The dimension of the virtual legs of the
                new root. Defaults to 2.
        
        """
        positivity_check(dimension, "dimension")
        tensor = eye(dimension).reshape((dimension,dimension,1))
        root_node = Node(identifier=identifier)
        self.add_root(root_node, tensor)

    def ket_id(self, node_id: str) -> str:
        """
        Returns the ket identifier for a given node identifier.

        Args:
            node_id (str): The identifier of the node.

        Returns:
            str: The ket identifier.
        """
        return node_id + self.ket_suffix

    def bra_id(self, node_id: str) -> str:
        """
        Returns the bra identifier for a given node identifier.

        Args:
            node_id (str): The identifier of the node.

        Returns:
            str: The bra identifier.
        """
        return node_id + self.bra_suffix

    def reverse_ket_id(self, ket_id: str) -> str:
        """
        Returns the node identifier for a given ket identifier.

        Args:
            ket_id (str): The ket identifier.

        Returns:
            str: The node identifier.

        """
        assert match(r".*"+self.ket_suffix, ket_id), \
            "The given identifier is not a ket identifier!"
        return ket_id[:-len(self.ket_suffix)]

    def reverse_bra_id(self, bra_id: str) -> str:
        """
        Returns the node identifier for a given bra identifier.

        Args:
            bra_id (str): The bra identifier.

        Returns:
            str: The node identifier.

        """
        assert match(r".*"+self.bra_suffix, bra_id), \
            "The given identifier is not a bra identifier!"
        return bra_id[:-len(self.bra_suffix)]

    def ket_to_bra_id(self, ket_id: str) -> str:
        """
        Returns the bra identifier for a given ket identifier.

        Args:
            ket_id (str): The ket identifier.

        Returns:
            str: The bra identifier.

        """
        return self.bra_id(self.reverse_ket_id(ket_id))

    def bra_to_ket_id(self, bra_id: str) -> str:
        """
        Returns the ket identifier for a given bra identifier.

        Args:
            bra_id (str): The bra identifier.

        Returns:
            str: The ket identifier.

        """
        return self.ket_id(self.reverse_bra_id(bra_id))

    def add_symmetric_children_to_parent(self,
                                         child_id: str,
                                         ket_tensor: ndarray,
                                         bra_tensor: ndarray,
                                         child_leg: int,
                                         parent_id: str,
                                         parent_leg: int,
                                         parent_bra_leg: None | int = None):
        """
        Adds a symmetric pair of children to a parent node.

        The children are added to the parent node with the same leg. Only if
        the parent node is the root node, the bra leg can be specified
        separately.

        Args:
            child_id (str): The identifier of the children.
            ket_tensor (ndarray): The tensor associated to the ket part.
            bra_tensor (ndarray): The tensor associated to the bra part.
            child_leg (int): The leg of the children.
            parent_id (str): The identifier of the parent node.
            parent_leg (int): The leg of the parent node.
            parent_bra_leg (None | int, optional): The bra leg of the parent
                node. Only if the parent node is the root node. Defaults to None.
        
        Raises:
            ValueError: If the parent_bra_leg is specified for a non-root node.
            AssertionError: If the bra and ket tensor do not have the same shape.

        """

        if parent_bra_leg is not None and parent_id != self.root_id and parent_bra_leg != parent_leg:
            errstr = "The parent_bra_leg can only be specified for the root node!"
            raise ValueError(errstr)
        if parent_bra_leg is None:
            parent_bra_leg = parent_leg
        if parent_id == self.root_id:
            parent_ket_id = self.root_id
            parent_bra_id = self.root_id
        else:
            parent_ket_id = self.ket_id(parent_id)
            parent_bra_id = self.bra_id(parent_id)
        assert bra_tensor.shape == ket_tensor.shape, \
           "The bra and ket tensor must have the same shape!"
        # Add Ket Node
        ket_node = Node(identifier=self.ket_id(child_id))
        self.add_child_to_parent(ket_node, ket_tensor, child_leg,
                                 parent_ket_id, parent_leg)
        # Add Bra Node
        bra_node = Node(identifier=self.bra_id(child_id))
        self.add_child_to_parent(bra_node, bra_tensor, child_leg,
                                 parent_bra_id, parent_bra_leg)

    def trace(self) -> complex:
        """
        Returns the trace of the TTNDO.

        Returns:
            complex: The trace of the TTNDO.
        """
        return trace_symmetric_ttndo(self)

    def norm(self) -> complex:
        """
        Returns the norm of the TTDNO, which is just the trace
        """
        return self.trace()

    def ttno_expectation_value(self,
                               operator: TreeTensorNetworkOperator) -> complex:
        """
        Computes the expectation value of the TTNDO with respect to a TTNO.

        Args:
            ttno (TreeTensorNetworkOperator): The TTNO to compute the expectation
                value with.

        Returns:
            complex: The expectation value of the TTNDO with respect to the TTNO.

        """
        return symmetric_ttndo_ttno_expectation_value(self, operator)

    def tensor_product_expectation_value(self,
                                         operator: TensorProduct
                                         ) -> complex:
        """
        Computes the expectation value of a tensor product of operators.

        Args:
            operator (TensorProduct): The tensor product of operators.

        Returns:
            complex: The resulting expectation value < TTNS | tensor_product | TTNS>

        """
        if len(operator) == 0:
            return self.scalar_product()
        ttn = deepcopy(self)
        for node_id, single_site_operator in operator.items():
            ket_id = self.ket_id(node_id)
            ttn.absorb_into_open_legs(ket_id, single_site_operator)
        return ttn.trace()

    def single_site_operator_expectation_value(self, node_id: str,
                                               operator: ndarray) -> complex:
        """
        The expectation value with regards to a single-site operator.

        The single-site operator acts on the specified node.

        Args:
            node_id (str): The identifier of the node, the operator is applied
                to.
            operator (ndarray): The operator of which we determine the
                expectation value. Note that the state will be contracted with
                axis/leg 1 of this operator.

        Returns:
            complex: The resulting expectation value < TTNS| Operator| TTN >.
        """
        tensor_product = TensorProduct({node_id: operator})
        return self.operator_expectation_value(tensor_product)
def Symmetric_ttndo_from_binary_ttns(ttns: TreeTensorNetworkState,
                root_id: str = "ttndo_root",
                root_bond_dim: int = 2
                ) -> SymmetricTTNDO:
    """
    Creates a TTNDO from a TTN.

    Args:
        ttns (TreeTensorNetworkState): The TTN to convert.
        root_id (str, optional): The identifier of the new root. Defaults to
            "ttndo_root".
        root_bond_dim (int, optional): The bond dimension of the new root.
            Defaults to 2.

    Returns:
        SymmetricTTNDO: The resulting TTNDO.
    """
    ttndo = SymmetricTTNDO()
    ttndo.add_trivial_root(root_id,
                           dimension=root_bond_dim)
    # Now we attach the root
    root_node, root_tensor = ttns.root
    ttns_root_id = root_node.identifier
    # We need to add a leg to be attached to the ttndo root.
    new_shape = tuple([1] + list(root_node.shape))
    root_tensor = deepcopy(root_tensor).reshape(new_shape)
    padding = [(0,0) for _ in range(len(root_tensor.shape))]
    padding[0] = (0,root_bond_dim-1)
    root_tensor = pad(root_tensor, padding)
    ttndo.add_symmetric_children_to_parent(ttns_root_id,
                                           root_tensor,
                                           root_tensor.conj(),
                                           0,
                                           root_id,
                                           0,
                                           parent_bra_leg=1
                                           )
    # Now we need to attach the children
    _rec_add_children(ttns, ttndo, root_node)
    return ttndo

def _rec_add_children(ttns: TreeTensorNetworkState,
                      ttndo: SymmetricTTNDO,
                      node: Node):
    """
    Recursively adds children to the TTNDO.
    """
    for child_id in node.children:
        child_node, child_tensor = ttns[child_id]
        # If the parent is the root, it got an additional leg now
        parent_leg = node.neighbour_index(child_id) + int(node.is_root())
        ttndo.add_symmetric_children_to_parent(child_id,
                                               child_tensor,
                                               child_tensor.conj(),
                                               child_node.parent_leg,
                                               node.identifier,
                                               parent_leg)
        _rec_add_children(ttns, ttndo, child_node)

class BINARYTTNDO(TreeTensorNetworkState):
    """
    A class representing an binary tree tensor network density operator (TTNDO) in vectorized form.
    
    The binary TTNDO structure distributes bra and ket nodes across the tree
    with lateral connections, creating a more flexible structure than the symmetric TTNDO.
    """

    def __init__(self, bra_suffix="_bra", ket_suffix="_ket"):
        """
        Initialize an BINARYTTNDO.
        
        Args:
            bra_suffix (str): Suffix to use for bra nodes. Defaults to "_bra".
            ket_suffix (str): Suffix to use for ket nodes. Defaults to "_ket".
        """
        super().__init__()
        self.bra_suffix = bra_suffix
        self.ket_suffix = ket_suffix

    def absorb_into_binary_ttndo_open_leg(self, node_id: str, matrix: ndarray):
        """
        Absorb a matrix into the input open leg of a node in the TTNDO.
        
        For BINARYTTNDO, each node has exactly two open legs. The input leg
        is assumed to be the one before the other leg (first open leg). The matrix
        is applied to this input leg.
        
        Args:
            node_id (str): The identifier of the node to which the matrix is applied.
            matrix (ndarray): The matrix to absorb. Must be a square matrix.
            
        Returns:
            BINARYTTNDO: The modified TTNDO for method chaining.
            
        Raises:
            AssertionError: If the matrix is not square or if the node doesn't have exactly 2 open legs.
        """
        node, node_tensor = self[node_id]
        open_legs = node.open_legs
        assert len(open_legs) == 2, f"Node {node_id} must have exactly 2 open legs, but has {len(open_legs)}."
        input_leg = open_legs[0]
        new_tensor = tensordot(node_tensor, matrix, axes=(input_leg, 1))
        self.tensors[node_id] = new_tensor        
        return self

    def contract(self, to_copy: bool = True) -> 'BINARYTTNDO':
        """
        Contract this binary TTNDO into a regular TTNDO.
        
        Returns:
            BINARYTTNDO: A new contracted TTNDO structure.
        """
        if to_copy:
            contracted = contract_physical_nodes(self, self.bra_suffix, self.ket_suffix)
            return contracted
        else:
            contract_physical_nodes(self, self.bra_suffix, self.ket_suffix, to_copy=False)

    def trace(self) -> complex:
        """
        Returns the trace of the BINARYTTNDO.

        Returns:
            complex: The trace of the TTNDO.
        """
        contracted_physically_binary_ttndo = self.contract()
        return trace_contracted_binary_ttndo(contracted_physically_binary_ttndo)


    def norm(self) -> complex:
        """
        Returns the norm of the BINARYTTNDO, which is just the trace.
        
        Returns:
            complex: The norm of the TTNDO.
        """
        return self.trace()

    def ttno_expectation_value(self, operator: TreeTensorNetworkOperator) -> complex:
        """
        Computes the expectation value of the BINARYTTNDO with respect to a TTNO.
        
        This is the trace of TTNO @ TTNDO: Tr(TTNO @ TTNDO)
        
        Args:
            operator (TreeTensorNetworkOperator): The TTNO to compute the expectation
                value with.

        Returns:
            complex: The expectation value of the TTNDO with respect to the TTNO.
        """
        contracted_ttndo = self.contract()
        return binary_ttndo_ttno_expectation_value(operator, contracted_ttndo)

    def tensor_product_expectation_value(self, operator: TensorProduct) -> complex:
        """
        Computes the expectation value of a tensor product of operators.
        
        The calculation first applies the operators to the TTNDO, then computes the trace.
        Args:
            operator (TensorProduct): The tensor product of operators.

        Returns:
            complex: The resulting expectation value < TTNDO | tensor_product | TTNDO >
        """
        if len(operator) == 0:
           return self.trace()
        
        # First contract to ensure we have the proper structure
        ttndo_copy = self.contract()

        # Apply operators 
        for node_id, single_site_operator in operator.items():
            # Directly absorb the operator into the node tensor
            ttndo_copy = ttndo_copy.absorb_into_binary_ttndo_open_leg(node_id, single_site_operator)
        # Calculate the trace of the modified TTNDO
        return trace_contracted_binary_ttndo(ttndo_copy)
    
    def single_site_operator_expectation_value(self, node_id: str, operator: ndarray) -> complex:
        """
        The expectation value with regards to a single-site operator.
        
        The single-site operator acts on the specified node.
        
        Args:
            node_id (str): The identifier of the node the operator is applied to.
            operator (ndarray): The operator to determine the expectation value of.
            
        Returns:
            complex: The resulting expectation value < TTNDO | Operator | TTNDO >.
        """
        tensor_product = TensorProduct({node_id: operator})
        return self.tensor_product_expectation_value(tensor_product)


def binary_ttndo_for_product_state(ttns: TreeTensorNetworkState,
                                    bond_dim: int,
                                    phys_tensor: ndarray,
                                    bra_suffix: str = "_bra",
                                    ket_suffix: str = "_ket") -> BINARYTTNDO:
    """
    Creates a physically binary TTNDO from a TTNS, where only physical nodes have dual representation.
    Virtual nodes have a single tensor, while physical nodes maintain bra and ket parts.
    
    Note: Trace and expectation value calculations for physically binary TTNDOs are not yet implemented
    and will be added in a future update. Currently, the structure supports contracting the dual nodes
    correctly but not computing traces or expectation values.
    
    Args:
        ttns: Original TTNS structure
        bond_dim: Bond dimension for the TTNDO
        phys_tensor: Physical tensor template for leaf nodes
        bra_suffix: Suffix for bra nodes
        ket_suffix: Suffix for ket nodes
        
    Returns:
        BINARYTTNDO: The resulting physically binary TTNDO
    """
    # Create new BINARYTTNDO to hold the structure
    ttndo = BINARYTTNDO(bra_suffix, ket_suffix)
    created_nodes = {}
    
    # Get original root information
    original_root_id = ttns.root_id
    original_root_node = ttns.nodes[original_root_id]
    
    # For the root, we use a single node with legs for all children
    root_shape = [bond_dim] * len(original_root_node.children)  # One leg for each child
    root_shape.append(1)  # Last leg is open/physical
    
    # Create root tensor with sparse initialization
    root_tensor = zeros(tuple(root_shape), dtype=complex)
    
    # Use sparse initialization: only set the [0,0,...,0] element to 1.0
    root_tensor[(0,) * (len(root_shape) - 1) + (0,)] = 1.0
    
    # Create and add the root node
    root_node = Node(identifier=original_root_id)
    ttndo.add_root(root_node, root_tensor)
    created_nodes[original_root_id] = True
    
    # Process children
    for i, child_id in enumerate(original_root_node.children):
        process_physical_branch(ttns, ttndo, child_id, original_root_id, i, 
                             created_nodes, bond_dim, phys_tensor, bra_suffix, ket_suffix)
    
    return ttndo

def process_physical_branch(ttns: TreeTensorNetworkState, 
                         ttndo: TreeTensorNetworkState,
                         orig_node_id: str, 
                         parent_id: str, 
                         parent_leg: int,
                         created_nodes: dict,
                         bond_dim: int, 
                         phys_tensor: ndarray,
                         bra_suffix: str, 
                         ket_suffix: str):
    """
    Process a branch in physically binary TTNDO - creates single virtual nodes
    and dual physical nodes.
    """
    orig_node = ttns.nodes[orig_node_id]
    
    # Skip if already created
    if orig_node_id in created_nodes:
        return
    
    # Check if physical node
    is_physical = len(orig_node.children) == 0
    
    if is_physical:
        # Process physical node (create bra-ket pair)
        create_physical_node_dual(ttndo, orig_node_id, parent_id, parent_leg,
                              created_nodes, bond_dim, ttns, bra_suffix, ket_suffix)
    else:
        # Virtual node - create single tensor
        create_single_virtual_node(ttndo, ttns, orig_node_id, parent_id, parent_leg, 
                               created_nodes, bond_dim, phys_tensor, bra_suffix, ket_suffix)

def create_single_virtual_node(ttndo: TreeTensorNetworkState,
                            ttns: TreeTensorNetworkState,
                            orig_node_id: str,
                            parent_id: str, 
                            parent_leg: int,
                            created_nodes: dict, 
                            bond_dim: int, 
                            phys_tensor: ndarray,
                            bra_suffix: str, 
                            ket_suffix: str):
    """
    Create a single virtual node (no bra/ket distinction).
    """
    orig_node = ttns.nodes[orig_node_id]
    num_children = len(orig_node.children)
    
    # Shape: parent, children, open
    node_shape = [bond_dim]  # parent leg
    node_shape.extend([bond_dim] * num_children)  # child legs 
    node_shape.append(1)  # open/physical leg
    
    # Create tensor with sparse initialization
    node_tensor = zeros(tuple(node_shape), dtype=complex)
    
    # Only set the [0,0,...,0] element to 1.0
    node_tensor[(0,) * (len(node_shape) - 1) + (0,)] = 1.0
    
    # Create node and connect to parent
    node = Node(identifier=orig_node_id)
    ttndo.add_child_to_parent(node, node_tensor, 0, parent_id, parent_leg, compatible= False)
    created_nodes[orig_node_id] = True
    
    # Process children
    for i, child_id in enumerate(orig_node.children):
        # First leg is parent, so child legs start at index 1
        child_leg_idx = i + 1
        process_physical_branch(ttns, ttndo, child_id, orig_node_id, child_leg_idx, 
                             created_nodes, bond_dim, phys_tensor, bra_suffix, ket_suffix)

def create_physical_node_dual(ttndo: TreeTensorNetworkState,
                           orig_node_id: str,
                           parent_id: str, 
                           parent_leg: int,
                           created_nodes: dict,
                           bond_dim: int, 
                           original_ttns: TreeTensorNetworkState,
                           bra_suffix: str, 
                           ket_suffix: str):
    """
    Create a physical node with dual representation (ket and bra tensors).
    
    Physical ket node: 3 legs (parent, lateral, physical)
    Physical bra node: 2 legs (parent, physical)
    Bra node connects to the ket node through the lateral connection.
    This ensures compatibility with contract_physical_nodes.
    """
    # Get original tensor from TTNS
    original_tensor = deepcopy(original_ttns.tensors[orig_node_id])
    phys_dim = original_tensor.shape[-1]
    parent_dim = original_tensor.shape[0]
    
    # Create ket node ID
    ket_id = orig_node_id + ket_suffix
    
    # KET PHYSICAL NODE - 3 legs: parent, lateral, physical
    ket_tensor = zeros((parent_dim, bond_dim, phys_dim), dtype=complex)
    
    # Copy values from original tensor with lateral bond in between
    for i in range(parent_dim):
        for j in range(phys_dim):
            ket_tensor[i, 0, j] = original_tensor[i, j]
    
    # Create and connect ket node to parent
    ket_node = Node(identifier=ket_id)
    ttndo.add_child_to_parent(ket_node, ket_tensor, 0, parent_id, parent_leg, compatible= False)
    created_nodes[ket_id] = True
    
    # Create bra node ID
    bra_id = orig_node_id + bra_suffix
    
    # BRA PHYSICAL NODE - 2 legs: parent, physical
    bra_tensor = zeros((bond_dim, phys_dim), dtype=complex)
    
    # Copy and conjugate values from original tensor
    for j in range(phys_dim):
        bra_tensor[0, j] = conj(original_tensor[0, j])
    
    # Create and connect bra node to ket node through lateral connection
    bra_node = Node(identifier=bra_id)
    ttndo.add_child_to_parent(bra_node, bra_tensor, 0, ket_id, 1, compatible= False)
    created_nodes[bra_id] = True

