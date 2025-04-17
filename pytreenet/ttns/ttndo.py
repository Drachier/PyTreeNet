"""
Provides a class that represents dennsity operators as a tree tensor network.

This leads to the concept of a tree tensor network density operator (TTNDO).

"""
from __future__ import annotations
from re import match
from copy import deepcopy

import numpy as np
from numpy import ndarray, pad, eye

from ..core.node import Node
from .ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..util.ttn_exceptions import positivity_check
from ..operators.tensorproduct import TensorProduct
from ..contractions.ttndo_contractions import (trace_symmetric_ttndo, 
                                              symmetric_ttndo_ttno_expectation_value, 
                                              trace_contracted_fully_intertwined_ttndo,
                                              trace_contracted_physically_intertwined_ttndo,
                                              fully_intertwined_ttndo_ttno_expectation_value,
                                              physically_intertwined_ttndo_ttno_expectation_value)

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
                         dimension: int = 5) -> None:
        """
        Adds a trivial root to the TTNDO.

        This root is an identity matrix (i.e. two legs) with a trivial physical
        leg. The identity matrix ensures proper contraction behavior.

        Args:
            identifier (str): The identifier of the new root.
            dimension (int, optional): The dimension of the virtual legs of the
                new root. Defaults to 5.
        
        """
        positivity_check(dimension, "dimension")
        # Create an identity matrix of the requested dimension
        # and reshape it into a 3D tensor with a trivial third dimension
        tensor = eye(dimension).reshape((dimension, dimension, 1))
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
                                 parent_ket_id, parent_leg,
                                 modify=True)
        # Add Bra Node
        bra_node = Node(identifier=self.bra_id(child_id))
        self.add_child_to_parent(bra_node, bra_tensor, child_leg,
                                 parent_bra_id, parent_bra_leg, modify=True)

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
            # Can be improved once shallow copying is possible
            #ttn = deepcopy(self)
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
            operator (np.ndarray): The operator of which we determine the
                expectation value. Note that the state will be contracted with
                axis/leg 1 of this operator.

        Returns:
            complex: The resulting expectation value < TTNS| Operator| TTN >.
        """
        tensor_product = TensorProduct({node_id: operator})
        return self.operator_expectation_value(tensor_product)

def from_ttns_symmetric(ttns: TreeTensorNetworkState,
                root_id: str = "ttndo_root",
                bond_dim: int = 5
                ) -> SymmetricTTNDO:
    """
    Creates a TTNDO from a TTN.

    Args:
        ttns (TreeTensorNetworkState): The TTN to convert.
        root_id (str, optional): The identifier of the new root. Defaults to
            "ttndo_root".
        bond_dim (int, optional): The bond dimension of the new root.
            Defaults to 5.

    Returns:
        SymmetricTTNDO: The resulting TTNDO.
    """
    ttndo = SymmetricTTNDO()
    ttndo.add_trivial_root(root_id,
                           dimension=bond_dim)
    # Now we attach the root
    root_node, root_tensor = ttns.root
    ttns_root_id = root_node.identifier
    # We need to add a leg to be attached to the ttndo root.
    new_shape = tuple([1] + list(root_node.shape))
    root_tensor = deepcopy(root_tensor).reshape(new_shape)
    padding = [(0,0) for _ in range(len(root_tensor.shape))]
    padding[0] = (0,bond_dim-1)
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
        
        # Get parent node identifiers in the TTNDO
        parent_ket_id = ttndo.ket_id(node.identifier) if node.identifier != ttndo.root_id else ttndo.root_id
        
        # Make a copy of the child tensor to avoid modifying the original
        child_tensor_copy = deepcopy(child_tensor)
        
        # Check shapes and bond dimensions between nodes
        parent_ket_node = ttndo.nodes[parent_ket_id]
        
        # Get the bond dimension at the connection point
        target_bond_dim = parent_ket_node.shape[parent_leg]
        
        # Check and adjust child tensor dimensions if needed
        child_bond_dim = child_tensor_copy.shape[child_node.parent_leg]
        if child_bond_dim != target_bond_dim:
            # Reshape or pad the tensor to match the expected dimension
            child_shape = list(child_tensor_copy.shape)
            child_shape[child_node.parent_leg] = target_bond_dim
            
            # Create padded tensor
            padded_tensor = np.zeros(child_shape, dtype=child_tensor_copy.dtype)
            
            # Create slice objects for copying the data
            slices = tuple(slice(None, min(d1, d2)) if i == child_node.parent_leg else slice(None) 
                         for i, (d1, d2) in enumerate(zip(child_tensor_copy.shape, child_shape)))
            
            # Copy data to padded tensor
            padded_tensor[slices] = child_tensor_copy[slices]
            child_tensor_copy = padded_tensor
        
        ttndo.add_symmetric_children_to_parent(child_id,
                                               child_tensor_copy,
                                               child_tensor_copy.conj(),
                                               child_node.parent_leg,
                                               node.identifier,
                                               parent_leg)
        _rec_add_children(ttns, ttndo, child_node)


class IntertwinedTTNDO(TreeTensorNetworkState):
    """
    A class representing an intertwined tree tensor network density operator (TTNDO).
    
    The intertwined TTNDO structure distributes bra and ket nodes across the tree
    with lateral connections, creating a more flexible structure than the symmetric TTNDO.
    """

    def __init__(self, bra_suffix="_bra", ket_suffix="_ket", form="full"):
        """
        Initialize an IntertwinedTTNDO.
        
        Args:
            bra_suffix (str): Suffix to use for bra nodes. Defaults to "_bra".
            ket_suffix (str): Suffix to use for ket nodes. Defaults to "_ket".
            form (str): Form of the TTNDO structure. Options are:
                - "full": Both virtual and physical nodes have dual representation
                - "physical": Only physical nodes have dual representation
        """
        super().__init__()
        self.bra_suffix = bra_suffix
        self.ket_suffix = ket_suffix
        self.form = form



    def absorb_into_intertwined_ttndo_open_leg(self, node_id: str, matrix: np.ndarray):
        """
        Absorb a matrix into the input open leg of a node in the TTNDO.
        
        For IntertwinedTTNDO, each node has exactly two open legs. The input leg
        is assumed to be the one before the other leg (first open leg). The matrix
        is applied to this input leg.
        
        Args:
            node_id (str): The identifier of the node to which the matrix is applied.
            matrix (np.ndarray): The matrix to absorb. Must be a square matrix.
            
        Returns:
            IntertwinedTTNDO: The modified TTNDO for method chaining.
            
        Raises:
            AssertionError: If the matrix is not square or if the node doesn't have exactly 2 open legs.
        """
        node, node_tensor = self[node_id]
        
        # Check that the node has exactly 2 open legs
        open_legs = node.open_legs
        assert len(open_legs) == 2, f"Node {node_id} must have exactly 2 open legs, but has {len(open_legs)}."
        
        # The input leg is the first open leg
        input_leg = open_legs[0]
        
        # The matrix must be square
        assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1], \
            f"Matrix must be square, but has shape {matrix.shape}."
        
        # The matrix dimension must match the input leg dimension
        assert matrix.shape[0] == node_tensor.shape[input_leg], \
            f"Matrix dimension {matrix.shape[0]} does not match input leg dimension {node_tensor.shape[input_leg]}."
        
        # Absorb the matrix into the input leg (axis 1 of matrix is the input leg)
        new_tensor = np.tensordot(node_tensor, matrix, axes=(input_leg, 1))
        
        # Update the tensor in the TTNDO
        self.tensors[node_id] = new_tensor
        
        return self

    def contract(self) -> 'IntertwinedTTNDO':
        """
        Contract this intertwined TTNDO into a regular TTNDO.
        
        Returns:
            IntertwinedTTNDO: A new contracted TTNDO structure.
        """
        contracted = contract_intertwined_ttndo(self, self.bra_suffix, self.ket_suffix)
        return contracted

    def trace(self) -> complex:
        """
        Returns the trace of the IntertwinedTTNDO.
        
        If the TTNDO is not yet contracted, it will be contracted first.

        Returns:
            complex: The trace of the TTNDO.
        """
        if self.form == "full":        
            contracted_fully_intertwined_ttndo = self.contract()
            return trace_contracted_fully_intertwined_ttndo(contracted_fully_intertwined_ttndo)
        elif self.form == "physical":
            contracted_physically_intertwined_ttndo = self.contract()
            return trace_contracted_physically_intertwined_ttndo(contracted_physically_intertwined_ttndo)
        else:
            raise ValueError(f"Invalid form: {self.form}")

    def norm(self) -> complex:
        """
        Returns the norm of the IntertwinedTTNDO, which is just the trace.
        
        Returns:
            complex: The norm of the TTNDO.
        """
        return self.trace()

    def ttno_expectation_value(self, operator: TreeTensorNetworkOperator) -> complex:
        """
        Computes the expectation value of the IntertwinedTTNDO with respect to a TTNO.
        
        This is the trace of TTNO @ TTNDO: Tr(TTNO @ TTNDO)
        
        Args:
            operator (TreeTensorNetworkOperator): The TTNO to compute the expectation
                value with.

        Returns:
            complex: The expectation value of the TTNDO with respect to the TTNO.
        """
        if self.form == "full":
            contracted_ttndo = self.contract()
            return fully_intertwined_ttndo_ttno_expectation_value(operator, contracted_ttndo)
        
        elif self.form == "physical":
            contracted_ttndo = self.contract()
            return physically_intertwined_ttndo_ttno_expectation_value(operator, contracted_ttndo)

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

        # Apply operators to appropriate nodes
        for node_id, single_site_operator in operator.items():
            # Directly absorb the operator into the node tensor
            ttndo_copy = ttndo_copy.absorb_into_intertwined_ttndo_open_leg(node_id, single_site_operator)
        # Calculate the trace of the modified TTNDO
        if self.form == "full":
            return trace_contracted_fully_intertwined_ttndo(ttndo_copy)
        elif self.form == "physical":
            return trace_contracted_physically_intertwined_ttndo(ttndo_copy)
        else:
            raise ValueError(f"Invalid form: {self.form}")
    
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

def from_ttns_fully_intertwined(ttns: TreeTensorNetworkState,
                          bond_dim: int,
                          phys_tensor: ndarray,
                          bra_suffix: str = "_bra",
                          ket_suffix: str = "_ket") -> IntertwinedTTNDO:
    """
    Creates an intertwined TTNDO from a TTNS, preserving the full original structure.
    
    Args:
        ttns: Original TTNS structure
        bond_dim: Bond dimension for the TTNDO
        phys_tensor: Physical tensor template for leaf nodes
        bra_suffix: Suffix for bra nodes
        ket_suffix: Suffix for ket nodes
        
    Returns:
        IntertwinedTTNDO: The resulting intertwined TTNDO
    """
    # Create new IntertwinedTTNDO to hold the structure
    ttndo = IntertwinedTTNDO(bra_suffix, ket_suffix)
    created_nodes = {}
    
    # Get original root information
    original_root_id = ttns.root_id
    original_root_node = ttns.nodes[original_root_id]
    
    # Count how many children will go to each root node type
    num_children = len(original_root_node.children)
    ket_children = sum(1 for i in range(num_children) if i % 2 == 0)
    bra_children = num_children - ket_children
    
    # Create root ket node 
    root_ket_shape = [bond_dim]  # First leg for lateral connection
    root_ket_shape.extend([bond_dim] * ket_children)  # Only legs for ket's actual children
    root_ket_shape.append(1)  # Last leg is open/physical
    
    # Create root ket node
    root_ket_id = original_root_id + ket_suffix
    root_ket_node = Node(identifier=root_ket_id)
    
    # Root tensor with dynamic shape based on number of children
    root_ket_tensor = np.zeros(tuple(root_ket_shape), dtype=complex)
    
    # Use sparse initialization: only set the [0,0,...,0] element to 1.0
    root_ket_tensor[(0,) * (len(root_ket_shape) - 1) + (0,)] = 1.0
    
    ttndo.add_root(root_ket_node, root_ket_tensor)
    created_nodes[root_ket_id] = True
    
    # Create root bra node with appropriate shape
    root_bra_id = original_root_id + bra_suffix
    root_bra_node = Node(identifier=root_bra_id)
    
    # Bra tensor has shape based on its actual children
    root_bra_shape = [bond_dim]  # Lateral connection
    root_bra_shape.extend([bond_dim] * bra_children)  # Only legs for bra's actual children
    root_bra_shape.append(1)  # Last leg is open/physical
    
    # Create bra tensor with sparse initialization
    root_bra_tensor = np.zeros(tuple(root_bra_shape), dtype=complex)
    
    # Use sparse initialization: only set the [0,0,...,0] element to 1.0
    root_bra_tensor[(0,) * (len(root_bra_shape) - 1) + (0,)] = 1.0
    
    # Connect bra node to ket node
    ttndo.add_child_to_parent(root_bra_node, root_bra_tensor, 0, root_ket_id, 0, modify=True)
    created_nodes[root_bra_id] = True
    
    # Process all children from the original tree
    ket_leg_idx = 1  # Start after lateral leg
    bra_leg_idx = 1  # Start after parent leg
    
    for i, child_id in enumerate(original_root_node.children):
        # For each child, connect to both ket and bra nodes
        # Even children go to ket node, odd children go to bra node
        if i % 2 == 0:
            process_branch(ttns, ttndo, child_id, root_ket_id, ket_leg_idx, "ket", 
                          created_nodes, bond_dim, phys_tensor, bra_suffix, ket_suffix)
            ket_leg_idx += 1  # Increment leg index
        else:
            process_branch(ttns, ttndo, child_id, root_bra_id, bra_leg_idx, "bra", 
                          created_nodes, bond_dim, phys_tensor, bra_suffix, ket_suffix)
            bra_leg_idx += 1  # Increment leg index
    
    return ttndo

def process_branch(ttns: TreeTensorNetworkState, 
                  ttndo: TreeTensorNetworkState,
                  orig_node_id: str, 
                  parent_id: str, 
                  parent_leg: int, 
                  node_type: str, 
                  created_nodes: dict,
                  bond_dim: int, 
                  phys_tensor: ndarray,
                  bra_suffix: str, 
                  ket_suffix: str):
    """
    Process a branch of the tree
    """
    orig_node = ttns.nodes[orig_node_id]
    
    # Determine node IDs
    node_id = orig_node_id + (ket_suffix if node_type == "ket" else bra_suffix)
    lateral_type = "bra" if node_type == "ket" else "ket"
    lateral_id = orig_node_id + (bra_suffix if node_type == "ket" else ket_suffix)
    
    # Skip if already created
    if node_id in created_nodes:
        return
    
    # Check if physical node
    is_physical = len(orig_node.children) == 0
    
    if is_physical:
        # Process physical node
        create_physical_node_pair(ttndo, node_id, lateral_id, parent_id, parent_leg, 
                                node_type, lateral_type, created_nodes, bond_dim, ttns)
    else:
        # Virtual node - primary vs lateral determines leg count
        create_virtual_node_pair(ttndo, ttns, orig_node_id, node_id, lateral_id, parent_id, 
                               parent_leg, node_type, lateral_type, created_nodes, 
                               bond_dim, phys_tensor, bra_suffix, ket_suffix)

def create_virtual_node_pair(ttndo: TreeTensorNetworkState,
                           ttns: TreeTensorNetworkState,
                           orig_node_id: str,
                           node_id: str, 
                           lateral_id: str, 
                           parent_id: str, 
                           parent_leg: int, 
                           node_type: str, 
                           lateral_type: str,
                           created_nodes: dict, 
                           bond_dim: int, 
                           phys_tensor: ndarray,
                           bra_suffix: str, 
                           ket_suffix: str):
    """
    Create a virtual node and its lateral counterpart.
    
    Primary virtual nodes: Enough legs for all children plus parent, lateral, open
    Lateral virtual nodes: Enough legs for all children plus parent, open
    
    Uses sparse initialization where only the first element [0,0,...,0] is set to 1.0
    to ensure proper tensor contraction behavior.
    """
    orig_node = ttns.nodes[orig_node_id]
    num_children = len(orig_node.children)
    
    # Count how many children will go to each node type
    primary_children = sum(1 for i in range(num_children) if i % 2 == 0)
    lateral_children = num_children - primary_children
    
    # PRIMARY VIRTUAL NODE - legs: parent, lateral, children, open
    primary_shape = [bond_dim, bond_dim]  # parent, lateral
    primary_shape.extend([bond_dim] * primary_children)  # only allocate legs for actual children
    primary_shape.append(1)  # open/physical leg
    
    primary_tensor = np.zeros(tuple(primary_shape), dtype=complex)
    
    # Sparse initialization: only set the [0,0,...,0] element to 1.0
    primary_tensor[(0,) * (len(primary_shape) - 1) + (0,)] = 1.0
    
    if node_type == "bra":
        primary_tensor = primary_tensor.conj()
    
    # Create node and connect to parent
    primary_node = Node(identifier=node_id)
    ttndo.add_child_to_parent(primary_node, primary_tensor, 0, parent_id, parent_leg, modify=True)
    created_nodes[node_id] = True
    
    # LATERAL VIRTUAL NODE - legs: parent, children, open
    lateral_shape = [bond_dim]  # parent
    lateral_shape.extend([bond_dim] * lateral_children)  # only allocate legs for actual children
    lateral_shape.append(1)  # open/physical leg
    
    lateral_tensor = np.zeros(tuple(lateral_shape), dtype=complex)
    
    # Sparse initialization: only set the [0,0,...,0] element to 1.0
    lateral_tensor[(0,) * (len(lateral_shape) - 1) + (0,)] = 1.0
    
    if lateral_type == "bra":
        lateral_tensor = lateral_tensor.conj()
    
    # Create lateral node and connect to primary node
    lateral_node = Node(identifier=lateral_id)
    ttndo.add_child_to_parent(lateral_node, lateral_tensor, 0, node_id, 1, modify=True)
    created_nodes[lateral_id] = True
    
    # Process ALL children according to the original tree structure
    primary_leg_idx = 2  # Start after parent and lateral legs
    lateral_leg_idx = 1  # Start after parent leg
    
    for i, child_id in enumerate(orig_node.children):
        # Even children go to primary node, odd children go to lateral node
        if i % 2 == 0:
            # Primary nodes have parent and lateral legs before child legs
            process_branch(ttns, ttndo, child_id, node_id, primary_leg_idx, node_type, 
                          created_nodes, bond_dim, phys_tensor, bra_suffix, ket_suffix)
            primary_leg_idx += 1  # Increment leg index for primary node
        else:
            # Lateral nodes have only parent leg before child legs
            process_branch(ttns, ttndo, child_id, lateral_id, lateral_leg_idx, lateral_type, 
                          created_nodes, bond_dim, phys_tensor, bra_suffix, ket_suffix)
            lateral_leg_idx += 1  # Increment leg index for lateral node

def create_physical_node_pair(ttndo: TreeTensorNetworkState,
                             node_id: str, 
                             lateral_id: str, 
                             parent_id: str, 
                             parent_leg: int,
                             node_type: str,
                             lateral_type: str,
                             created_nodes: dict,
                             bond_dim: int, 
                             original_ttns: TreeTensorNetworkState):
    """
    Create a pair of physical nodes (ket and bra) using original TTNS tensors
    
    Primary physical node: 3 legs (parent, lateral, physical)
    Lateral physical node: 2 legs (parent, physical)
    """
    # Get original tensor from TTNS
    orig_node_id = node_id.split('_bra')[0].split('_ket')[0]  # Remove suffixes
    original_tensor = deepcopy(original_ttns.tensors[orig_node_id])
    phys_dim = original_tensor.shape[-1]
    parent_dim = original_tensor.shape[0]
    
    # PRIMARY PHYSICAL NODE - 3 legs: parent, lateral, physical
    primary_tensor = np.zeros((parent_dim, bond_dim, phys_dim), dtype=complex)
    
    # Only use index 0 for the lateral dimension to maintain quantum structure
    for i in range(parent_dim):
        for k in range(phys_dim):
            primary_tensor[i, 0, k] = original_tensor[i, k]
    
    if node_type == "bra":
        primary_tensor = primary_tensor.conj()
    
    # Create node and connect to parent
    primary_node = Node(identifier=node_id)
    ttndo.add_child_to_parent(primary_node, primary_tensor, 0, parent_id, parent_leg, modify=True)
    created_nodes[node_id] = True
    
    # LATERAL PHYSICAL NODE - 2 legs: parent, physical
    lateral_tensor = np.zeros((parent_dim, phys_dim), dtype=complex)
    
    # Copy values from original tensor
    for i in range(parent_dim):
        for j in range(phys_dim):
            lateral_tensor[i, j] = original_tensor[i, j]
    
    if lateral_type == "bra":
        lateral_tensor = lateral_tensor.conj()
    
    # Create node and connect to primary node
    lateral_node = Node(identifier=lateral_id)
    ttndo.add_child_to_parent(lateral_node, lateral_tensor, 0, node_id, 1, modify=True)
    created_nodes[lateral_id] = True

def contract_intertwined_ttndo(ttndo: TreeTensorNetworkState, 
                              bra_suffix: str = "_bra", 
                              ket_suffix: str = "_ket") -> TreeTensorNetworkState:
    """
    Contracts an intertwined TTNDO into a regular TTNDO.
    
    This function identifies lateral connections between bra and ket nodes and
    contracts them to form a regular tensor network where each node has two open legs.
    The result maintains the original tree structure but each node now has two open legs
    representing the density operator.
    
    Works with both "full" form (all nodes have bra/ket pairs) and "physical" form
    (only physical nodes have bra/ket pairs).
    
    Args:
        ttndo (TreeTensorNetworkState): The intertwined TTNDO to contract
        bra_suffix (str): The suffix used for bra nodes (default: "_bra")
        ket_suffix (str): The suffix used for ket nodes (default: "_ket")
        
    Returns:
        TreeTensorNetworkState: A contracted TTNS where each node has two open legs
    """
    # Helper function to identify virtual nodes
    def is_virtual_node(node_id):
        """Check if a node is a virtual node (not a physical node)."""
        return not (node_id.startswith("qubit") or node_id.startswith("site"))
    
    # Detect which form of TTNDO we're dealing with
    is_physical_form = hasattr(ttndo, 'form') and ttndo.form == "physical"
    
    # Determine if we should preserve virtual nodes based on ttndo.form
    preserve_virtual_nodes = is_physical_form
    
    # Create a deep copy to avoid modifying the original
    result_ttn = deepcopy(ttndo)
    
    # Dictionary to track which nodes have been processed
    processed_nodes = {}
    
    # Process nodes in a bottom-up manner to ensure we contract from leaves to root
    linearized_nodes = result_ttn.linearise()

    # First pass: Identify all node pairs that need to be contracted
    node_pairs = []
    for node_id in linearized_nodes:
        # Skip already processed nodes
        if node_id in processed_nodes:
            continue
            
        # Skip nodes that don't have a bra/ket suffix
        if not (node_id.endswith(bra_suffix) or node_id.endswith(ket_suffix)):
            continue
            
        # Get the corresponding lateral node ID
        if node_id.endswith(ket_suffix):
            base_id = node_id[:-len(ket_suffix)]
            lateral_id = base_id + bra_suffix
        else:  # node_id.endswith(bra_suffix)
            base_id = node_id[:-len(bra_suffix)]
            lateral_id = base_id + ket_suffix
        
        # Skip if lateral node doesn't exist or already processed
        if lateral_id not in result_ttn.nodes or lateral_id in processed_nodes:
            continue
            
        # Add to pairs for contraction
        node_pairs.append((node_id, lateral_id, base_id))
        processed_nodes[node_id] = True
        processed_nodes[lateral_id] = True
    
    # Second pass: Determine the correct contraction order (leaves first, then up)
    node_pairs.sort(key=lambda pair: pair[0])
    
    # Third pass: Contract node pairs in the determined order (from leaves to root)
    for _, (node_id, lateral_id, base_id) in enumerate(node_pairs):
        # Check if nodes still exist 
        if node_id not in result_ttn.nodes or lateral_id not in result_ttn.nodes:
            continue
            
        node = result_ttn.nodes[node_id]
        lateral_node = result_ttn.nodes[lateral_id]
        
        # Find the lateral connection
        lateral_connection = False
        
        # Check if nodes are directly connected
        node_neighbors = node.neighbouring_nodes()
        if lateral_id in node_neighbors:
            lateral_connection = True
                
        # If not found, check if the lateral node is connected to the primary node
        if not lateral_connection:
            lateral_node_neighbors = lateral_node.neighbouring_nodes()
            if node_id in lateral_node_neighbors:
                lateral_connection = True
        
        # If directly connected, contract the nodes
        if lateral_connection:
            # Always pass the ket node as the first parameter to contract_nodes
            # This follows the TTNO convention where ket legs come before bra legs.
            try:
                if node_id.endswith(ket_suffix):  # node_id is ket
                    result_ttn.contract_nodes(node_id, lateral_id, new_identifier=base_id)
                else:  # lateral_id is ket
                    result_ttn.contract_nodes(lateral_id, node_id, new_identifier=base_id)
            except Exception:
                raise
        else:
            # Nodes are not directly connected 
            print(f"Warning: Nodes {node_id} and {lateral_id} are not directly connected")
    
    # If using physical form with preservation of virtual nodes, we're done
    if preserve_virtual_nodes:
        return result_ttn
    
    # Otherwise, for full intertwined form, ensure all virtual nodes have two open legs
    for node_id in list(result_ttn.nodes.keys()):
        # Only process virtual nodes without suffixes
        if (not is_virtual_node(node_id) or 
            node_id.endswith(bra_suffix) or 
            node_id.endswith(ket_suffix)):
            continue
            
        # Get the tensor and node
        node = result_ttn.nodes[node_id]
        
        # If the node already has exactly 2 open legs, skip it
        if len(node.open_legs) == 2:
            continue

    return result_ttn

def from_ttns_physically_intertwined(ttns: TreeTensorNetworkState,
                                   bond_dim: int,
                                   phys_tensor: ndarray,
                                   bra_suffix: str = "_bra",
                                   ket_suffix: str = "_ket") -> IntertwinedTTNDO:
    """
    Creates a physically intertwined TTNDO from a TTNS, where only physical nodes have dual representation.
    Virtual nodes have a single tensor, while physical nodes maintain bra and ket parts.
    
    Note: Trace and expectation value calculations for physically intertwined TTNDOs are not yet implemented
    and will be added in a future update. Currently, the structure supports contracting the dual nodes
    correctly but not computing traces or expectation values.
    
    Args:
        ttns: Original TTNS structure
        bond_dim: Bond dimension for the TTNDO
        phys_tensor: Physical tensor template for leaf nodes
        bra_suffix: Suffix for bra nodes
        ket_suffix: Suffix for ket nodes
        
    Returns:
        IntertwinedTTNDO: The resulting physically intertwined TTNDO
    """
    # Create new IntertwinedTTNDO to hold the structure
    ttndo = IntertwinedTTNDO(bra_suffix, ket_suffix, form="physical")
    created_nodes = {}
    
    # Get original root information
    original_root_id = ttns.root_id
    original_root_node = ttns.nodes[original_root_id]
    
    # For the root, we use a single node with legs for all children
    root_shape = [bond_dim] * len(original_root_node.children)  # One leg for each child
    root_shape.append(1)  # Last leg is open/physical
    
    # Create root tensor with sparse initialization
    root_tensor = np.zeros(tuple(root_shape), dtype=complex)
    
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
    Process a branch in physically intertwined TTNDO - creates single virtual nodes
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
    
    The virtual node has legs for its parent and all its children, plus one open leg.
    Uses sparse initialization where only the first element [0,0,...,0] is set to 1.0.
    """
    orig_node = ttns.nodes[orig_node_id]
    num_children = len(orig_node.children)
    
    # Shape: parent, children, open
    node_shape = [bond_dim]  # parent leg
    node_shape.extend([bond_dim] * num_children)  # child legs 
    node_shape.append(1)  # open/physical leg
    
    # Create tensor with sparse initialization
    node_tensor = np.zeros(tuple(node_shape), dtype=complex)
    
    # Sparse initialization: only set the [0,0,...,0] element to 1.0
    node_tensor[(0,) * (len(node_shape) - 1) + (0,)] = 1.0
    
    # Create node and connect to parent
    node = Node(identifier=orig_node_id)
    ttndo.add_child_to_parent(node, node_tensor, 0, parent_id, parent_leg, modify=True)
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
    This ensures compatibility with contract_intertwined_ttndo.
    """
    # Get original tensor from TTNS
    original_tensor = deepcopy(original_ttns.tensors[orig_node_id])
    phys_dim = original_tensor.shape[-1]
    parent_dim = original_tensor.shape[0]
    
    # Create ket node ID
    ket_id = orig_node_id + ket_suffix
    
    # KET PHYSICAL NODE - 3 legs: parent, lateral, physical
    ket_tensor = np.zeros((parent_dim, bond_dim, phys_dim), dtype=complex)
    
    # Copy values from original tensor with lateral bond in between
    for i in range(parent_dim):
        for j in range(phys_dim):
            ket_tensor[i, 0, j] = original_tensor[i, j]
    
    # Create and connect ket node to parent
    ket_node = Node(identifier=ket_id)
    ttndo.add_child_to_parent(ket_node, ket_tensor, 0, parent_id, parent_leg, modify=True)
    created_nodes[ket_id] = True
    
    # Create bra node ID
    bra_id = orig_node_id + bra_suffix
    
    # BRA PHYSICAL NODE - 2 legs: parent, physical
    bra_tensor = np.zeros((bond_dim, phys_dim), dtype=complex)
    
    # Copy and conjugate values from original tensor
    for j in range(phys_dim):
        bra_tensor[0, j] = np.conj(original_tensor[0, j])
    
    # Create and connect bra node to ket node through lateral connection
    bra_node = Node(identifier=bra_id)
    ttndo.add_child_to_parent(bra_node, bra_tensor, 0, ket_id, 1, modify=True)
    created_nodes[bra_id] = True
