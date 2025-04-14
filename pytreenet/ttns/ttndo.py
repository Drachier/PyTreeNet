"""
Provides a class that represents dennsity operators as a tree tensor network.

This leads to the concept of a tree tensor network density operator (TTNDO).

"""
from re import match
from copy import deepcopy

from numpy import eye, ndarray, pad

from ..core.node import Node
from .ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..util.ttn_exceptions import positivity_check
from ..operators.tensorproduct import TensorProduct
from ..contractions.ttndo_contractions import trace_ttndo, ttndo_ttno_expectation_value

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
        return trace_ttndo(self)

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
        return ttndo_ttno_expectation_value(self, operator)

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
        for node_id, single_site_operator in operator.items():
            ket_id = self.ket_id(node_id)
            # Can be improved once shallow copying is possible
            ttn = deepcopy(self)
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

def from_ttns(ttns: TreeTensorNetworkState,
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
