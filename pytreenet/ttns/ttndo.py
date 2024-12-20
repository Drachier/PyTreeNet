"""
Provides a class that represents dennsity operators as a tree tensor network.

This leads to the concept of a tree tensor network density operator (TTNDO).

"""
from numpy import eye, ndarray
from re import match

from ..core.node import Node
from .ttns import TreeTensorNetworkState
from ..util.ttn_exceptions import positivity_check

class TreeTensorNetworkDensityOperator(TreeTensorNetworkState):
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
                new root. Defaults to 1.
        
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
