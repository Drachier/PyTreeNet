"""
Implements a generation function to generate binary TTNS.
"""
from copy import deepcopy
from typing import Self

from numpy import ndarray, zeros

from ..ttns import TreeTensorNetworkState
from ..core.node import Node
from ..util.ttn_exceptions import positivity_check
from .special_nodes import constant_bd_trivial_node

__all__ = ["generate_binary_ttns"]

class HelperNode:
    """
    A small helper class that stores a node, as well as its position in the tree.
    """

    def __init__(self, node: Node, level: int, position: int):
        self.node = node
        self.level = level
        self.position = position

    def children_level(self) -> int:
        """
        Returns the level of the children of the node.
        """
        return self.level + 1

    def children_position(self) -> tuple[int, int]:
        """
        Returns the horizontal position of the children of the node.
        """
        return 2*self.position, 2*self.position + 1

    def parent_legs(self, ttns: TreeTensorNetworkState
                    ) -> tuple[int, int]:
        """
        Returns the legs of the current node that serve as parent leg for the
        children.
        """
        if self.node.identifier == ttns.root_id:
            return 0, 1
        return 1, 2

    def generate_children_nodes(self,
                                virtual_prefix: str
                                ) -> tuple[Self, Self]:
        """
        Generates the children of the current node.
        """
        current_level = self.children_level()
        left_child_position, right_child_position = self.children_position()
        left_child_id = create_virtual_node_id(current_level,
                                               left_child_position,
                                               virtual_prefix)
        left_child_node = Node(identifier=left_child_id)
        right_child_id = create_virtual_node_id(current_level,
                                                right_child_position,
                                                virtual_prefix)
        right_child_node = Node(identifier=right_child_id)
        left_helper = HelperNode(left_child_node,
                                 current_level,
                                 left_child_position)
        right_helper = HelperNode(right_child_node,
                                  current_level,
                                  right_child_position)
        return left_helper, right_helper

    def generate_children_tensors(self,
                                  bond_dim: int
                                  ) -> tuple[ndarray, ndarray]:
        """
        Generates the tensors of the children of the current node.
        """
        l_tensor = create_non_root_virt_tensor(bond_dim)
        r_tensor = create_non_root_virt_tensor(bond_dim)
        return l_tensor, r_tensor

def create_virtual_node_id(level: int, position: int, prefix: str) -> str:
    """
    Creates a virtual node identifier.
    """
    return prefix + str(level) + "_" + str(position)

def create_non_root_virt_tensor(bond_dim: int) -> ndarray:
    """
    Creates a virtual tensor for a non-root node.
    """
    virt_tensor = constant_bd_trivial_node(bond_dim, 3)
    return virt_tensor

def add_all_nodes(num_phys: int,
                  bond_dim: int,
                  virtual_prefix: str
                  ) -> list[Node, TreeTensorNetworkState]:
    """
    Adds all nodes as virtual nodes to the tree tensor network state.

    Args:
        num_phys (int): The number of physical nodes that should be added.
        bond_dim (int): The bond dimension of the tree tensor network state.
        virtual_prefix (str): The prefix for the virtual nodes.

    Returns:
        list[Node]: A list of all nodes that should be physical nodes.
        TreeTensorNetworkState: The tree tensor network state with all nodes.
    
    """
    ttns = TreeTensorNetworkState()
    root_node = Node(identifier=virtual_prefix + "0_0")
    root_tensor = constant_bd_trivial_node(bond_dim, 2)
    ttns.add_root(root_node, root_tensor)
    phys_nodes = [HelperNode(root_node, 0, 0)]
    while len(phys_nodes) != num_phys:
        hnode = phys_nodes.pop(0)
        if len(phys_nodes) == num_phys:
            break
        l_helper, r_helper = hnode.generate_children_nodes(virtual_prefix)
        l_tensor, r_tensor = hnode.generate_children_tensors(bond_dim)
        parent_legs = hnode.parent_legs(ttns)
        parent_id = hnode.node.identifier
        ttns.add_child_to_parent(l_helper.node,
                                l_tensor,
                                0,
                                parent_id,
                                parent_legs[0])
        ttns.add_child_to_parent(r_helper.node,
                                r_tensor,
                                0,
                                parent_id,
                                parent_legs[1])
        # This modification is intended
        phys_nodes.append(l_helper)
        phys_nodes.append(r_helper)
    phys_nodes = [hnode.node for hnode in phys_nodes]
    return phys_nodes, ttns

def transform_phys_nodes(ttns: TreeTensorNetworkState,
                         phys_nodes: list[Node],
                         phys_tensor: ndarray,
                         phys_prefix: str
                         ) -> TreeTensorNetworkState:
    """
    Transforms the physical nodes to physical nodes in the tree tensor network state.

    The nodes are changed in place.

    Args:
        ttns (TreeTensorNetworkState): The tree tensor network state.
        phys_nodes (list[Node]): The physical nodes.
        phys_tensor (ndarray): The tensor for the physical sites.
        phys_prefix (str): The prefix for the physical nodes.
    
    Returns:
        TreeTensorNetworkState: The tree tensor network state with the physical nodes.

    """
    for i, node in enumerate(phys_nodes):
        new_id = phys_prefix + str(i)
        ttns.replace_node(new_id,
                          node.identifier,
                          deepcopy(phys_tensor))
    return ttns

def generate_binary_ttns(num_phys: int,
                         bond_dim: int,
                         phys_tensor: ndarray,
                         phys_prefix: str = "site",
                         virtual_prefix: str = "node"
                         ) -> TreeTensorNetworkState:
    """
    Generates a binary tree tensor network state.

    The TTNS will have constant bond dimension and all physical sites will have
    the same tensor.
    
    Args:
        num_phys (int): The number of physical sites.
        bond_dim (int): The bond dimension of the tree tensor network state.
        phys_tensor (ndarray): The tensor for the physical sites.
        phys_prefix (str): The prefix for the physical nodes.
        virtual_prefix (str): The prefix for the virtual nodes.
    
    Returns:
        TreeTensorNetworkState: The generated tree tensor network state.

    """
    positivity_check(num_phys, "number of physical sites")
    positivity_check(bond_dim, "bond dimension")
    phys_nodes, ttns = add_all_nodes(num_phys, bond_dim, virtual_prefix)
    ttns = transform_phys_nodes(ttns, phys_nodes, phys_tensor, phys_prefix)
    return ttns
