"""
This module contains functions used to contract two nodes.
"""
from copy import copy

import numpy as np
from numpy.typing import NDArray

from ..core.node import Node
from ..core.graph_node import determine_parentage

def contract_nodes(node1: Node,
                   tensor1: NDArray[np.complex128],
                   node2: Node,
                   tensor2: NDArray[np.complex128],
                   new_identifier: str = ""
                   ) -> tuple[Node, NDArray[np.complex128]]:
    """
    Contract two nodes and return the new node and its tensor.

    Args:
        node1 (Node): The first node to contract.
        tensor1 (NDArray[np.complex128]): The tensor associated with the first
            node.
        node2 (Node): The second node to contract.
        tensor2 (NDArray[np.complex128]): The tensor associated with the
            second node.
        new_identifier (str, optional): Identifier for the new node. Defaults to
            ``parent_id + "contr" + child_id``.
    
    Returns:
        tuple[Node, NDArray[np.complex128]]: A tuple containing the new node
            and its tensor. The resulting leg order as defined by the node is
            the following:

            ``(parent_parent_leg, node1_children_legs, node2_children_legs,
            node1_open_legs, node2_open_legs)``

            This is not the leg order of the tensor itself.

    """
    if new_identifier == "":
        new_identifier = default_new_identifier(node1, node2)
    parent_node, child_node = determine_parentage(node1, node2)
    tensor1_is_parent = parent_node.identifier == node1.identifier
    if tensor1_is_parent:
        parent_tensor = tensor1
        child_tensor = tensor2
    else:
        parent_tensor = tensor2
        child_tensor = tensor1
    new_tensor = data_contraction(parent_node, parent_tensor,
                                    child_node, child_tensor)
    new_node = contracted_node_creation(new_tensor,
                                        parent_node,
                                        child_node,
                                        tensor1_is_parent,
                                        new_identifier = new_identifier)
    return new_node, new_tensor

def data_contraction(parent_node: Node,
                     parent_tensor: NDArray[np.complex128],
                     child_node: Node,
                     child_tensor: NDArray[np.complex128]
                     ) -> NDArray[np.complex128]:
    """
    Contracts the underlying tensors of two nodes and returns the resulting
        tensor.

    Args:
        parent_node (Node): The parent node in the contraction.
        parent_tensor (NDArray[np.complex128]): The tensor associated with the
            parent node.
        child_node (Node): The child node in the contraction.
        child_tensor (NDArray[np.complex128]): The tensor associated with the
            child node.

    Returns:
        NDArray[np.complex128]: The resulting tensor after contracting the two tensors.

    """
    axes = (parent_node.neighbour_index(child_node.identifier),
            child_node.parent_leg)
    new_tensor = np.tensordot(parent_tensor, child_tensor, axes=axes)
    return new_tensor

def contracted_node_creation(new_tensor: NDArray[np.complex128],
                             parent_node: Node,
                             child_node: Node,
                             node1_is_parent: bool,
                             new_identifier: str = ""
                             ) -> Node:
    """
    Create the new node corresponding to the contraction of two given nodes.

    Args:
        new_tensor (NDArray[np.complex128]): The tensor resulting from the
            contraction of the two nodes.
        parent_node (Node): The parent node in the contraction.
        child_node (Node): The child node in the contraction.
        node1_is_parent (bool): A boolean indicating whether the first
            node is the parent node in the contraction. Required to
            determine the correct leg order.
        new_identifier (str, optional): Identifier for the new node. Defaults to
            ``parent_id + "contr" + child_id``.
    
    Returns:
        Node: The new node created from the contraction of the two nodes.

    """
    if new_identifier == "":
        new_identifier = default_new_identifier(parent_node, child_node)
    new_node = Node(tensor=new_tensor,
                    identifier=new_identifier)
    # Actual tensor leg of new_tensor now have the form
    # (parent_of_parent, remaining_children_of_parent, open_of_parent,
    # children_of_child, open_of_child)
    # However, the newly created node is basically a node with only open legs.
    if not parent_node.is_root():
        new_node.open_leg_to_parent(parent_node.parent, 0)
    parent_children = copy(parent_node.children)
    parent_children.remove(child_node.identifier)
    parent_child_dict = {identifier: leg_value + parent_node.nparents()
                            for leg_value, identifier in enumerate(parent_children)}
    child_children_dict = {identifier: leg_value + parent_node.nlegs() - 1
                            for leg_value, identifier in enumerate(child_node.children)}
    # Add the children in the correct order
    if node1_is_parent:
        parent_child_dict.update(child_children_dict)
        new_node.open_legs_to_children(parent_child_dict)
    else:
        child_children_dict.update(parent_child_dict)
        new_node.open_legs_to_children(child_children_dict)
    # Correct the order of the open legs, if necessary
    ## Extra is only to separate the dealing with the chlidren and open legs
    if not node1_is_parent:
        new_nvirt = new_node.nvirt_legs()
        range_parent = range(new_nvirt, new_nvirt + parent_node.nopen_legs())
        range_child = range(new_nvirt + parent_node.nopen_legs(), new_node.nlegs())
        new_node.exchange_open_leg_ranges(range_parent, range_child)
    return new_node

def default_new_identifier(node1: Node,
                             node2: Node) -> str:
    """
    Generate a default identifier for the new node based on the parent and
        child node identifiers.

    Args:
        node1 (Node): The parent node in the contraction.
        node2 (Node): The child node in the contraction.

    Returns:
        str: A string identifier for the new node.
    """
    return f"{node1.identifier}contr{node2.identifier}"
