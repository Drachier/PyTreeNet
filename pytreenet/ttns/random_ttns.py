"""
This module contains functions for generating random Tree Tensor Network States
(TTNS).

Each node in the TTNS is a tensor, and the connections between nodes are 
represented by the indices of these tensors.

This module is part of a larger package for working with Tree Tensor Networks.
"""
from __future__ import annotations

from .ttns import TreeTensorNetworkState
from ..node import random_tensor_node

def random_small_ttns() -> TreeTensorNetworkState:
    """
    Generates a small TreeTensorNetworkState of three nodes:
    The root (`"root"`) and its two children (`"c1"` and `"c2"`). The associated 
    tensors are random, but their dimensions are set.

                |2
                |
                r
               / \\
         3|  5/  6\\   |4
          |  /     \\  |
           c1        c2
    """
    random_ttns = TreeTensorNetworkState()
    root_node, root_tensor = random_tensor_node((5,6,2),"root")
    random_ttns.add_root(root_node, root_tensor)
    c1_node, c1_tensor = random_tensor_node((5,3),"c1")
    random_ttns.add_child_to_parent(c1_node, c1_tensor, 0, "root", 0)
    c2_node, c2_tensor = random_tensor_node((6,4),"c2")
    random_ttns.add_child_to_parent(c2_node, c2_tensor, 0, "root", 1)
    return random_ttns

def random_big_ttns(option: str) -> TreeTensorNetworkState:
    """
    Generates a big TTNS with identifiers of the form `"site" + int`.
     The identifiers and dimensions are set, but the associated tensors
     are random.

    Args:
        option (str): A parameter to choose between different topologies and
         dimensions.
    """

    if option == "same_dimension":
        # All dimensions virtual and physical are initially the same
        # We need a ttn to work on.
        node1, tensor1 = random_tensor_node((2,2,2,2), identifier="site1")
        node2, tensor2 = random_tensor_node((2,2,2), identifier="site2")
        node3, tensor3 = random_tensor_node((2,2), identifier="site3")
        node4, tensor4 = random_tensor_node((2,2,2), identifier="site4")
        node5, tensor5 = random_tensor_node((2,2), identifier="site5")
        node6, tensor6 = random_tensor_node((2,2,2,2), identifier="site6")
        node7, tensor7 = random_tensor_node((2,2), identifier="site7")
        node8, tensor8 = random_tensor_node((2,2), identifier="site8")

        random_ttns = TreeTensorNetworkState()
        random_ttns.add_root(node1, tensor1)
        random_ttns.add_child_to_parent(node2, tensor2, 0, "site1", 0)
        random_ttns.add_child_to_parent(node3, tensor3, 0, "site2", 1)
        random_ttns.add_child_to_parent(node4, tensor4, 0, "site1", 1)
        random_ttns.add_child_to_parent(node5, tensor5, 0, "site4", 1)
        random_ttns.add_child_to_parent(node6, tensor6, 0, "site1", 2)
        random_ttns.add_child_to_parent(node7, tensor7, 0, "site6", 1)
        random_ttns.add_child_to_parent(node8, tensor8, 0, "site6", 2)
    return random_ttns

def random_big_ttns_two_root_children(mode:str="same_dimension") -> TreeTensorNetworkState:
    """
    Provides a ttns of the form
                0
               / \\
              /   \\
             1     6
            / \\    \\
           /   \\    \\
          2     3     7
               / \\
              /   \\
             4     5
    
    """
    if mode == "same_dimension":
        shapes = [(2,2,2),(2,2,2,2),(2,2),(2,2,2,2),
                  (2,2),(2,2),(2,2,2),(2,2)]
    elif mode == "different_virt_dimensions":
        shapes = [(7,6,2),(7,4,5,2),(4,2),(5,2,3,2),
                  (2,2),(3,2),(6,3,2),(3,2)]

    nodes = [random_tensor_node(shape, identifier="site"+str(i))
             for i, shape in enumerate(shapes)]
    random_ttns = TreeTensorNetworkState()
    random_ttns.add_root(nodes[0][0], nodes[0][1])
    random_ttns.add_child_to_parent(nodes[1][0],nodes[1][1],0,"site0",0)
    random_ttns.add_child_to_parent(nodes[2][0],nodes[2][1],0,"site1",1)
    random_ttns.add_child_to_parent(nodes[3][0],nodes[3][1],0,"site1",2)
    random_ttns.add_child_to_parent(nodes[4][0],nodes[4][1],0,"site3",1)
    random_ttns.add_child_to_parent(nodes[5][0],nodes[5][1],0,"site3",2)
    random_ttns.add_child_to_parent(nodes[6][0],nodes[6][1],0,"site0",1)
    random_ttns.add_child_to_parent(nodes[7][0],nodes[7][1],0,"site6",1)
    return random_ttns
