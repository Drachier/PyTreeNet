"""
This module contains functions generating random Tree Tensor Networks.

There are a variety of given tree topologies, which can be filled with random
tensors.
"""
from __future__ import annotations
from typing import Union, List, Tuple
from enum import Enum

from ..core.ttn import TreeTensorNetwork
from .random_node import random_tensor_node
from .random_ttns import RandomTTNSMode

def random_small_ttns(mode: RandomTTNSMode = RandomTTNSMode.DIFFVIRT) -> TreeTensorNetwork:
    """
    Generates a small TreeTensorNetworkState of three nodes:
    The root (`"root"`) and its two children (`"c1"` and `"c2"`). The associated 
    tensors are random, but their dimensions are set.

    Args:
        mode (RandomTTNSMode): The mode of random generation of the TTNS. If mode
            is DIFFVIRT, the virtual bond dimensions are as follows::

                        |2
                        |
                        r
                       / \\
                 3|  5/  6\\   |4
                  |  /     \\  |
                   c1        c2

            Otherwise all virtual bond dimensions default to 2. If the mode is
            SAMEPHYS all phyiscal dimensions will default to 2.

    Returns:
        TreeTensorNetwork: A tree tensor network with the above topology and
            randomly filled tensors.
    """
    random_ttns = TreeTensorNetwork()
    if mode == RandomTTNSMode.DIFFVIRT:
        root_node, root_tensor = random_tensor_node((5,6,2),"root")
        random_ttns.add_root(root_node, root_tensor)
        c1_node, c1_tensor = random_tensor_node((5,3),"c1")
        random_ttns.add_child_to_parent(c1_node, c1_tensor, 0, "root", 0)
        c2_node, c2_tensor = random_tensor_node((6,4),"c2")
        random_ttns.add_child_to_parent(c2_node, c2_tensor, 0, "root", 1)
    elif mode == RandomTTNSMode.SAMEPHYS:
        root_node, root_tensor = random_tensor_node((5,6,2),"root")
        random_ttns.add_root(root_node, root_tensor)
        c1_node, c1_tensor = random_tensor_node((5,2),"c1")
        random_ttns.add_child_to_parent(c1_node, c1_tensor, 0, "root", 0)
        c2_node, c2_tensor = random_tensor_node((6,2),"c2")
        random_ttns.add_child_to_parent(c2_node, c2_tensor, 0, "root", 1)
    else:
        root_node, root_tensor = random_tensor_node((2,2,2),"root")
        random_ttns.add_root(root_node, root_tensor)
        c1_node, c1_tensor = random_tensor_node((2,3),"c1")
        random_ttns.add_child_to_parent(c1_node, c1_tensor, 0, "root", 0)
        c2_node, c2_tensor = random_tensor_node((2,4),"c2")
        random_ttns.add_child_to_parent(c2_node, c2_tensor, 0, "root", 1)
    return random_ttns

def random_big_ttns(mode: RandomTTNSMode = RandomTTNSMode.SAME) -> TreeTensorNetwork:
    """
    Generates a big TTNS
    
    The node identifiers of the form `"site" + int`. The identifiers and
    dimensions are set, but the associated tensors are random.

    Args:
        mode (RandomTTNSMode): The mode of random generation of the TTNS.
            Currently the only mode supported is SAME.

    Returns:
        TreeTensorNetwork: A random TTNS with the following topology::

                1------6-----7
               / \\     \\
              /   \\     \\
             /     \\     8 
            2       4
            |       |
            |       |
            3       5
        
    """
    if mode == RandomTTNSMode.SAME:
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

        random_ttns = TreeTensorNetwork()
        random_ttns.add_root(node1, tensor1)
        random_ttns.add_child_to_parent(node2, tensor2, 0, "site1", 0)
        random_ttns.add_child_to_parent(node3, tensor3, 0, "site2", 1)
        random_ttns.add_child_to_parent(node4, tensor4, 0, "site1", 1)
        random_ttns.add_child_to_parent(node5, tensor5, 0, "site4", 1)
        random_ttns.add_child_to_parent(node6, tensor6, 0, "site1", 2)
        random_ttns.add_child_to_parent(node7, tensor7, 0, "site6", 1)
        random_ttns.add_child_to_parent(node8, tensor8, 0, "site6", 2)
        return random_ttns
    errstr = "The only supported mode is RandomTTNSMode.SAME"
    raise NotImplementedError(errstr)

def random_big_ttns_two_root_children(mode: Union[RandomTTNSMode,List[Tuple[int]]] = RandomTTNSMode.SAME
                                      ) -> TreeTensorNetwork:
    """
    Returns a random big TTNS where the root has only two children.

    For testing it is important to know that the children of 1 will be in a
    different order, if the TTNS is orthogonalised.

    Args:
        mode (RandomTTNSMode): The mode of random generation of the TTNS. If it
            is SAME all legs will be chosen as 2. For DIFFVIRT the virtual
            bond dimensons are all different. Alternatively, a list of tuples
            representing the shapes of the tensors can be passed.
    
    Returns:
        TreeTensorNetwork: A random TTNS with the topology::

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
    if mode == RandomTTNSMode.SAME:
        shapes = [(2,2,2),(2,2,2,2),(2,2),(2,2,2,2),
                  (2,2),(2,2),(2,2,2),(2,2)]
    elif mode == RandomTTNSMode.DIFFVIRT:
        shapes = [(7,6,2),(7,4,5,2),(4,2),(5,2,3,2),
                  (2,2),(3,2),(6,3,2),(3,2)]
    elif mode == RandomTTNSMode.TRIVIALVIRTUAL:
        shapes = [(1,1,2),(1,1,1,2),(1,2),(1,1,1,2),
                  (1,2),(1,2),(1,1,2),(1,2)]
    elif isinstance(mode, list):
        assert len(mode) == 8, "The list must have 8 elements!"
        shapes = mode
    else:
        errstr = "Only RandomTTNSMode.SAME, RandomTTNSMode.DIFFVIRT or a list of shapes is supported!"
        raise NotImplementedError(errstr)

    nodes = [random_tensor_node(shape, identifier="site"+str(i))
             for i, shape in enumerate(shapes)]
    random_ttns = TreeTensorNetwork()
    random_ttns.add_root(nodes[0][0], nodes[0][1])
    random_ttns.add_child_to_parent(nodes[1][0],nodes[1][1],0,"site0",0)
    random_ttns.add_child_to_parent(nodes[2][0],nodes[2][1],0,"site1",1)
    random_ttns.add_child_to_parent(nodes[3][0],nodes[3][1],0,"site1",2)
    random_ttns.add_child_to_parent(nodes[4][0],nodes[4][1],0,"site3",1)
    random_ttns.add_child_to_parent(nodes[5][0],nodes[5][1],0,"site3",2)
    random_ttns.add_child_to_parent(nodes[6][0],nodes[6][1],0,"site0",1)
    random_ttns.add_child_to_parent(nodes[7][0],nodes[7][1],0,"site6",1)
    return random_ttns
