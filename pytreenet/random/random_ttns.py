"""
This module contains functions generating random Tree Tensor Network States.

There are a variety of given tree topologies, which can be filled with random
tensors.
"""
from __future__ import annotations
from typing import Union, List, Tuple
from enum import Enum

from ..ttns.ttns import TreeTensorNetworkState
from .random_node import random_tensor_node
from .random_matrices import crandn
from ..core.tree_structure import LinearisationMode

class RandomTTNSMode(Enum):
    """
    An enumeration for the different modes of random generation of TTNS.

    The modes are usually concerned with the different ways to choose the
    virtual bond dimensions.

    #. SAME: All bond dimensions are chosen equal
    #. DIFFVIRT: Virtual dimensions are chosen different. The exact size depends
        on the topology used.
    #. SAMEPHYS: Forces the same physical dimensions for the TTNS.

    """
    SAME = "same_dimension"
    DIFFVIRT = "different_virt_dimensions"
    SAMEPHYS = "same_phys_dim"
    DIFFPHYS = "different_phys_dim"
    SAMEVIRT = "same_virt_dim"
    TRIVIALVIRTUAL = "trivial_virtual"

def random_small_ttns(mode: RandomTTNSMode = RandomTTNSMode.DIFFVIRT,
                      ids: Union[List[str],None] = None,
                      seed=None) -> TreeTensorNetworkState:
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
        ids (List[str]|None): A list of identifiers for the nodes. If None, the
            the identifiers will be set to `"root"`, `"c1"` and `"c2"`. If a
            list is provided, the identifiers will be set in that order.
        seed: An optional seed for the random number generator.

    Returns:
        TreeTensorNetwork: A tree tensor network with the above topology and
            randomly filled tensors.
    """
    if ids is None:
        ids = ["root", "c1", "c2"]
    root_id = ids[0]
    c1_id = ids[1]
    c2_id = ids[2]
    random_ttns = TreeTensorNetworkState()
    if mode == RandomTTNSMode.DIFFVIRT:
        root_node, root_tensor = random_tensor_node((5,6,2),root_id, seed=seed)
        random_ttns.add_root(root_node, root_tensor)
        c1_node, c1_tensor = random_tensor_node((5,3),c1_id, seed=seed)
        random_ttns.add_child_to_parent(c1_node, c1_tensor, 0, root_id, 0)
        c2_node, c2_tensor = random_tensor_node((6,4),c2_id, seed=seed)
        random_ttns.add_child_to_parent(c2_node, c2_tensor, 0, root_id, 1)
    elif mode == RandomTTNSMode.SAMEPHYS:
        root_node, root_tensor = random_tensor_node((5,6,2),root_id, seed=seed)
        random_ttns.add_root(root_node, root_tensor)
        c1_node, c1_tensor = random_tensor_node((5,2),c1_id, seed=seed)
        random_ttns.add_child_to_parent(c1_node, c1_tensor, 0, root_id, 0)
        c2_node, c2_tensor = random_tensor_node((6,2),c2_id, seed=seed)
        random_ttns.add_child_to_parent(c2_node, c2_tensor, 0, root_id, 1)
    else:
        root_node, root_tensor = random_tensor_node((2,2,2),root_id, seed=seed)
        random_ttns.add_root(root_node, root_tensor)
        c1_node, c1_tensor = random_tensor_node((2,3),c1_id, seed=seed)
        random_ttns.add_child_to_parent(c1_node, c1_tensor, 0, root_id, 0)
        c2_node, c2_tensor = random_tensor_node((2,4),c2_id, seed=seed)
        random_ttns.add_child_to_parent(c2_node, c2_tensor, 0, root_id, 1)
    return random_ttns

def random_big_ttns(mode: RandomTTNSMode = RandomTTNSMode.SAME,
                   seed=None
                   ) -> TreeTensorNetworkState:
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
        shapes = [(2,2,2,2),(2,2,2),(2,2),(2,2,2),
                  (2,2),(2,2,2,2),(2,2),(2,2)]
    elif mode == RandomTTNSMode.DIFFVIRT:
        shapes = [(4,3,6,1),(4,1,3),(1,2),(3,2,2),
                  (2,2),(6,3,2,1),(3,4),(2,3)]
    else:
        errstr = "The only supported mode is RandomTTNSMode.SAME"
        raise NotImplementedError(errstr)
    nodes = [random_tensor_node(shape, identifier="site"+str(i+1), seed=seed)
                 for i, shape in enumerate(shapes)]
    random_ttns = TreeTensorNetworkState()
    random_ttns.add_root(nodes[0][0], nodes[0][1])
    random_ttns.add_child_to_parent(nodes[1][0], nodes[1][1], 0, "site1", 0)
    random_ttns.add_child_to_parent(nodes[2][0], nodes[2][1], 0, "site2", 1)
    random_ttns.add_child_to_parent(nodes[3][0], nodes[3][1], 0, "site1", 1)
    random_ttns.add_child_to_parent(nodes[4][0], nodes[4][1], 0, "site4", 1)
    random_ttns.add_child_to_parent(nodes[5][0], nodes[5][1], 0, "site1", 2)
    random_ttns.add_child_to_parent(nodes[6][0], nodes[6][1], 0, "site6", 1)
    random_ttns.add_child_to_parent(nodes[7][0], nodes[7][1], 0, "site6", 2)
    return random_ttns

def random_big_ttns_two_root_children(mode: Union[RandomTTNSMode,List[Tuple[int]]] = RandomTTNSMode.SAME,
                                      seed=None
                                      ) -> TreeTensorNetworkState:
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
        TreeTensorNetworkState: A random TTNS with the topology::

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
    elif mode == RandomTTNSMode.DIFFPHYS:
        shapes = [(7,6,1),(7,4,5,1),(4,3),(5,2,3,4),
                  (2,5),(3,2),(6,3,2),(3,2)]
    elif mode == RandomTTNSMode.TRIVIALVIRTUAL:
        shapes = [(1,1,2),(1,1,1,2),(1,2),(1,1,1,2),
                  (1,2),(1,2),(1,1,2),(1,2)]
    elif isinstance(mode, list):
        assert len(mode) == 8, "The list must have 8 elements!"
        shapes = mode
    else:
        errstr = "Only RandomTTNSMode.SAME, RandomTTNSMode.DIFFVIRT or a list of shapes is supported!"
        raise NotImplementedError(errstr)

    nodes = [random_tensor_node(shape, identifier="site"+str(i), seed=seed)
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

def random_like(ttns: TreeTensorNetworkState,
                bond_dim: int | None = None,
                **kwargs
                ) -> TreeTensorNetworkState:
    """
    Generates a random TTNS with the same topology as the given one.

    Args:
        ttns (TreeTensorNetworkState): The TTNS whose topology is to be copied.
        bond_dim (int|None): If provided, all virtual bond dimensions will be
            set maximally to this value. Otherwise, the bond dimensions of
            the given TTNS are used.
        **kwargs: Additional keyword arguments passed to the random tensor
            generator.

    Returns:
        TreeTensorNetworkState: A random TTNS with the same topology as the
            given one.
    """
    order = ttns.linearise(LinearisationMode.PARENTS_FIRST)
    new_tensors = {}
    for node_id in order:
        old_node, _ = ttns[node_id] # get the tensors out to transpose them
        if bond_dim is not None:
            neigh_shape = old_node.neighbour_dims()
            shape = [min(bond_dim, dim) for dim in neigh_shape]
            shape += old_node.open_dimensions()
            shape = tuple(shape)
        else:
            shape = old_node.shape
        new_tensor = crandn(shape, **kwargs)
        new_tensors[node_id] = new_tensor
    random_ttns = TreeTensorNetworkState.from_tensors(ttns,
                                                      new_tensors)
    return random_ttns
