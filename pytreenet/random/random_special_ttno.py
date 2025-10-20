"""
This module implements the creation of random TTNO of special types.
"""
from __future__ import annotations

from ..special_ttn.special_states import TTNStructure
from ..special_ttn.mps import MatrixProductOperator
from ..special_ttn.star import StarTreeOperator
from ..special_ttn.fttn import ForkTreeProductOperator
from .random_matrices import crandn
from ..ttno.ttno_class import TTNO
from ..core.node import Node

def random_ttno(structure: TTNStructure,
                sys_size: int,
                phys_dim: int,
                bond_dim: int,
                **kwargs
                ) -> TTNO:
    """
    Creates a random TTNO of a special type.

    Args:
        structure (TTNStructure): The structure of the TTNO.
        sys_size (int): Size parameter for the TTNO structure.
            For MPS, this is the number of sites.
            For T-star, this is the arm length.
            For binary tree, this is the depth.
            For FTPS, this is the side length.
        phys_dim (int): Physical dimension of each site.
        bond_dim (int): Bond dimension between sites.
        **kwargs: Additional keyword arguments for the random number
            generation.
    
    Returns:
        TreeTensorNetworkState: The generated random TTNO.
    """
    args = (sys_size, phys_dim, bond_dim)
    if structure == TTNStructure.MPS:
        ttns = random_mpo(*args, **kwargs)
    elif structure == TTNStructure.TSTAR:
        ttns = random_tstar_operator(*args, **kwargs)
    elif structure == TTNStructure.BINARY:
        ttns = random_binary_operator(*args, **kwargs)
    elif structure == TTNStructure.FTPS:
        ttns = random_ftpo(*args, **kwargs)
    else:
        raise ValueError(f"Unknown TTN structure: {structure}!")
    return ttns

def random_mpo(num_sites: int,
               phys_dim: int,
               bond_dim: int,
               **kwargs

               ) -> MatrixProductOperator:
    """
    Creates a random MPO (Matrix Product Operator).

    Args:
        num_sites (int): Number of sites in the MPO.
        phys_dim (int): Physical dimension of each site.
        bond_dim (int): Bond dimension between sites.
        **kwargs: Additional keyword arguments for the random number
            generation.

    Returns:
        MatrixProductOperator: The generated random MPO.
    """
    if num_sites < 2:
        shapes = [(phys_dim, phys_dim)]
    else:
        shapes = [(bond_dim, phys_dim, phys_dim)]
        shapes += [(bond_dim, bond_dim, phys_dim, phys_dim)
                   for _ in range(num_sites - 2)]
        shapes += [(bond_dim, phys_dim, phys_dim)]
    tensors = [crandn(shape, **kwargs) for shape in shapes]
    mpo = MatrixProductOperator.from_tensor_list(tensors,
                                              root_site=num_sites//2)
    return mpo

def random_tstar_operator(arm_length: int,
                           phys_dim: int,
                           bond_dim: int,
                           **kwargs
                           ) -> StarTreeOperator:
    """
    Creates a random T-star TTN.

    Args:
        arm_length (int): Length of each arm in the T-star.
        phys_dim (int): Physical dimension of each site.
        bond_dim (int): Bond dimension between sites.
        **kwargs: Additional keyword arguments for the random number
            generation.
    
    Returns:
        TreeTensorNetworkOperator: The generated random T-star TTN.
    """
    if arm_length < 1:
        raise ValueError("Arm length must be at least 1!")
    arm_shapes = []
    if arm_length > 1:
        arm_shapes += [(bond_dim, bond_dim, phys_dim, phys_dim)
                       for _ in range(arm_length - 1)]
    arm_shapes.append((bond_dim, phys_dim, phys_dim))
    arms = [[crandn(shape, **kwargs) for shape in arm_shapes]
            for _ in range(3)]
    centre_tensor = crandn((bond_dim, bond_dim, bond_dim, 1, 1), **kwargs)
    tstar = StarTreeOperator.from_tensor_lists(centre_tensor,
                                                  arms)
    return tstar

def random_binary_operator(depth: int,
                        phys_dim: int,
                        bond_dim: int,
                        **kwargs
                        ) -> TTNO:
    """
    Creates a random perfectly balanced binary TTNO.

    Args:
        depth (int): Depth of the binary tree.
        phys_dim (int): Physical dimension of each site.
        bond_dim (int): Bond dimension between sites.
        **kwargs: Additional keyword arguments for the random number
            generation.
    
    Returns:
        TreeTensorNetworkOperator: The generated random binary TTNS.
    """
    if depth < 1:
        raise ValueError("Depth must be at least 1!")
    ttns = TTNO()
    if depth == 1:
        shape = (bond_dim, phys_dim, phys_dim)
    else:
        shape = (bond_dim, bond_dim, 1, 1)
    root_tensor = crandn(shape, **kwargs)
    ttns.add_root(Node(identifier="N0"),
                       tensor=root_tensor)
    for d in range(1, depth):
        leafs = ttns.get_leaves()
        index = 0
        for leaf in leafs:
            for cindex in range(2):
                if d == 1:
                    parent_index = cindex
                else:
                    parent_index = cindex + 1
                if d == depth - 1:
                    shape = (bond_dim, phys_dim, phys_dim)
                else:
                    shape = (bond_dim, bond_dim, bond_dim, 1, 1)
                tensor = crandn(shape, **kwargs)
                ttns.add_child_to_parent(Node(identifier=f"N{d}_{index}"),
                                         tensor,
                                         0,
                                         leaf,
                                         parent_index)
                index += 1
    return ttns

def random_ftpo(side_length: int,
                phys_dim: int,
                bond_dim: int,
                **kwargs
                ) -> ForkTreeProductOperator:
    """
    Creates a random Fork Tree Product Operator (FTPO).

    The main chain of the FTPO will not have any physical legs.

    Args:
        side_length (int): Length of the main chain and each subchain.
        phys_dim (int): Physical dimension of each site.
        bond_dim (int): Bond dimension between sites.
        **kwargs: Additional keyword arguments for the random number
            generation.

    Returns:
        ForkTreeProductOperator: The generated random FTPO.
    """
    if side_length <= 1:
        raise ValueError("Side length must be at least 2!")
    main_shapes = [(bond_dim, bond_dim, 1, 1)]
    main_shapes += [(bond_dim, bond_dim, bond_dim, 1, 1)
                    for _ in range(side_length - 2)]
    main_shapes += [(bond_dim, bond_dim, 1, 1)]
    main_tensors = [crandn(shape, **kwargs)
                    for shape in main_shapes]
    subchain_shapes = [(bond_dim, bond_dim, phys_dim, phys_dim)
                       for _ in range(side_length)]
    subchain_shapes += [(bond_dim, phys_dim, phys_dim)]
    subchain_tensors = [[crandn(shape, **kwargs)
                         for shape in subchain_shapes]
                        for _ in range(side_length)]
    ftpo = ForkTreeProductOperator.from_tensors(main_tensors,
                                             subchain_tensors)
    return ftpo
