"""
This module implements the creation of random TTNS of special types.
"""
from __future__ import annotations

from ..special_ttn.special_states import TTNStructure
from ..special_ttn.mps import MatrixProductState
from ..special_ttn.star import StarTreeTensorState
from ..special_ttn.fttn import ForkTreeProductState
from .random_matrices import crandn
from ..ttns.ttns import TTNS
from ..core.node import Node

def random_ttns(structure: TTNStructure,
                sys_size: int,
                phys_dim: int,
                bond_dim: int,
                normalise: bool = False,
                **kwargs
                ) -> TTNS:
    """
    Creates a random TTNS of a special type.

    Args:
        structure (TTNStructure): The structure of the TTNS.
        sys_size (int): Size parameter for the TTNS structure.
            For MPS, this is the number of sites.
            For T-star, this is the arm length.
            For binary tree, this is the depth.
            For FTPS, this is the side length.
        phys_dim (int): Physical dimension of each site.
        bond_dim (int): Bond dimension between sites.
        **kwargs: Additional keyword arguments for the random number
            generation.
    
    Returns:
        TreeTensorNetworkState: The generated random TTNS.
    """
    args = (sys_size, phys_dim, bond_dim)
    if structure == TTNStructure.MPS:
        ttns = random_mps(*args, **kwargs)
    elif structure == TTNStructure.TSTAR:
        ttns = random_tstar_state(*args, **kwargs)
    elif structure == TTNStructure.BINARY:
        ttns = random_binary_state(*args, **kwargs)
    elif structure == TTNStructure.FTPS:
        ttns = random_ftps(*args, **kwargs)
    else:
        raise ValueError(f"Unknown TTN structure: {structure}!")
    if normalise:
        vec = ttns.completely_contract_tree(to_copy=True)[0].flatten()
        ttns.normalise()
    return ttns

def random_mps(num_sites: int,
               phys_dim: int,
               bond_dim: int,
               **kwargs
               ) -> MatrixProductState:
    """
    Creates a random MPS (Matrix Product State).

    Args:
        num_sites (int): Number of sites in the MPS.
        phys_dim (int): Physical dimension of each site.
        bond_dim (int): Bond dimension between sites.
        **kwargs: Additional keyword arguments for the random number
            generation.

    Returns:
        MatrixProductState: The generated random MPS.
    """
    if num_sites < 2:
        shapes = [(phys_dim, )]
    else:
        shapes = [(bond_dim, phys_dim)]
        shapes += [(bond_dim, bond_dim, phys_dim)
                   for _ in range(num_sites - 2)]
        shapes += [(bond_dim, phys_dim)]
    tensors = [crandn(shape, **kwargs) for shape in shapes]
    mps = MatrixProductState.from_tensor_list(tensors,
                                              root_site=num_sites//2)
    return mps

def random_tstar_state(arm_length: int,
                        phys_dim: int,
                        bond_dim: int,
                        **kwargs
                        ) -> StarTreeTensorState:
    """
    Creates a random T-star TTN.

    Args:
        arm_length (int): Length of each arm in the T-star.
        phys_dim (int): Physical dimension of each site.
        bond_dim (int): Bond dimension between sites.
        **kwargs: Additional keyword arguments for the random number
            generation.
    
    Returns:
        TreeTensorNetworkState: The generated random T-star TTN.
    """
    if arm_length < 1:
        raise ValueError("Arm length must be at least 1!")
    arm_shapes = []
    if arm_length > 1:
        arm_shapes += [(bond_dim, bond_dim, phys_dim)
                       for _ in range(arm_length - 1)]
    arm_shapes.append((bond_dim, phys_dim))
    arms = [[crandn(shape, **kwargs) for shape in arm_shapes]
            for _ in range(3)]
    centre_tensor = crandn((bond_dim, bond_dim, bond_dim, 1), **kwargs)
    tstar = StarTreeTensorState.from_tensor_lists(centre_tensor,
                                                  arms)
    return tstar

def random_binary_state(depth: int,
                        phys_dim: int,
                        bond_dim: int,
                        **kwargs
                        ) -> TTNS:
    """
    Creates a random perfectly balanced binary TTNS.

    Args:
        depth (int): Depth of the binary tree.
        phys_dim (int): Physical dimension of each site.
        bond_dim (int): Bond dimension between sites.
        **kwargs: Additional keyword arguments for the random number
            generation.
    
    Returns:
        TreeTensorNetworkState: The generated random binary TTNS.
    """
    if depth < 1:
        raise ValueError("Depth must be at least 1!")
    ttns = TTNS()
    if depth == 1:
        shape = (bond_dim, phys_dim)
    else:
        shape = (bond_dim, bond_dim, 1)
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
                    shape = (bond_dim, phys_dim)
                else:
                    shape = (bond_dim, bond_dim, bond_dim, 1)
                tensor = crandn(shape, **kwargs)
                ttns.add_child_to_parent(Node(identifier=f"N{d}_{index}"),
                                         tensor,
                                         0,
                                         leaf,
                                         parent_index)
                index += 1
    return ttns

def random_ftps(side_length: int,
                phys_dim: int,
                bond_dim: int,
                **kwargs
                ) -> ForkTreeProductState:
    """
    Creates a random Fork Tree Product State (FTPS).

    The main chain of the FTPS will not have any physical legs.

    Args:
        side_length (int): Length of the main chain and each subchain.
        phys_dim (int): Physical dimension of each site.
        bond_dim (int): Bond dimension between sites.
        **kwargs: Additional keyword arguments for the random number
            generation.

    Returns:
        ForkTreeProductState: The generated random FTPS.
    """
    if side_length <= 1:
        raise ValueError("Side length must be at least 2!")
    main_shapes = [(bond_dim, bond_dim, 1)]
    main_shapes += [(bond_dim, bond_dim, bond_dim, 1)
                    for _ in range(side_length - 2)]
    main_shapes += [(bond_dim, bond_dim, 1)]
    main_tensors = [crandn(shape, **kwargs)
                    for shape in main_shapes]
    subchain_shapes = [(bond_dim, bond_dim, phys_dim)
                       for _ in range(side_length)]
    subchain_shapes += [(bond_dim, phys_dim)]
    subchain_tensors = [[crandn(shape, **kwargs)
                         for shape in subchain_shapes]
                        for _ in range(side_length)]
    ftps = ForkTreeProductState.from_tensors(main_tensors,
                                             subchain_tensors)
    return ftps
