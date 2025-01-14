"""
Provides random TTNDO structures for testing purposes.
"""

from pytreenet.ttns.ttndo import SymmetricTTNDO
from pytreenet.random import crandn

def generate_one_child_layer_ttdndo() -> SymmetricTTNDO:
    """
    Generates a TTDNO with a single child layer.

    So it consists of a root and a single bra and ket child each.
    The virtual dimensions are 4 and the physical dimensions are 3.

    Returns:
        SymmetricTTNDO: The generated TTDNO.

    """
    root_id = "root"
    ttndo = SymmetricTTNDO()
    ttndo.add_trivial_root(root_id, dimension=4)
    child_ly1_id = "child_ly1"
    ket_tensor = crandn((4,3))
    bra_tensor = crandn((4,3))
    ttndo.add_symmetric_children_to_parent(child_ly1_id,
                                            ket_tensor, bra_tensor,
                                            0, root_id, 0, 1)
    return ttndo

def generate_three_child_layers_ttdndo() -> SymmetricTTNDO:
    """
    Generate a TTDNO with three child layers.

    This also includes a node with three children and a node with a child
    and an open dimension.

    Returns:
        SymmetricTTNDO: The generated TTDNO.
        
    
    Ket side:
        (bra side)--R
                    \\6   3
                     K0------K13---2
                5  4/ \\5 
                --K10   K11
                  /3     |2
                B20
                |2

    """
    rooti_id = "root"
    ttndo = SymmetricTTNDO()
    ttndo.add_trivial_root(rooti_id, dimension=6)
    child_ly0_id = "child_ly0"
    ket_tensor = crandn((6,4,5,3,1))
    bra_tensor = crandn((6,4,5,3,1))
    ttndo.add_symmetric_children_to_parent(child_ly0_id,
                                            ket_tensor, bra_tensor,
                                            0, rooti_id, 0, 1)
    child_ly1_ids = ["child_ly1" + str(i) for i in range(3)]
    ket_tensor = crandn((4,3,5))
    bra_tensor = crandn((4,3,5))
    ttndo.add_symmetric_children_to_parent(child_ly1_ids[0],
                                            ket_tensor, bra_tensor,
                                            0, child_ly0_id, 1)
    ket_tensor = crandn((5,2))
    bra_tensor = crandn((5,2))
    ttndo.add_symmetric_children_to_parent(child_ly1_ids[1],
                                            ket_tensor, bra_tensor,
                                            0, child_ly0_id, 2)
    ket_tensor = crandn((3,2))
    bra_tensor = crandn((3,2))
    ttndo.add_symmetric_children_to_parent(child_ly1_ids[2],
                                            ket_tensor, bra_tensor,
                                            0, child_ly0_id, 3)
    child_ly2_id = "child_ly2"
    ket_tensor = crandn((3,2))
    bra_tensor = crandn((3,2))
    ttndo.add_symmetric_children_to_parent(child_ly2_id,
                                            ket_tensor, bra_tensor,
                                            0, child_ly1_ids[0], 1)
    return ttndo
