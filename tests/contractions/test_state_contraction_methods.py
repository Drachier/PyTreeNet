from unittest import TestCase, main as unitmain

from numpy import tensordot, allclose

from pytreenet.random.random_node import random_tensor_node, crandn

from pytreenet.contractions.state_state_contraction import (contract_bra_to_ket_and_blocks_ignore_one_leg)

class TestContract_bra_to_ket_and_blocks_ignore_one_leg(TestCase):

    def test_two_neighbours_same_id_same_order(self):
        """
        Tests the contraction, if two neighbours exist.
        The neighbours are attached to the two nodes in the same way and have
        the identifiers.

        """
        # Create the nodes
        ket_id = "ket"
        ket_node, _ = random_tensor_node((4,3,2), identifier=ket_id)
        bra_id = "bra"
        bra_node, bra_tensor = random_tensor_node((4,3,2), identifier=bra_id)
        # Add Neighbours
        parent_id = "parent"
        ket_node.open_leg_to_parent(parent_id, 0)
        bra_node.open_leg_to_parent(parent_id, 0)
        neighbour1_id = "neighbour1"
        ket_node.open_leg_to_child(neighbour1_id, 1)
        bra_node.open_leg_to_child(neighbour1_id, 1)
        # Create ketblock tensor, i.e. the tensor of the ket node contracted
        # with the neighbour 2 block.
        ketblock_tensor = crandn((4,2,3))
        # Reference
        reference_tensor = tensordot(ketblock_tensor,
                                     bra_tensor,
                                     axes=([1,2],[2,1]))
        # Found
        found_tensor = contract_bra_to_ket_and_blocks_ignore_one_leg(bra_tensor,
                                                                     ketblock_tensor,
                                                                     bra_node,
                                                                     ket_node,
                                                                     parent_id)
        # Test
        self.assertTrue(allclose(reference_tensor, found_tensor))

    def test_two_neighbours_same_id_different_order(self):
        """
        Tests the contraction, if two neighbours exist.
        The neighbours are attached to the two nodes in different ways and but
        have the same identifiers.

        """
        # Create the nodes
        ket_id = "ket"
        ket_node, _ = random_tensor_node((4,3,2), identifier=ket_id)
        bra_id = "bra"
        bra_node, bra_tensor = random_tensor_node((4,3,2), identifier=bra_id)
        # Add Neighbours. Changing ket node order, allows us to keep the bra_tensor as is.
        neighbour1_id = "neighbour1"
        neighbour2_id = "neighbour2"
        ket_node.open_leg_to_child(neighbour2_id, 1)
        ket_node.open_leg_to_child(neighbour1_id, 1) # Leg 0 was pushed to 1
        bra_node.open_leg_to_child(neighbour1_id, 0)
        bra_node.open_leg_to_child(neighbour2_id, 1)
        # Create ketblock tensor, i.e. the tensor of the ket node contracted
        # with the neighbour 2 block.
        ketblock_tensor = crandn((4,2,3))
        # Reference
        reference_tensor = tensordot(ketblock_tensor,
                                     bra_tensor,
                                     axes=([1,2],[2,1]))
        # Found
        found_tensor = contract_bra_to_ket_and_blocks_ignore_one_leg(bra_tensor,
                                                                     ketblock_tensor,
                                                                     bra_node,
                                                                     ket_node,
                                                                     neighbour1_id)
        # Test
        self.assertTrue(allclose(reference_tensor, found_tensor))

    def test_two_neighbours_different_id_same_order(self):
        """
        Tests the contraction, if two neighbours exist.
        The neighbours are attached to the two nodes in the same way and have
        different identifiers.
        """
        # Create the nodes
        ket_id = "ket"
        ket_node, _ = random_tensor_node((4,3,2), identifier=ket_id)
        bra_id = "bra"
        bra_node, bra_tensor = random_tensor_node((4,3,2), identifier=bra_id)
        # Add Neighbours. Changing ket node order, allows us to keep the bra_tensor as is.
        ket_neighbour1_id = "ket_neighbour1"
        ket_neighbour2_id = "ket_neighbour2"
        ket_node.open_leg_to_child(ket_neighbour1_id, 0)
        ket_node.open_leg_to_child(ket_neighbour2_id, 1)
        bra_neighbour1_id = "bra_neighbour1"
        bra_neighbour2_id = "bra_neighbour2"
        bra_node.open_leg_to_child(bra_neighbour1_id, 0)
        bra_node.open_leg_to_child(bra_neighbour2_id, 1)
        # id transformation
        # We just switched the two ids around
        def id_trafo(x):
            if x == ket_neighbour1_id:
                return bra_neighbour1_id
            if x == ket_neighbour2_id:
                return bra_neighbour2_id
            raise ValueError("Unknown id!")
        # Create ketblock tensor, i.e. the tensor of the ket node contracted
        # with the neighbour 2 block.
        ketblock_tensor = crandn((4,2,3))
        # Reference
        reference_tensor = tensordot(ketblock_tensor,
                                     bra_tensor,
                                     axes=([1,2],[2,1]))
        # Found
        found_tensor = contract_bra_to_ket_and_blocks_ignore_one_leg(bra_tensor,
                                                                     ketblock_tensor,
                                                                     bra_node,
                                                                     ket_node,
                                                                     ket_neighbour1_id,
                                                                     id_trafo=id_trafo)
        # Test
        self.assertTrue(allclose(reference_tensor, found_tensor))

    def test_two_neighbours_different_id_different_order(self):
        """
        Tests the contraction, if two neighbours exist.
        The neighbours are attached to the two nodes in different ways and have
        different identifiers.
        """
        # Create the nodes
        ket_id = "ket"
        ket_node, _ = random_tensor_node((4,3,2), identifier=ket_id)
        bra_id = "bra"
        bra_node, bra_tensor = random_tensor_node((4,3,2), identifier=bra_id)
        # Add Neighbours. Changing ket node order, allows us to keep the bra_tensor as is.
        ket_neighbour1_id = "ket_neighbour1"
        ket_neighbour2_id = "ket_neighbour2"
        ket_node.open_leg_to_child(ket_neighbour2_id, 1)
        ket_node.open_leg_to_child(ket_neighbour1_id, 1)
        bra_neighbour1_id = "bra_neighbour1"
        bra_neighbour2_id = "bra_neighbour2"
        bra_node.open_leg_to_child(bra_neighbour1_id, 0)
        bra_node.open_leg_to_child(bra_neighbour2_id, 1)
        # id transformation
        # We just switched the two ids around
        def id_trafo(x):
            if x == ket_neighbour1_id:
                return bra_neighbour1_id
            if x == ket_neighbour2_id:
                return bra_neighbour2_id
            raise ValueError("Unknown id!")
        # Create ketblock tensor, i.e. the tensor of the ket node contracted
        # with the neighbour 2 block.
        ketblock_tensor = crandn((4,2,3))
        # Reference
        reference_tensor = tensordot(ketblock_tensor,
                                     bra_tensor,
                                     axes=([1,2],[2,1]))
        # Found
        found_tensor = contract_bra_to_ket_and_blocks_ignore_one_leg(bra_tensor,
                                                                     ketblock_tensor,
                                                                     bra_node,
                                                                     ket_node,
                                                                     ket_neighbour1_id,
                                                                     id_trafo=id_trafo)
        # Test
        self.assertTrue(allclose(reference_tensor, found_tensor))

    # Now we need to test it with three neighbours, which gives rise to a more
    # complicated leg order during contraction.

    def test_three_neighbours_same_id_same_order(self):
        """
        Tests the contraction, if three neighbours exist.
        The neighbours are attached to the two nodes in the same way and have
        the identifiers.

        """
        # Create the nodes
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,4,3,2), identifier=ket_id)
        bra_id = "bra"
        bra_node, bra_tensor = random_tensor_node((5,4,3,2), identifier=bra_id)
        # Add Neighbours
        neighbour1_id = "neighbour1"
        neighbour2_id = "neighbour2"
        neighbour3_id = "neighbour3"
        ket_node.open_leg_to_child(neighbour1_id, 0)
        ket_node.open_leg_to_child(neighbour2_id, 1)
        ket_node.open_leg_to_child(neighbour3_id, 2)
        bra_node.open_leg_to_child(neighbour1_id, 0)
        bra_node.open_leg_to_child(neighbour2_id, 1)
        bra_node.open_leg_to_child(neighbour3_id, 2)
        # Create ketblock tensor, i.e. the tensor of the ket node contracted
        # with the neighbour 2 block.
        ketblock_tensor = crandn((5,2,4,3))
        # Reference
        reference_tensor = tensordot(ketblock_tensor,
                                     bra_tensor,
                                     axes=([1,2,3],[3,1,2]))
        # Found
        found_tensor = contract_bra_to_ket_and_blocks_ignore_one_leg(bra_tensor,
                                                                     ketblock_tensor,
                                                                     bra_node,
                                                                     ket_node,
                                                                     neighbour1_id)
        # Test
        self.assertTrue(allclose(reference_tensor, found_tensor))

    def test_three_neighbours_same_id_different_order(self):
        """
        Tests the contraction, if three neighbours exist.
        The neighbours are attached to the two nodes in the same way and have
        the identifiers.

        """
        # Create the nodes
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,4,3,2), identifier=ket_id)
        bra_id = "bra"
        bra_node, bra_tensor = random_tensor_node((5,4,3,2), identifier=bra_id)
        # Add Neighbours, changing ket node order, allows us to keep the bra_tensor as is.
        neighbour1_id = "neighbour1"
        neighbour2_id = "neighbour2"
        neighbour3_id = "neighbour3"
        ket_node.open_leg_to_child(neighbour1_id, 0)
        ket_node.open_leg_to_child(neighbour3_id, 2)
        ket_node.open_leg_to_child(neighbour2_id, 2)
        bra_node.open_leg_to_child(neighbour1_id, 0)
        bra_node.open_leg_to_child(neighbour2_id, 1)
        bra_node.open_leg_to_child(neighbour3_id, 2)
        # Create ketblock tensor, i.e. the tensor of the ket node contracted
        # with the neighbour 2 block.
        ketblock_tensor = crandn((5,2,3,4))
        # Reference
        reference_tensor = tensordot(ketblock_tensor,
                                     bra_tensor,
                                     axes=([1,2,3],[3,2,1]))
        # Found
        found_tensor = contract_bra_to_ket_and_blocks_ignore_one_leg(bra_tensor,
                                                                     ketblock_tensor,
                                                                     bra_node,
                                                                     ket_node,
                                                                     neighbour1_id)
        # Test
        self.assertTrue(allclose(reference_tensor, found_tensor))

    def test_three_neighbours_different_id_same_order(self):
        """
        Tests the contraction, if three neighbours exist.
        The neighbours are attached to the two nodes in the same way and have
        the identifiers.

        """
        # Create the nodes
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,4,3,2), identifier=ket_id)
        bra_id = "bra"
        bra_node, bra_tensor = random_tensor_node((5,4,3,2), identifier=bra_id)
        # Add Neighbours
        ket_neighbour1_id = "ket_neighbour1"
        ket_neighbour2_id = "ket_neighbour2"
        ket_neighbour3_id = "ket_neighbour3"
        ket_node.open_leg_to_child(ket_neighbour1_id, 0)
        ket_node.open_leg_to_child(ket_neighbour2_id, 1)
        ket_node.open_leg_to_child(ket_neighbour3_id, 2)
        bra_neighbour1_id = "bra_neighbour1"
        bra_neighbour2_id = "bra_neighbour2"
        bra_neighbour3_id = "bra_neighbour3"
        bra_node.open_leg_to_child(bra_neighbour1_id, 0)
        bra_node.open_leg_to_child(bra_neighbour2_id, 1)
        bra_node.open_leg_to_child(bra_neighbour3_id, 2)
        def id_trafo(x):
            if x == ket_neighbour1_id:
                return bra_neighbour1_id
            if x == ket_neighbour2_id:
                return bra_neighbour2_id
            if x == ket_neighbour3_id:
                return bra_neighbour3_id
            raise ValueError("Unknown id!")
        # Create ketblock tensor, i.e. the tensor of the ket node contracted
        # with the neighbour 2 block.
        ketblock_tensor = crandn((5,2,4,3))
        # Reference
        reference_tensor = tensordot(ketblock_tensor,
                                     bra_tensor,
                                     axes=([1,2,3],[3,1,2]))
        # Found
        found_tensor = contract_bra_to_ket_and_blocks_ignore_one_leg(bra_tensor,
                                                                     ketblock_tensor,
                                                                     bra_node,
                                                                     ket_node,
                                                                     ket_neighbour1_id,
                                                                     id_trafo=id_trafo)
        # Test
        self.assertTrue(allclose(reference_tensor, found_tensor))

    def test_three_neighbours_different_id_different_order(self):
        """
        Tests the contraction, if three neighbours exist.
        The neighbours are attached to the two nodes in the same way and have
        the identifiers.

        """
        # Create the nodes
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,4,3,2), identifier=ket_id)
        bra_id = "bra"
        bra_node, bra_tensor = random_tensor_node((5,4,3,2), identifier=bra_id)
        # Add Neighbours, changing ket node order, allows us to keep the bra_tensor as is.
        ket_neighbour1_id = "ket_neighbour1"
        ket_neighbour2_id = "ket_neighbour2"
        ket_neighbour3_id = "ket_neighbour3"
        ket_node.open_leg_to_child(ket_neighbour1_id, 0)
        ket_node.open_leg_to_child(ket_neighbour3_id, 1)
        ket_node.open_leg_to_child(ket_neighbour2_id, 2)
        bra_neighbour1_id = "bra_neighbour1"
        bra_neighbour2_id = "bra_neighbour2"
        bra_neighbour3_id = "bra_neighbour3"
        bra_node.open_leg_to_child(bra_neighbour1_id, 0)
        bra_node.open_leg_to_child(bra_neighbour2_id, 1)
        bra_node.open_leg_to_child(bra_neighbour3_id, 2)
        def id_trafo(x):
            if x == ket_neighbour1_id:
                return bra_neighbour1_id
            if x == ket_neighbour2_id:
                return bra_neighbour2_id
            if x == ket_neighbour3_id:
                return bra_neighbour3_id
            raise ValueError("Unknown id!")
        # Create ketblock tensor, i.e. the tensor of the ket node contracted
        # with the neighbour 2 block.
        ketblock_tensor = crandn((5,2,3,4))
        # Reference
        reference_tensor = tensordot(ketblock_tensor,
                                     bra_tensor,
                                     axes=([1,2,3],[3,2,1]))
        # Found
        found_tensor = contract_bra_to_ket_and_blocks_ignore_one_leg(bra_tensor,
                                                                     ketblock_tensor,
                                                                     bra_node,
                                                                     ket_node,
                                                                     ket_neighbour1_id,
                                                                     id_trafo=id_trafo)
        # Test
        self.assertTrue(allclose(reference_tensor, found_tensor))

if __name__ == '__main__':
    unitmain()
