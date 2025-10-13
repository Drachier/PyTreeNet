import unittest

import numpy as np

from pytreenet.core.node import Node
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict
from pytreenet.random import random_tensor_node, crandn

from pytreenet.contractions.contraction_util import (determine_index_with_ignored_leg,
                                                     get_equivalent_legs,
                                                     contract_all_neighbour_blocks_to_ket)


class TestContractionUtil(unittest.TestCase):
    def setUp(self) -> None:
        self.identifier = "I am an identifier that identifies stuff."
        self.node, self.tensor = random_tensor_node((6, 5, 4, 3, 2),
                                                    identifier=self.identifier)
        self.node.add_parent("parent")
        for i in range(3):
            self.node.add_child("child"+str(i))

        self.dictionary = PartialTreeCachDict()
        identifiers = ["parent", "child0", "child1", "child2"]
        shapes = [(6, 2), (5, 3), (4, ), (3, 5, 4)]
        for i, ident in enumerate(identifiers):
            tensor = crandn(shapes[i])
            self.dictionary.add_entry(ident,
                                      self.identifier,
                                      tensor)

    def test_determine_index_with_ignored_leg_smaller(self):
        """
        If the neighbour_index is smaller we should get 0.
        """
        self.assertEqual(0,
                         determine_index_with_ignored_leg(self.node,
                                                          "child0",
                                                          "child2"))
        self.assertEqual(0,
                         determine_index_with_ignored_leg(self.node,
                                                          "child1",
                                                          "child2"))
        self.assertEqual(0,
                         determine_index_with_ignored_leg(self.node,
                                                          "parent",
                                                          "child2"))
        self.assertEqual(0,
                         determine_index_with_ignored_leg(self.node,
                                                          "parent",
                                                          "child0"))

    def test_determine_index_with_ignored_leg_larger(self):
        """
        If the neighbour index is larger we should get 1.
        """
        self.assertEqual(1,
                         determine_index_with_ignored_leg(self.node,
                                                          "child2",
                                                          "child0"))
        self.assertEqual(1,
                         determine_index_with_ignored_leg(self.node,
                                                          "child2",
                                                          "child1"))
        self.assertEqual(1,
                         determine_index_with_ignored_leg(self.node,
                                                          "child2",
                                                          "parent"))
        self.assertEqual(1,
                         determine_index_with_ignored_leg(self.node,
                                                          "child0",
                                                          "parent"))

    def test_determine_index_with_ignored_leg_equal(self):
        """
        If the two identifiers are equal an Assertion error is thrown,
         as this does not make sense.
        """
        for identifier in ["child0", "child1", "child2", "parent"]:
            with self.assertRaises(AssertionError):
                determine_index_with_ignored_leg(self.node,
                                                 identifier,
                                                 identifier)

    def test_contract_all_neighbour_blocks_to_ket_diff_order(self):
        """
        Contract all neighbour blocks to the ket tensor, but with a custom
        order of the neighbours.
        """
        order = ["child1", "child2", "parent", "child0"]
        found = contract_all_neighbour_blocks_to_ket(self.tensor,
                                                     self.node,
                                                     self.dictionary,
                                                     order=order)
        # Reference
        correct_tensor = contract_all_neighbour_blocks_to_ket(self.tensor,
                                                              self.node,
                                                              self.dictionary)
        correct_tensor = correct_tensor.transpose(0,3,4,1,2)
        # Test
        correct_shape = (2,5,4,2,3)
        self.assertEqual(correct_shape, found.shape)
        self.assertTrue(np.allclose(correct_tensor, found))


class TestGetEquivalentLegs(unittest.TestCase):
    """
    Test the function get_equivalent_legs.
    """

    def setUp(self) -> None:
        self.identifier = "I am an identifier that identifies stuff."
        self.node, self.tensor = random_tensor_node((6, 5, 4, 3, 2),
                                                    identifier=self.identifier)
        self.node.add_parent("parent")
        for i in range(3):
            self.node.add_child("child"+str(i))

    def test_no_ignoring_same_order(self):
        """
        If the two nodes have neighbours in the same order,
         the legs should be the same.
        """
        node2 = Node(identifier="node2")
        node2.add_parent("parent")
        for i in range(3):
            node2.add_child("child"+str(i))
        correct_legs = [0, 1, 2, 3]
        legs1, legs2 = get_equivalent_legs(self.node,
                                           node2)
        self.assertEqual(correct_legs, legs1)
        self.assertEqual(correct_legs, legs2)

    def test_no_ignoring_different_order(self):
        """
        The two nodes might have a difference in their child order.
        """
        node2 = Node(identifier="node2")
        node2.add_parent("parent")
        node2.add_child("child2")
        node2.add_child("child0")
        node2.add_child("child1")
        legs1, legs2 = get_equivalent_legs(self.node,
                                           node2)
        self.assertEqual([0, 1, 2, 3], legs1)
        self.assertEqual([0, 2, 3, 1], legs2)

    def test_ignore_one_same_order(self):
        """
        If we ignore one leg, its index should not appear in the result.
        """
        node2 = Node(identifier="node2")
        node2.add_parent("parent")
        for i in range(3):
            node2.add_child("child"+str(i))
        legs1, legs2 = get_equivalent_legs(self.node,
                                           node2,
                                           ignore_legs="child0")
        self.assertEqual([0, 2, 3], legs1)
        self.assertEqual([0, 2, 3], legs2)

    def test_ignore_one_different(self):
        """
        if we ignore one leg, its index should not appear in the result.
        """
        node2 = Node(identifier="node2")
        node2.add_parent("parent")
        node2.add_child("child2")
        node2.add_child("child0")
        node2.add_child("child1")
        legs1, legs2 = get_equivalent_legs(self.node,
                                           node2,
                                           ignore_legs="child0")
        self.assertEqual([0, 2, 3], legs1)
        self.assertEqual([0, 3, 1], legs2)

    def test_ignore_two(self):
        """
        If we ignore two legs, the result should be empty.
        """
        node2 = Node(identifier="node2")
        node2.add_parent("parent")
        node2.add_child("child2")
        node2.add_child("child0")
        node2.add_child("child1")
        legs1, legs2 = get_equivalent_legs(self.node,
                                           node2,
                                           ignore_legs=["child0", "child2"])
        self.assertEqual([0, 2], legs1)
        self.assertEqual([0, 3], legs2)

    # From here on the two nodes have different identifiers

    def id_trafo(self, ket_identifier: str) -> str:
        """
        A simple identifier transformation.
        """
        return ket_identifier + "_other"

    def test_get_equivalent_legs_no_ignoring_same_order_diff_ids(self):
        """
        If the two nodes have neighbours in the same order,
         the legs should be the same.
        """
        node2 = Node(identifier="node2_other")
        node2.add_parent("parent_other")
        for i in range(3):
            node2.add_child("child"+str(i)+"_other")
        correct_legs = [0, 1, 2, 3]
        legs1, legs2 = get_equivalent_legs(self.node,
                                           node2,
                                           id_trafo=self.id_trafo)
        self.assertEqual(correct_legs, legs1)
        self.assertEqual(correct_legs, legs2)

    def test_no_ignoring_different_order_diff_ids(self):
        """
        The two nodes might have a difference in their child order.
        """
        node2 = Node(identifier="node2_other")
        node2.add_parent("parent_other")
        node2.add_child("child2_other")
        node2.add_child("child0_other")
        node2.add_child("child1_other")
        legs1, legs2 = get_equivalent_legs(self.node,
                                           node2,
                                           id_trafo=self.id_trafo)
        self.assertEqual([0, 1, 2, 3], legs1)
        self.assertEqual([0, 2, 3, 1], legs2)

    def test_ignore_one_same_order_diff_ids(self):
        """
        If we ignore one leg, its index should not appear in the result.
        """
        node2 = Node(identifier="node2_other")
        node2.add_parent("parent_other")
        for i in range(3):
            node2.add_child("child"+str(i)+"_other")
        legs1, legs2 = get_equivalent_legs(self.node,
                                           node2,
                                           ignore_legs="child0",
                                           id_trafo=self.id_trafo)
        self.assertEqual([0, 2, 3], legs1)
        self.assertEqual([0, 2, 3], legs2)

    def test_ignore_one_different_diff_ids(self):
        """
        if we ignore one leg, its index should not appear in the result.
        """
        node2 = Node(identifier="node2_other")
        node2.add_parent("parent_other")
        node2.add_child("child2_other")
        node2.add_child("child0_other")
        node2.add_child("child1_other")
        legs1, legs2 = get_equivalent_legs(self.node,
                                           node2,
                                           ignore_legs="child0",
                                           id_trafo=self.id_trafo)
        self.assertEqual([0, 2, 3], legs1)
        self.assertEqual([0, 3, 1], legs2)

    def test_ignore_two_diff_ids(self):
        """
        If we ignore two legs, the result should be empty.
        """
        node2 = Node(identifier="node2_other")
        node2.add_parent("parent_other")
        node2.add_child("child2_other")
        node2.add_child("child0_other")
        node2.add_child("child1_other")
        legs1, legs2 = get_equivalent_legs(self.node,
                                           node2,
                                           ignore_legs=["child0", "child2"],
                                           id_trafo=self.id_trafo)
        self.assertEqual([0, 2], legs1)
        self.assertEqual([0, 3], legs2)


if __name__ == "__main__":
    unittest.main()
