import unittest

import numpy as np

import pytreenet as ptn
from pytreenet.contractions import contraction_util

class TestContractionUtil(unittest.TestCase):
    def setUp(self) -> None:
        self.identifier = "I am an identifier that identifiers stuff."
        self.node, self.tensor = ptn.random_tensor_node((6,5,4,3,2),
                                                        identifier=self.identifier)
        self.node.add_parent("parent")
        for i in range(3):
            self.node.add_child("child"+str(i))

        self.dictionary = contraction_util.PartialTreeCachDict()
        identifiers = ["parent", "child0", "child1", "child2"]
        shapes = [(6,2),(5,3),(4, ),(3,5,4)]
        for i, ident in enumerate(identifiers):
            tensor = ptn.crandn(shapes[i])
            self.dictionary.add_entry(ident,
                                      self.identifier,
                                      tensor)

    def test_determine_index_with_ignored_leg_smaller(self):
        """
        If the neighbour_index is smaller we should get 0.
        """
        self.assertEqual(0,
                         contraction_util.determine_index_with_ignored_leg(self.node,
                                                                      "child0",
                                                                      "child2"))
        self.assertEqual(0,
                         contraction_util.determine_index_with_ignored_leg(self.node,
                                                                      "child1",
                                                                      "child2"))
        self.assertEqual(0,
                         contraction_util.determine_index_with_ignored_leg(self.node,
                                                                      "parent",
                                                                      "child2"))
        self.assertEqual(0,
                         contraction_util.determine_index_with_ignored_leg(self.node,
                                                                      "parent",
                                                                      "child0"))

    def test_determine_index_with_ignored_leg_larger(self):
        """
        If the neighbour index is larger we should get 1.
        """
        self.assertEqual(1,
                         contraction_util.determine_index_with_ignored_leg(self.node,
                                                                      "child2",
                                                                      "child0"))
        self.assertEqual(1,
                         contraction_util.determine_index_with_ignored_leg(self.node,
                                                                      "child2",
                                                                      "child1"))
        self.assertEqual(1,
                         contraction_util.determine_index_with_ignored_leg(self.node,
                                                                      "child2",
                                                                      "parent"))
        self.assertEqual(1,
                         contraction_util.determine_index_with_ignored_leg(self.node,
                                                                      "child0",
                                                                      "parent"))

    def test_determine_index_with_ignored_leg_equal(self):
        """
        If the two identifiers are equal an Assertion error is thrown,
         as this does not make sense.
        """
        for identifier in ["child0", "child1", "child2", "parent"]:
            with self.assertRaises(AssertionError):
                contraction_util.determine_index_with_ignored_leg(self.node,
                                                             identifier,
                                                             identifier)

    def test_get_equivalent_legs_no_ignoring_same_order(self):
        """
        If the two nodes have neighbours in the same order,
         the legs should be the same.
        """
        node2 = ptn.Node(identifier="node2")
        node2.add_parent("parent")
        for i in range(3):
            node2.add_child("child"+str(i))
        correct_legs = [0,1,2,3]
        legs1, legs2 = contraction_util.get_equivalent_legs(self.node,
                                                            node2)
        self.assertEqual(correct_legs, legs1)
        self.assertEqual(correct_legs, legs2)

    def test_get_equivalent_legs_no_ignoring_different_order(self):
        """
        The two nodes might have a difference in their child order.
        """
        node2 = ptn.Node(identifier="node2")
        node2.add_parent("parent")
        node2.add_child("child2")
        node2.add_child("child0")
        node2.add_child("child1")
        legs1, legs2 = contraction_util.get_equivalent_legs(self.node,
                                                            node2)
        self.assertEqual([0,1,2,3], legs1)
        self.assertEqual([0,2,3,1], legs2)

    def test_get_equivalent_legs_ignore_one_same_order(self):
        """
        If we ignore one leg, its index should not appear in the result.
        """
        node2 = ptn.Node(identifier="node2")
        node2.add_parent("parent")
        for i in range(3):
            node2.add_child("child"+str(i))
        legs1, legs2 = contraction_util.get_equivalent_legs(self.node,
                                                            node2,
                                                            ignore_legs="child0")
        self.assertEqual([0,2,3], legs1)
        self.assertEqual([0,2,3], legs2)

    def test_get_equivalent_legs_ignore_one_different(self):
        """
        if we ignore one leg, its index should not appear in the result.
        """
        node2 = ptn.Node(identifier="node2")
        node2.add_parent("parent")
        node2.add_child("child2")
        node2.add_child("child0")
        node2.add_child("child1")
        legs1, legs2 = contraction_util.get_equivalent_legs(self.node,
                                                            node2,
                                                            ignore_legs="child0")
        self.assertEqual([0,2,3], legs1)
        self.assertEqual([0,3,1], legs2)

    def test_get_equivalent_legs_ignore_two(self):
        """
        If we ignore two legs, the result should be empty.
        """
        node2 = ptn.Node(identifier="node2")
        node2.add_parent("parent")
        node2.add_child("child2")
        node2.add_child("child0")
        node2.add_child("child1")
        legs1, legs2 = contraction_util.get_equivalent_legs(self.node,
                                                            node2,
                                                            ignore_legs=["child0", "child2"])
        self.assertEqual([0,2], legs1)
        self.assertEqual([0,3], legs2)
    
    def test_contract_neighbour_block_to_ket_parent(self):
        """
                                    ______
                               ____|      |
                           |     1 |      |
                         __|__     |parent|
                    ____|     |____|      |
                        |  A  |  0 |      |
                        |_____|    |______|
        """
        correct_result = np.tensordot(self.tensor,
                                      self.dictionary.get_entry("parent",
                                                               self.identifier),
                                      axes=([0],[0]))
        found_tensor = contraction_util.contract_neighbour_block_to_ket(self.tensor,
                                                                        self.node,
                                                                        "parent",
                                                                        self.dictionary)
        self.assertTrue(np.allclose(correct_result, found_tensor))
    
    def test_contract_neighbour_block_to_ket_child0(self):
        """
                                    ______
                               ____|      |
                           |     1 |      |
                         __|__     |  c0  |
                    ____|     |____|      |
                        |  A  |  0 |      |
                        |_____|    |______|
        """
        correct_result = np.tensordot(self.tensor,
                                      self.dictionary.get_entry("child0",
                                                               self.identifier),
                                      axes=([1],[0]))
        found_tensor = contraction_util.contract_neighbour_block_to_ket(self.tensor,
                                                                        self.node,
                                                                        "child0",
                                                                        self.dictionary)
        self.assertTrue(np.allclose(correct_result, found_tensor))
    
    def test_contract_neighbour_block_to_ket_child1(self):
        """
                           |        ______
                         __|__     |      |
                    ____|     |____|  c1  |
                        |  A  |  0 |      |
                        |_____|    |______|
        """
        correct_result = np.tensordot(self.tensor,
                                      self.dictionary.get_entry("child1",
                                                               self.identifier),
                                      axes=([2],[0]))
        found_tensor = contraction_util.contract_neighbour_block_to_ket(self.tensor,
                                                                        self.node,
                                                                        "child1",
                                                                        self.dictionary)
        self.assertTrue(np.allclose(correct_result, found_tensor))

    def test_contract_neighbour_block_to_ket_child2(self):
        """
                                    ______
                               ____|      |
                                 2 |      |
                                   |      |
                               ____|      |
                           |     1 |  c2  |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |  0 |      |
                        |_____|    |______|
        """
        correct_result = np.tensordot(self.tensor,
                                      self.dictionary.get_entry("child2",
                                                               self.identifier),
                                      axes=([3],[0]))
        found_tensor = contraction_util.contract_neighbour_block_to_ket(self.tensor,
                                                                        self.node,
                                                                        "child2",
                                                                        self.dictionary)
        self.assertTrue(np.allclose(correct_result, found_tensor))

    def test_contract_neighbour_block_to_ket_ignore_one_leg_parent(self):
        """
        We contract one after the other that is not the parent.
        """
        ignore_leg = "parent"
        correct_tensor = np.tensordot(self.tensor,
                                      self.dictionary.get_entry("child0",
                                                               self.identifier),
                                      axes=([1],[0]))
        found_tensor = contraction_util.contract_neighbour_block_to_ket_ignore_one_leg(self.tensor,
                                                                                       self.node,
                                                                                       "child0",
                                                                                       ignore_leg,
                                                                                       self.dictionary)
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        correct_tensor = np.tensordot(found_tensor,
                                      self.dictionary.get_entry("child1",
                                                               self.identifier),
                                      axes=([1],[0]))
        found_tensor = contraction_util.contract_neighbour_block_to_ket_ignore_one_leg(found_tensor,
                                                                                       self.node,
                                                                                       "child1",
                                                                                       ignore_leg,
                                                                                       self.dictionary)
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        correct_tensor = np.tensordot(found_tensor,
                                      self.dictionary.get_entry("child2",
                                                               self.identifier),
                                      axes=([1],[0]))
        found_tensor = contraction_util.contract_neighbour_block_to_ket_ignore_one_leg(found_tensor,
                                                                                       self.node,
                                                                                       "child2",
                                                                                       ignore_leg,
                                                                                       self.dictionary)
        self.assertTrue(np.allclose(correct_tensor, found_tensor))

    def test_contract_neighbour_block_to_ket_ignore_one_leg_child1(self):
        """
        We contract one after the other that is not the parent.
        """
        ignore_leg = "child1"
        correct_tensor = np.tensordot(self.tensor,
                                      self.dictionary.get_entry("parent",
                                                               self.identifier),
                                      axes=([0],[0]))
        found_tensor = contraction_util.contract_neighbour_block_to_ket_ignore_one_leg(self.tensor,
                                                                                       self.node,
                                                                                       "parent",
                                                                                       ignore_leg,
                                                                                       self.dictionary)
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        correct_tensor = np.tensordot(found_tensor,
                                      self.dictionary.get_entry("child0",
                                                               self.identifier),
                                      axes=([0],[0]))
        found_tensor = contraction_util.contract_neighbour_block_to_ket_ignore_one_leg(found_tensor,
                                                                                       self.node,
                                                                                       "child0",
                                                                                       ignore_leg,
                                                                                       self.dictionary)
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        correct_tensor = np.tensordot(found_tensor,
                                      self.dictionary.get_entry("child2",
                                                               self.identifier),
                                      axes=([1],[0]))
        found_tensor = contraction_util.contract_neighbour_block_to_ket_ignore_one_leg(found_tensor,
                                                                                       self.node,
                                                                                       "child2",
                                                                                       ignore_leg,
                                                                                       self.dictionary)
        self.assertTrue(np.allclose(correct_tensor, found_tensor))

    def test_contract_all_but_one_neihgbour_block_to_ket_parent(self):
        """
        We contract all but one neighbour block to the ket tensor.
        """
        ignore_leg = "parent"
        correct_tensor = contraction_util.contract_neighbour_block_to_ket_ignore_one_leg(self.tensor,
                                                                                         self.node,
                                                                                         "child0",
                                                                                         ignore_leg,
                                                                                         self.dictionary)
        correct_tensor = contraction_util.contract_neighbour_block_to_ket_ignore_one_leg(correct_tensor,
                                                                                         self.node,
                                                                                         "child1",
                                                                                         ignore_leg,
                                                                                         self.dictionary)
        correct_tensor = contraction_util.contract_neighbour_block_to_ket_ignore_one_leg(correct_tensor,
                                                                                         self.node,
                                                                                         "child2",
                                                                                         ignore_leg,
                                                                                         self.dictionary)
        found_tensor = contraction_util.contract_all_but_one_neighbour_block_to_ket(self.tensor,
                                                                                    self.node,
                                                                                    ignore_leg,
                                                                                    self.dictionary)
        self.assertTrue(np.allclose(correct_tensor, found_tensor))

    def test_contract_all_but_one_neihgbour_block_to_ket_child1(self):
        """
        We contract all but one neighbour block to the ket tensor.
        """
        ignore_leg = "child1"
        correct_tensor = contraction_util.contract_neighbour_block_to_ket_ignore_one_leg(self.tensor,
                                                                                         self.node,
                                                                                         "parent",
                                                                                         ignore_leg,
                                                                                         self.dictionary)
        correct_tensor = contraction_util.contract_neighbour_block_to_ket_ignore_one_leg(correct_tensor,
                                                                                         self.node,
                                                                                         "child0",
                                                                                         ignore_leg,
                                                                                         self.dictionary)
        correct_tensor = contraction_util.contract_neighbour_block_to_ket_ignore_one_leg(correct_tensor,
                                                                                         self.node,
                                                                                         "child2",
                                                                                         ignore_leg,
                                                                                         self.dictionary)
        found_tensor = contraction_util.contract_all_but_one_neighbour_block_to_ket(self.tensor,
                                                                                    self.node,
                                                                                    ignore_leg,
                                                                                    self.dictionary)
        self.assertTrue(np.allclose(correct_tensor, found_tensor))

    def test_contract_all_neighbour_blocks_to_ket(self):
        """
        We contract all neighbour blocks to the ket tensor.
        """
        correct_tensor = contraction_util.contract_neighbour_block_to_ket(self.tensor,
                                                                         self.node,
                                                                         "parent",
                                                                         self.dictionary)
        correct_tensor = contraction_util.contract_neighbour_block_to_ket(correct_tensor,
                                                                         self.node,
                                                                         "child0",
                                                                         self.dictionary,
                                                                         0)
        correct_tensor = contraction_util.contract_neighbour_block_to_ket(correct_tensor,
                                                                         self.node,
                                                                         "child1",
                                                                         self.dictionary,
                                                                         0)
        correct_tensor = contraction_util.contract_neighbour_block_to_ket(correct_tensor,
                                                                         self.node,
                                                                         "child2",
                                                                         self.dictionary,
                                                                         0)
        found_tensor = contraction_util.contract_all_neighbour_blocks_to_ket(self.tensor,
                                                                            self.node,
                                                                            self.dictionary)
        self.assertTrue(np.allclose(correct_tensor, found_tensor))

if __name__ == "__main__":
    unittest.main()
