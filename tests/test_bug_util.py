import unittest

from pytreenet.random.random_node import random_tensor_node
from pytreenet.random.random_matrices import crandn

from pytreenet.time_evolution.bug import (new_basis_tensor_qr_legs,
                                          find_new_basis_tensor_leg_permutation,
                                          compute_new_basis_tensor)

class TestNewBasisTensorQRLegs(unittest.TestCase):

    def test_leaf(self):
        """
        Test the QR leg finding of a leaf tensor.
        """
        node, _ = random_tensor_node((3,2))
        node.add_parent("parent")
        q_legs, r_legs = new_basis_tensor_qr_legs(node)
        self.assertEqual(q_legs, [1])
        self.assertEqual(r_legs, [0])

    def test_two_children(self):
        """
        Test the QR leg finding of a tensor with two children.
        """
        node, _ = random_tensor_node((5,4,3,2))
        node.add_parent("parent")
        node.add_children(["child1","child2"])
        q_legs, r_legs = new_basis_tensor_qr_legs(node)
        self.assertEqual(q_legs, [1,2,3])
        self.assertEqual(r_legs, [0])

    def test_no_phys_leg(self):
        """
        Test the QR leg finding of a tensor with no physical legs.
        """
        node, _ = random_tensor_node((4,3,2))
        node.add_parent("parent")
        node.add_children(["child1","child2"])
        q_legs, r_legs = new_basis_tensor_qr_legs(node)
        self.assertEqual(q_legs, [1,2])
        self.assertEqual(r_legs, [0])

    def test_root(self):
        """
        Test the QR leg finding of a root tensor.
        """
        node, _ = random_tensor_node((4,3,2))
        node.add_children(["child1","child2"])
        self.assertRaises(AssertionError, new_basis_tensor_qr_legs, node)

class TestFindNewBasisTensorLegPermutation(unittest.TestCase):

    def test_leaf(self):
        """
        Test the permutation finding of a leaf tensor.
        """
        node, _ = random_tensor_node((3,2))
        node.add_parent("parent")
        permutation = find_new_basis_tensor_leg_permutation(node)
        self.assertEqual(permutation, [1,0])

    def test_two_children(self):
        """
        Test the permutation finding of a tensor with two children.
        """
        node, _ = random_tensor_node((5,4,3,2))
        node.add_parent("parent")
        node.add_children(["child1","child2"])
        permutation = find_new_basis_tensor_leg_permutation(node)
        self.assertEqual(permutation, [3,0,1,2])

    def test_no_phys_leg(self):
        """
        Test the permutation finding of a tensor with no physical legs.
        """
        node, _ = random_tensor_node((4,3,2))
        node.add_parent("parent")
        node.add_children(["child1","child2"])
        permutation = find_new_basis_tensor_leg_permutation(node)
        self.assertEqual(permutation, [2,0,1])


if __name__ == '__main__':
    unittest.main()
