from unittest import TestCase, main as unitmain
from copy import deepcopy

from pytreenet.random.random_matrices import crandn
from pytreenet.core.quantum_numbers.qn_node import QNNode

class TestQNodeValidity(TestCase):
    """
    Test the method checking validity of quantum numbers.
    """

    def test_wrong_dims(self):
        """
        QN are invalid if their dimension does not match the tensor's leg.
        """
        tensor = crandn(2,3,4)
        qn = [[0,1],[0,1,2],[0,1,2]]
        node = QNNode(tensor=tensor, identifier="test", qn=qn)
        self.assertFalse(node.qn_valid())

    def test_valid_qn(self):
        """
        QN are valid if their dimension matches the tensor's leg.
        """
        tensor = crandn(2,3,4)
        qn = [[0,1],[0,1,2],[0,1,2,3]]
        node = QNNode(tensor=tensor, identifier="test", qn=qn)
        self.assertTrue(node.qn_valid())

class TestGetNeighbourQN(TestCase):
    """
    Test the ability to obtain the quantum numbers of a given neighbour node.
    """

    def test_standard(self):
        """
        The neighbours in this test do not cause leg permutations.
        """
        tensor = crandn(2,3,4)
        qn = [[0,1],[0,1,2],[0,1,2,3]]
        node = QNNode(tensor=tensor, identifier="node1", qn=qn)
        node.open_leg_to_parent("parent", 0)
        node.open_leg_to_child("child", 1)
        self.assertEqual(node.get_neighbour_qn("parent"), qn[0])
        self.assertEqual(node.get_neighbour_qn("child"), qn[1])

    def test_with_permutation(self):
        """
        In this test the addition of parent and child will cause a permutation
        in the tensor legs.
        """
        tensor = crandn(2,3,4)
        qn = [[0,1],[0,1,2],[0,1,2,3]]
        node = QNNode(tensor=tensor, identifier="node1", qn=deepcopy(qn))
        node.open_leg_to_parent("parent", 1)
        node.open_leg_to_child("child", 2)
        self.assertEqual(node.get_neighbour_qn("parent"), qn[1])
        self.assertEqual(node.get_neighbour_qn("child"), qn[2])

if __name__ == "__main__":
    unitmain()
