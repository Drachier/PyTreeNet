
from unittest import TestCase, main as unitmain

from pytreenet.special_ttn.special_nodes import (trivial_virtual_node,
                                                 constant_bd_trivial_node)

class TestTrivialVirtualNode(TestCase):

    def test_trivial_virtual_node(self):
        """
        Tests the generation of a trivial virtual node.
        """
        shape = (2,3,4)
        tensor = trivial_virtual_node(shape)
        self.assertEqual((2,3,4,1), tensor.shape)
        self.assertEqual(1, tensor[0,0,0])
        tensor[0,0,0] = 0
        self.assertTrue((tensor == 0).all())

    def test_constant_trivial_virtual_node(self):
        """
        Tests the generation of a trivial virtual noded with constant bond
        dimension around it.
        """
        bond_dim = 3
        num_legs = 4
        tensor = constant_bd_trivial_node(bond_dim,
                                          num_legs)
        self.assertEqual((3,3,3,3,1), tensor.shape)
        self.assertEqual(1, tensor[0,0,0,0])
        tensor[0,0,0,0] = 0
        self.assertTrue((tensor == 0).all())

if __name__ == "__main__":
    unitmain()
