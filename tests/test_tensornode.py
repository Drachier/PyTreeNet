import unittest
import numpy as np

import pytreenet as ptn

class TestTensorNode(unittest.TestCase):

    def setUp(self):
        tensor1 = ptn.crandn((2,3,4))
        self.node1 = ptn.TensorNode(tensor=tensor1, tag="First node", identifier='1')
        self.original_open_leg_number = len(self.node1.open_legs)

    def test_attributes(self):
        self.assertEqual(self.node1.open_legs, [0,1,2])
        self.assertEqual(self.node1.tag, "First node")
        self.assertEqual(self.node1.identifier, "1")

        tensor2 = ptn.crandn((2,3,4))
        tag_None_node = ptn.TensorNode(tensor=tensor2, identifier="id2")
        self.assertEqual(tag_None_node.identifier, "id2")
        self.assertEqual(tag_None_node.tag, "id2")

        tensor3 = ptn.crandn((2,3,4))
        all_None_node = ptn.TensorNode(tensor=tensor3)
        self.assertEqual(all_None_node.open_legs, [0,1,2])
        self.assertTrue(np.allclose(all_None_node.tensor, tensor3))
        self.assertFalse(all_None_node.identifier == None)
        self.assertTrue(all_None_node.identifier == all_None_node.tag)

    def test_open_leg_to_parent(self):
        self.node1.open_leg_to_parent(1, "Parent")
        self.assertEqual(self.node1.parent_leg, ["Parent", 1])
        new_open_leg_number = len(self.node1.open_legs)
        self.assertEqual(self.original_open_leg_number - 1, new_open_leg_number)

    def test_open_legs_to_children(self):
        self.node1.open_legs_to_children([1,2], ["id1", "id2"])
        new_open_leg_number = len(self.node1.open_legs)
        self.assertEqual(self.original_open_leg_number - 2, new_open_leg_number)
        reference_dic = {"id1" :1, "id2" :2}
        self.assertEqual(self.node1.children_legs, reference_dic)
        self.assertEqual(self.node1.open_legs, [0])

    def test_parent_leg_to_open_leg(self):
        self.node1.open_leg_to_parent(1, "Parent")
        self.node1.parent_leg_to_open_leg()
        self.assertEqual(self.node1.parent_leg,[])
        self.assertEqual(self.node1.open_legs,[0,2,1])

    def test_children_legs_to_open_legs(self):
        self.node1.open_legs_to_children([0,2], ["id0", "id2"])
        self.node1.children_legs_to_open_legs(["id0", "id2"])
        self.assertEqual(self.node1.children_legs, dict())
        self.assertEqual(self.node1.open_legs,[1,0,2])

    def test_is_root(self):
        self.assertTrue(self.node1.is_root())
        self.node1.open_leg_to_parent(1, "Parent")
        self.assertFalse(self.node1.is_root())

    def test_has_x_childre(self):
        self.node1.open_legs_to_children([0,2], ["id0", "id2"])
        self.assertTrue(self.node1.has_x_children(1))
        self.assertTrue(self.node1.has_x_children(2))
        self.assertFalse(self.node1.has_x_children(3))

    def test_is_leaf(self):
        self.assertTrue(self.node1.is_leaf())
        self.node1.open_legs_to_children([0,2], ["id0", "id2"])
        self.assertFalse(self.node1.is_leaf())

if __name__ == "__main__":
    unittest.main()