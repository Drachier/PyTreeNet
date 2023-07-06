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
        
    def test_absorb_tensor(self):
        rand_tensor1 = ptn.crandn((2,5))
        self.node1.absorb_tensor(rand_tensor1, 0, 0)
        self.assertEqual(self.node1.tensor.shape, (5,3,4))
        
        rand_tensor2 = ptn.crandn((2,3))
        self.node1.absorb_tensor(rand_tensor2, 1, 1)
        self.assertEqual(self.node1.tensor.shape, (5,2,4))
        
        rand_tensor2 = ptn.crandn((4,3))
        self.node1.absorb_tensor(rand_tensor2, 0, 2)
        self.assertEqual(self.node1.tensor.shape, (5,2,3))

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
        
    def test_conjugate_node(self):
        conj_node = ptn.conjugate_node(self.node1)
        
        self.assertEqual("conj_1", conj_node.identifier)
        self.assertEqual("conj_First node", conj_node.tag)
        self.assertTrue(np.allclose(conj_node.tensor, np.conj(self.node1.tensor)))

class TestComplicatedTensorNode(unittest.TestCase):
    def setUp(self):
        tensor = ptn.crandn((2,3,4,5,6,7))
        
        self.node = ptn.TensorNode(tensor, identifier="MyNode")
        
    def test_order_legs_wolastlegindex_woparent(self):        
        # Identifiers correspond to leg dimension
        open_leg_list = [0,2,3,5]
        id_list = ["id2", "id4", "id5", "id7"]
        self.node.open_legs_to_children(open_leg_list, id_list)
           
        self.node.order_legs()
        
        self.assertEqual(self.node.open_legs, [4,5])
        
        correct_children_legs = {"id2": 0, "id4": 1, "id5": 2, "id7": 3}
        self.assertEqual(self.node.children_legs, correct_children_legs)
        
        correct_tensor_shape = (2,4,5,7,3,6)
        self.assertEqual(self.node.tensor.shape, correct_tensor_shape)
        
    def test_order_leg_wolastlegindex_wparent(self):
        # Identifiers correspond to leg dimension
        open_leg_list = [0,3,5]
        id_list = ["id2", "id5", "id7"]
        self.node.open_legs_to_children(open_leg_list,  id_list)   
        self.node.open_leg_to_parent(1, "id3")
        
        self.node.order_legs()
        
        correct_open_legs = [4,5]
        self.assertEqual(self.node.open_legs, correct_open_legs)
        
        correct_parent_leg = ["id3", 3]
        self.assertEqual(self.node.parent_leg, correct_parent_leg)
        
        correct_children_legs = {"id2": 0, "id5": 1, "id7": 2}
        self.assertEqual(self.node.children_legs, correct_children_legs)
        
        correct_tensor_shape = (2,5,7,3,4,6)
        self.assertEqual(self.node.tensor.shape, correct_tensor_shape)
        
    def test_order_legs_wlastlegindexopen_woparent(self):
        # Identifiers correspond to leg dimension
        open_leg_list = [0,2,3,5]
        id_list = ["id2", "id4", "id5", "id7"]
        self.node.open_legs_to_children(open_leg_list, id_list)
           
        self.node.order_legs(last_leg_index=1)
        
        correct_open_legs = [5, 4]
        self.assertEqual(self.node.open_legs, correct_open_legs)
        
        correct_children_legs = {"id2": 0, "id4": 1, "id5": 2, "id7": 3}
        self.assertEqual(self.node.children_legs, correct_children_legs)
        
        correct_tensor_shape = (2,4,5,7,6,3)
        self.assertEqual(self.node.tensor.shape, correct_tensor_shape)
        
    def test_order_legs_wlastlegindexchil_woparent(self):
        # Identifiers correspond to leg dimension
        open_leg_list = [0,2,3,5]
        id_list = ["id2", "id4", "id5", "id7"]
        self.node.open_legs_to_children(open_leg_list, id_list)
           
        self.node.order_legs(last_leg_index=2)
        
        correct_open_legs = [3, 4]
        self.assertEqual(self.node.open_legs, correct_open_legs)
        
        correct_children_legs = {"id2": 0, "id4": 5, "id5": 1, "id7": 2}
        self.assertEqual(self.node.children_legs, correct_children_legs)
        
        correct_tensor_shape = (2,5,7,3,6,4)
        self.assertEqual(self.node.tensor.shape, correct_tensor_shape)
        
    def test_order_legs_wlastindexopen_wparent(self):
        # Identifiers correspond to leg dimension
        open_leg_list = [0,3,5]
        id_list = ["id2", "id5", "id7"]
        self.node.open_legs_to_children(open_leg_list, id_list)   
        self.node.open_leg_to_parent(1, "id3")
        
        self.node.order_legs(last_leg_index=2)
        
        correct_open_legs = [5,4]
        self.assertEqual(self.node.open_legs, correct_open_legs)
        
        correct_parent_leg = ["id3", 3]
        self.assertEqual(self.node.parent_leg, correct_parent_leg)
        
        correct_children_legs = {"id2": 0, "id5": 1, "id7": 2}
        self.assertEqual(self.node.children_legs, correct_children_legs)
        
        correct_tensor_shape = (2,5,7,3,6,4)
        self.assertEqual(self.node.tensor.shape, correct_tensor_shape)        
    
    def test_order_legs_wlastindexparent(self):
        # Identifiers correspond to leg dimension
        open_leg_list = [0,3,5]
        id_list = ["id2", "id5", "id7"]
        self.node.open_legs_to_children(open_leg_list, id_list)   
        self.node.open_leg_to_parent(1, "id3")
        
        self.node.order_legs(last_leg_index=1)
        
        correct_open_legs = [3,4]
        self.assertEqual(self.node.open_legs, correct_open_legs)
        
        correct_parent_leg = ["id3", 5]
        self.assertEqual(self.node.parent_leg, correct_parent_leg)
        
        correct_children_legs = {"id2": 0, "id5": 1, "id7": 2}
        self.assertEqual(self.node.children_legs, correct_children_legs)
        
        correct_tensor_shape = (2,5,7,4,6,3)
        self.assertEqual(self.node.tensor.shape, correct_tensor_shape)
        
    def test_shape(self):
        found_shape = self.node.shape
        correct_shape = (2,3,4,5,6,7)
        
        self.assertEqual(found_shape, correct_shape)
        
    def test_shape_of_legs(self):
        leg_indices = [0,2,4,5]
        
        # Test for tuple
        found_shape = self.node.shape_of_legs(leg_indices)
        
        correct_shape = (2,4,6,7)
        
        self.assertEqual(correct_shape, found_shape)
        
        # Test for list
        found_shape = self.node.shape_of_legs(leg_indices, dtype="list")
        
        correct_shape = [2,4,6,7]
        
        self.assertEqual(correct_shape, found_shape)
        
        # Test error        
        self.assertRaises(ValueError, self.node.shape_of_legs,
                                      leg_indices, "pancake")
        
    def test_leg_number_functions(self):
        open_leg_list = [0,2,3,5]
        id_list = ["id2", "id4", "id5", "id7"]
        self.node.open_legs_to_children(open_leg_list, id_list)
        
        # Total legs
        found_total_legs = self.node.nlegs()
        correct_number = 6
        self.assertEqual(correct_number, found_total_legs)
        
        # Children legs
        found_children_legs = self.node.nchild_legs()
        correct_number = 4
        self.assertEqual(correct_number, found_children_legs)
        
        # Virtual legs without parent
        found_virtual_legs = self.node.nvirt_legs()
        self.assertEqual(correct_number, found_virtual_legs)
        
        # Virtual legs with parent
        self.node.child_leg_to_open_leg("id5")
        self.node.open_leg_to_parent(3, "id5")
        
        found_virtual_legs = self.node.nvirt_legs()
        self.assertEqual(correct_number, found_virtual_legs)
        
        # Open Legs
        found_open_legs = self.node.nopen_legs()
        correct_number = 2
        self.assertEqual(correct_number, found_open_legs)
        
if __name__ == "__main__":
    unittest.main()