import unittest

from copy import deepcopy

from pytreenet.core.node import Node
from pytreenet.util.ttn_exceptions import NotCompatibleException
from pytreenet.random import crandn

class TestNodeInit(unittest.TestCase):

    def test_init(self):
        shapes = [(), (2, ), (2, 2), (2, 3, 4, 5)]
        for shape in shapes:
            random_tensor = crandn(shape)
            node = Node(tensor=random_tensor)
            self.assertEqual(len(shape), len(node.leg_permutation))
            self.assertEqual(list(range(len(shape))), node.leg_permutation)
            self.assertEqual(shape, node.shape)

    def test_empty_init(self):
        empty = Node()
        self.assertTrue(empty.leg_permutation is None)
        self.assertTrue(empty.shape is None)

class TestNodeMethods(unittest.TestCase):

    def setUp(self):
        # All are tested with no and two open legs
        self.nodes = {}
        self.ids = ["empty0", "empty2", "leaf0", "leaf2", "node1p1c0", "node1p1c2",
                    "node1p2c0", "node1p2c2", "root1c0", "root1c2", "root2c0", "root2c2"]

        # Empty
        tensor0 = crandn(())
        tensor2 = crandn((2, 3))
        self.nodes["empty0"] = Node(tensor0, identifier="empty0")
        self.nodes["empty2"] = Node(tensor2, identifier="empty2")

        # Leaf
        tensor0 = crandn((2,))
        tensor2 = crandn((2, 3, 4))
        parent_id = "parent_id"
        node0 = Node(tensor0, identifier="leaf0")
        node2 = Node(tensor2, identifier="leaf2")
        node0.add_parent(parent_id)
        node2.add_parent(parent_id)
        self.nodes["leaf0"] = node0
        self.nodes["leaf2"] = node2

        # 1 parent 1 child
        tensor0 = crandn((2, 3))
        tensor2 = crandn((2, 3, 4, 5))
        node0 = Node(tensor0, identifier="node1p1c0")
        node2 = Node(tensor2, identifier="node1p1c2")
        parent_id = "parent_id"
        node0.add_parent(parent_id)
        node2.add_parent(parent_id)
        child_id = "child_id"
        node0.add_child(child_id)
        node2.add_child(child_id)
        self.nodes["node1p1c0"] = node0
        self.nodes["node1p1c2"] = node2

        # 1 parent 2 children
        tensor0 = crandn((2, 3, 4))
        tensor2 = crandn((2, 3, 4, 5, 6))
        node0 = Node(tensor0, identifier="node1p2c0")
        node2 = Node(tensor2, identifier="node1p2c2")
        parent_id = "parent_id"
        node0.add_parent(parent_id)
        node2.add_parent(parent_id)
        children_ids = ["child_id", "child2"]
        node0.add_children(children_ids)
        node2.add_children(children_ids)
        self.nodes["node1p2c0"] = node0
        self.nodes["node1p2c2"] = node2

        # Root 1 child
        tensor0 = crandn((2))
        tensor2 = crandn((2, 3, 4))
        node0 = Node(tensor0, identifier="root1c0")
        node2 = Node(tensor2, identifier="root1c2")
        child_id = "child_id"
        node0.add_child(child_id)
        node2.add_child(child_id)
        self.nodes["root1c0"] = node0
        self.nodes["root1c2"] = node2

        # Root 2 children
        tensor0 = crandn((2, 3))
        tensor2 = crandn((2, 3, 4, 5))
        node0 = Node(tensor0, identifier="root2c0")
        node2 = Node(tensor2, identifier="root2c2")
        children_ids = ["child_id", "child2"]
        node0.add_children(children_ids)
        node2.add_children(children_ids)
        self.nodes["root2c0"] = node0
        self.nodes["root2c2"] = node2

    def test_open_leg_to_parent_no_open_leg(self):
        # Have no parent but no open leg
        id_list = ["empty0", "root1c0", "root2c0"]
        for ids in id_list:
            self.assertRaises(ValueError, self.nodes[ids].open_leg_to_parent,
                              "Any", 0)

    def test_open_leg_to_parent_not_root(self):
        # Already have a parent
        id_list = ["node1p2c0","node1p2c2","node1p1c0","node1p1c2"]
        for ids in id_list:
            self.assertRaises(NotCompatibleException,
                              self.nodes[ids].open_leg_to_parent,
                              "id!",2)

    def test_open_leg_to_parent_occupied_leg(self):
        id_list = ["root1c2","root2c2"]
        for ids in id_list:
            self.assertRaises(NotCompatibleException,
                              self.nodes[ids].open_leg_to_parent,
                              "id!",0)

    def test_open_leg_to_parent_None_id(self):
        id_list = ["root1c2","root2c2"]
        for ids in id_list:
            self.assertRaises(ValueError,
                              self.nodes[ids].open_leg_to_parent,
                              None,2)

    def test_open_leg_to_parent_1st_open_leg(self):
        # Haven an open leg and no parent
        id_list = ["empty2", "root1c2", "root2c2"]
        open_leg_value = [0, 1, 2]
        open_leg_value = dict(zip(id_list, open_leg_value))
        shape = [(2,3),(3,2,4),(4,2,3,5)]
        shape = dict(zip(id_list,shape))

        for ids in id_list:
            node = self.nodes[ids]
            node.open_leg_to_parent("parent_id", open_leg_value[ids])
            self.assertEqual(open_leg_value[ids],node.leg_permutation[0])
            self.assertEqual(shape[ids],node.shape)
            self.assertEqual("parent_id", node.parent)

    def test_open_leg_to_parent_2nd_open_leg(self):
        # Haven an open leg and no parent
        id_list = ["empty2", "root1c2", "root2c2"]
        open_leg_value = [1, 2, 3]
        open_leg_value = dict(zip(id_list, open_leg_value))
        shape = [(3,2),(4,2,3),(5,2,3,4)]
        shape = dict(zip(id_list,shape))

        for ids in id_list:
            node = self.nodes[ids]
            node.open_leg_to_parent("parent_id", open_leg_value[ids])
            self.assertEqual(open_leg_value[ids], node.leg_permutation[0])
            self.assertEqual("parent_id", node.parent)
            self.assertEqual(shape[ids],node.shape)
            self.assertEqual("parent_id", node.parent)

    def test_open_leg_to_child_no_open_leg(self):
        # Have no open leg
        id_list = ["empty0", "root1c0", "root2c0","node1p1c0","node1p2c0"]
        for ids in id_list:
            self.assertRaises(ValueError, self.nodes[ids].open_leg_to_child,
                              "Any", 0)

    def test_open_leg_to_child_occupied_leg(self):
        # Have no open leg
        id_list = ["root1c2", "root2c2","node1p1c2","node1p2c2"]
        for ids in id_list:
            self.assertRaises(NotCompatibleException, self.nodes[ids].open_leg_to_child,
                              "Any", 0)

    def test_open_leg_to_child_1st_open_leg(self):
        # Haven an open leg
        id_list = ["empty2", "leaf2", "node1p1c2", "node1p2c2", "root1c2", "root2c2"]
        open_leg_value = [0, 1, 2, 3, 1, 2]
        open_leg_value = dict(zip(id_list, open_leg_value))
        shape = [(2,3),(2,3,4),(2,3,4,5),(2,3,4,5,6),(2,3,4),(2,3,4,5)]
        shape = dict(zip(id_list,shape))

        for ids in id_list:
            node = self.nodes[ids]
            node.open_leg_to_child("new_child_id", open_leg_value[ids])
            # 1st open leg stays in the same position
            self.assertEqual(open_leg_value[ids], node.leg_permutation[open_leg_value[ids]])
            self.assertTrue("new_child_id" in node.children)
            self.assertEqual(shape[ids],node.shape)

    def test_open_leg_to_child_2nd_open_leg(self):
        # Haven an open leg
        id_list = ["empty2", "leaf2", "node1p1c2", "node1p2c2", "root1c2", "root2c2"]
        open_leg_value = [1, 2, 3, 4, 2, 3]
        open_leg_value = dict(zip(id_list, open_leg_value))
        new_position = [0, 1, 2, 3, 1, 2]
        new_position = dict(zip(id_list, new_position))
        shape = [(3,2),(2,4,3),(2,3,5,4),(2,3,4,6,5),(2,4,3),(2,3,5,4)]
        shape = dict(zip(id_list,shape))

        for ids in id_list:
            node = self.nodes[ids]
            node.open_leg_to_child("new_child_id", open_leg_value[ids])
            self.assertEqual(open_leg_value[ids], node.leg_permutation[new_position[ids]])
            self.assertTrue("new_child_id" in node.children)
            self.assertEqual(shape[ids],node.shape)

    def test_open_legs_to_children_legs_root(self):
        tensor = crandn((2,3,4,5,6))
        children_ids = ["id1", "id2", "id3"]

        # Test from 0
        root = Node(tensor=tensor, identifier="root")
        open_legs = [0,1,2]
        child_dict = dict(zip(children_ids, open_legs))
        root.open_legs_to_children(child_dict)
        self.assertEqual([0,1,2,3,4], root.leg_permutation)
        self.assertEqual(children_ids, root.children)
        self.assertEqual((2,3,4,5,6),root.shape)

        # Test not from 0
        root = Node(tensor=tensor, identifier="root")
        open_legs = [2,3,4]
        child_dict = dict(zip(children_ids, open_legs))
        root.open_legs_to_children(child_dict)
        self.assertEqual([2,3,4,0,1], root.leg_permutation)
        self.assertEqual(children_ids, root.children)
        self.assertEqual((4,5,6,2,3),root.shape)

        # Test unordered
        root = Node(tensor=tensor, identifier="root")
        open_legs = [0,3,2]
        child_dict = dict(zip(children_ids, open_legs))
        root.open_legs_to_children(child_dict)
        self.assertEqual([0,3,2,1,4], root.leg_permutation)
        self.assertEqual(children_ids, root.children)
        self.assertEqual((2,5,4,3,6),root.shape)

    def test_open_legs_to_children_legs_non_root_parent_at_0(self):
        tensor = crandn((2,3,4,5,6))
        children_ids = ["id1", "id2", "id3"]

        # Test ordered
        node = Node(tensor=tensor, identifier="node")
        node.open_leg_to_parent("Vader", 0)
        open_legs = [1,2,3]
        child_dict = dict(zip(children_ids, open_legs))
        node.open_legs_to_children(child_dict)
        self.assertEqual([0,1,2,3,4], node.leg_permutation)
        self.assertEqual(children_ids, node.children)
        self.assertEqual((2,3,4,5,6),node.shape)

        # Test not from 1
        node = Node(tensor=tensor, identifier="node")
        node.open_leg_to_parent("Vader", 0)
        open_legs = [2,3,4]
        child_dict = dict(zip(children_ids, open_legs))
        node.open_legs_to_children(child_dict)
        self.assertEqual([0,2,3,4,1], node.leg_permutation)
        self.assertEqual(children_ids, node.children)
        self.assertEqual((2,4,5,6,3),node.shape)

        # Test unordered
        node = Node(tensor=tensor, identifier="node")
        node.open_leg_to_parent("Vader", 0)
        open_legs = [3,1,4]
        child_dict = dict(zip(children_ids, open_legs))
        node.open_legs_to_children(child_dict)
        self.assertEqual([0,3,1,4,2], node.leg_permutation)
        self.assertEqual(children_ids, node.children)
        self.assertEqual((2,5,3,6,4),node.shape)

    def test_open_legs_to_children_legs_non_root_parent_at_1(self):
        tensor = crandn((2,3,4,5,6))
        children_ids = ["id1", "id2", "id3"]

        # Test ordered
        node = Node(tensor=tensor, identifier="node")
        node.open_leg_to_parent("Vader", 1)
        open_legs = [1,2,3]
        child_dict = dict(zip(children_ids, open_legs))
        node.open_legs_to_children(child_dict)
        self.assertEqual([1,0,2,3,4], node.leg_permutation)
        self.assertEqual(children_ids, node.children)
        self.assertEqual((3,2,4,5,6),node.shape)

        # Test not from 1
        node = Node(tensor=tensor, identifier="node")
        node.open_leg_to_parent("Vader", 1)
        open_legs = [2,3,4]
        child_dict = dict(zip(children_ids, open_legs))
        node.open_legs_to_children(child_dict)
        self.assertEqual([1,2,3,4,0], node.leg_permutation)
        self.assertEqual(children_ids, node.children)
        self.assertEqual((3,4,5,6,2),node.shape)

        # Test unordered
        node = Node(tensor=tensor, identifier="node")
        node.open_leg_to_parent("Vader", 1)
        open_legs = [3,1,4]
        child_dict = dict(zip(children_ids, open_legs))
        node.open_legs_to_children(child_dict)
        self.assertEqual([1,3,0,4,2], node.leg_permutation)
        self.assertEqual(children_ids, node.children)
        self.assertEqual((3,5,2,6,4),node.shape)

    def test_parent_leg_to_open_leg(self):
        # Have a parent leg
        id_list = ["leaf0", "leaf2", "node1p1c0", "node1p1c2",
                   "node1p2c0", "node1p2c2"]
        new_position = [0, 2, 1, 3, 2, 4]
        new_position = dict(zip(id_list, new_position))
        shape = [(2, ),(3,4,2),(3,2),(3,4,5,2),
                 (3,4,2),(3,4,5,6,2)]
        shape = dict(zip(id_list,shape))

        for ids in id_list:
            node = self.nodes[ids]
            node.parent_leg_to_open_leg()
            self.assertEqual(0, node.leg_permutation[new_position[ids]])
            self.assertEqual(shape[ids],node.shape)
            self.assertTrue(node.is_root())

    def test_child_leg_to_open_leg_1st_child(self):
        id_list = ["node1p1c0", "node1p1c2", "node1p2c0", "node1p2c2",
                   "root1c0","root1c2","root2c0","root2c2"]
        value = [1,1,1,1,0,0,0,0]
        value = dict(zip(id_list, value))
        new_position = [1, 3, 2, 4, 0, 2, 1, 3]
        new_position = dict(zip(id_list, new_position))
        shape = [(2,3),(2,4,5,3),(2,4,3),(2,4,5,6,3),
                 (2, ),(3,4,2),(3,2),(3,4,5,2)]
        shape = dict(zip(id_list,shape))

        for ids in id_list:
            node = self.nodes[ids]
            node.child_leg_to_open_leg("child_id")
            self.assertEqual(value[ids], node.leg_permutation[new_position[ids]])
            self.assertEqual(shape[ids],node.shape)

    def test_child_leg_to_open_leg_2nd_child(self):
        id_list = ["node1p2c0", "node1p2c2","root2c0","root2c2"]
        value = [2,2,1,1]
        value = dict(zip(id_list, value))
        new_position = [2, 4, 1, 3]
        new_position = dict(zip(id_list, new_position))
        shape = [(2,3,4),(2,3,5,6,4),(2,3),(2,4,5,3)]
        shape = dict(zip(id_list,shape))

        for ids in id_list:
            node = self.nodes[ids]
            node.child_leg_to_open_leg("child2")
            self.assertEqual(value[ids], node.leg_permutation[new_position[ids]])
            self.assertEqual(shape[ids],node.shape)

    def test_children_legs_to_open_legs_2nd_child(self):
        id_list = ["node1p2c0", "node1p2c2","root2c0","root2c2"]
        value = [1,1,0,0]
        value = dict(zip(id_list, value))
        new_position = [1, 3, 0, 2]
        new_position = dict(zip(id_list, new_position))
        shape = [(2,3,4),(2,5,6,3,4),(2,3),(4,5,2,3)]
        shape = dict(zip(id_list,shape))

        for ids in id_list:
            node = self.nodes[ids]
            node.children_legs_to_open_legs(["child_id","child2"])
            self.assertEqual(value[ids], node.leg_permutation[new_position[ids]])
            self.assertEqual(value[ids]+1, node.leg_permutation[new_position[ids]+1])
            self.assertEqual(shape[ids],node.shape)

    def test_nlegs(self):
        correct_numbers = [0, 2, 1, 3, 2, 4, 3, 5, 1, 3, 2, 4]
        correct_numbers = dict(zip(self.ids, correct_numbers))

        for ids, node in self.nodes.items():
            self.assertEqual(correct_numbers[ids], node.nlegs())

    def test_nchild_legs(self):
        correct_numbers = [0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2]
        correct_numbers = dict(zip(self.ids, correct_numbers))

        for ids, node in self.nodes.items():
            self.assertEqual(correct_numbers[ids], node.nchild_legs())

    def test_nvirt_legs(self):
        correct_numbers = [0, 0, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2]
        correct_numbers = dict(zip(self.ids, correct_numbers))

        for ids, node in self.nodes.items():
            self.assertEqual(correct_numbers[ids], node.nvirt_legs())

    def test_nopen_legs(self):
        correct_numbers = [0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2]
        correct_numbers = dict(zip(self.ids, correct_numbers))

        for ids, node in self.nodes.items():
            self.assertEqual(correct_numbers[ids], node.nopen_legs())

    def test_swap_two_child_legs(self):
        # Have two children
        id_list = ["node1p2c0", "node1p2c2", "root2c0", "root2c2"]
        leg_value1 = [self.nodes[ids].neighbour_index("child_id")
                      for ids in id_list]
        leg_value2 = [self.nodes[ids].neighbour_index("child2")
                      for ids in id_list]
        leg_values = zip(leg_value1, leg_value2)
        leg_values = dict(zip(id_list, leg_values))

        for ids in id_list:
            node = self.nodes[ids]
            node.swap_two_child_legs("child_id", "child2")
            self.assertEqual(leg_values[ids][0], node.leg_permutation[leg_values[ids][1]])
            self.assertEqual(leg_values[ids][1], node.leg_permutation[leg_values[ids][0]])

    def test_open_dimension(self):
        open_dimensions = [1,6,1,12,1,20,1,30,1,12,1,20]
        open_dimensions = dict(zip(self.ids, open_dimensions))
        for ids in self.ids:
            self.assertEqual(open_dimensions[ids],
                             self.nodes[ids].open_dimension())

    def test_eq_to_self(self):
        for ids, node in self.nodes.items():
            for ids2, node2 in self.nodes.items():
                if ids == ids2:
                    self.assertTrue(node == node2)
                else:
                    self.assertFalse(node == node2)

class TestNodeEq(unittest.TestCase):
    """
    Testing equality only according to shape. The neigbours are already 
        tested in the parent class tests.
        c.f. test_graph_node.TestGraphNode.test_eq
    """
    def setUp(self) -> None:
        self.shape = (2, 3, 4)
        tensor1 = crandn(self.shape)
        tensor2 = crandn(self.shape)
        identifier = "id"
        self.node1 = Node(identifier=identifier, tensor=tensor1)
        self.node2 = Node(identifier=identifier, tensor=tensor2)

    def test_eq_shape_same_ext_shape_same_internal_shape(self):
        """
        Test equality with same shape.
        """
        self.assertTrue(self.node1 == self.node2)
        self.assertTrue(self.node2 == self.node1)

    def test_eq_shape_same_ext_shape_different_internal_shape(self):
        """
        It does not matter, if the internal shape is different.
        """
        self.node2._shape = (4,3,2)
        self.node2._leg_permutation = [2,1,0]
        self.assertTrue(self.node1 == self.node2)
        self.assertTrue(self.node2 == self.node1)

    def test_eq_shape_different_ext_shape_same_internal_shape(self):
        """
        It does not matter, if the external shape is different.
        """
        self.node2._shape = (4,3,2)
        self.assertFalse(self.node1 == self.node2)
        self.assertFalse(self.node2 == self.node1)

    def test_eq_shape_different_ext_shape_different_internal_shape(self):
        """
        It does not matter, if the external shape is different.
        """
        self.node2._shape = (4,3,2)
        self.node2._leg_permutation = [0,2,1]
        self.assertFalse(self.node1 == self.node2)
        self.assertFalse(self.node2 == self.node1)

class Test_replace_tensor(unittest.TestCase):

    def test_trivial(self):
        """
        Tests the method with a tensor that fits without issue.
        """
        tensor = crandn((2,3,4))
        node = Node(tensor, identifier="id")
        ref = deepcopy(node)
        new_tensor = crandn((2,3,4))
        node.replace_tensor(new_tensor)
        # Accordingly the nothing should have changed
        self.assertEqual(ref, node)

    def test_with_perm(self):
        """
        Tests the method where the tensor needs a permutation to fit, 
        but the node has a trivial permutation.
        """
        tensor = crandn((2,3,4))
        node = Node(tensor, identifier="id")
        new_tensor = crandn((4,2,3))
        perm = [1,2,0]
        node.replace_tensor(new_tensor, permutation=perm)
        self.assertEqual(perm, node._leg_permutation)
        self.assertEqual(new_tensor.shape, node._shape)

    def test_with_double_perm(self):
        """
        Tests the method were the tensor needs a permutation to fit and the
        node already has a permutation.
        """
        tensor = crandn((2,3,4))
        node = Node(tensor, identifier="id")
        node.open_leg_to_parent("parent", 1)
        node.open_leg_to_child("child", 2)
        new_tensor = crandn((4,3,2))
        perm = [1,0,2]
        node.replace_tensor(new_tensor, permutation=perm)
        self.assertEqual(perm, node._leg_permutation)
        self.assertEqual(new_tensor.shape, node._shape)

    def test_with_node_perm(self):
        """
        Tests the method were the tensor does not need a permutation to fit,
        but the node has a permutation.
        """
        tensor = crandn((2,3,4))
        node = Node(tensor, identifier="id")
        node.open_leg_to_parent("parent", 1)
        node.open_leg_to_child("child", 2)
        new_tensor = crandn((3,4,2))
        node.replace_tensor(new_tensor)
        self.assertEqual([0,1,2], node._leg_permutation)
        self.assertEqual(new_tensor.shape, node._shape)

    def test_invalid_no_perm(self):
        """
        Tests the method were the tensor does not fit and no permutation is
        given.
        """
        tensor = crandn((2,3,4))
        node = Node(tensor, identifier="id")
        new_tensor = crandn((3,4,2))
        self.assertRaises(NotCompatibleException, node.replace_tensor,
                          new_tensor)

    def test_invalid_node_perm(self):
        """
        Testst the method were the tensor does not fit and has not permutation
        given, but the node has a permutation.
        """
        tensor = crandn((2,3,4))
        node = Node(tensor, identifier="id")
        node.open_leg_to_parent("parent", 1)
        node.open_leg_to_child("child", 2)
        new_tensor = crandn((2,3,4))
        self.assertRaises(NotCompatibleException, node.replace_tensor,
                          new_tensor)

    def test_invalid_perm(self):
        """
        Tests the method were the tensor does not fit and a permutation is
        given but wrong.
        """
        tensor = crandn((2,3,4))
        node = Node(tensor, identifier="id")
        new_tensor = crandn((3,4,2))
        perm = [1,2,0]
        self.assertRaises(NotCompatibleException, node.replace_tensor,
                          new_tensor, perm)

    def test_invalid_perm_orig_shape(self):
        """
        Tests the method were the tensor does fit, but a wrong permutation is
        given.
        """
        tensor = crandn((2,3,4))
        node = Node(tensor, identifier="id")
        new_tensor = crandn((2,3,4))
        perm = [1,2,0]
        self.assertRaises(NotCompatibleException, node.replace_tensor,
                          new_tensor, perm)

    def test_invalid_two_perms(self):
        """
        Tests the method were the tensor does not fit and a wrong permutation
        is given, while the node also has a permutation.
        """
        tensor = crandn((2,3,4))
        node = Node(tensor, identifier="id")
        node.open_leg_to_parent("parent", 1)
        node.open_leg_to_child("child", 2)
        new_tensor = crandn((2,4,3))
        perm = [1,0,2]
        self.assertRaises(NotCompatibleException, node.replace_tensor,
                          new_tensor, perm)

    def test_invalid_two_perms_orig_shape(self):
        """
        Tests the method were the tensor does fit, but a wrong permutation is
        given, while the node also has a permutation.
        """
        tensor = crandn((2,3,4))
        node = Node(tensor, identifier="id")
        node.open_leg_to_parent("parent", 1)
        node.open_leg_to_child("child", 2)
        new_tensor = crandn((3,4,2))
        perm = [1,0,2]
        self.assertRaises(NotCompatibleException, node.replace_tensor,
                          new_tensor, perm)


if __name__ == "__main__":
    unittest.main()
