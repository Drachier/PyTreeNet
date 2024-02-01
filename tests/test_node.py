import unittest

import pytreenet as ptn

class TestNodeInit(unittest.TestCase):

    def test_init(self):
        shapes = [(), (2, ), (2, 2), (2, 3, 4, 5)]
        for shape in shapes:
            random_tensor = ptn.crandn(shape)
            node = ptn.Node(tensor=random_tensor)
            self.assertEqual(len(shape), len(node.leg_permutation))
            self.assertEqual(list(range(len(shape))), node.leg_permutation)
            self.assertEqual(shape, node.shape)

    def test_empty_init(self):
        empty = ptn.Node()
        self.assertTrue(empty.leg_permutation is None)
        self.assertTrue(empty.shape is None)

class TestNodeMethods(unittest.TestCase):

    def setUp(self):
        # All are tested with no and two open legs
        self.nodes = {}
        self.ids = ["empty0", "empty2", "leaf0", "leaf2", "node1p1c0", "node1p1c2",
                    "node1p2c0", "node1p2c2", "root1c0", "root1c2", "root2c0", "root2c2"]

        # Empty
        tensor0 = ptn.crandn(())
        tensor2 = ptn.crandn((2, 3))
        self.nodes["empty0"] = ptn.Node(tensor0, identifier="empty0")
        self.nodes["empty2"] = ptn.Node(tensor2, identifier="empty2")

        # Leaf
        tensor0 = ptn.crandn((2,))
        tensor2 = ptn.crandn((2, 3, 4))
        parent_id = "parent_id"
        node0 = ptn.Node(tensor0, identifier="leaf0")
        node2 = ptn.Node(tensor2, identifier="leaf2")
        node0.add_parent(parent_id)
        node2.add_parent(parent_id)
        self.nodes["leaf0"] = node0
        self.nodes["leaf2"] = node2

        # 1 parent 1 child
        tensor0 = ptn.crandn((2, 3))
        tensor2 = ptn.crandn((2, 3, 4, 5))
        node0 = ptn.Node(tensor0, identifier="node1p1c0")
        node2 = ptn.Node(tensor2, identifier="node1p1c2")
        parent_id = "parent_id"
        node0.add_parent(parent_id)
        node2.add_parent(parent_id)
        child_id = "child_id"
        node0.add_child(child_id)
        node2.add_child(child_id)
        self.nodes["node1p1c0"] = node0
        self.nodes["node1p1c2"] = node2

        # 1 parent 2 children
        tensor0 = ptn.crandn((2, 3, 4))
        tensor2 = ptn.crandn((2, 3, 4, 5, 6))
        node0 = ptn.Node(tensor0, identifier="node1p2c0")
        node2 = ptn.Node(tensor2, identifier="node1p2c2")
        parent_id = "parent_id"
        node0.add_parent(parent_id)
        node2.add_parent(parent_id)
        children_ids = ["child_id", "child2"]
        node0.add_children(children_ids)
        node2.add_children(children_ids)
        self.nodes["node1p2c0"] = node0
        self.nodes["node1p2c2"] = node2

        # Root 1 child
        tensor0 = ptn.crandn((2))
        tensor2 = ptn.crandn((2, 3, 4))
        node0 = ptn.Node(tensor0, identifier="root1c0")
        node2 = ptn.Node(tensor2, identifier="root1c2")
        child_id = "child_id"
        node0.add_child(child_id)
        node2.add_child(child_id)
        self.nodes["root1c0"] = node0
        self.nodes["root1c2"] = node2

        # Root 2 children
        tensor0 = ptn.crandn((2, 3))
        tensor2 = ptn.crandn((2, 3, 4, 5))
        node0 = ptn.Node(tensor0, identifier="root2c0")
        node2 = ptn.Node(tensor2, identifier="root2c2")
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
            self.assertRaises(ptn.NotCompatibleException,
                              self.nodes[ids].open_leg_to_parent,
                              "id!",2)

    def test_open_leg_to_parent_occupied_leg(self):
        id_list = ["root1c2","root2c2"]
        for ids in id_list:
            self.assertRaises(ptn.NotCompatibleException,
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
            self.assertRaises(ptn.NotCompatibleException, self.nodes[ids].open_leg_to_child,
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
        tensor = ptn.crandn((2,3,4,5,6))
        children_ids = ["id1", "id2", "id3"]

        # Test from 0
        root = ptn.Node(tensor=tensor, identifier="root")
        open_legs = [0,1,2]
        child_dict = dict(zip(children_ids, open_legs))
        root.open_legs_to_children(child_dict)
        self.assertEqual([0,1,2,3,4], root.leg_permutation)
        self.assertEqual(children_ids, root.children)
        self.assertEqual((2,3,4,5,6),root.shape)

        # Test not from 0
        root = ptn.Node(tensor=tensor, identifier="root")
        open_legs = [2,3,4]
        child_dict = dict(zip(children_ids, open_legs))
        root.open_legs_to_children(child_dict)
        self.assertEqual([2,3,4,0,1], root.leg_permutation)
        self.assertEqual(children_ids, root.children)
        self.assertEqual((4,5,6,2,3),root.shape)

        # Test unordered
        root = ptn.Node(tensor=tensor, identifier="root")
        open_legs = [0,3,2]
        child_dict = dict(zip(children_ids, open_legs))
        root.open_legs_to_children(child_dict)
        self.assertEqual([0,3,2,1,4], root.leg_permutation)
        self.assertEqual(children_ids, root.children)
        self.assertEqual((2,5,4,3,6),root.shape)

    def test_open_legs_to_children_legs_non_root_parent_at_0(self):
        tensor = ptn.crandn((2,3,4,5,6))
        children_ids = ["id1", "id2", "id3"]

        # Test ordered
        node = ptn.Node(tensor=tensor, identifier="node")
        node.open_leg_to_parent("Vader", 0)
        open_legs = [1,2,3]
        child_dict = dict(zip(children_ids, open_legs))
        node.open_legs_to_children(child_dict)
        self.assertEqual([0,1,2,3,4], node.leg_permutation)
        self.assertEqual(children_ids, node.children)
        self.assertEqual((2,3,4,5,6),node.shape)

        # Test not from 1
        node = ptn.Node(tensor=tensor, identifier="node")
        node.open_leg_to_parent("Vader", 0)
        open_legs = [2,3,4]
        child_dict = dict(zip(children_ids, open_legs))
        node.open_legs_to_children(child_dict)
        self.assertEqual([0,2,3,4,1], node.leg_permutation)
        self.assertEqual(children_ids, node.children)
        self.assertEqual((2,4,5,6,3),node.shape)

        # Test unordered
        node = ptn.Node(tensor=tensor, identifier="node")
        node.open_leg_to_parent("Vader", 0)
        open_legs = [3,1,4]
        child_dict = dict(zip(children_ids, open_legs))
        node.open_legs_to_children(child_dict)
        self.assertEqual([0,3,1,4,2], node.leg_permutation)
        self.assertEqual(children_ids, node.children)
        self.assertEqual((2,5,3,6,4),node.shape)

    def test_open_legs_to_children_legs_non_root_parent_at_1(self):
        tensor = ptn.crandn((2,3,4,5,6))
        children_ids = ["id1", "id2", "id3"]

        # Test ordered
        node = ptn.Node(tensor=tensor, identifier="node")
        node.open_leg_to_parent("Vader", 1)
        open_legs = [1,2,3]
        child_dict = dict(zip(children_ids, open_legs))
        node.open_legs_to_children(child_dict)
        self.assertEqual([1,0,2,3,4], node.leg_permutation)
        self.assertEqual(children_ids, node.children)
        self.assertEqual((3,2,4,5,6),node.shape)

        # Test not from 1
        node = ptn.Node(tensor=tensor, identifier="node")
        node.open_leg_to_parent("Vader", 1)
        open_legs = [2,3,4]
        child_dict = dict(zip(children_ids, open_legs))
        node.open_legs_to_children(child_dict)
        self.assertEqual([1,2,3,4,0], node.leg_permutation)
        self.assertEqual(children_ids, node.children)
        self.assertEqual((3,4,5,6,2),node.shape)

        # Test unordered
        node = ptn.Node(tensor=tensor, identifier="node")
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

    def test_get_child_leg_1st_child(self):
        # Have a child
        id_list = ["node1p1c0", "node1p1c2", "node1p2c0", "node1p2c2",
                   "root1c0", "root1c2", "root2c0", "root2c2"]
        leg_value = [1, 1, 1, 1, 0, 0, 0, 0]
        leg_value = dict(zip(id_list, leg_value))

        for ids in id_list:
            node = self.nodes[ids]
            self.assertEqual(leg_value[ids], node.get_child_leg("child_id"))

    def test_get_child_leg_2nd_child(self):
        # Have two children
        id_list = ["node1p2c0", "node1p2c2", "root2c0", "root2c2"]
        leg_value = [2, 2, 1, 1]
        leg_value = dict(zip(id_list, leg_value))

        for ids in id_list:
            node = self.nodes[ids]
            self.assertEqual(leg_value[ids], node.get_child_leg("child2"))

    def test_swap_two_child_legs(self):
        # Have two children
        id_list = ["node1p2c0", "node1p2c2", "root2c0", "root2c2"]
        leg_value1 = [self.nodes[ids].get_child_leg("child_id")
                      for ids in id_list]
        leg_value2 = [self.nodes[ids].get_child_leg("child2")
                      for ids in id_list]
        leg_values = zip(leg_value1, leg_value2)
        leg_values = dict(zip(id_list, leg_values))

        for ids in id_list:
            node = self.nodes[ids]
            node.swap_two_child_legs("child_id", "child2")
            self.assertEqual(leg_values[ids][0], node.leg_permutation[leg_values[ids][1]])
            self.assertEqual(leg_values[ids][1], node.leg_permutation[leg_values[ids][0]])

    def test_open_dimension(self):
        open_dimensions = [0,6,0,12,0,20,0,30,0,12,0,20]
        open_dimensions = dict(zip(self.ids, open_dimensions))
        for ids in self.ids:
            self.assertEqual(open_dimensions[ids],
                             self.nodes[ids].open_dimension())

if __name__ == "__main__":
    unittest.main()
