import unittest

import pytreenet as ptn

class TestNodeInit(unittest.TestCase):

    def test_init(self):
        shapes = [(), (2, ), (2, 2), (2, 3, 4, 5)]
        for shape in shapes:
            random_tensor = ptn.crandn(shape)
            leg_node = ptn.Node(tensor=random_tensor)
            self.assertEqual(len(shape), len(leg_node.leg_permutation))
            self.assertEqual(list(range(len(shape))), leg_node.leg_permutation)

    def test_from_node(self):
        # All are tested with exactly two open legs
        # Test empty
        tensor = ptn.crandn((2, 2))
        node = ptn.Node()
        leg_node = ptn.Node.from_node(tensor, node)

        self.assertTrue(leg_node.is_root())
        self.assertTrue(leg_node.is_leaf())
        self.assertEqual(list(range(len(tensor.shape))), leg_node.leg_permutation)

        # Test leaf
        tensor = ptn.crandn((2, 3, 4))
        node = ptn.Node()
        parent_id = "parent_id"
        node.add_parent(parent_id)
        leg_node = ptn.Node.from_node(tensor, node)

        self.assertEqual(leg_node.parent, parent_id)
        self.assertTrue(leg_node.is_leaf())
        self.assertEqual(list(range(len(tensor.shape))), leg_node.leg_permutation)

        # Test 1 parent 1 child
        tensor = ptn.crandn((2, 3, 4, 5))
        node = ptn.GraphNode()
        parent_id = "parent_id"
        child_id = "child_id"
        node.add_parent(parent_id)
        node.add_child(child_id)
        leg_node = ptn.Node.from_node(tensor, node)

        self.assertEqual(leg_node.parent, parent_id)
        self.assertEqual(leg_node.children, [child_id])
        self.assertEqual(list(range(len(tensor.shape))), leg_node.leg_permutation)

        # Test 1 parent 2 children
        tensor = ptn.crandn((2, 3, 4, 5))
        node = ptn.GraphNode()
        parent_id = "parent_id"
        children_ids = ["child_id", "child2"]
        node.add_parent(parent_id)
        node.add_children(children_ids)
        leg_node = ptn.Node.from_node(tensor, node)

        self.assertEqual(leg_node.parent, parent_id)
        self.assertEqual(leg_node.children, children_ids)
        self.assertEqual(list(range(len(tensor.shape))), leg_node.leg_permutation)

        # Test root 1 child
        tensor = ptn.crandn((2, 3, 4, 5))
        node = ptn.GraphNode()
        child_id = "child2"
        node.add_child(child_id)
        leg_node = ptn.Node.from_node(tensor, node)

        self.assertTrue(leg_node.is_root())
        self.assertTrue(leg_node.is_parent_of(child_id))
        self.assertEqual(list(range(len(tensor.shape))), leg_node.leg_permutation)

        # Test root 2 children
        tensor = ptn.crandn((2, 3, 4, 5))
        node = ptn.GraphNode()
        children_ids = ["child_id", "child2"]
        node.add_children(children_ids)
        leg_node = ptn.Node.from_node(tensor, node)

        self.assertTrue(leg_node.is_root())
        for child_id in children_ids:
            self.assertTrue(leg_node.is_parent_of(child_id))
        self.assertEqual(list(range(len(tensor.shape))), leg_node.leg_permutation)


class TestNodeMethods(unittest.TestCase):

    def setUp(self):
        # All are tested with no and two open legs
        self.nodes = {}
        self.ids = ["empty0", "empty2", "leaf0", "leaf2", "node1p1c0", "node1p1c2",
                    "node1p2c0", "node1p2c2", "root1c0", "root1c2", "root2c0", "root2c2"]

        # Empty
        tensor0 = ptn.crandn(())
        tensor2 = ptn.crandn((2, 2))
        node = ptn.GraphNode()
        self.nodes["empty0"] = ptn.Node.from_node(tensor0, node)
        self.nodes["empty2"] = ptn.Node.from_node(tensor2, node)

        # Leaf
        tensor0 = ptn.crandn((2,))
        tensor2 = ptn.crandn((2, 3, 4))
        node = ptn.GraphNode()
        parent_id = "parent_id"
        node.add_parent(parent_id)
        self.nodes["leaf0"] = ptn.Node.from_node(tensor0, node)
        self.nodes["leaf2"] = ptn.Node.from_node(tensor2, node)

        # 1 parent 1 child
        tensor0 = ptn.crandn((2, 3))
        tensor2 = ptn.crandn((2, 3, 4, 5))
        node = ptn.GraphNode()
        parent_id = "parent_id"
        node.add_parent(parent_id)
        child_id = "child_id"
        node.add_child(child_id)
        self.nodes["node1p1c0"] = ptn.Node.from_node(tensor0, node)
        self.nodes["node1p1c2"] = ptn.Node.from_node(tensor2, node)

        # 1 parent 2 children
        tensor0 = ptn.crandn((2, 3, 4))
        tensor2 = ptn.crandn((2, 3, 4, 5, 6))
        node = ptn.GraphNode()
        parent_id = "parent_id"
        node.add_parent(parent_id)
        children_ids = ["child_id", "child2"]
        node.add_children(children_ids)
        self.nodes["node1p2c0"] = ptn.Node.from_node(tensor0, node)
        self.nodes["node1p2c2"] = ptn.Node.from_node(tensor2, node)

        # Root 1 child
        tensor0 = ptn.crandn((2))
        tensor2 = ptn.crandn((2, 3, 4))
        node = ptn.GraphNode()
        child_id = "child_id"
        node.add_child(child_id)
        self.nodes["root1c0"] = ptn.Node.from_node(tensor0, node)
        self.nodes["root1c2"] = ptn.Node.from_node(tensor2, node)

        # Root 2 children
        tensor0 = ptn.crandn((2, 3))
        tensor2 = ptn.crandn((2, 3, 4, 5))
        node = ptn.GraphNode()
        children_ids = ["child_id", "child2"]
        node.add_children(children_ids)
        self.nodes["root2c0"] = ptn.Node.from_node(tensor0, node)
        self.nodes["root2c2"] = ptn.Node.from_node(tensor2, node)

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

    def test_open_leg_to_parent_1st_open_leg(self):
        # Haven an open leg and no parent
        id_list = ["empty2", "root1c2", "root2c2"]
        open_leg_value = [0, 1, 2]
        open_leg_value = dict(zip(id_list, open_leg_value))

        for ids in id_list:
            node = self.nodes[ids]
            node.open_leg_to_parent("parent_id", open_leg_value[ids])
            self.assertEqual(open_leg_value[ids], node.leg_permutation[0])
            self.assertEqual("parent_id", node.parent)

        # Have no parent but no open leg
        id_list = ["empty0", "root1c0", "root2c0"]
        for ids in id_list:
            self.assertRaises(ValueError, self.nodes[ids].open_leg_to_parent, "Any", 0)

    def test_open_leg_to_parent_2nd_open_leg(self):
        # Haven an open leg and no parent
        id_list = ["empty2", "root1c2", "root2c2"]
        open_leg_value = [1, 2, 3]
        open_leg_value = dict(zip(id_list, open_leg_value))

        for ids in id_list:
            node = self.nodes[ids]
            node.open_leg_to_parent("parent_id", open_leg_value[ids])
            self.assertEqual(open_leg_value[ids], node.leg_permutation[0])
            self.assertEqual("parent_id", node.parent)

    def test_open_leg_to_child_1st_open_leg(self):
        # Haven an open leg
        id_list = ["empty2", "leaf2", "node1p1c2", "node1p2c2", "root1c2", "root2c2"]
        open_leg_value = [0, 1, 2, 3, 1, 2]
        open_leg_value = dict(zip(id_list, open_leg_value))

        for ids in id_list:
            node = self.nodes[ids]
            node.open_leg_to_child("new_child_id", open_leg_value[ids])
            # 1st open leg stays in the same position
            self.assertEqual(open_leg_value[ids], node.leg_permutation[open_leg_value[ids]])
            self.assertTrue("new_child_id" in node.children)

    def test_open_leg_to_child_2nd_open_leg(self):
        # Haven an open leg
        id_list = ["empty2", "leaf2", "node1p1c2", "node1p2c2", "root1c2", "root2c2"]
        open_leg_value = [1, 2, 3, 4, 2, 3]
        open_leg_value = dict(zip(id_list, open_leg_value))
        new_position = [0, 1, 2, 3, 1, 2]
        new_position = dict(zip(id_list, new_position))

        for ids in id_list:
            node = self.nodes[ids]
            node.open_leg_to_child("new_child_id", open_leg_value[ids])
            self.assertEqual(open_leg_value[ids], node.leg_permutation[new_position[ids]])
            self.assertTrue("new_child_id" in node.children)

    def test_parent_leg_to_open_leg(self):
        # Have a parent leg
        id_list = ["leaf0", "leaf2", "node1p1c0", "node1p1c2", "node1p2c0", "node1p2c2"]
        new_position = [0, 2, 1, 3, 2, 4]
        new_position = dict(zip(id_list, new_position))

        for ids in id_list:
            node = self.nodes[ids]
            node.parent_leg_to_open_leg()
            self.assertEqual(0, node.leg_permutation[new_position[ids]])

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

    def test_child_leg_to_open_leg_1st_child(self):
        # Have a child
        id_list = ["node1p1c0", "node1p1c2", "node1p2c0", "node1p2c2",
                   "root1c0", "root1c2", "root2c0", "root2c2"]
        leg_value = {ids: self.nodes[ids].get_child_leg("child_id")
                     for ids in id_list}

        for ids in id_list:
            node = self.nodes[ids]
            node.child_leg_to_open_leg("child_id")
            self.assertEqual(leg_value[ids], node.leg_permutation[-1])

    def test_child_leg_to_open_leg_2nd_child(self):
        # Have two children
        id_list = ["node1p2c0", "node1p2c2", "root2c0", "root2c2"]
        leg_value = {ids: self.nodes[ids].get_child_leg("child2")
                     for ids in id_list}

        for ids in id_list:
            node = self.nodes[ids]
            node.child_leg_to_open_leg("child2")
            self.assertEqual(leg_value[ids], node.leg_permutation[-1])

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


if __name__ == "__main__":
    unittest.main()
