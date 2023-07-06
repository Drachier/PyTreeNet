import unittest

import pytreenet as ptn


class TestInitilisation(unittest.TestCase):

    def test_initialisation(self):

        # Without extra attributes
        empty_node = ptn.GraphNode()
        self.assertNotEqual("", empty_node.identifier)
        self.assertNotEqual("", empty_node.tag)
        self.assertEqual(None, empty_node.parent)
        self.assertEqual([], empty_node.children)

        # With identifier and tag
        id = "id"
        tag = "tag"
        node = ptn.GraphNode(tag=tag, identifier=id)
        self.assertEqual(id, node.identifier)
        self.assertEqual(tag, node.tag)


class TestGraphNode(unittest.TestCase):

    def setUp(self):
        self.parent_id = "parent"
        self.child1_id = "c1"
        self.child2_id = "c2"

        self.node = ptn.Node(identifier="this")

    def test_add_parent(self):

        self.assertEqual(None, self.node.parent)
        self.node.add_parent(self.parent_id)
        self.assertEqual(self.parent_id, self.node.parent)

        # Test adding additional parent -> Exception
        self.assertRaises(ValueError, self.node.add_parent, "some")

    def test_remove_parent(self):

        self.node.add_parent(self.parent_id)
        self.assertEqual(self.parent_id, self.node.parent)
        self.node.remove_parent()
        self.assertEqual(None, self.node.parent)

    def test_add_child(self):

        self.assertEqual([], self.node.children)

        # First child
        self.node.add_child(self.child1_id)
        self.assertEqual([self.child1_id], self.node.children)

        # Second child
        self.node.add_child(self.child2_id)
        self.assertEqual([self.child1_id, self.child2_id], self.node.children)


if __name__ == "__main__":
    unittest.main()
