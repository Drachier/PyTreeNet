import unittest

import pytreenet as ptn

class TestInitilisation(unittest.TestCase):

    def test_initialisation(self):

        # Without extra attributes
        empty_node = ptn.GraphNode()
        self.assertNotEqual("", empty_node.identifier)
        self.assertEqual(None, empty_node.parent)
        self.assertEqual([], empty_node.children)

        # With identifier
        ids = "id"
        node = ptn.GraphNode(identifier=ids)
        self.assertEqual(ids, node.identifier)

class TestGraphNode(unittest.TestCase):

    def setUp(self):
        self.parent_id = "parent"
        self.child1_id = "c1"
        self.child2_id = "c2"

        self.node = ptn.GraphNode(identifier="this")

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

    def test_add_children(self):
        self.assertEqual([], self.node.children)
        children = [self.child1_id, self.child2_id]
        self.node.add_children(children)
        self.assertEqual(children, self.node.children)

        # Add another child
        self.node.add_children(["third_child"])
        children.append("third_child")
        self.assertEqual(children, self.node.children)

    def test_remove_child(self):
        self.assertRaises(ValueError, self.node.remove_child, "some")

        # If children exist
        self.node.add_children([self.child1_id, self.child2_id])
        self.node.remove_child(self.child1_id)
        self.assertEqual([self.child2_id], self.node.children)
        self.node.remove_child(self.child2_id)
        self.assertEqual([], self.node.children)

    def test_child_index(self):
        self.assertRaises(ValueError, self.node.child_index, "some")

        # If children exist
        self.node.add_children([self.child1_id, self.child2_id])
        self.assertEqual(0, self.node.child_index(self.child1_id))
        self.assertEqual(1, self.node.child_index(self.child2_id))

    def test_replace_child(self):
        self.node.add_children([self.child1_id, self.child2_id])

        # Same identifier -> Nothing happens
        self.node.replace_child(self.child1_id, self.child1_id)
        self.assertEqual([self.child1_id, self.child2_id], self.node.children)
        self.node.replace_child(self.child1_id, "other")
        self.assertEqual(["other", self.child2_id], self.node.children)
        self.node.replace_child(self.child2_id, "other2")
        self.assertEqual(["other", "other2"], self.node.children)

    def test_is_root(self):
        # It is
        self.assertTrue(self.node.is_root())

        # Isn't
        self.node.add_parent("1338")
        self.assertFalse(self.node.is_root())

    def test_is_leaf(self):
        # It is
        self.assertTrue(self.node.is_leaf())

        # Isn't
        self.node.add_child("ABCDEFG")
        self.assertFalse(self.node.is_leaf())

    def test_is_child_of(self):
        # Not if there is no parent
        self.assertFalse(self.node.is_child_of("hi"))

        # Wrong node
        self.node.add_parent("shark")
        self.assertFalse(self.node.is_child_of("hi"))

        # It is
        self.assertTrue(self.node.is_child_of("shark"))

    def test_is_parent_of(self):
        # Not without children
        self.assertFalse(self.node.is_parent_of("fish"))

        # Wrong child
        self.node.add_children([self.child1_id, self.child2_id])
        self.assertFalse(self.node.is_parent_of("More Fish!"))

        # It is
        self.assertTrue(self.node.is_parent_of(self.child1_id))
        self.assertTrue(self.node.is_parent_of(self.child2_id))

    def test_nparents(self):
        # There are only two options
        self.assertEqual(0, self.node.nparents())
        self.node.add_parent("Hydrogen")
        self.assertEqual(1, self.node.nparents())

    def test_nchildren(self):
        # None
        self.assertEqual(0, self.node.nchildren())

        # Two!
        self.node.add_children([self.child1_id, self.child2_id])
        self.assertEqual(2, self.node.nchildren())

    def test_nneighbours(self):
        # None
        self.assertEqual(0, self.node.nneighbours())

        # 1 parent
        self.node.add_parent("Daemons from hell!")
        self.assertEqual(1, self.node.nneighbours())

        # And two children
        self.node.add_children([self.child1_id, self.child2_id])
        self.assertEqual(3, self.node.nneighbours())

        # Only children
        self.node.remove_parent()
        self.assertEqual(2, self.node.nneighbours())

    def test_has_x_children(self):
        # None
        self.assertFalse(self.node.has_x_children(3))
        self.assertTrue(self.node.has_x_children(0))

        # Some (also known as 2)
        self.node.add_children([self.child1_id, self.child2_id])
        self.assertFalse(self.node.has_x_children(0))
        self.assertFalse(self.node.has_x_children(10928))
        self.assertTrue(self.node.has_x_children(2))

    def test_neighbouring_nodes(self):
        # None
        self.assertEqual([], self.node.neighbouring_nodes())

        # Parent
        self.node.add_parent("A String")
        self.assertEqual(["A String"], self.node.neighbouring_nodes())

        # And children
        self.node.add_children([self.child1_id, self.child2_id])
        self.assertEqual(["A String", self.child1_id, self.child2_id],
                         self.node.neighbouring_nodes())

        # Only children
        self.node.remove_parent()
        self.assertEqual([self.child1_id, self.child2_id],
                         self.node.neighbouring_nodes())

    def test_eq(self):
        # Empty Nodes
        self.assertFalse(ptn.GraphNode(identifier="Not this") == self.node)
        other_node = ptn.GraphNode(identifier="this")
        self.assertTrue(other_node, self.node)

        # With parent
        other_node.add_parent("parent")
        self.assertFalse(other_node == self.node)
        self.node.add_parent("Cat")
        self.assertFalse(other_node == self.node)
        other_node.remove_parent()
        other_node.add_parent("Cat")
        self.assertTrue(other_node == self.node)

        # With children
        self.node.add_children([self.child1_id, self.child2_id])
        self.assertFalse(other_node == self.node)
        other_node.add_child("Carneval!")
        self.assertFalse(other_node == self.node)
        other_node.remove_child("Carneval!")
        other_node.add_children([self.child1_id, self.child2_id])
        self.assertTrue(other_node == self.node)

if __name__ == "__main__":
    unittest.main()
