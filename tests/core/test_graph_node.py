import unittest

from pytreenet.core.graph_node import (GraphNode,
                                       find_children_permutation,
                                       find_child_permutation_neighbour_index,
                                       determine_parentage)
from pytreenet.util.ttn_exceptions import NoConnectionException

class TestInitilisation(unittest.TestCase):

    def test_initialisation(self):

        # Without extra attributes
        empty_node = GraphNode()
        self.assertNotEqual("", empty_node.identifier)
        self.assertEqual(None, empty_node.parent)
        self.assertEqual([], empty_node.children)

        # With identifier
        ids = "id"
        node = GraphNode(identifier=ids)
        self.assertEqual(ids, node.identifier)

class TestGraphNode(unittest.TestCase):

    def setUp(self):
        self.parent_id = "parent"
        self.child1_id = "c1"
        self.child2_id = "c2"

        self.node = GraphNode(identifier="this")

    def test_add_parent(self):
        self.assertEqual(None, self.node.parent)
        self.node.add_parent(self.parent_id)
        self.assertEqual(self.parent_id, self.node.parent)

        # Test adding additional parent -> Exception
        self.assertRaises(AssertionError, self.node.add_parent, "some")

    def test_remove_parent(self):
        self.node.add_parent(self.parent_id)
        self.assertEqual(self.parent_id, self.node.parent)
        self.node.remove_parent()
        self.assertEqual(None, self.node.parent)

    def test_remove_non_existing_parent(self):
        self.node.remove_parent()
        self.assertEqual(None,self.node.parent)

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

    def test_remove_child_not_existing(self):
        self.assertRaises(ValueError, self.node.remove_child, "some")

    def test_remove_child_existing(self):
        # If children exist
        self.node.add_children([self.child1_id, self.child2_id])
        self.node.remove_child(self.child1_id)
        self.assertEqual([self.child2_id], self.node.children)
        self.node.remove_child(self.child2_id)
        self.assertEqual([], self.node.children)

    def test_child_index_not_existing(self):
        self.assertRaises(ValueError, self.node.child_index, "some")

    def test_child_index_existing(self):
        # If children exist
        self.node.add_children([self.child1_id, self.child2_id])
        self.assertEqual(0, self.node.child_index(self.child1_id))
        self.assertEqual(1, self.node.child_index(self.child2_id))

    def test_neighbour_index_not_existing(self):
        self.assertRaises(NoConnectionException, self.node.neighbour_index, "some")

    def test_neighbour_index_children_only(self):
        self.node.add_children([self.child1_id,self.child2_id])
        self.assertEqual(0,self.node.neighbour_index(self.child1_id))
        self.assertEqual(1,self.node.neighbour_index(self.child2_id))

    def test_neighbour_index_parent_only(self):
        self.node.add_parent(self.parent_id)
        self.assertEqual(0,self.node.neighbour_index(self.parent_id))

    def test_neighbour_index_parent_and_children(self):
        self.node.add_parent(self.parent_id)
        self.node.add_children([self.child1_id,self.child2_id])
        self.assertEqual(0,self.node.neighbour_index(self.parent_id))
        self.assertEqual(1,self.node.neighbour_index(self.child1_id))
        self.assertEqual(2,self.node.neighbour_index(self.child2_id))

    def test_replace_child_non_existent(self):
        self.node.add_children([self.child1_id, self.child2_id])
        self.assertRaises(ValueError,self.node.replace_child,
                          "false","other")

    def test_replace_child_same_identifier(self):
        self.node.add_children([self.child1_id, self.child2_id])
        # Same identifier -> Nothing happens
        self.node.replace_child(self.child1_id, self.child1_id)
        self.assertEqual([self.child1_id, self.child2_id], self.node.children)

    def test_replace_child(self):
        self.node.add_children([self.child1_id, self.child2_id])
        self.node.replace_child(self.child1_id, "other")
        self.assertEqual(["other", self.child2_id], self.node.children)
        self.node.replace_child(self.child2_id, "other2")
        self.assertEqual(["other", "other2"], self.node.children)

    def test_replace_neighbour_non_existing(self):
        self.node.add_parent(self.parent_id)
        self.node.add_children([self.child1_id,self.child2_id])
        self.assertRaises(NoConnectionException,self.node.replace_neighbour,
                          "false","other")

    def test_replace_neighbour_parent(self):
        self.node.add_parent(self.parent_id)
        self.node.replace_neighbour(self.parent_id,"new")
        self.assertEqual("new",self.node.parent)

    def test_replace_neighbour_parent_with_children(self):
        self.node.add_parent(self.parent_id)
        self.node.add_children([self.child1_id,self.child2_id])
        self.node.replace_neighbour(self.parent_id,"new")
        self.assertEqual("new",self.node.parent)

    def test_replace_neighbour_child(self):
        self.node.add_children([self.child1_id,self.child2_id])
        self.node.replace_neighbour(self.child1_id,"new")
        self.assertEqual(["new",self.child2_id],self.node.children)
        self.node.replace_neighbour(self.child2_id,"new2")
        self.assertEqual(["new","new2"],self.node.children)

    def test_replace_neigbour_children_with_parent(self):
        self.node.add_parent(self.parent_id)
        self.node.add_children([self.child1_id,self.child2_id])
        self.node.replace_neighbour(self.child1_id,"new")
        self.assertEqual(["new",self.child2_id],self.node.children)
        self.node.replace_neighbour(self.child2_id,"new2")
        self.assertEqual(["new","new2"],self.node.children)

    def test_is_root_true(self):
        # It is
        self.assertTrue(self.node.is_root())

    def test_is_root_false(self):
        # Isn't
        self.node.add_parent("1338")
        self.assertFalse(self.node.is_root())

    def test_is_leaf_true(self):
        # It is
        self.assertTrue(self.node.is_leaf())

    def test_is_leaf_false(self):
        # Isn't
        self.node.add_child("ABCDEFG")
        self.assertFalse(self.node.is_leaf())

    def test_is_child_of_root(self):
        # Not if there is no parent
        self.assertFalse(self.node.is_child_of("hi"))

    def test_is_child_of_false(self):
        # Wrong node
        self.node.add_parent("shark")
        self.assertFalse(self.node.is_child_of("hi"))

    def test_is_child_of_true(self):
        # It is
        self.node.add_parent("shark")
        self.assertTrue(self.node.is_child_of("shark"))

    def test_is_parent_of_leaf(self):
        # Not without children
        self.assertFalse(self.node.is_parent_of("fish"))

    def test_is_parent_of_false(self):
        # Wrong child
        self.node.add_children([self.child1_id, self.child2_id])
        self.assertFalse(self.node.is_parent_of("More Fish!"))

    def test_is_parent_of_true(self):
        # It is
        self.node.add_children([self.child1_id, self.child2_id])
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
        self.assertFalse(GraphNode(identifier="Not this") == self.node)
        other_node = GraphNode(identifier="this")
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

    def test_eq_root_chenanigans(self):
        """
        Tests the special cases, where roots are involved.
        """
        # Different roots
        other_node = GraphNode(identifier="other")
        self.assertFalse(other_node == self.node)

        # Same root
        other_node = GraphNode(identifier="this")
        self.assertTrue(other_node == self.node)

        # One root, but not the other
        other_node = GraphNode(identifier="this")
        other_node.add_parent("parent")
        self.assertFalse(other_node == self.node)
        self.assertFalse(self.node == other_node)

        # With children
        other_node = GraphNode(identifier="this")
        self.node.add_children([self.child1_id, self.child2_id])
        self.assertFalse(other_node == self.node)
        other_node.add_children([self.child1_id, self.child2_id])
        self.assertTrue(other_node == self.node)

    def test_neighbour_id_non_root(self):
        """
        Test obtaining the correct neighbour id given an index.
        The node is not a root.
        """
        self.node.add_parent(self.parent_id)
        self.node.add_child(self.child1_id)
        self.node.add_child(self.child2_id)
        self.assertEqual(self.parent_id,self.node.neighbour_id(0))
        self.assertEqual(self.child1_id,self.node.neighbour_id(1))
        self.assertEqual(self.child2_id,self.node.neighbour_id(2))

    def test_neighbour_id_root(self):
        """
        Test obtaining the correct neighbour id given an index.
        The node is a root.
        """
        self.node.add_child(self.child1_id)
        self.node.add_child(self.child2_id)
        self.assertEqual(self.child1_id,self.node.neighbour_id(0))
        self.assertEqual(self.child2_id,self.node.neighbour_id(1))

    def test_neighbour_id_invalid_index(self):
        """
        Test obtaining the correct neighbour id given an invalid index.
        """
        self.node.add_child(self.child1_id)
        self.node.add_child(self.child2_id)
        self.assertRaises(IndexError,self.node.neighbour_id,-1)
        self.assertRaises(IndexError,self.node.neighbour_id,9)

class TestNodeFunctions(unittest.TestCase):

    def test_find_children_permutation_trivial(self):
        old_node = GraphNode()
        old_node.add_children(["c1", "c2", "c3"])
        new_node = GraphNode()
        new_node.add_children(["c1", "c2", "c3"])
        perm = find_children_permutation(old_node, new_node)
        self.assertEqual([0, 1, 2], perm)

    def test_find_children_permutation_elementary_permutation(self):
        old_node = GraphNode()
        old_node.add_children(["c1", "c2", "c3"])
        new_node = GraphNode()
        new_node.add_children(["c1", "c3", "c2"])
        perm = find_children_permutation(old_node, new_node)
        self.assertEqual([0, 2, 1], perm)

    def test_find_children_permutation_complex(self):
        old_node = GraphNode()
        old_node.add_children(["c1", "c2", "c3"])
        new_node = GraphNode()
        new_node.add_children(["c2", "c3", "c1"])
        perm = find_children_permutation(old_node, new_node)
        self.assertEqual([1,2,0], perm)

    def test_find_children_permutation_trivial_modify(self):
        old_node = GraphNode()
        old_node.add_children(["c1", "c2", "c3"])
        modifier = lambda x: "c" + x
        new_node = GraphNode()
        new_node.add_children(["1", "2", "3"])
        perm = find_children_permutation(old_node, new_node,
                                         modify_function=modifier)
        self.assertEqual([0, 1, 2], perm)

    def test_find_children_permutation_elementary_permutation_modify(self):
        old_node = GraphNode()
        old_node.add_children(["c1", "c2", "c3"])
        modifier = lambda x: "c" + x
        new_node = GraphNode()
        new_node.add_children(["1", "3", "2"])
        perm = find_children_permutation(old_node, new_node,
                                         modify_function=modifier)
        self.assertEqual([0, 2, 1], perm)

    def test_find_children_permutation_complex_modify(self):
        old_node = GraphNode()
        old_node.add_children(["c1", "c2", "c3"])
        modifier = lambda x: "c" + x
        new_node = GraphNode()
        new_node.add_children(["2", "3", "1"])
        perm = find_children_permutation(old_node, new_node,
                                         modify_function=modifier)
        self.assertEqual([1,2,0], perm)

    def test_find_children_permutation_complex_parent(self):
        old_node = GraphNode()
        old_node.add_parent("parent")
        old_node.add_children(["c1", "c2", "c3"])
        new_node = GraphNode()
        new_node.add_parent("parent")
        new_node.add_children(["c2", "c3", "c1"])
        perm = find_children_permutation(old_node, new_node)
        self.assertEqual([1,2,0], perm)

    def test_find_children_permutation_complex_parent_modify(self):
        old_node = GraphNode()
        old_node.add_parent("parent")
        old_node.add_children(["c1", "c2", "c3"])
        modifier = lambda x: "c" + x
        new_node = GraphNode()
        new_node.add_parent("parent")
        new_node.add_children(["2", "3", "1"])
        perm = find_children_permutation(old_node, new_node,
                                         modify_function=modifier)
        self.assertEqual([1,2,0], perm)

    def test_find_child_permutation_neighbour_index_complex_modify(self):
        old_node = GraphNode()
        old_node.add_children(["c1", "c2", "c3"])
        modifier = lambda x: "c" + x
        new_node = GraphNode()
        new_node.add_children(["2", "3", "1"])
        perm = find_child_permutation_neighbour_index(old_node, new_node,
                                         modify_function=modifier)
        self.assertEqual([1,2,0], perm)

    def test_find_child_permutation_neighbour_index_complex_parent(self):
        old_node = GraphNode()
        old_node.add_parent("parent")
        old_node.add_children(["c1", "c2", "c3"])
        new_node = GraphNode()
        new_node.add_parent("parent")
        new_node.add_children(["c2", "c3", "c1"])
        perm = find_child_permutation_neighbour_index(old_node, new_node)
        self.assertEqual([2,3,1], perm)

    def test_find_child_permutation_neighbour_index_complex_parent_modify(self):
        old_node = GraphNode()
        old_node.add_parent("parent")
        old_node.add_children(["c1", "c2", "c3"])
        modifier = lambda x: "c" + x
        new_node = GraphNode()
        new_node.add_parent("parent")
        new_node.add_children(["2", "3", "1"])
        perm = find_child_permutation_neighbour_index(old_node, new_node,
                                                     modify_function=modifier)
        self.assertEqual([2,3,1], perm)

    def test_determine_parentage_parent1(self):
        """
        Tests the parentage determination when the first node is the parent.
        """
        node_1 = GraphNode(identifier="parent")
        node_2 = GraphNode(identifier="child")
        node_1.add_child(node_2.identifier)
        node_2.add_parent(node_1.identifier)
        parentage = determine_parentage(node_1, node_2)
        self.assertEqual("parent", parentage[0].identifier)
        self.assertEqual("child", parentage[1].identifier)

    def test_determine_parentage_parent2(self):
        """
        Tests the parentage determination when the second node is the parent.
        """
        node_1 = GraphNode(identifier="child")
        node_2 = GraphNode(identifier="parent")
        node_1.add_parent(node_2.identifier)
        node_2.add_child(node_1.identifier)
        parentage = determine_parentage(node_1, node_2)
        self.assertEqual("parent", parentage[0].identifier)
        self.assertEqual("child", parentage[1].identifier)

    def test_determine_parentage_no_nghbs(self):
        """
        Tests the parentage determination when there are not neighbours.
        """
        node_1 = GraphNode(identifier="node1")
        node_2 = GraphNode(identifier="node2")
        self.assertRaises(NoConnectionException, determine_parentage,
                          node_1, node_2)


if __name__ == "__main__":
    unittest.main()
