from unittest import TestCase, main

from pytreenet.core.node import relative_leg_permutation
from pytreenet.random.random_node import random_tensor_node

class Test_find_child_permutation_neighbour_index(TestCase):

    def test_on_leaf(self):
        """
        Tests function on a leaf with only a prent leg.
        """
        old_shape = (2, )
        old_node, _ = random_tensor_node(old_shape, "old")
        old_node.open_leg_to_parent("parent", 0)
        new_shape = old_shape
        new_node, _ = random_tensor_node(new_shape, "new")
        new_node.open_leg_to_parent("parent", 0)
        found = relative_leg_permutation(old_node, new_node)
        correct = [0]
        self.assertEqual(found, correct)

    def test_on_leaf_phys_dim(self):
        """
        Tests function on a leaf with a parent and a physical leg.
        """
        old_shape = (2,3)
        old_node, _ = random_tensor_node(old_shape, "old")
        old_node.open_leg_to_parent("parent", 0)
        new_shape = old_shape
        new_node, _ = random_tensor_node(new_shape, "new")
        new_node.open_leg_to_parent("parent", 0)
        found = relative_leg_permutation(old_node, new_node)
        correct = [0, 1]
        self.assertEqual(found, correct)

    def test_on_root(self):
        """
        Tests the function on a root node with two children.
        """
        old_shape = (2, 3)
        old_node, _ = random_tensor_node(old_shape, "old")
        old_node.open_legs_to_children({"c1":0, "c2":1})
        new_shape = old_shape
        new_node, _ = random_tensor_node(new_shape, "new")
        new_node.open_legs_to_children({"c1":0, "c2":1})
        found = relative_leg_permutation(old_node, new_node)
        correct = [0, 1]
        self.assertEqual(found, correct)

    def test_on_root_phys_dim(self):
        """
        Tests the function on a root node with two children and a physical leg.
        """
        old_shape = (2, 3, 4)
        old_node, _ = random_tensor_node(old_shape, "old")
        old_node.open_legs_to_children({"c1":0, "c2":1})
        new_shape = old_shape
        new_node, _ = random_tensor_node(new_shape, "new")
        new_node.open_legs_to_children({"c1":0, "c2":1})
        found = relative_leg_permutation(old_node, new_node)
        correct = [0, 1, 2]
        self.assertEqual(found, correct)

    def test_on_root_phys_dim_permuted(self):
        """
        Test the function on a root node with two children and a physical leg,
        where the children are permuted between the two node.
        """
        old_shape = (2, 3, 4)
        old_node, _ = random_tensor_node(old_shape, "old")
        old_node.open_legs_to_children({"c1":0, "c2":1})
        new_shape = (3,2,4)
        new_node, _ = random_tensor_node(new_shape, "new")
        new_node.open_legs_to_children({"c2":0, "c1":1})
        found = relative_leg_permutation(old_node, new_node)
        correct = [1,0,2]
        self.assertEqual(found, correct)

    def test_on_root_phys_dim_permuted_test(self):
        """
        Test the function on a root node with two children and a physical leg,
        where the children are permuted between the two node.
        """
        old_shape = (2, 3, 4)
        old_node, _ = random_tensor_node(old_shape, "old")
        old_node.open_legs_to_children({"c1":0, "c2":1})
        new_shape = (2,3,4)
        new_node, _ = random_tensor_node(new_shape, "new")
        # Note that only the order of addition of the leg is relevant.
        new_node.open_legs_to_children({"c2":1, "c1":0})
        found = relative_leg_permutation(old_node, new_node)
        correct = [1,0,2]
        self.assertEqual(found, correct)

    def test_on_node(self):
        """
        Tests the function on a node with a parent and three children.
        """
        old_shape = (2, 3, 4, 5)
        old_node, _ = random_tensor_node(old_shape, "old")
        old_node.open_leg_to_parent("parent", 0)
        old_node.open_legs_to_children({"c1":1, "c2":2, "c3":3})
        new_shape = old_shape
        new_node, _ = random_tensor_node(new_shape, "new")
        new_node.open_leg_to_parent("parent", 0)
        new_node.open_legs_to_children({"c1":1, "c2":2, "c3":3})
        found = relative_leg_permutation(old_node, new_node)
        correct = [0, 1, 2, 3]
        self.assertEqual(found, correct)

    def test_on_node_phys_dim(self):
        """
        Tests the function on a node with a parent and three children and two
        physical legs.
        """
        old_shape = (2, 3, 4, 5, 6, 7)
        old_node, _ = random_tensor_node(old_shape, "old")
        old_node.open_leg_to_parent("parent", 0)
        old_node.open_legs_to_children({"c1":1, "c2":2, "c3":3})
        new_shape = old_shape
        new_node, _ = random_tensor_node(new_shape, "new")
        new_node.open_leg_to_parent("parent", 0)
        new_node.open_legs_to_children({"c1":1, "c2":2, "c3":3})
        found = relative_leg_permutation(old_node, new_node)
        correct = [0, 1, 2, 3, 4, 5]
        self.assertEqual(found, correct)

    def test_on_node_phys_dim_permuted(self):
        """
        Tests the function on a node with a parent and three children and two
        physical legs, where the children are permuted between the two nodes.
        """
        old_shape = (2, 3, 4, 5, 6, 7)
        old_node, _ = random_tensor_node(old_shape, "old")
        old_node.open_leg_to_parent("parent", 0)
        old_node.open_legs_to_children({"c1":1, "c2":2, "c3":3})
        new_shape = (2, 5, 3, 4, 6, 7)
        new_node, _ = random_tensor_node(new_shape, "new")
        new_node.open_leg_to_parent("parent", 0)
        new_node.open_legs_to_children({"c3":1, "c1":2, "c2":3})
        found = relative_leg_permutation(old_node, new_node)
        correct = [0, 3, 1, 2, 4, 5]
        self.assertEqual(found, correct)

    def test_on_root_phys_dim_permuted_modify(self):
        """
        Tests function on a root, where the children are permuted between the two
        nodes and their identifiers are different. The nodes also have physical
        legs. 
        """
        old_shape = (2, 3, 4)
        old_node, _ = random_tensor_node(old_shape, "old")
        old_node.open_legs_to_children({"c1":0, "c2":1})
        new_shape = (3,2,4)
        new_node, _ = random_tensor_node(new_shape, "new")
        new_node.open_legs_to_children({"2":0, "1":1})
        mod_fct = lambda x: "c"+x
        found = relative_leg_permutation(old_node, new_node,
                                        modify_function=mod_fct)
        correct = [1,0,2]
        self.assertEqual(found, correct)

    def test_on_node_phys_dim_permuted_modify(self):
        """
        Tests the function on a node with a parent and three children and two
        physical legs, where the children are permuted between the two nodes.
        """
        old_shape = (2, 3, 4, 5, 6, 7)
        old_node, _ = random_tensor_node(old_shape, "old")
        old_node.open_leg_to_parent("parent", 0)
        old_node.open_legs_to_children({"c1":1, "c2":2, "c3":3})
        new_shape = (2, 5, 3, 4, 6, 7)
        new_node, _ = random_tensor_node(new_shape, "new")
        new_node.open_leg_to_parent("parent", 0)
        new_node.open_legs_to_children({"3":1, "1":2, "2":3})
        mod_fct = lambda x: "c"+x
        found = relative_leg_permutation(old_node, new_node,
                                         modify_function=mod_fct)
        correct = [0, 3, 1, 2, 4, 5]
        self.assertEqual(found, correct)

    def test_non_implemented_part(self):
        """
        The function is not implemented, if the parent might also be permuted.
        """
        old_shape = (2, 3, 4, 5, 6, 7)
        old_node, _ = random_tensor_node(old_shape, "old")
        old_node.open_leg_to_parent("parent", 0)
        old_node.open_legs_to_children({"c1":1, "c2":2, "c3":3})
        new_shape = (2, 5, 3, 4, 6, 7)
        new_node, _ = random_tensor_node(new_shape, "new")
        new_node.open_leg_to_parent("parent", 0)
        new_node.open_legs_to_children({"3":1, "1":2, "2":3})
        mod_fct = lambda x: "c"+x
        self.assertRaises(NotImplementedError, relative_leg_permutation,
                          old_node, new_node, modify_function=mod_fct,
                          modified_parent=True)

if __name__ == "__main__":
    main()
