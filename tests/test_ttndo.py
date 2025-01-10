from unittest import TestCase, main

from pytreenet.ttns.ttndo import SymmetricTTNDO
from pytreenet.random import crandn

class TestTTNDOInit(TestCase):
    """
    Test the initialisation of the SymmetricTTNDO class.
    """

    def test_init_no_keywords(self):
        """
        Test the initialisation of the SymmetricTTNDO class
        without keywords for the suffixes.
        """
        ttns = SymmetricTTNDO()
        self.assertEqual(ttns.bra_suffix, "_bra")
        self.assertEqual(ttns.ket_suffix, "_ket")

    def test_init_with_keywords(self):
        """
        Test the initialisation of the SymmetricTTNDO class
        with keywords for the suffixes.
        """
        ttns = SymmetricTTNDO(bra_suffix="_bra1", ket_suffix="_ket1")
        self.assertEqual(ttns.bra_suffix, "_bra1")
        self.assertEqual(ttns.ket_suffix, "_ket1")

class TestTTNDOStringMethods(TestCase):
    """
    Test the methods that deal with the identifier strings in some way.
    """

    def setUp(self):
        self.ttndo = SymmetricTTNDO()

    def test_ket_id(self):
        """
        Test the ket_id method.
        """
        self.assertEqual(self.ttndo.ket_id("node"), "node_ket")

    def test_bra_id(self):
        """
        Test the bra_id method.
        """
        self.assertEqual(self.ttndo.bra_id("node"), "node_bra")

    def test_reverse_ket_id(self):
        """
        Test the reverse_ket_id method.
        """
        self.assertEqual(self.ttndo.reverse_ket_id("node_ket"), "node")

    def test_reverse_bra_id(self):
        """
        Test the reverse_bra_id method.
        """
        self.assertEqual(self.ttndo.reverse_bra_id("node_bra"), "node")

    def test_ket_to_bra_id(self):
        """
        Test the ket_to_bra_id method.
        """
        self.assertEqual(self.ttndo.ket_to_bra_id("node_ket"), "node_bra")

    def test_bra_to_ket_id(self):
        """
        Test the bra_to_ket_id method.
        """
        self.assertEqual(self.ttndo.bra_to_ket_id("node_bra"), "node_ket")

class TestTTNDORootConstructionMethods(TestCase):
    """
    Test the methods that build the nodes of the TTNDO.
    """

    def setUp(self):
        self.ttndo = SymmetricTTNDO()

    def test_add_trivial_root_default(self):
        """
        Test the add_trivial_root method.
        """
        root_id = "root"
        self.ttndo.add_trivial_root(root_id)
        self.assertEqual(self.ttndo.root_id, root_id)
        self.assertEqual(self.ttndo.tensors[root_id].shape, (2,2,1))

    def test_add_trivial_root_custom(self):
        """
        Test the add_trivial_root method with a custom dimension.
        """
        root_id = "root"
        self.ttndo.add_trivial_root(root_id, dimension=3)
        self.assertEqual(self.ttndo.root_id, root_id)
        self.assertEqual(self.ttndo.tensors[root_id].shape, (3,3,1))

    def test_add_trivial_root_error(self):
        """
        Test the add_trivial_root method with an invalid dimension.
        """
        self.assertRaises(ValueError, self.ttndo.add_trivial_root, "root", dimension=0)

class TestTTNDOChildrenAddingMethod(TestCase):
    """
    Test the methods that add children to the nodes of the TTNDO.
    """

    def setUp(self):
        self.ttndo = SymmetricTTNDO()
        self.ttndo.add_trivial_root("root")

    def test_add_symmetric_children_to_root(self):
        """
        Test the the adding of children to the root.
        """
        child_id = "child"
        shape = (2,3,4)
        ket_tensor = crandn(shape)
        bra_tensor = crandn(shape)
        self.ttndo.add_symmetric_children_to_parent(child_id, ket_tensor,
                                                    bra_tensor, 0,
                                                    self.ttndo.root_id, 0, 1)
        ket_id = self.ttndo.ket_id(child_id)
        bra_id = self.ttndo.bra_id(child_id)
        # Nodes and Tensors are added
        self.assertIn(ket_id, self.ttndo.tensors)
        self.assertIn(bra_id, self.ttndo.tensors)
        self.assertIn(ket_id, self.ttndo.nodes)
        self.assertIn(bra_id, self.ttndo.nodes)
        # Check parentage
        self.assertTrue(self.ttndo.is_child_of(ket_id, self.ttndo.root_id))
        self.assertTrue(self.ttndo.is_child_of(bra_id, self.ttndo.root_id))

    def test_add_symmetric_children_to_non_root(self):
        """
        Test the adding of children to a non-root node.
        """
        # Prepare
        parent_id = "parent"
        parent_shape = (2,3,4)
        parent_ket_tensor = crandn(parent_shape)
        parent_bra_tensor = crandn(parent_shape)
        self.ttndo.add_symmetric_children_to_parent(parent_id, parent_ket_tensor,
                                                    parent_bra_tensor, 0,
                                                    self.ttndo.root_id, 0, 1)
        ket_parent_id = self.ttndo.ket_id(parent_id)
        bra_parent_id = self.ttndo.bra_id(parent_id)
        # Test
        child_id = "child"
        shape = (3,4,5)
        ket_tensor = crandn(shape)
        bra_tensor = crandn(shape)
        self.ttndo.add_symmetric_children_to_parent(child_id, ket_tensor,
                                                    bra_tensor, 0,
                                                    parent_id, 1)
        ket_id = self.ttndo.ket_id(child_id)
        bra_id = self.ttndo.bra_id(child_id)
        # Nodes and Tensors are added
        self.assertIn(ket_id, self.ttndo.tensors)
        self.assertIn(bra_id, self.ttndo.tensors)
        self.assertIn(ket_id, self.ttndo.nodes)
        self.assertIn(bra_id, self.ttndo.nodes)
        # Check parentage
        self.assertTrue(self.ttndo.is_child_of(ket_id, ket_parent_id))
        self.assertTrue(self.ttndo.is_child_of(bra_id, bra_parent_id))

    def test_wrong_shapes(self):
        """
        Test the adding of children with wrong shapes raises an error.
        """
        child_id = "child"
        shape = (2,3,4)
        shape2 = (2,3,5)
        ket_tensor = crandn(shape)
        bra_tensor = crandn(shape2)
        self.assertRaises(AssertionError, self.ttndo.add_symmetric_children_to_parent,
                          child_id, ket_tensor, bra_tensor, 0,
                          self.ttndo.root_id, 0, 1)

    def test_bra_leg_for_non_root(self):
        # Prepare
        parent_id = "parent"
        parent_shape = (2,3,4)
        parent_ket_tensor = crandn(parent_shape)
        parent_bra_tensor = crandn(parent_shape)
        self.ttndo.add_symmetric_children_to_parent(parent_id, parent_ket_tensor,
                                                    parent_bra_tensor, 0,
                                                    self.ttndo.root_id, 0, 1)
        # Test
        child_id = "child"
        shape = (3,4,5)
        ket_tensor = crandn(shape)
        bra_tensor = crandn(shape)
        self.assertRaises(ValueError, self.ttndo.add_symmetric_children_to_parent,
                          child_id, ket_tensor, bra_tensor, 0,
                          parent_id, 1, 2)

if __name__ == '__main__':
    main()
