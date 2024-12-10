import unittest
from copy import copy, deepcopy

from numpy import zeros, allclose, eye, tensordot
from numpy.linalg import qr

from pytreenet.core.leg_specification import LegSpecification
from pytreenet.random.random_node import random_tensor_node
from pytreenet.random.random_ttns import (random_small_ttns,
                                          random_big_ttns_two_root_children,
                                          RandomTTNSMode)
from pytreenet.random.random_matrices import crandn
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict
from pytreenet.time_evolution.time_evo_util.bug_util import (basis_change_tensor_id,
                                                            reverse_basis_change_tensor_id,
                                                            new_basis_tensor_qr_legs,
                                                            concat_along_parent_leg,
                                                            _compute_new_basis_tensor_qr,
                                                            compute_new_basis_tensor,
                                                            compute_basis_change_tensor,
                                                            find_new_basis_replacement_leg_specs,)

class Test_identifier_functions(unittest.TestCase):

    def test_basis_change_tensor_id(self):
        """
        Test the basis change tensor identifier function.
        """
        node_id = "node"
        found = basis_change_tensor_id(node_id)
        correct = node_id + "_basis_change_tensor"
        self.assertEqual(found, correct)

    def test_reverse_basis_change_tensor_id(self):
        """
        Test the reverse basis change tensor identifier function.
        """
        node_id = "node"
        full_node_id = basis_change_tensor_id(node_id)
        found = reverse_basis_change_tensor_id(full_node_id)
        self.assertEqual(found, node_id)

class Test_concat_along_parent_leg(unittest.TestCase):

    def test_on_leaf(self):
        """
        Test the concatenation of a leaf tensor along the parent leg.
        """
        old_shape = (3,2)
        node, old_tensor = random_tensor_node(old_shape)
        node.add_parent("parent")
        new_tensor = crandn(old_tensor.shape)
        concat_tensor = concat_along_parent_leg(node, old_tensor, new_tensor)
        new_shape = (6,2)
        self.assertEqual(concat_tensor.shape, new_shape)
        # Build the expected tensor
        expected_tensor = zeros(new_shape,dtype=old_tensor.dtype)
        expected_tensor[:old_shape[0],:] = old_tensor
        expected_tensor[old_shape[0]:,:] = new_tensor
        self.assertTrue(allclose(concat_tensor, expected_tensor))

    def test_on_two_children(self):
        """
        Test the concatenation of a tensor with two children along the parent
        leg.
        """
        old_shape = (5,4,3,2)
        node, old_tensor = random_tensor_node(old_shape)
        node.add_parent("parent")
        node.add_children(["child1","child2"])
        new_tensor = crandn(old_tensor.shape)
        concat_tensor = concat_along_parent_leg(node, old_tensor, new_tensor)
        new_shape = (10,4,3,2)
        self.assertEqual(concat_tensor.shape, new_shape)
        # Build the expected tensor
        expected_tensor = zeros(new_shape, dtype=old_tensor.dtype)
        expected_tensor[:old_shape[0],:,:,:] = old_tensor
        expected_tensor[old_shape[0]:,:,:,:] = new_tensor
        self.assertTrue(allclose(concat_tensor, expected_tensor))

    def test_on_no_phys_leg(self):
        """
        Test the concatenation of a tensor with no physical legs along the
        parent leg.
        """
        old_shape = (4,3,2)
        node, old_tensor = random_tensor_node(old_shape)
        node.add_parent("parent")
        node.add_children(["child1","child2"])
        new_tensor = crandn(old_tensor.shape)
        concat_tensor = concat_along_parent_leg(node, old_tensor, new_tensor)
        new_shape = (8,3,2)
        self.assertEqual(concat_tensor.shape, new_shape)
        # Build the expected tensor
        expected_tensor = zeros(new_shape, dtype=old_tensor.dtype)
        expected_tensor[:old_shape[0],:,:] = old_tensor
        expected_tensor[old_shape[0]:,:,:] = new_tensor
        self.assertTrue(allclose(concat_tensor, expected_tensor))

class TestNewBasisTensorQRLegs(unittest.TestCase):

    def test_leaf(self):
        """
        Test the QR leg finding of a leaf tensor.
        """
        node, _ = random_tensor_node((3,2))
        node.add_parent("parent")
        q_legs, r_legs = new_basis_tensor_qr_legs(node)
        self.assertEqual(q_legs, (1, ))
        self.assertEqual(r_legs, (0, ))

    def test_two_children(self):
        """
        Test the QR leg finding of a tensor with two children.
        """
        node, _ = random_tensor_node((5,4,3,2))
        node.add_parent("parent")
        node.add_children(["child1","child2"])
        q_legs, r_legs = new_basis_tensor_qr_legs(node)
        self.assertEqual(q_legs, (1,2,3))
        self.assertEqual(r_legs, (0, ))

    def test_no_phys_leg(self):
        """
        Test the QR leg finding of a tensor with no physical legs.
        """
        node, _ = random_tensor_node((4,3,2))
        node.add_parent("parent")
        node.add_children(["child1","child2"])
        q_legs, r_legs = new_basis_tensor_qr_legs(node)
        self.assertEqual(q_legs, (1,2))
        self.assertEqual(r_legs, (0,))

    def test_root(self):
        """
        Test the QR leg finding of a root tensor.
        """
        node, _ = random_tensor_node((4,3,2))
        node.add_children(["child1","child2"])
        self.assertRaises(AssertionError, new_basis_tensor_qr_legs, node)

class Test_compute_new_basis_qr(unittest.TestCase):
    """
    Tests the computation of the new basis tensor directly from the QR
    decomposition.
    
    """

    def test_on_leaf(self):
        """
        Test the computation of the new basis tensor for a leaf tensor.
        """
        old_shape = (3,2)
        node, _ = random_tensor_node(old_shape)
        node.add_parent("parent")
        new_shape = (6,2)
        combined_tensor = crandn(new_shape)
        resulting_tensor = _compute_new_basis_tensor_qr(node, combined_tensor)
        # Check the shape
        self.assertEqual(resulting_tensor.shape[0], old_shape[1])
        # Check the isometry
        identity = eye(resulting_tensor.shape[-1])
        found = resulting_tensor @ resulting_tensor.T.conj()
        self.assertTrue(allclose(identity,found))
        # Compute reference tensor
        ref, _ = qr(combined_tensor.T)
        self.assertTrue(allclose(resulting_tensor, ref))

    def test_on_two_children(self):
        """
        Test the computation of the new basis tensor for a tensor with two
        children.
        """
        old_shape = (5,4,3,2)
        node, _ = random_tensor_node(old_shape)
        node.add_parent("parent")
        node.add_children(["child1","child2"])
        new_shape = (10,4,3,2)
        combined_tensor = crandn(new_shape)
        resulting_tensor = _compute_new_basis_tensor_qr(node, combined_tensor)
        # Check the shape
        self.assertEqual(resulting_tensor.shape[0:3], old_shape[1:])
        # Check the isometry
        identity = eye(resulting_tensor.shape[-1])
        found = tensordot(resulting_tensor,
                          resulting_tensor.conj(),
                          axes=([0,1,2],[0,1,2])
                          )
        self.assertTrue(allclose(identity,found))
        # Compute reference tensor
        ref, _ = qr(combined_tensor.reshape(new_shape[0],-1).T)
        ref = ref.reshape(new_shape[1],new_shape[2],new_shape[3],ref.shape[-1])
        self.assertTrue(allclose(resulting_tensor, ref))

    def test_on_no_phys_leg(self):
        """
        Test the computation of the new basis tensor for a tensor with no
        physical legs.
        """
        old_shape = (4,3,2)
        node, _ = random_tensor_node(old_shape)
        node.add_parent("parent")
        node.add_children(["child1","child2"])
        new_shape = (8,3,2)
        combined_tensor = crandn(new_shape)
        resulting_tensor = _compute_new_basis_tensor_qr(node, combined_tensor)
        # Check the shape
        self.assertEqual(resulting_tensor.shape[0:2], old_shape[1:])
        # Check the isometry
        identity = eye(resulting_tensor.shape[-1])
        found = tensordot(resulting_tensor,
                          resulting_tensor.conj(),
                          axes=([0,1],[0,1])
                          )
        self.assertTrue(allclose(identity,found))
        # Compute reference tensor
        ref, _ = qr(combined_tensor.reshape(new_shape[0],-1).T)
        ref = ref.reshape(new_shape[1],new_shape[2],ref.shape[-1])
        self.assertTrue(allclose(resulting_tensor, ref))

class Test_compute_new_basis_tensor(unittest.TestCase):

    def test_on_leaf(self):
        """
        Test the computation of the new basis tensor for a leaf tensor.
        """
        old_shape = (3,2)
        node, old_tensor = random_tensor_node(old_shape)
        node.add_parent("parent")
        updated_tensor = crandn(old_shape)
        new_basis_tensor = compute_new_basis_tensor(node,
                                                    old_tensor,
                                                    updated_tensor)
        # Check the shape
        self.assertEqual(new_basis_tensor.shape[1], old_shape[1])
        # Check the isometry
        identity = eye(new_basis_tensor.shape[0])
        found = new_basis_tensor.conj().T @ new_basis_tensor
        self.assertTrue(allclose(identity,found))
        # Compute reference tensor
        comb = concat_along_parent_leg(node, old_tensor, updated_tensor)
        ref = _compute_new_basis_tensor_qr(node, comb)
        ref = ref.T
        self.assertTrue(allclose(new_basis_tensor, ref))

    def test_on_two_children(self):
        """
        Test the computation of the new basis tensor for a tensor with two
        children.
        """
        old_shape = (5,4,3,2)
        node, old_tensor = random_tensor_node(old_shape)
        node.add_parent("parent")
        node.add_children(["child1","child2"])
        updated_tensor = crandn(old_shape)
        new_basis_tensor = compute_new_basis_tensor(node,
                                                    old_tensor,
                                                    updated_tensor)
        # Check the shape
        self.assertEqual(new_basis_tensor.shape[1:], old_shape[1:])
        # Check the isometry
        identity = eye(new_basis_tensor.shape[0])
        found = tensordot(new_basis_tensor,
                          new_basis_tensor.conj(),
                          axes=([1,2,3],[1,2,3])
                          )
        self.assertTrue(allclose(identity,found))
        # Compute reference tensor
        comb = concat_along_parent_leg(node, old_tensor, updated_tensor)
        ref = _compute_new_basis_tensor_qr(node, comb)
        ref = ref.transpose([3,0,1,2])
        self.assertTrue(allclose(new_basis_tensor, ref))

    def test_on_no_phys_leg(self):
        """
        Test the computation of the new basis tensor for a tensor with no
        physical legs.
        """
        old_shape = (4,3,2)
        node, old_tensor = random_tensor_node(old_shape)
        node.add_parent("parent")
        node.add_children(["child1","child2"])
        updated_tensor = crandn(old_shape)
        new_basis_tensor = compute_new_basis_tensor(node,
                                                    old_tensor,
                                                    updated_tensor)
        # Check the shape
        self.assertEqual(new_basis_tensor.shape[1:], old_shape[1:])
        # Check the isometry
        identity = eye(new_basis_tensor.shape[0])
        found = tensordot(new_basis_tensor,
                          new_basis_tensor.conj(),
                          axes=([1,2],[1,2])
                          )
        self.assertTrue(allclose(identity,found))
        # Compute reference tensor
        comb = concat_along_parent_leg(node, old_tensor, updated_tensor)
        ref = _compute_new_basis_tensor_qr(node, comb)
        ref = ref.transpose([2,0,1])
        self.assertTrue(allclose(new_basis_tensor, ref))

class Test_compute_basis_change_tensor_Complicated(unittest.TestCase):

    def setUp(self):
        shapes_old = [(7,6,2),(7,4,5,3),(4,2),(5,4,6,2),
                      (4,3),(6,3),(6,4,3),(4,2)]
        self.old_ttns = random_big_ttns_two_root_children(mode=shapes_old)
        shapes_new = [(8,5,2),(8,2,6,3),(2,2),(6,3,3,2),
                      (3,3),(3,3),(5,2,3),(2,2)]
        self.new_ttns = random_big_ttns_two_root_children(mode=shapes_new)
        # Mix up the leg order a bit
        self.new_ttns.canonical_form("site5")
        self.basis_change_cache = PartialTreeCachDict()

    def test_on_leaf_site2(self):
        """
        Test the computation of the basis change tensor for the leaf site2.
        """
        node_id = "site2"
        node_old, tensor_old = self.old_ttns[node_id]
        node_new, tensor_new = self.new_ttns[node_id]
        ref_old = deepcopy(tensor_old)
        ref_new = deepcopy(tensor_new)
        found = compute_basis_change_tensor(node_old,
                                            node_new,
                                            tensor_old,
                                            tensor_new,
                                            self.basis_change_cache)
        # Check
        correct = tensordot(ref_old,ref_new.conj(),axes=(1,1))
        self.assertTrue(allclose(found, correct))

    def test_on_site6(self):
        """
        Test the computation of the basis change tensor for site6.
        """
        node_id = "site6"
        node_old, tensor_old = self.old_ttns[node_id]
        node_new, tensor_new = self.new_ttns[node_id]
        ref_old = deepcopy(tensor_old)
        ref_new = deepcopy(tensor_new)
        # We need the basis change tensor for site7
        node_id7 = "site7"
        nold7, told7 = self.old_ttns[node_id7]
        nnew7, tnew7 = self.new_ttns[node_id7]
        basis_change_tensor7 = compute_basis_change_tensor(nold7,
                                                           nnew7,
                                                           told7,
                                                           tnew7,
                                                           self.basis_change_cache)
        self.basis_change_cache.add_entry(node_id7, node_id,
                                          basis_change_tensor7)
        ref_cache = deepcopy(self.basis_change_cache)
        # Compute the basis change tensor
        found = compute_basis_change_tensor(node_old,
                                            node_new,
                                            tensor_old,
                                            tensor_new,
                                            self.basis_change_cache)
        # Check
        correct = tensordot(ref_old,ref_new.conj(),axes=(2,2))
        correct = tensordot(correct,
                            ref_cache.get_entry(node_id7, node_id),
                            axes=([1,3],[0,1]))
        self.assertEqual(found.shape, (6,5))
        self.assertTrue(allclose(found, correct))

    def test_on_site3(self):
        """
        Test the computation of the basis change tensor for site3.
        """
        node_id = "site3"
        node_old, tensor_old = self.old_ttns[node_id]
        node_new, tensor_new = self.new_ttns[node_id]
        ref_old = deepcopy(tensor_old)
        ref_new = deepcopy(tensor_new)
        # We need to compute the basis change tensor for the children first
        node_ids = ["site4","site5"]
        for child_id in node_ids:
            nold, told = self.old_ttns[child_id]
            nnew, tnew = self.new_ttns[child_id]
            basis_change_tensor = compute_basis_change_tensor(nold,
                                                            nnew,
                                                            told,
                                                            tnew,
                                                            self.basis_change_cache)
            self.basis_change_cache.add_entry(child_id, node_id,
                                            basis_change_tensor)
        ref_cache = deepcopy(self.basis_change_cache)
        # Compute the basis change tensor
        found = compute_basis_change_tensor(node_old,
                                            node_new,
                                            tensor_old,
                                            tensor_new,
                                            self.basis_change_cache)
        # Check
        # old child order is site4, site5
        # new child order is site5, site4
        correct = tensordot(ref_old,ref_new.conj(),axes=(3,3))
        correct = tensordot(correct,
                            ref_cache.get_entry("site5", node_id),
                            axes=([2,4],[0,1]))
        correct = tensordot(correct,
                            ref_cache.get_entry("site4", node_id),
                            axes=([1,3],[0,1]))
        self.assertEqual(found.shape, (5,6))
        self.assertTrue(allclose(found, correct))

    def test_on_site1(self):
        """
        Test the computation of the basis change tensor for site3.
        """
        node_id = "site1"
        node_old, tensor_old = self.old_ttns[node_id]
        node_new, tensor_new = self.new_ttns[node_id]
        ref_old = deepcopy(tensor_old)
        ref_new = deepcopy(tensor_new)
        # We need to compute the basis change tensor for the children first
        node_ids = [("site2","site1"),
                    ("site4","site3"),
                    ("site5","site3"),
                    ("site3","site1")]
        for child_id, parent_id in node_ids:
            nold, told = self.old_ttns[child_id]
            nnew, tnew = self.new_ttns[child_id]
            basis_change_tensor = compute_basis_change_tensor(nold,
                                                            nnew,
                                                            told,
                                                            tnew,
                                                            self.basis_change_cache)
            self.basis_change_cache.add_entry(child_id, parent_id,
                                            basis_change_tensor)
        ref_cache = deepcopy(self.basis_change_cache)
        # Compute the basis change tensor
        found = compute_basis_change_tensor(node_old,
                                            node_new,
                                            tensor_old,
                                            tensor_new,
                                            self.basis_change_cache)
        # Check
        # old child order is site2, site3
        # new child order is site3, site2
        correct = tensordot(ref_old,ref_new.conj(),axes=(3,3))
        correct = tensordot(correct,
                            ref_cache.get_entry("site3", node_id),
                            axes=([2,4],[0,1]))
        correct = tensordot(correct,
                            ref_cache.get_entry("site2", node_id),
                            axes=([1,3],[0,1]))
        self.assertEqual(found.shape, (7,8))
        self.assertTrue(allclose(found, correct))

class Testfind_new_basis_replacement_leg_specs(unittest.TestCase):

    def setUp(self):
        self.m_legs = LegSpecification("parent",
                                       [],[])

    def test_on_leaf(self):
        """
        Test the finding of the new basis replacement legs for a leaf tensor.
        """
        node, _ = random_tensor_node((3,2))
        node.add_parent("parent")
        new_legs = find_new_basis_replacement_leg_specs(node)
        ref_u_legs = LegSpecification(None,[],[1])
        self.assertEqual(new_legs[0], self.m_legs)
        self.assertEqual(new_legs[1], ref_u_legs)

    def test_on_two_children(self):
        """
        Test the finding of the new basis replacement legs for a tensor with two
        children.
        """
        node, _ = random_tensor_node((5,4,3,2))
        node.add_parent("parent")
        children = ["child1","child2"]
        node.add_children(children)
        new_legs = find_new_basis_replacement_leg_specs(node)
        ref_u_legs = LegSpecification(None,copy(children),[3])
        self.assertEqual(new_legs[0], self.m_legs)
        self.assertEqual(new_legs[1], ref_u_legs)

    def test_on_no_phys_leg(self):
        """
        Test the finding of the new basis replacement legs for a tensor with no
        physical legs.
        """
        node, _ = random_tensor_node((4,3,2))
        node.add_parent("parent")
        children = ["child1","child2"]
        node.add_children(children)
        new_legs = find_new_basis_replacement_leg_specs(node)
        ref_u_legs = LegSpecification(None,copy(children),[])
        self.assertEqual(new_legs[0], self.m_legs)
        self.assertEqual(new_legs[1], ref_u_legs)

if __name__ == '__main__':
    unittest.main()
