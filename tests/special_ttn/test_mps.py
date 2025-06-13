import unittest

import numpy as np

import pytreenet as ptn
from pytreenet.random import crandn

class TestMPT_Attaching(unittest.TestCase):
    def setUp(self) -> None:
        self.mpt = ptn.MatrixProductTree()
        self.mpt.add_root(ptn.Node(identifier="site2"), crandn((4, 5, 2)))

        self.nodes = [ptn.Node(identifier="site"+str(i)) for i in (0, 1, 3, 4)]
        shapes = [(3, 2), (3, 4, 2), (5, 3, 2), (3, 2)]
        self.tensors = [crandn(shape) for shape in shapes]

    def test_add_left_first(self):
        self.mpt.attach_node_left_end(self.nodes[1], self.tensors[1])
        self.mpt.attach_node_right_end(self.nodes[2], self.tensors[2])

        self.assertEqual([self.nodes[1]], self.mpt.left_nodes)
        self.assertEqual([self.nodes[2]], self.mpt.right_nodes)

        self.assertTrue("site1" in self.mpt)
        self.assertTrue("site3" in self.mpt)

        ref_mpt = np.tensordot(self.mpt.root[1],
                               self.tensors[1],
                               axes=(self.mpt.root[0].neighbour_index("site1"), 1))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[2],
                               axes=(0, 0))

        found_mpt = self.mpt.completely_contract_tree()[0]
        self.assertTrue(np.allclose(ref_mpt, found_mpt))

    def test_add_right_first(self):
        self.mpt.attach_node_right_end(self.nodes[2], self.tensors[2])
        self.mpt.attach_node_left_end(self.nodes[1], self.tensors[1])

        self.assertEqual([self.nodes[1]], self.mpt.left_nodes)
        self.assertEqual([self.nodes[2]], self.mpt.right_nodes)

        self.assertTrue("site1" in self.mpt)
        self.assertTrue("site3" in self.mpt)

        ref_mpt = np.tensordot(self.mpt.root[1],
                               self.tensors[1],
                               axes=(self.mpt.root[0].neighbour_index("site1"), 1))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[2],
                               axes=(0, 0))
        ref_mpt = np.transpose(ref_mpt,
                               (0, 3, 4, 1, 2))

        found_mpt = self.mpt.completely_contract_tree()[0]
        self.assertTrue(np.allclose(ref_mpt, found_mpt))

    def test_add_two_left_nodes(self):
        self.mpt.attach_node_left_end(self.nodes[1], self.tensors[1])
        self.mpt.attach_node_left_end(
            self.nodes[0], self.tensors[0], final=True)

        identifiers = [node.identifier for node in self.mpt.left_nodes]
        self.assertTrue("site0" in identifiers)
        self.assertTrue("site1" in identifiers)

        self.assertTrue("site0" in self.mpt)
        self.assertTrue("site1" in self.mpt)

        ref_mpt = np.tensordot(self.mpt.root[1],
                               self.tensors[1],
                               axes=(self.mpt.root[0].neighbour_index("site1"), 1))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[0],
                               axes=(2, 0))

        found_mpt = self.mpt.completely_contract_tree()[0]
        self.assertTrue(np.allclose(ref_mpt, found_mpt))

    def test_add_two_right_nodes(self):
        self.mpt.attach_node_right_end(self.nodes[2], self.tensors[2])
        self.mpt.attach_node_right_end(self.nodes[3], self.tensors[3])

        identifiers = [node.identifier for node in self.mpt.right_nodes]
        self.assertTrue("site3" in identifiers)
        self.assertTrue("site4" in identifiers)

        self.assertTrue("site3" in self.mpt)
        self.assertTrue("site4" in self.mpt)

        ref_mpt = np.tensordot(self.mpt.root[1],
                               self.tensors[2],
                               axes=(self.mpt.root[0].neighbour_index("site3"), 0))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[3],
                               axes=(2, 0))

        found_mpt = self.mpt.completely_contract_tree()[0]
        self.assertTrue(np.allclose(ref_mpt, found_mpt))

    def test_add_four_nodes(self):
        self.mpt.attach_node_left_end(self.nodes[1], self.tensors[1])
        self.mpt.attach_node_left_end(
            self.nodes[0], self.tensors[0], final=True)
        self.mpt.attach_node_right_end(self.nodes[2], self.tensors[2])
        self.mpt.attach_node_right_end(self.nodes[3], self.tensors[3])

        left_identifiers = [node.identifier for node in self.mpt.left_nodes]
        self.assertTrue("site0" in left_identifiers)
        self.assertTrue("site1" in left_identifiers)
        right_identifiers = [node.identifier for node in self.mpt.right_nodes]
        self.assertTrue("site3" in right_identifiers)
        self.assertTrue("site4" in right_identifiers)
        self.assertTrue("site0" in self.mpt)
        self.assertTrue("site1" in self.mpt)
        self.assertTrue("site3" in self.mpt)
        self.assertTrue("site4" in self.mpt)

        ref_mpt = np.tensordot(self.mpt.root[1],
                               self.tensors[1],
                               axes=(self.mpt.root[0].neighbour_index("site1"), 1))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[0],
                               axes=(2, 0))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[2],
                               axes=(0, 0))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[3],
                               axes=(3, 0))

        found_mpt = self.mpt.completely_contract_tree()[0]
        self.assertTrue(np.allclose(ref_mpt, found_mpt))


class TestMPT_classmethods(unittest.TestCase):
    def setUp(self) -> None:
        shapes = [(3, 2), (3, 4, 2), (4, 5, 2), (5, 3, 2), (3, 2)]
        self.tensors = [crandn(shape) for shape in shapes]

    def test_from_tensor_list_root_at_2(self):
        mpt = ptn.MatrixProductTree.from_tensor_list(self.tensors,
                                                     root_site=2)

        ref_mpt = np.tensordot(self.tensors[2],
                               self.tensors[1],
                               axes=(0, 1))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[0],
                               axes=(2, 0))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[3],
                               axes=(0, 0))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[4],
                               axes=(3, 0))

        found_mpt = mpt.completely_contract_tree()[0]
        self.assertTrue(np.allclose(ref_mpt, found_mpt))

    def test_from_tensor_list_root_at_0(self):
        mpt = ptn.MatrixProductTree.from_tensor_list(self.tensors,
                                                     root_site=0)

        ref_mpt = np.tensordot(self.tensors[0],
                               self.tensors[1],
                               axes=(0, 0))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[2],
                               axes=(1, 0))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[3],
                               axes=(2, 0))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[4],
                               axes=(3, 0))

        found_mpt = mpt.completely_contract_tree()[0]
        self.assertTrue(np.allclose(ref_mpt, found_mpt))

    def test_from_tensor_list_root_at_last_node(self):
        mpt = ptn.MatrixProductTree.from_tensor_list(self.tensors,
                                                     root_site=4)

        ref_mpt = np.tensordot(self.tensors[4],
                               self.tensors[3],
                               axes=(0, 1))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[2],
                               axes=(1, 1))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[1],
                               axes=(2, 1))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[0],
                               axes=(3, 0))

        found_mpt = mpt.completely_contract_tree()[0]
        self.assertTrue(np.allclose(ref_mpt, found_mpt))


class TestMPS(unittest.TestCase):

    def test_neg_dimension(self):
        self.assertRaises(ValueError,
                          ptn.MatrixProductState.constant_product_state,
                          0, -1, 10)

    def test_state_larger_dimension(self):
        self.assertRaises(ValueError,
                          ptn.MatrixProductState.constant_product_state,
                          3, 2, 10)

    def test_neg_state(self):
        self.assertRaises(ValueError,
                          ptn.MatrixProductState.constant_product_state,
                          -2, 2, 10)

    def test_neg_num_sites(self):
        self.assertRaises(ValueError,
                          ptn.MatrixProductState.constant_product_state,
                          0, 2, -2)

    def test_constant_prod_state_all_bond_dim_1(self):
        mps = ptn.MatrixProductState.constant_product_state(0,2,5)
        for node in mps.nodes.values():
            if not node.is_root():
                self.assertEqual(1, node.shape[0])

        ref_mps = np.zeros(2**5)
        ref_mps[0] = 1
        found_mps = mps.completely_contract_tree()[0]
        found_mps = np.reshape(found_mps,32)
        self.assertTrue(np.allclose(ref_mps,found_mps))

    def test_to_little_custom_bond_dime(self):
        self.assertRaises(ValueError,
                          ptn.MatrixProductState.constant_product_state,
                          0,2,5,bond_dimensions=[0])

    def test_to_many_custom_bond_dime(self):
        self.assertRaises(ValueError,
                          ptn.MatrixProductState.constant_product_state,
                          0,2,5,bond_dimensions=[0,1,2,3,4,5])

    def test_contant_prod_state_custom_bond_dim(self):
        bond_dim = [2,4,4,2]
        mps = ptn.MatrixProductState.constant_product_state(0,2,5,
                                                            root_site=2,
                                                            bond_dimensions=bond_dim)
        for node_id in ("site0","site4"):
            self.assertEqual(2,mps.nodes[node_id].shape[0])
        for node_id in ("site1","site3"):
            self.assertEqual(4,mps.nodes[node_id].shape[0])

        ref_mps = np.zeros(2**5)
        ref_mps[0] = 1
        found_mps = mps.completely_contract_tree()[0]
        found_mps = np.reshape(found_mps,32)
        self.assertTrue(np.allclose(ref_mps,found_mps))

if __name__ == "__main__":
    unittest.main()
