
import unittest

import numpy as np

import pytreenet as ptn

class TestCachedSiteTensor(unittest.TestCase):
    def setUp(self) -> None:
        self.conversion_dict = {"root_op1": ptn.random_hermitian_matrix(),
                                "root_op2": ptn.random_hermitian_matrix(),
                                "I2": np.eye(2),
                                "c1_op": ptn.random_hermitian_matrix(size=3),
                                "I3": np.eye(3),
                                "c2_op": ptn.random_hermitian_matrix(size=4),
                                "I4": np.eye(4)}
        self.ref_tree = ptn.random_small_ttns()
        tensor_prod = [ptn.TensorProduct({"c1": "I3", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op2", "c2": "c2_op"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "I2", "c2": "c2_op"})
                       ]
        ham = ptn.Hamiltonian(tensor_prod, self.conversion_dict)
        self.hamiltonian = ptn.TTNO.from_hamiltonian(ham, self.ref_tree)

        ket_node, ket_tensor = self.ref_tree["root"]
        ham_node, ham_tensor = self.hamiltonian["root"]
        self.root_cache = ptn.CachedSiteTensor(ket_node, ham_node,
                                               ket_tensor, ham_tensor)

        ket_node, ket_tensor = self.ref_tree["c1"]
        ham_node, ham_tensor = self.hamiltonian["c1"]
        self.c1_cache = ptn.CachedSiteTensor(ket_node, ham_node,
                                               ket_tensor, ham_tensor)

        ket_node, ket_tensor = self.ref_tree["c2"]
        ham_node, ham_tensor = self.hamiltonian["c2"]
        self.c2_cache = ptn.CachedSiteTensor(ket_node, ham_node,
                                               ket_tensor, ham_tensor)

    def test_cached_legs(self):
        self.assertEqual(2, self.root_cache.cached_legs())
        self.assertEqual(1, self.c1_cache.cached_legs())
        self.assertEqual(1, self.c2_cache.cached_legs())

    def test_state_phys_leg(self):
        self.assertEqual(2, self.root_cache._node_state_phys_leg())
        self.assertEqual(1, self.c1_cache._node_state_phys_leg())
        self.assertEqual(1, self.c2_cache._node_state_phys_leg())

    def test_state_operator_input_leg(self):
        self.assertEqual(3, self.root_cache._node_operator_input_leg())
        self.assertEqual(2, self.c1_cache._node_operator_input_leg())
        self.assertEqual(2, self.c2_cache._node_operator_input_leg())

    def test_state_operator_output_leg(self):
        self.assertEqual(2, self.root_cache._node_operator_output_leg())
        self.assertEqual(1, self.c1_cache._node_operator_output_leg())
        self.assertEqual(1, self.c2_cache._node_operator_output_leg())

    def test_contract_tensor_sandwich_root(self):
        ref_tensor = np.tensordot(self.ref_tree.tensors["root"],
                                  self.hamiltonian.tensors["root"],
                                  axes=(2,3))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.ref_tree.tensors["root"].conj(),
                                  axes=(4,2))

        found_tensor = self.root_cache.contract_tensor_sandwich()
        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_contract_tensor_sandwich_c1(self):
        ref_tensor = np.tensordot(self.ref_tree.tensors["c1"],
                                  self.hamiltonian.tensors["c1"],
                                  axes=(1,2))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.ref_tree.tensors["c1"].conj(),
                                  axes=(2,1))

        found_tensor = self.c1_cache.contract_tensor_sandwich()
        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_contract_tensor_sandwich_c2(self):
        ref_tensor = np.tensordot(self.ref_tree.tensors["c2"],
                                  self.hamiltonian.tensors["c2"],
                                  axes=(1,2))
        ref_tensor = np.tensordot(ref_tensor,
                                  self.ref_tree.tensors["c2"].conj(),
                                  axes=(2,1))

        found_tensor = self.c2_cache.contract_tensor_sandwich()
        self.assertTrue(np.allclose(ref_tensor, found_tensor))

    def test_leg_order(self):
        self.assertEqual([0,2,4,1,3,5], self.root_cache.leg_order())
        self.assertEqual([0,1,2], self.c1_cache.leg_order())
        self.assertEqual([0,1,2], self.c2_cache.leg_order())

    def test_new_shape(self):
        tensor_root = self.root_cache.contract_tensor_sandwich().transpose(self.root_cache.leg_order())
        self.assertEqual([50,72], self.root_cache.new_shape(tensor_root))
        tensor_c1 = self.c1_cache.contract_tensor_sandwich().transpose(self.c1_cache.leg_order())
        self.assertEqual([50], self.c1_cache.new_shape(tensor_c1))
        tensor_c2 = self.c2_cache.contract_tensor_sandwich().transpose(self.c2_cache.leg_order())
        self.assertEqual([72], self.c2_cache.new_shape(tensor_c2))

    def test_compute_root(self):
        ref_root = self.root_cache.contract_tensor_sandwich().transpose(self.root_cache.leg_order())
        ref_root = ref_root.reshape(self.root_cache.new_shape(ref_root))
        self.assertTrue(np.allclose(ref_root, self.root_cache.compute()))

    def test_compute_c1(self):
        ref_c1 = self.c1_cache.contract_tensor_sandwich().transpose(self.c1_cache.leg_order())
        ref_c1 = ref_c1.reshape(self.c1_cache.new_shape(ref_c1))
        self.assertTrue(np.allclose(ref_c1, self.c1_cache.compute()))

    def test_compute_c2(self):
        ref_c2 = self.c2_cache.contract_tensor_sandwich().transpose(self.c2_cache.leg_order())
        ref_c2 = ref_c2.reshape(self.c2_cache.new_shape(ref_c2))
        self.assertTrue(np.allclose(ref_c2, self.c2_cache.compute()))
    
if __name__ == "__main__":
    unittest.main()