import unittest

import numpy as np

from scipy.linalg import expm

import pytreenet as ptn

class TestTrotterSplitting(unittest.TestCase):

    def setUp(self):

        # We need a ttn to work on.
        self.node1, self.tensor1 = ptn.random_tensor_node((5, 6, 3, 2), identifier="site1")
        self.node2, self.tensor2 = ptn.random_tensor_node((5, 4, 2), identifier="site2")
        self.node3, self.tensor3 = ptn.random_tensor_node((4, 2), identifier="site3")
        self.node4, self.tensor4 = ptn.random_tensor_node((6, 3, 2), identifier="site4")
        self.node5, self.tensor5 = ptn.random_tensor_node((3, 2), identifier="site5")
        self.node6, self.tensor6 = ptn.random_tensor_node((3, 5, 4, 2), identifier="site6")
        self.node7, self.tensor7 = ptn.random_tensor_node((5, 2), identifier="site7")
        self.node8, self.tensor8 = ptn.random_tensor_node((4, 2), identifier="site8")

        self.ttn = ptn.TreeTensorNetwork()

        self.ttn.add_root(self.node1, self.tensor1)
        self.ttn.add_child_to_parent(self.node2, self.tensor2, 0, "site1", 0)
        self.ttn.add_child_to_parent(self.node3, self.tensor3, 0, "site2", 1)
        self.ttn.add_child_to_parent(self.node4, self.tensor4, 0, "site1", 1)
        self.ttn.add_child_to_parent(self.node5, self.tensor5, 0, "site4", 1)
        self.ttn.add_child_to_parent(self.node6, self.tensor6, 0, "site1", 2)
        self.ttn.add_child_to_parent(self.node7, self.tensor7, 0, "site6", 1)
        self.ttn.add_child_to_parent(self.node8, self.tensor8, 0, "site6", 2)

        # We need a Hamiltonian and will use a simple Ising model
        X, _, Z = ptn.pauli_matrices()
        self.loc_operatorZ = Z
        self.loc_operatorX = X

        nn = self.ttn.nearest_neighbours()
        self.tensor_products = []

        for nn_pair in nn:
            term = ptn.TensorProduct({nn_pair[0]: self.loc_operatorZ, nn_pair[1]: self.loc_operatorZ})
            self.tensor_products.append(term)

        self.swaps_before = [ptn.SWAPlist([("site1", "site2"), ("site2", "site3")])
                             for i in self.tensor_products]
        self.swaps_after = [ptn.SWAPlist([("site3", "site2"), ("site2", "site1")])
                            for i in self.tensor_products]

        # And finally a splitting
        self.splitting_int = [3, 0, 1, 4, 2, 5, 6]
        self.splitting_tuple = [(3, 1), (0, 1), (1, 1), (4, 1), (2, 1), (5, 1), (6, 1)]

        # Remainders from the testing of the exponent in TEBD
        # two_site_operator = np.kron(self.loc_operatorZ, self.loc_operatorZ)
        # correct_exponent = expm((-1j * time_step_size) * two_site_operator)

        # correct_pairs = self.ttn.nearest_neighbours()

        # for index in self.splitting:
        #     found_exponent = tebd1.exponents[index]

        #     self.assertTrue(np.allclose(correct_exponent,
        #                                 found_exponent["operator"]))

        #     self.assertTrue(tuple(found_exponent["site_ids"]) in correct_pairs)

    def test_init_full(self):
        test_trottersplitting = ptn.TrotterSplitting(self.tensor_products,
                                                     splitting=self.splitting_tuple,
                                                     swaps_before=self.swaps_before,
                                                     swaps_after=self.swaps_after)

        self.assertEqual(test_trottersplitting.tensor_products, self.tensor_products)
        self.assertEqual(test_trottersplitting.splitting, self.splitting_tuple)
        self.assertEqual(test_trottersplitting.swaps_before, self.swaps_before)
        self.assertEqual(test_trottersplitting.swaps_after, self.swaps_after)

    def test_init_splitting_int(self):
        # Tests correctness, if the splitting is given as a list of integers.
        test_trottersplitting = ptn.TrotterSplitting(self.tensor_products,
                                                     splitting=self.splitting_int,
                                                     swaps_before=self.swaps_before,
                                                     swaps_after=self.swaps_after)
        self.assertEqual(test_trottersplitting.tensor_products, self.tensor_products)
        self.assertEqual(test_trottersplitting.splitting, self.splitting_tuple)
        self.assertEqual(test_trottersplitting.swaps_before, self.swaps_before)
        self.assertEqual(test_trottersplitting.swaps_after, self.swaps_after)

    def test_init_splitting_mixed(self):
        # Tests correctness, if the splitting is given as a mixture of int and tuples
        mixed_splitting = [(3, 1), (0, 1), 1, 4, (2, 1), 5, (6, 1)]
        test_trottersplitting = ptn.TrotterSplitting(self.tensor_products,
                                                     splitting=mixed_splitting,
                                                     swaps_before=self.swaps_before,
                                                     swaps_after=self.swaps_after)
        self.assertEqual(test_trottersplitting.tensor_products, self.tensor_products)
        self.assertEqual(test_trottersplitting.splitting, self.splitting_tuple)
        self.assertEqual(test_trottersplitting.swaps_before, self.swaps_before)
        self.assertEqual(test_trottersplitting.swaps_after, self.swaps_after)

    def test_init_no_splitting(self):
        test_trottersplitting = ptn.TrotterSplitting(self.tensor_products,
                                                     swaps_before=self.swaps_before,
                                                     swaps_after=self.swaps_after)
        self.assertEqual(test_trottersplitting.tensor_products, self.tensor_products)
        ref_splitting = [(i, 1) for i in range(len(self.splitting_tuple))]
        self.assertEqual(test_trottersplitting.splitting, ref_splitting)
        self.assertEqual(test_trottersplitting.swaps_before, self.swaps_before)
        self.assertEqual(test_trottersplitting.swaps_after, self.swaps_after)

    def test_init_no_swaps_before(self):
        test_trottersplitting = ptn.TrotterSplitting(self.tensor_products,
                                                     splitting=self.splitting_tuple,
                                                     swaps_after=self.swaps_after)
        self.assertEqual(test_trottersplitting.tensor_products, self.tensor_products)
        self.assertEqual(test_trottersplitting.splitting, self.splitting_tuple)
        self.assertEqual(test_trottersplitting.swaps_before,
                         [ptn.SWAPlist([])] * len(self.splitting_tuple))
        self.assertEqual(test_trottersplitting.swaps_after, self.swaps_after)

    def test_init_no_swaps_after(self):
        test_trottersplitting = ptn.TrotterSplitting(self.tensor_products,
                                                     splitting=self.splitting_tuple,
                                                     swaps_before=self.swaps_before)
        self.assertEqual(test_trottersplitting.tensor_products, self.tensor_products)
        self.assertEqual(test_trottersplitting.splitting, self.splitting_tuple)
        self.assertEqual(test_trottersplitting.swaps_before, self.swaps_before)
        self.assertEqual(test_trottersplitting.swaps_after,
                         [ptn.SWAPlist([])] * len(self.splitting_tuple))

if __name__ == "__main__":
    unittest.main()
