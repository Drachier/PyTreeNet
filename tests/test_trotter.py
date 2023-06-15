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
        self.operators = []

        for nn_pair in nn:
            term = {nn_pair[0]: self.loc_operatorZ, nn_pair[1]: self.loc_operatorZ}
            self.operators.append(term)

        self.swaps_before = [ptn.SWAPlist([("site1", "site2"), ("site2", "site3")]) for i in self.operators]
        self.swaps_after = [ptn.SWAPlist([("site3", "site2"), ("site2", "site1")]) for i in self.operators]

        # And finally a splitting
        self.splitting = [3, 0, 1, 4, 2, 5, 6]

        # Remainders from the testing of the exponent in TEBD
        # two_site_operator = np.kron(self.loc_operatorZ, self.loc_operatorZ)
        # correct_exponent = expm((-1j * time_step_size) * two_site_operator)

        # correct_pairs = self.ttn.nearest_neighbours()

        # for index in self.splitting:
        #     found_exponent = tebd1.exponents[index]

        #     self.assertTrue(np.allclose(correct_exponent,
        #                                 found_exponent["operator"]))

        #     self.assertTrue(tuple(found_exponent["site_ids"]) in correct_pairs)

    def test_init(self):
        # Test extension of splitting
        test_TrotterSplitting = ptn.TrotterSplitting(self.operators, self.splitting,
                                                     self.swaps_before, self.swaps_after)

        self.assertEqual(test_TrotterSplitting.operators, self.operators)

        ref_splitting = [(3, 1), (0, 1), (1, 1), (4, 1), (2, 1), (5, 1), (6, 1)]
        self.assertEqual(test_TrotterSplitting.splitting, ref_splitting)

        # Test extension of mixed splitting
        mixed_splitting = [(3, 0.5), (0, 1.2), 1, 4, (2, 1), 5, (6, 1.4)]
        test_TrotterSplitting = ptn.TrotterSplitting(self.operators, mixed_splitting,
                                                     self.swaps_before, self.swaps_after)

        ref_splitting = [(3, 0.5), (0, 1.2), (1, 1), (4, 1), (2, 1), (5, 1), (6, 1.4)]

        self.assertEqual(ref_splitting, test_TrotterSplitting.splitting)

        # Test empty SWAPs
        test_TrotterSplitting = ptn.TrotterSplitting(self.operators, self.splitting)

        self.assertEqual([ptn.SWAPlist([]) for i in self.splitting], test_TrotterSplitting.swaps_before)
        self.assertEqual([ptn.SWAPlist([]) for i in self.splitting], test_TrotterSplitting.swaps_after)

    def test_exponentiate_splitting(self):
        test_TrotterSplitting = ptn.TrotterSplitting(self.operators, self.splitting,
                                                     self.swaps_before, self.swaps_after)

        delta_time = 0.1
        unitaries = test_TrotterSplitting.exponentiate_splitting(self.ttn, delta_time)


if __name__ == "__main__":
    unittest.main()
