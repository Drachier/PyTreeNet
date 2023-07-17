import unittest

import pytreenet as ptn

class TestTEBD(unittest.TestCase):

    def setUp(self):

        # We need a ttn to work on.
        self.node1, self.tensor1 = ptn.random_tensor_node((5,6,3,2), identifier="site1")
        self.node2, self.tensor2 = ptn.random_tensor_node((5,4,2), identifier="site2")
        self.node3, self.tensor3 = ptn.random_tensor_node((4,2), identifier="site3")
        self.node4, self.tensor4 = ptn.random_tensor_node((6,3,2), identifier="site4")
        self.node5, self.tensor5 = ptn.random_tensor_node((3,2), identifier="site5")
        self.node6, self.tensor6 = ptn.random_tensor_node((3,5,4,2), identifier="site6")
        self.node7, self.tensor7 = ptn.random_tensor_node((5,2), identifier="site7")
        self.node8, self.tensor8 = ptn.random_tensor_node((4,2), identifier="site8")

        self.ttn = ptn.TreeTensorNetworkState()

        self.ttn.add_root(self.node1, self.tensor1)
        self.ttn.add_child_to_parent(self.node2, self.tensor2, 0, "site1", 0)
        self.ttn.add_child_to_parent(self.node3, self.tensor3, 0, "site2", 1)
        self.ttn.add_child_to_parent(self.node4, self.tensor4, 0, "site1", 1)
        self.ttn.add_child_to_parent(self.node5, self.tensor5, 0, "site4", 1)
        self.ttn.add_child_to_parent(self.node6, self.tensor6, 0, "site1", 2)
        self.ttn.add_child_to_parent(self.node7, self.tensor7, 0, "site6", 1)
        self.ttn.add_child_to_parent(self.node8, self.tensor8, 0, "site6", 2)

        # We need a toy model and will use a simple Ising model
        X, _, Z = ptn.pauli_matrices()
        self.loc_operatorZ = Z
        self.loc_operatorX = X

        nn = self.ttn.nearest_neighbours()
        time_operators = []
        for nn_pair in nn:
            term = ptn.TensorProduct({nn_pair[0]: self.loc_operatorZ,
                                      nn_pair[1]: self.loc_operatorZ})
            time_operators.append(term)

        # And a splitting
        splitting = [3, 0, 1, 4, 2, 5, 6]

        # To build a TrotterSplitting
        self.trotter_splitting_woswaps = ptn.TrotterSplitting(time_operators,
                                                              splitting=splitting)

        swaps_before = [ptn.SWAPlist([]) for i in splitting]
        swaps_before[0].extend([("site3", "site2"), ("site2", "site1")])
        swaps_before[4].extend([("site4", "site1"), ("site6", "site7")])
        swaps_after = [ptn.SWAPlist([]) for i in range(len(splitting))]
        swaps_after[0].extend([("site8", "site6"), ("site6", "site7")])
        swaps_after[4].extend([("site1", "site2"), ("site2", "site3")])
        self.trotter_splitting_wswaps = ptn.TrotterSplitting(time_operators,
                                                              splitting=splitting,
                                                              swaps_before=swaps_before,
                                                              swaps_after=swaps_after)

        # We want to evaluate the two pauli_matrices locally and the tensor
        # poduct over all sites
        self.operators = []
        Z_all = {}
        X_all = {}
        for node_id in self.ttn.nodes.keys():
            dictZ = ptn.TensorProduct({node_id: self.loc_operatorZ})
            self.operators.append(dictZ)
            dictX = ptn.TensorProduct({node_id: self.loc_operatorX})
            self.operators.append(dictX)
            Z_all[node_id] = self.loc_operatorZ
            X_all[node_id] = self.loc_operatorX
        self.operators.append(ptn.TensorProduct(Z_all))
        self.operators.append(ptn.TensorProduct(X_all))

        self.time_step_size = 0.1
        self.final_time = 1

    def test_init(self):
        tebd1 = ptn.TEBD(self.ttn, self.trotter_splitting_woswaps,
                         self.time_step_size,
                         self.final_time,
                         self.operators)
        self.assertEqual(7, len(tebd1.exponents))
        tebd2 = ptn.TEBD(self.ttn, self.trotter_splitting_wswaps,
                         self.time_step_size,
                         self.final_time,
                         self.operators)
        self.assertEqual(15, len(tebd2.exponents))

    def test_run_one_time_step(self):
        # Setting up tebd
        tebd = ptn.TEBD(self.ttn,
                        self.trotter_splitting_wswaps,
                        self.time_step_size,
                        self.final_time,
                        self.operators)
        tebd.run_one_time_step()

    def test_evaluate_operators(self):
        # Setting up tebd
        tebd = ptn.TEBD(self.ttn,
                        self.trotter_splitting_wswaps,
                        self.time_step_size,
                        self.final_time,
                        self.operators)
        tebd.evaluate_operators()

if __name__ == "__main__":
    unittest.main()
