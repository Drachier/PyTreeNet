import unittest

import numpy as np

from scipy.linalg import expm

import pytreenet as ptn

class TestTEBD(unittest.TestCase):
    
    def setUp(self):
        
        # We need a ttn to work on.
        self.node1 = ptn.random_tensor_node((5,6,3,2), identifier="site1")
        self.node2 = ptn.random_tensor_node((5,4,2), identifier="site2")
        self.node3 = ptn.random_tensor_node((4,2), identifier="site3")
        self.node4 = ptn.random_tensor_node((6,3,2), identifier="site4")
        self.node5 = ptn.random_tensor_node((3,2), identifier="site5")
        self.node6 = ptn.random_tensor_node((3,5,4,2), identifier="site6")
        self.node7 = ptn.random_tensor_node((5,2), identifier="site7")
        self.node8 = ptn.random_tensor_node((4,2), identifier="site8")
    
        self.ttn = ptn.TreeTensorNetwork()
        
        self.ttn.add_root(self.node1)
        self.ttn.add_child_to_parent(self.node2, 0, "site1", 0)
        self.ttn.add_child_to_parent(self.node3, 0, "site2", 1)
        self.ttn.add_child_to_parent(self.node4, 0, "site1", 1)
        self.ttn.add_child_to_parent(self.node5, 0, "site4", 1)
        self.ttn.add_child_to_parent(self.node6, 0, "site1", 2)
        self.ttn.add_child_to_parent(self.node7, 0, "site6", 1)
        self.ttn.add_child_to_parent(self.node8, 0, "site6", 2)
        
        # We need a Hamiltonian and will use a simple Ising model
        X, _, Z = ptn.pauli_matrices()
        self.loc_operatorZ = Z
        self.loc_operatorX = X
        
        nn = self.ttn.nearest_neighbours()
        
        self.hamiltonian = ptn.Hamiltonian()
        
        for nn_pair in nn:
            term = {nn_pair[0]: self.loc_operatorZ, nn_pair[1]: self.loc_operatorZ}
            self.hamiltonian.add_term(term)
            
        # We want to evaluate the two pauli_matrices locally and the tensor
        # poduct over all sites
        self.operators = []
        
        Z_all = dict()
        X_all = dict()
        
        for node_id in self.ttn.nodes:
            dictZ = {node_id: self.loc_operatorZ}
            self.operators.append(dictZ)
            
            dictX = {node_id: self.loc_operatorX}
            self.operators.append(dictX)
            
            Z_all[node_id] = self.loc_operatorZ
            X_all[node_id] = self.loc_operatorX
            
        self.operators.append(Z_all)
        self.operators.append(X_all)
        
        # And finally a splitting
        self.splitting = [3, 0, 1, 4, 2, 5, 6]
    
    def test_init(self):
        """
        This includes the exponentiation.
        """
        time_step_size = 0.1
        final_time = 1
        
        tebd1 = ptn.TEBD(self.ttn, self.hamiltonian, time_step_size,
                         final_time, self.splitting)
        
        two_site_operator = np.kron(self.loc_operatorZ, self.loc_operatorZ)
        correct_exponent = expm((-1j * time_step_size) * two_site_operator)
        
        correct_pairs = self.ttn.nearest_neighbours()
        
        for index in self.splitting:
            found_exponent = tebd1.exponents[index]
            
            self.assertTrue(np.allclose(correct_exponent,
                                        found_exponent["operator"]))
            
            self.assertTrue(tuple(found_exponent["site_ids"]) in correct_pairs)
    
    def test_find_node_for_legs_of_two_site_tensor_find_node_for_legs_of_two_site_tensor(self):
        
        correct_leg_indices_list = [([0,2,3],[1,4]), ([0,2,3],[1,4]),
                                    ([0,2,3],[1,4,5]),
                                    ([0,2],[1]), ([0,2],[1]),
                                    ([0,2,3],[1]), ([0,2,3],[1])]
        
        for i, pair in enumerate(self.ttn.nearest_neighbours()):
            node1 = self.ttn.nodes[pair[0]]
            node2 = self.ttn.nodes[pair[1]]
            
            found_leg_indices = ptn.TEBD._find_node_for_legs_of_two_site_tensor(node1, node2)
            
            correct_leg_indices = correct_leg_indices_list[i]
            
            self.assertEqual(correct_leg_indices, found_leg_indices)
            
    def test_permutation_svdresults(self):
        test_node = ptn.random_tensor_node((2,3,4,5))
        mock_identifiers = ["Identify", "Actually two identifiers in a coat", "CT-7567", ":)"]
        test_node.open_legs_to_children([1,2], mock_identifiers[1:3])
        test_node.open_leg_to_parent(3, mock_identifiers[-1])
        
        found_permutation = ptn.TEBD._permutation_svdresult_u_to_fit_node(test_node)
        correct_permutation = [1,2,0,3]
        
        self.assertEqual(correct_permutation, found_permutation)
              
    def test_run_one_time_step(self):
        # Setting up tebd
        time_step_size = 0.1
        final_time = 1
        
        tebd1 = ptn.TEBD(self.ttn, self.hamiltonian, time_step_size,
                         final_time, custom_splitting=self.splitting)
        
        tebd1.run_one_time_step()
    
    def test_evaluate_operators(self):
        # Setting up tebd
        time_step_size = 0.1
        final_time = 1

        tebd1 = ptn.TEBD(self.ttn, self.hamiltonian, time_step_size,
                         final_time, custom_splitting=self.splitting,
                         operators=self.operators)
        
        tebd1.evaluate_operators()
        
    def test_run(self):
        # Setting up tebd
        time_step_size = 0.1
        final_time = 1

        tebd1 = ptn.TEBD(self.ttn, self.hamiltonian, time_step_size,
                         final_time, custom_splitting=self.splitting,
                         operators=self.operators)
        
        tebd1.run()
        print(tebd1.results)

if __name__ == "__main__":
    unittest.main()