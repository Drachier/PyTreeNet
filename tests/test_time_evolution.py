import unittest

import numpy as np

import pytreenet as ptn

class TestTimeEvolutionInit(unittest.TestCase):

    def setUp(self):
        # Initialise initial state
        self.initial_state = ptn.TreeTensorNetworkState()
        self.initial_state.add_root(ptn.Node(identifier="root"), ptn.crandn((5,6,2)))
        self.initial_state.add_child_to_parent(ptn.Node(identifier="c1"),
            ptn.crandn((5,3)), 0, "root", 0)
        self.initial_state.add_child_to_parent(ptn.Node(identifier="c2"),
            ptn.crandn((6,4)), 0, "root", 1)

        # Operators
        single_site_operator = ptn.TensorProduct({"root": ptn.crandn((2,2))})
        two_site_operator = ptn.TensorProduct({"c1": ptn.crandn((3,3)),
                                               "c2": ptn.crandn((4,4))})
        three_site_operator = ptn.TensorProduct({"c1": ptn.crandn((3,3)),
                                                  "c2": ptn.crandn((4,4)),
                                                  "root": ptn.crandn((2,2))})
        self.operators = [single_site_operator,
                          two_site_operator,
                          three_site_operator]

        # Time Parameters
        self.time_step_size = 0.1
        self.final_time = 1.0

        self.time_evol = ptn.TimeEvolution(self.initial_state, self.time_step_size,
            self.final_time, self.operators)

    def test_initial_state_init(self):
        for identifier in ["root", "c1", "c2"]:
            self.assertTrue(identifier in self.time_evol.initial_state.nodes)

    def test_time_init(self):
        self.assertEqual(self.time_step_size, self.time_evol.time_step_size)
        self.assertEqual(self.final_time, self.time_evol.final_time)
        self.assertEqual(10, self.time_evol.num_time_steps)

    def test_compute_num_time_steps(self):
        num_time_steps = self.time_evol._compute_num_time_steps()
        self.assertEqual(10, num_time_steps)

    def test_operator_init(self):
        self.assertEqual(len(self.operators), len(self.time_evol.operators))
        for ten_prod in self.operators:
            self.assertTrue(ten_prod in self.time_evol.operators)

    def test_time_step_size_check(self):
        self.assertRaises(ValueError, ptn.TimeEvolution, self.initial_state,
            -0.1, self.final_time, self.operators)

    def test_final_time_check(self):
        self.assertRaises(ValueError, ptn.TimeEvolution, self.initial_state,
            self.time_step_size, -1, self.operators)

    def test_only_one_operator(self):
        time_evo = ptn.TimeEvolution(self.initial_state, self.time_step_size,
            self.final_time, self.operators[0])
        self.assertTrue(isinstance(time_evo.operators, list))
        self.assertEqual(time_evo.operators[0], self.operators[0])

    def test_results_shape(self):
        for i in range(0,4):
            time_evo = ptn.TimeEvolution(self.initial_state, self.time_step_size,
                self.final_time, self.operators[0:i])
            self.assertEqual((i+1, 11), time_evo.results.shape)

class TestTimeEvolutionMethods(unittest.TestCase):
    def setUp(self):
        # Initialise initial state
        self.initial_state = ptn.TreeTensorNetworkState()
        self.initial_state.add_root(ptn.Node(identifier="root"), ptn.crandn((5,6,2)))
        self.initial_state.add_child_to_parent(ptn.Node(identifier="c1"),
            ptn.crandn((5,3)), 0, "root", 0)
        self.initial_state.add_child_to_parent(ptn.Node(identifier="c2"),
            ptn.crandn((6,4)), 0, "root", 1)

        # Operators
        single_site_operator = ptn.TensorProduct({"root": ptn.crandn((2,2))})
        two_site_operator = ptn.TensorProduct({"c1": ptn.crandn((3,3)),
                                               "c2": ptn.crandn((4,4))})
        three_site_operator = ptn.TensorProduct({"c1": ptn.crandn((3,3)),
                                                  "c2": ptn.crandn((4,4)),
                                                  "root": ptn.crandn((2,2))})
        self.operators = [single_site_operator,
                          two_site_operator,
                          three_site_operator]

        # Time Parameters
        self.time_step_size = 0.1
        self.final_time = 1.0

        self.time_evol = ptn.TimeEvolution(self.initial_state, self.time_step_size,
            self.final_time, self.operators)

    def test_run_one_time_step(self):
        self.assertRaises(NotImplementedError, self.time_evol.run_one_time_step)

    def test_evaluate_operators_for_single_site(self):
        found_results = self.time_evol.evaluate_operators()

        # Find reference results
        state_vector = self.time_evol.initial_state.completely_contract_tree(to_copy=True)
        state_vector = state_vector.tensors[state_vector.root_id].reshape(24)
        op1 = self.operators[0].pad_with_identities(self.initial_state).into_operator()
        result1 = state_vector.conj().T @ op1.operator.T @ state_vector

        self.assertAlmostEqual(found_results[0], result1)

if __name__ == "__main__":
    unittest.main()
