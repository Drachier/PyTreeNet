import unittest

import numpy as np

import pytreenet as ptn
from pytreenet.random import crandn

class TestTimeEvolutionInit(unittest.TestCase):

    def setUp(self):
        # Initialise initial state
        self.initial_state = ptn.TreeTensorNetworkState()
        self.initial_state.add_root(ptn.Node(identifier="root"), crandn((5,6,2)))
        self.initial_state.add_child_to_parent(ptn.Node(identifier="c1"),
            crandn((5,3)), 0, "root", 0)
        self.initial_state.add_child_to_parent(ptn.Node(identifier="c2"),
            crandn((6,4)), 0, "root", 1)

        # Operators
        single_site_operator = ptn.TensorProduct({"root": crandn((2,2))})
        two_site_operator = ptn.TensorProduct({"c1": crandn((3,3)),
                                               "c2": crandn((4,4))})
        three_site_operator = ptn.TensorProduct({"c1": crandn((3,3)),
                                                  "c2": crandn((4,4)),
                                                  "root": crandn((2,2))})
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

    def test_results(self):
        '''
        The results are only initialised once a time evolution is run.
        '''
        self.assertEqual(None, self.time_evol._results)

class TestTimeEvolutionMethods(unittest.TestCase):
    def setUp(self):
        # Initialise initial state
        self.initial_state = ptn.TreeTensorNetworkState()
        self.initial_state.add_root(ptn.Node(identifier="root"), crandn((5,6,2)))
        self.initial_state.add_child_to_parent(ptn.Node(identifier="c1"),
            crandn((5,3)), 0, "root", 0)
        self.initial_state.add_child_to_parent(ptn.Node(identifier="c2"),
            crandn((6,4)), 0, "root", 1)

        # Operators
        single_site_operator = ptn.TensorProduct({"root": crandn((2,2))})
        two_site_operator = ptn.TensorProduct({"c1": crandn((3,3)),
                                               "c2": crandn((4,4))})
        three_site_operator = ptn.TensorProduct({"c1": crandn((3,3)),
                                                  "c2": crandn((4,4)),
                                                  "root": crandn((2,2))})
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
        self.assertRaises(NotImplementedError, self.time_evol.evaluate_operators)

    def test_init_results(self):
        """
        Initialising the results with standard evaluation_time.
        """
        self.time_evol.init_results()
        results = self.time_evol._results
        correct_shape = (4,11)
        self.assertEqual(correct_shape, results.shape)
        correct_results = np.zeros(correct_shape)
        self.assertTrue(np.allclose(correct_results, results))

    def test_init_results_with_custom_evaluation_time(self):
        """
        Initialising the results with custom evaluation_time.
        """
        evaluation_time = 2
        self.time_evol.init_results(evaluation_time=evaluation_time)
        results = self.time_evol._results
        correct_shape = (4,6)
        self.assertEqual(correct_shape, results.shape)
        correct_results = np.zeros(correct_shape)
        self.assertTrue(np.allclose(correct_results, results))

    def test_init_results_with_inf(self):
        """
        Initialising the results with evalutation_time = "inf".
        Thus only the last result is saved.
        """
        evaluation_time = "inf"
        self.time_evol.init_results(evaluation_time=evaluation_time)
        results = self.time_evol._results
        correct_shape = (4,1)
        self.assertEqual(correct_shape, results.shape)
        correct_results = np.zeros(correct_shape)
        self.assertTrue(np.allclose(correct_results, results))

    def test_check_result_exists(self):
        """
        The results are only initialised once a time evolution is run.
        Therefore, this method should raise an error, if called before
        the time evolution is run.
        """
        self.assertRaises(AssertionError, self.time_evol.check_result_exists)

    def test_check_result_exists_after_run(self):
        """
        The results are only initialised once a time evolution is run.
        Therefore, this method should not raise an error, if called after
        the time evolution is run.
        """
        # As the time-evolution is an abstract method, we run the method
        # _init_results before calling check_result_exists
        self.time_evol.init_results()
        self.time_evol.check_result_exists()

    def test_results_real_true(self):
        """
        Should be True for real results.
        """
        self.time_evol.init_results()
        test_results = np.real(crandn(self.time_evol.results.shape))
        self.time_evol._results = test_results
        self.assertTrue(self.time_evol.results_real())

    def test_results_real_false(self):
        """
        Should be False for complex results.
        """
        self.time_evol.init_results()
        test_results = crandn(self.time_evol.results.shape)
        self.time_evol._results = test_results
        self.assertFalse(self.time_evol.results_real())

    def test_times(self):
        """
        Should return the times at which the operators were evaluated.
        They are always assumed to be real.
        """
        self.time_evol.init_results()
        times = crandn((11, ))
        self.time_evol._results[-1] = times
        self.assertFalse(np.allclose(times, self.time_evol.times()))
        times = np.real(times)
        self.assertTrue(np.allclose(times, self.time_evol.times()))

    def test_operator_results_no_realise(self):
        """
        Should return the operator results without realising them.
        """
        self.time_evol.init_results()
        results = crandn(self.time_evol.results.shape)
        self.time_evol._results = results
        self.assertTrue(np.allclose(results[0:-1], self.time_evol.operator_results()))

    def test_operator_results_realise(self):
        """
        Should return the operator results and realise them.
        """
        self.time_evol.init_results()
        results = crandn(self.time_evol.results.shape)
        self.time_evol._results = results
        self.assertTrue(np.allclose(np.real(results[0:-1]), self.time_evol.operator_results(realise=True)))

if __name__ == "__main__":
    unittest.main()
