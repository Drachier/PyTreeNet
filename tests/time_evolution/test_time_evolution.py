import unittest

import numpy as np
from scipy.linalg import expm
import pytest

import pytreenet as ptn
from pytreenet.time_evolution.time_evolution import EvoDirection, TimeEvoMode, TimeEvoMethod
from pytreenet.random import crandn
from pytreenet.random.random_matrices import random_hermitian_matrix

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
            self.assertTrue(ten_prod in self.time_evol.operators.values())

    def test_time_step_size_check(self):
        self.assertRaises(ValueError, ptn.TimeEvolution, self.initial_state,
            -0.1, self.final_time, self.operators)

    def test_final_time_check(self):
        self.assertRaises(ValueError, ptn.TimeEvolution, self.initial_state,
            self.time_step_size, -1, self.operators)

    def test_only_one_operator(self):
        time_evo = ptn.TimeEvolution(self.initial_state, self.time_step_size,
            self.final_time, self.operators[0])
        self.assertTrue(isinstance(time_evo.operators, dict))
        self.assertEqual(time_evo.operators["0"], self.operators[0])

    def test_results(self):
        '''
        The results are only initialised once a time evolution is run.
        '''
        self.assertFalse(self.time_evol.results.is_initialized())

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
        self.assertRaises(NotImplementedError, self.time_evol.evaluate_operator, self.operators[0])

    def test_init_results(self):
        """
        Initialising the results with standard evaluation_time.
        """
        self.time_evol.init_results()
        results = self.time_evol.results
        # Check that results are initialized
        self.assertTrue(results.is_initialized())
        # Check that we have the right number of operators + times
        self.assertEqual(len(results.results), 4)  # 3 operators + times
        # Check that times array has correct shape
        times = results.times()
        self.assertEqual(times.shape, (11,))  # 10 time steps + initial
        # Check that all results are initialized to zeros
        for key in results.results:
            if key != "times":
                result_array = results.results[key]
                self.assertEqual(result_array.shape, (11,))
                self.assertTrue(np.allclose(result_array, np.zeros_like(result_array)))

    def test_init_results_with_custom_evaluation_time(self):
        """
        Initialising the results with custom evaluation_time.
        """
        evaluation_time = 2
        self.time_evol.init_results(evaluation_time=evaluation_time)
        results = self.time_evol.results
        self.assertTrue(results.is_initialized())
        # Check times array shape: num_time_steps // evaluation_time + 1
        times = results.times()
        expected_length = self.time_evol.num_time_steps // evaluation_time + 1
        self.assertEqual(times.shape, (expected_length,))

    def test_init_results_with_inf(self):
        """
        Initialising the results with evalutation_time = "inf".
        Thus only the last result is saved.
        """
        evaluation_time = "inf"
        self.time_evol.init_results(evaluation_time=evaluation_time)
        results = self.time_evol.results
        self.assertTrue(results.is_initialized())
        # Check times array shape: should be (1,) for inf evaluation
        times = results.times()
        self.assertEqual(times.shape, (1,))

    def test_check_result_exists(self):
        """
        The results are only initialised once a time evolution is run.
        Therefore, this method should raise an error, if called before
        the time evolution is run.
        """
        self.assertFalse(self.time_evol.results.is_initialized())

    def test_check_result_exists_after_run(self):
        """
        The results are only initialised once a time evolution is run.
        Therefore, this method should not raise an error, if called after
        the time evolution is run.
        """
        # As the time-evolution is an abstract method, we run the method
        # init_results before calling check_result_exists
        self.time_evol.init_results()
        self.assertTrue(self.time_evol.results.is_initialized())

    def test_results_real_true(self):
        """
        Should be True for real results.
        """
        self.time_evol.init_results()
        # Set a real result for operator "0"
        test_results = np.real(crandn((11,)))
        self.time_evol.results.results["0"] = test_results
        self.assertTrue(self.time_evol.results.result_real("0"))

    def test_results_real_false(self):
        """
        Should be False for complex results.
        """
        self.time_evol.init_results()
        # Set a complex result for operator "0"
        test_results = crandn((11,))
        self.time_evol.results.results["0"] = test_results
        self.assertFalse(self.time_evol.results.result_real("0"))

    def test_times(self):
        """
        Should return the times at which the operators were evaluated.
        They are always assumed to be real.
        """
        self.time_evol.init_results()
        times = np.real(crandn((11,)))
        self.time_evol.results.results["times"] = times
        retrieved_times = self.time_evol.results.times()
        self.assertTrue(np.allclose(times, retrieved_times))

    def test_operator_results_no_realise(self):
        """
        Should return the operator results without realising them.
        """
        self.time_evol.init_results()
        results = crandn((11,))
        self.time_evol.results.results["0"] = results
        retrieved = self.time_evol.results.operator_result("0")
        self.assertTrue(np.allclose(results, retrieved))

    def test_operator_results_realise(self):
        """
        Should return the operator results and realise them.
        """
        self.time_evol.init_results()
        results = crandn((11,))
        self.time_evol.results.results["0"] = results
        retrieved = self.time_evol.results.operator_result("0", realise=True)
        self.assertTrue(np.allclose(np.real(results), retrieved))


### Test EvoDirection Enum Class
@pytest.mark.parametrize("test_input,expected", [(True, EvoDirection.FORWARD),
                                                (False, EvoDirection.BACKWARD)])
def test_from_bool(test_input, expected):
    """
    Test the from_bool method of the EvoDirection enum class.
    """
    assert EvoDirection.from_bool(test_input) == expected

@pytest.mark.parametrize("test_input,expected", [(EvoDirection.FORWARD, -1),
                                                (EvoDirection.BACKWARD, 1)])
def test_exp_sign(test_input, expected):
    """
    Test the exp_sign method of the EvoDirection enum class.
    """
    assert EvoDirection.exp_sign(test_input) == expected

@pytest.mark.parametrize("test_input,expected", [(EvoDirection.FORWARD, -1j),
                                                (EvoDirection.BACKWARD, 1j)])
def test_exp_factor(test_input, expected):
    """
    Test the exp_factor method of the EvoDirection enum class.
    """
    assert EvoDirection.exp_factor(test_input) == expected


### Test TimeEvoMethod and TimeEvoMode Classes
def test_fastest_equivalent():
    """
    Test that the 'fastest' mode is equivalent to 'chebyshev'.
    """
    assert TimeEvoMethod.fastest_equivalent() == TimeEvoMethod.CHEBYSHEV

@pytest.mark.parametrize("method,expected", [
    (TimeEvoMethod.EXPM, False),
    (TimeEvoMethod.CHEBYSHEV, False),
    (TimeEvoMethod.SPARSE, False),
    (TimeEvoMethod.RK45, True),
    (TimeEvoMethod.RK23, True),
    (TimeEvoMethod.DOP853, True),
    (TimeEvoMethod.BDF, True),
])
def test_is_scipy(method, expected):
    """
    Test the is_scipy method of the TimeEvoMethod enum class.
    """
    assert method.is_scipy() == expected

@pytest.mark.parametrize("method", [
    TimeEvoMethod.RK45,
    TimeEvoMethod.RK23,
    TimeEvoMethod.DOP853,
    TimeEvoMethod.BDF,
])
def test_timeevo_mode_action_scipy(method):
    """
    Test that TimeEvoMode with scipy methods can use time_evolve_action.
    """
    shape = (2, 3, 4)
    psi_init = crandn(shape)
    dt = 0.1
    exponent_mat = random_hermitian_matrix(np.prod(shape).item())
    exponent_tens = exponent_mat.reshape(2*list(shape))
    def multiply_fn(_, x):
        return np.tensordot(exponent_tens, x, axes=([3,4,5], [0,1,2]))
    
    # Create TimeEvoMode instance
    mode = TimeEvoMode(method, {'atol': 1e-9, 'rtol': 1e-9})
    found = mode.time_evolve_action(psi_init, multiply_fn, dt)
    
    # Reference is the exponentiation of the matrix
    reference = expm(-1j * exponent_mat * dt) @ psi_init.flatten()
    # Check if the results are close
    assert shape == found.shape
    np.testing.assert_allclose(found.flatten(), reference)

@pytest.mark.parametrize("method", [
    TimeEvoMethod.EXPM,
    TimeEvoMethod.CHEBYSHEV,
    TimeEvoMethod.SPARSE,
    TimeEvoMethod.RK45,
    TimeEvoMethod.RK23,
    TimeEvoMethod.DOP853,
    TimeEvoMethod.BDF
])
def test_timeevo_mode_time_evolve(method):
    """
    Test the time_evolve method of the TimeEvoMode class that simply uses
    the full Hamiltonian matrix.
    """
    shape = (2, 3, 4)
    psi_init = crandn(shape)
    dt = 0.1
    exponent_mat = random_hermitian_matrix(np.prod(shape).item())
    
    # Create TimeEvoMode instance with appropriate options
    if method.is_scipy():
        mode = TimeEvoMode(method, {'atol': 1e-9, 'rtol': 1e-9})
    else:
        mode = TimeEvoMode(method)
    
    found = mode.time_evolve(psi_init, exponent_mat, dt)
    
    # Reference is the exponentiation of the matrix
    reference = expm(-1j * exponent_mat * dt) @ psi_init.flatten()
    # Check if the results are close
    assert shape == found.shape
    np.testing.assert_allclose(found.flatten(), reference)

def test_timeevo_mode_fastest_equivalent():
    """
    Test that TimeEvoMode with FASTEST method works correctly.
    """
    mode_fastest = TimeEvoMode(TimeEvoMethod.FASTEST)
    
    # Test that fastest equivalent returns correct mode
    fastest_equiv = mode_fastest.fastest_equivalent()
    assert fastest_equiv.method == TimeEvoMethod.CHEBYSHEV
    
    # Test is_scipy behavior
    assert mode_fastest.is_scipy() == False
    assert mode_fastest.action_evolvable() == False

def test_timeevo_mode_default_solver_options():
    """
    Test that default solver options are set correctly.
    """
    # Scipy methods should get default atol/rtol
    rk45_mode = TimeEvoMode(TimeEvoMethod.RK45)
    assert 'atol' in rk45_mode.solver_options
    assert 'rtol' in rk45_mode.solver_options
    assert rk45_mode.solver_options['atol'] == 1e-06
    assert rk45_mode.solver_options['rtol'] == 1e-06
    
    # Non-scipy methods should get empty dict
    cheby_mode = TimeEvoMode(TimeEvoMethod.CHEBYSHEV)
    assert cheby_mode.solver_options == {}

def test_timeevo_mode_custom_solver_options():
    """
    Test that custom solver options override defaults.
    """
    custom_options = {'atol': 1e-10, 'rtol': 1e-10, 'max_step': 0.01}
    mode = TimeEvoMode(TimeEvoMethod.RK45, custom_options)
    assert mode.solver_options == custom_options

def test_timeevo_mode_class_methods():
    """
    Test the class method constructors.
    """
    # Test a few class method constructors
    mode_fastest = TimeEvoMode.create_fastest()
    assert mode_fastest.method == TimeEvoMethod.FASTEST
    
    mode_rk45 = TimeEvoMode.create_rk45()
    assert mode_rk45.method == TimeEvoMethod.RK45
    assert 'atol' in mode_rk45.solver_options
    
    mode_rk45_custom = TimeEvoMode.create_rk45(atol=1e-10, rtol=1e-10)
    assert mode_rk45_custom.solver_options['atol'] == 1e-10

if __name__ == "__main__":
    unittest.main()
