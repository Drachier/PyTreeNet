import unittest

import numpy as np
from scipy.linalg import expm
import pytest

import pytreenet as ptn
from pytreenet.time_evolution.time_evolution import EvoDirection, TimeEvoMode
from pytreenet.time_evolution.results import Results
from pytreenet.random import crandn
from pytreenet.random.random_matrices import random_hermitian_matrix
from pytreenet.time_evolution.time_evolution import EvoDirection, TimeEvoMode, TimeEvoMethod
from pytreenet.core.node import Node
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.time_evolution.time_evolution import TimeEvolution

class TestTimeEvolutionInit(unittest.TestCase):

    def setUp(self):
        # Initialise initial state
        self.initial_state = TreeTensorNetworkState()
        self.initial_state.add_root(Node(identifier="root"), crandn((5,6,2)))
        self.initial_state.add_child_to_parent(Node(identifier="c1"),
            crandn((5,3)), 0, "root", 0)
        self.initial_state.add_child_to_parent(Node(identifier="c2"),
            crandn((6,4)), 0, "root", 1)

        # Operators
        single_site_operator = TensorProduct({"root": crandn((2,2))})
        two_site_operator = TensorProduct({"c1": crandn((3,3)),
                                               "c2": crandn((4,4))})
        three_site_operator = TensorProduct({"c1": crandn((3,3)),
                                                  "c2": crandn((4,4)),
                                                  "root": crandn((2,2))})
        self.operators = [single_site_operator,
                          two_site_operator,
                          three_site_operator]

        # Time Parameters
        self.time_step_size = 0.1
        self.final_time = 1.0

        self.time_evol = TimeEvolution(self.initial_state, self.time_step_size,
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
        for i, ten_prod in enumerate(self.operators):
            self.assertIn(str(i), self.time_evol.operators)
            self.assertEqual(ten_prod, self.time_evol.operators[str(i)])

    def test_time_step_size_check(self):
        self.assertRaises(ValueError, TimeEvolution, self.initial_state,
            -0.1, self.final_time, self.operators)

    def test_final_time_check(self):
        self.assertRaises(ValueError, TimeEvolution, self.initial_state,
            self.time_step_size, -1, self.operators)

    def test_only_one_operator(self):
        time_evo = TimeEvolution(self.initial_state, self.time_step_size,
            self.final_time, self.operators[0])
        self.assertTrue(isinstance(time_evo.operators, dict))
        self.assertEqual(len(time_evo.operators), 1)
        self.assertEqual(time_evo.operators["0"], self.operators[0])

    def test_results(self):
        '''
        The results are only initialised once a time evolution is run.
        '''
        self.time_evol.results.close_to(Results())

class TestTimeEvolutionMethods(unittest.TestCase):
    def setUp(self):
        # Initialise initial state
        self.initial_state = TreeTensorNetworkState()
        self.initial_state.add_root(Node(identifier="root"), crandn((5,6,2)))
        self.initial_state.add_child_to_parent(Node(identifier="c1"),
            crandn((5,3)), 0, "root", 0)
        self.initial_state.add_child_to_parent(Node(identifier="c2"),
            crandn((6,4)), 0, "root", 1)

        # Operators
        single_site_operator = TensorProduct({"root": crandn((2,2))})
        two_site_operator = TensorProduct({"c1": crandn((3,3)),
                                               "c2": crandn((4,4))})
        three_site_operator = TensorProduct({"c1": crandn((3,3)),
                                                  "c2": crandn((4,4)),
                                                  "root": crandn((2,2))})
        self.operators = [single_site_operator,
                          two_site_operator,
                          three_site_operator]

        # Time Parameters
        self.time_step_size = 0.1
        self.final_time = 1.0

        self.time_evol = TimeEvolution(self.initial_state, self.time_step_size,
            self.final_time, self.operators)

    def test_run_one_time_step(self):
        self.assertRaises(NotImplementedError, self.time_evol.run_one_time_step)

    def test_init_results(self):
        """
        Initialising the results with standard evaluation_time.
        """
        self.time_evol.init_results()
        results = self.time_evol.results
        self.assertTrue(results.is_initialized())
        self.assertEqual(len(self.time_evol.operators) + 1, results.num_results())
        self.assertEqual(self.time_evol.num_time_steps + 1, results.results_length())

    def test_init_results_with_custom_evaluation_time(self):
        """
        Initialising the results with custom evaluation_time.
        """
        evaluation_time = 2
        self.time_evol.init_results(evaluation_time=evaluation_time)
        results = self.time_evol.results
        self.assertTrue(results.is_initialized())
        self.assertEqual(len(self.time_evol.operators) + 1, results.num_results())
        self.assertEqual(self.time_evol.num_time_steps // 2 + 1, results.results_length())

    def test_init_results_with_inf(self):
        """
        Initialising the results with evalutation_time = "inf".
        Thus only the last result is saved.
        """
        evaluation_time = "inf"
        self.time_evol.init_results(evaluation_time=evaluation_time)
        results = self.time_evol.results
        self.assertTrue(results.is_initialized())
        self.assertEqual(len(self.time_evol.operators) + 1, results.num_results())
        self.assertEqual(2, results.results_length())


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
    # Use relaxed tolerances for ODE solvers, with extra tolerance for BDF
    if method == TimeEvoMethod.BDF or method == TimeEvoMethod.RK23:
        np.testing.assert_allclose(found.flatten(), reference, rtol=1e-2, atol=1e-3)
    else:
        np.testing.assert_allclose(found.flatten(), reference, rtol=1e-6, atol=1e-5)

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
    # Use relaxed tolerances for ODE solvers, with method-specific tolerances
    if method.is_scipy():
        if method == TimeEvoMethod.BDF:
            np.testing.assert_allclose(found.flatten(), reference, rtol=4e-2, atol=2e-3)
        elif method == TimeEvoMethod.RK23:
            np.testing.assert_allclose(found.flatten(), reference, rtol=2e-3, atol=5e-6)
        else:
            np.testing.assert_allclose(found.flatten(), reference, rtol=1e-6, atol=1e-6)
    else:
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
