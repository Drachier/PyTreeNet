"""
Tests the Magnus expansion implementation of a time evolution.
"""
import unittest

from scipy.integrate import solve_ivp
from scipy.linalg import expm
import numpy as np
import numpy.testing as npt

from pytreenet.time_evolution.magnus import (Magnus,
                                             ChebyshevMagnus,
                                             MagnusConfig,
                                             _magnus_integrand_order1,
                                             _magnus_integrand_order2,
                                             _magnus_integrand_order3_1,
                                             _magnus_integrand_order3_2)
from pytreenet.special_ttn.mps import MatrixProductState
from pytreenet.operators.models.two_site_model import IsingModel
from pytreenet.ttno.ttno_class import TTNO
from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.core.addition.linear_combination import (LinCombParams,
                                                        ApplicationMethod,
                                                        AdditionMethod)

class TestMagnus(unittest.TestCase):
    
    def setUp(self):
        self.control_function = lambda t: 2 * t + 3
        self.mps = MatrixProductState.constant_product_state(0,2,3,
                                                             root_site=1)
        self.time_step_size = 0.1
        self.magnus_config = MagnusConfig(c_order=3,
                                          m_order=3)
        ising_model = IsingModel(ext_magn=0.0)
        ham = ising_model.generate_chain_model(3)
        self.time_indep_ttno = TTNO.from_hamiltonian(ham,
                                                     self.mps)
        ising_model = IsingModel(factor=0.0,
                                 ext_magn=1.0)
        ham = ising_model.generate_chain_model(3)
        self.time_dep_ttno = TTNO.from_hamiltonian(ham,
                                              self.mps)
        self.operators = {"site"+str(i): TensorProduct({"site"+str(i): pauli_matrices()[0]})
                          for i in range(3)}
        self.lincomb_params = LinCombParams(ApplicationMethod.DIRECT,
                                            AdditionMethod.DIRECT)

    def test_one_time_step_no_time_dependence(self):
        """
        Test one time step of the Magnus expansion against a numerical
        solution of the dense computation. The control function is
        constant, so there is no time dependence.
        """
        control_function = lambda t: 1.0
        magnus = Magnus(self.mps,
                        self.time_indep_ttno,
                        self.time_dep_ttno,
                        control_function,
                        self.time_step_size,
                        self.time_step_size,
                        self.operators,
                        self.lincomb_params,
                        config=MagnusConfig(c_order=4, m_order=3))
        magnus.run_one_time_step()

        # Computing the reference solution
        num_sites = 3
        time_indep_ham = np.zeros((2**num_sites, 2**num_sites), dtype=complex)
        term1 = np.kron(np.kron(pauli_matrices()[0], pauli_matrices()[0]),
                        np.eye(2))
        term2 = np.kron(np.kron(np.eye(2), pauli_matrices()[0]),
                        pauli_matrices()[0])
        time_indep_ham += term1 + term2
        time_dep_ham = np.zeros((2**num_sites, 2**num_sites), dtype=complex)
        term1 = np.kron(np.kron(pauli_matrices()[2], np.eye(2)),
                        np.eye(2))
        term2 = np.kron(np.kron(np.eye(2), pauli_matrices()[2]),
                        np.eye(2))
        term3 = np.kron(np.kron(np.eye(2), np.eye(2)),
                        pauli_matrices()[2])
        time_dep_ham += term1 + term2 + term3
        total_ham = time_indep_ham + control_function(0) * time_dep_ham
        initial_state = np.zeros(2**num_sites, dtype=complex)
        initial_state[0] = 1.0
        reference_state = expm(-1j * total_ham * self.time_step_size) @ initial_state
        # Comparing the results
        magnus_state = magnus.state.completely_contract_tree(order=["site0", "site1", "site2"])[0]
        magnus_state = magnus_state.reshape(-1)
        print(np.linalg.norm(magnus_state))
        print(np.linalg.norm(reference_state))
        npt.assert_allclose(magnus_state, reference_state)

    def test_one_time_step_with_time_dependence(self):
        """
        Test one time step of the Magnus expansion against a numerical
        solution of the dense computation. The control function is
        non-constant, so there is time dependence.
        """
        control_function = self.control_function
        magnus = Magnus(self.mps,
                        self.time_indep_ttno,
                        self.time_dep_ttno,
                        control_function,
                        self.time_step_size,
                        self.time_step_size,
                        self.operators,
                        self.lincomb_params,
                        config=MagnusConfig(c_order=4, m_order=3))
        magnus.run_one_time_step()

        # Computing the reference solution
        num_sites = 3
        time_indep_ham = np.zeros((2**num_sites, 2**num_sites), dtype=complex)
        term1 = np.kron(np.kron(pauli_matrices()[0], pauli_matrices()[0]),
                        np.eye(2))
        term2 = np.kron(np.kron(np.eye(2), pauli_matrices()[0]),
                        pauli_matrices()[0])
        time_indep_ham += term1 + term2
        time_dep_ham = np.zeros((2**num_sites, 2**num_sites), dtype=complex)
        term1 = np.kron(np.kron(pauli_matrices()[2], np.eye(2)),
                        np.eye(2))
        term2 = np.kron(np.kron(np.eye(2), pauli_matrices()[2]),
                        np.eye(2))
        term3 = np.kron(np.kron(np.eye(2), np.eye(2)),
                        pauli_matrices()[2])
        time_dep_ham += term1 + term2 + term3
        def se_rhs(t, y):
            return -1j * (time_indep_ham + control_function(t) * time_dep_ham) @ y
        initial_state = np.zeros(2**num_sites, dtype=complex)
        initial_state[0] = 1.0
        sol = solve_ivp(se_rhs, (0, self.time_step_size), initial_state,
                        method="RK45", rtol=1e-10, atol=1e-10)
        reference_state = sol.y[:, -1]
        # Comparing the results
        magnus_state = magnus.state.completely_contract_tree(order=["site0", "site1", "site2"])[0]
        magnus_state = magnus_state.reshape(-1)
        print(np.linalg.norm(magnus_state))
        print(np.linalg.norm(reference_state))
        npt.assert_allclose(magnus_state, reference_state)

class TestControlFunctionIntegrals(unittest.TestCase):
    """
    Tests the functions that compute the integrals of the control functions.
    """

    def setUp(self):
        self.cfs = [lambda t: t, lambda t: t**2+3, lambda t: np.sin(t)]
        self.time_step_size = 0.1
        self.magnusses: list[Magnus] = []
        for cf in self.cfs:
            mps = MatrixProductState.constant_product_state(0,2,3,
                                                                root_site=1)
            magnus_config = MagnusConfig(c_order=3,
                                            m_order=3)
            ising_model = IsingModel(ext_magn=0.0)
            ham = ising_model.generate_chain_model(3)
            time_indep_ttno = TTNO.from_hamiltonian(ham,
                                                        mps)
            ising_model = IsingModel(factor=0.0,
                                    ext_magn=1.0)
            ham = ising_model.generate_chain_model(3)
            time_dep_ttno = TTNO.from_hamiltonian(ham,
                                                mps)
            operators = {"site"+str(i): TensorProduct({"site"+str(i): pauli_matrices()[0]})
                            for i in range(3)}
            lincomb_params = LinCombParams(ApplicationMethod.DIRECT,
                                                AdditionMethod.DIRECT)
            magnus = Magnus(mps,
                            time_indep_ttno,
                            time_dep_ttno,
                            cf,
                            self.time_step_size,
                            1.0,
                            operators,
                            lincomb_params,
                            config=magnus_config)
            self.magnusses.append(magnus)

    def test_integration_limits_zerotime(self):
        """
        Test the integration limits of the control function integrals at
        time t=0.
        """
        for magnus in self.magnusses:
            lower_limit, upper_limit = magnus._integration_limits()
            self.assertAlmostEqual(lower_limit, 0.0)
            self.assertAlmostEqual(upper_limit, self.time_step_size)

    def test_integration_limits_nonzerotime(self):
        """
        Test the integration limits of the control function integrals at
        time t>0.
        """
        for magnus in self.magnusses:
            magnus._current_time_step = 5
            lower_limit, upper_limit = magnus._integration_limits()
            self.assertAlmostEqual(lower_limit, 5 * self.time_step_size)
            self.assertAlmostEqual(upper_limit, 6 * self.time_step_size)

    def test_0th_integral(self):
        """
        Test the zeroth order integral of the control function.
        """
        for magnus in self.magnusses:
            integral = magnus._compute_control_function_integrals()
            correct_integral = -1j * self.time_step_size
            self.assertAlmostEqual(integral[0], correct_integral)

    def test_1st_integral(self):
        """
        Test the first order integral of the control function.
        """
        correct_integrals = [-1j * 0.5 * self.time_step_size**2,
                             -1j * (1/3) * self.time_step_size**3 - 1j * 3 * self.time_step_size,
                             -1j * (1 - np.cos(self.time_step_size))]
        for magnus, correct_integral in zip(self.magnusses, correct_integrals):
            integral = magnus._compute_control_function_integrals()
            self.assertAlmostEqual(integral[1], correct_integral)

    def test_2nd_integral(self):
        """
        Test the second order integral of the control function.
        """
        correct_integrals = [(1/12) * self.time_step_size**3,
                             (1/12) * self.time_step_size**4,
                             (-1/2) * (self.time_step_size + self.time_step_size * np.cos(self.time_step_size) - 2 * np.sin(self.time_step_size))]
        for magnus, correct_integral in zip(self.magnusses, correct_integrals):
            integral = magnus._compute_control_function_integrals()
            self.assertAlmostEqual(integral[2], correct_integral)

    def test_3_1_integral(self):
        """
        Test the third order integral of the control function (first variant).
        """
        correct_integrals = [0,
                             1j / 240 * self.time_step_size**5]
        for magnus, correct_integral in zip(self.magnusses[0:2], correct_integrals[0:2]):
            integral = magnus._compute_control_function_integrals()
            self.assertAlmostEqual(integral[3], correct_integral)

class TestChebyshevMagnus(unittest.TestCase):
    """
    Tests the helper class ChebyshevMagnus, which is used to compute the
    Chebyshev expansion of the Magnus expansion.
    """

    def test_invalid_init(self):
        """
        An invalid initialization of the ChebyshevMagnus class should raise a
        ValueError.
        """
        self.assertRaises(ValueError, ChebyshevMagnus, order=-1, prefactors=[1.0])
        self.assertRaises(ValueError, ChebyshevMagnus, order=0, prefactors=[])

    def test_0th_order(self):
        """
        Test the zeroth order Chebyshev expansion of the Magnus expansion.
        """
        correct = ChebyshevMagnus(order=0, prefactors=[1.0])
        found = ChebyshevMagnus.zeroth_order()
        self.assertEqual(correct, found)

    def test_1st_order(self):
        """
        Test the first order Chebyshev expansion of the Magnus expansion.
        """
        correct = ChebyshevMagnus(order=1, prefactors=[0.0, 1.0])
        found = ChebyshevMagnus.first_order()
        self.assertEqual(correct, found)

    def test_next_order_2nd(self):
        """
        Test the second order Chebyshev expansion of the Magnus expansion.
        """
        zeroth_order = ChebyshevMagnus.zeroth_order()
        first_order = ChebyshevMagnus.first_order()
        correct = ChebyshevMagnus(order=2, prefactors=[1.0, 0.0, 2.0])
        found = ChebyshevMagnus.next_order(first_order, zeroth_order)
        self.assertEqual(correct, found)

    def test_next_order_3rd(self):
        """
        Test the third order Chebyshev expansion of the Magnus expansion.
        """
        zeroth_order = ChebyshevMagnus.zeroth_order()
        first_order = ChebyshevMagnus.first_order()
        second_order = ChebyshevMagnus.next_order(first_order, zeroth_order)
        correct = ChebyshevMagnus(order=3, prefactors=[0.0, 3.0, 0.0, 4.0])
        found = ChebyshevMagnus.next_order(second_order, first_order)
        self.assertEqual(correct, found)

    def test_next_order_4th(self):
        """
        Test the fourth order Chebyshev expansion of the Magnus expansion.
        """
        zeroth_order = ChebyshevMagnus.zeroth_order()
        first_order = ChebyshevMagnus.first_order()
        second_order = ChebyshevMagnus.next_order(first_order, zeroth_order)
        third_order = ChebyshevMagnus.next_order(second_order, first_order)
        correct = ChebyshevMagnus(order=4, prefactors=[1.0, 0.0, 8.0, 0.0, 8.0])
        found = ChebyshevMagnus.next_order(third_order, second_order)
        self.assertEqual(correct, found)

    def test_next_order_non_follow(self):
        """
        Test the next order Chebyshev expansion of the Magnus expansion
        with non-following orders.
        """
        zeroth_order = ChebyshevMagnus.zeroth_order()
        first_order = ChebyshevMagnus.first_order()
        second_order = ChebyshevMagnus.next_order(first_order, zeroth_order)
        self.assertRaises(ValueError,
                          ChebyshevMagnus.next_order,
                          second_order, zeroth_order)
        
class TestMagnusIntegrandOrderFunctions(unittest.TestCase):
    """
    Tests the functions that compute the order of the Magnus integrand.
    """

    def setUp(self):
        self.cf1 = lambda t: t
        self.cf2 = lambda t: t**2+3
        self.cf3 = lambda t: np.sin(t)
        self.cfs = [self.cf1, self.cf2, self.cf3]

    def test_order1(self):
        """
        Test the first order Magnus integrand function.
        """
        for cf in self.cfs:
            vals = np.arange(0, 10, 0.1)
            found_vals = [_magnus_integrand_order1(v, cf) for v in vals]
            correct_vals = np.array([cf(v) for v in vals])
            npt.assert_array_almost_equal(correct_vals, found_vals)

    def test_order2(self):
        """
        Test the second order Magnus integrand function.
        """
        for cf in self.cfs:
            vals = np.arange(0, 10, 0.1)
            found_vals = [_magnus_integrand_order2(v1, v2, cf)
                          for v1, v2 in zip(vals, vals)]
            correct_vals = np.array([cf(v1) - cf(v2)
                                     for v1, v2 in zip(vals, vals)])
            npt.assert_array_almost_equal(correct_vals, found_vals)

    def test_order3_1(self):
        """
        Test the third order Magnus integrand function (first variant).
        """
        for cf in self.cfs:
            vals = np.arange(0, 10, 0.1)
            found_vals = [_magnus_integrand_order3_1(v1, v2, v3, cf)
                          for v1, v2, v3 in zip(vals, vals, vals)]
            correct_vals = np.array([cf(v1) - 2*cf(v2) + cf(v3)
                                     for v1, v2, v3 in zip(vals, vals, vals)])
            npt.assert_array_almost_equal(correct_vals, found_vals)

    def test_order3_2(self):
        """
        Test the third order Magnus integrand function (second variant).
        """
        for cf in self.cfs:
            vals = np.arange(0, 10, 0.1)
            found_vals = [_magnus_integrand_order3_2(v1, v2, v3, cf)
                          for v1, v2, v3 in zip(vals, vals, vals)]
            correct_vals = np.array([2*cf(v1)*cf(v3) - cf(v1)*cf(v2) - cf(v2)*cf(v3)
                                     for v1, v2, v3 in zip(vals, vals, vals)])
            npt.assert_array_almost_equal(correct_vals, found_vals)


if __name__ == "__main__":
    unittest.main()