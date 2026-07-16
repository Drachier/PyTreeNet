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
                                             MagnusConfig)
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

if __name__ == "__main__":
    unittest.main()