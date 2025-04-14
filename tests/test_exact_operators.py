
from unittest import TestCase, main as unitmain
from copy import deepcopy

from numpy import eye, asarray, zeros, kron, allclose, ndarray

from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.random import crandn

from pytreenet.operators.exact_operators import (exact_ising_hamiltonian,
                                                 exact_single_site_operator,
                                                 assert_square_matrix,
                                                 flipped_exact_ising_hamiltonian,
                                                 exact_local_operators,
                                                 exact_zero_state,
                                                 exact_vectorised_operator,
                                                 exact_state_to_density_matrix,
                                                 _hamiltonian_part,
                                                 _jump_operator_terms,
                                                 exact_lindbladian)

class TestAssertSquareMatrix(TestCase):
    """
    Test the assertion of a matrix being square.
    """

    def test_pass(self):
        """
        Test the assertion of a square matrix.
        """
        matrix = crandn((2,2))
        assert_square_matrix(matrix)

    def test_pass_different_dim(self):
        """
        Test the assertion of a square matrix with different dimensions.
        """
        matrix = crandn((3,3))
        assert_square_matrix(matrix)

    def test_fail_non_square(self):
        """
        Test the assertion of a non-square matrix.
        """
        matrix = crandn((2,3))
        self.assertRaises(AssertionError, assert_square_matrix, matrix)

    def test_fail_vector(self):
        """
        Test the assertion of a vector.
        """
        matrix = crandn(2)
        self.assertRaises(AssertionError, assert_square_matrix, matrix)

    def test_fail_high_dim_tensor(self):
        """
        Test the assertion of a high dimensional tensor.
        """
        matrix = crandn((2,2,2))
        self.assertRaises(AssertionError, assert_square_matrix, matrix)

class TestExactSingleSiteOperator(TestCase):
    """
    Test the creation of an exact single site operator.
    """

    def test_assert_fail(self):
        """
        Test the assertion of a too large index
        """
        num_sites = 1
        local_operator = crandn(2,2)
        self.assertRaises(AssertionError, exact_single_site_operator,
                          local_operator,
                          num_sites,
                          num_sites)

    def test_single_site(self):
        """
        Test the creation of a single site operator for a single site.
        """
        num_sites = 1
        local_operator = crandn(2,2)
        found = exact_single_site_operator(local_operator,
                                           0,
                                           num_sites)
        correct = local_operator
        self.assertTrue(allclose(correct, found))

    def test_two_sites_at_0(self):
        """
        Tests the creation of a single site operator for two sites, where the
        operator is applied at site 0. 
        """
        num_sites = 2
        site_index = 0
        local_operator = crandn(2,2)
        correct = kron(local_operator, eye(2))
        found = exact_single_site_operator(local_operator,
                                           site_index,
                                           num_sites)
        self.assertTrue(allclose(correct, found))

    def test_two_sites_at_1(self):
        """
        Tests the creation of a single site operator for two sites, where the
        operator is applied at site 1. 
        """
        num_sites = 2
        site_index = 1
        local_operator = crandn(2,2)
        correct = kron(eye(2), local_operator)
        found = exact_single_site_operator(local_operator,
                                           site_index,
                                           num_sites)
        self.assertTrue(allclose(correct, found))

    def test_three_sites_at_0(self):
        """
        Tests the creation of a single site operator for three sites, where the
        operator is applied at site 0. 
        """
        num_sites = 3
        site_index = 0
        local_operator = crandn(2,2)
        correct = kron(local_operator, eye(4))
        found = exact_single_site_operator(local_operator,
                                           site_index,
                                           num_sites)
        self.assertTrue(allclose(correct, found))

    def test_three_sites_at_1(self):
        """
        Tests the creation of a single site operator for three sites, where the
        operator is applied at site 1. 
        """
        num_sites = 3
        site_index = 1
        local_operator = crandn(2,2)
        correct = kron(eye(2), kron(local_operator, eye(2)))
        found = exact_single_site_operator(local_operator,
                                           site_index,
                                           num_sites)
        self.assertTrue(allclose(correct, found))

    def test_three_sites_at_2(self):
        """
        Tests the creation of a single site operator for three sites, where the
        operator is applied at site 2. 
        """
        num_sites = 3
        site_index = 2
        local_operator = crandn(2,2)
        correct = kron(eye(4), local_operator)
        found = exact_single_site_operator(local_operator,
                                           site_index,
                                           num_sites)
        self.assertTrue(allclose(correct, found))

class TestExactIsingHamiltonian(TestCase):
    """
    Test the creation of an exact Ising Hamiltonian.
    """

    def test_single_site(self):
        """
        Test the exact Ising Hamiltonian for a single site.
        """
        coupling_strength = 1.0
        g = 0.5
        num_sites = 1
        correct = -1* g * pauli_matrices()[2]
        found = exact_ising_hamiltonian(coupling_strength,
                                        g,
                                        num_sites)
        self.assertTrue(allclose(correct, found))

    def test_two_sites(self):
        """
        Test the exact Ising Hamiltonian for two sites.
        """
        coupling_strength = 1.0
        g = 0.5
        num_sites = 2
        identity = eye(2)
        sigma_x, _, sigma_z = pauli_matrices()
        single_site_terms = -1 * g * kron(identity, sigma_z)
        single_site_terms += -1 * g * kron(sigma_z, identity)
        two_site_terms = -1 * coupling_strength * kron(sigma_x, sigma_x)
        correct = single_site_terms + two_site_terms
        found = exact_ising_hamiltonian(coupling_strength,
                                        g,
                                        num_sites)
        self.assertTrue(allclose(correct, found))

    def test_three_sites(self):
        """
        Test the exact Ising Hamiltonian for three sites.
        """
        coupling_strength = 1.0
        g = 0.5
        num_sites = 3
        sigma_x, _, sigma_z = pauli_matrices()
        single_site_terms = -1 * g * kron(sigma_z, eye(4))
        single_site_terms += -1 * g * kron(eye(2), kron(sigma_z, eye(2)))
        single_site_terms += -1 * g * kron(eye(4), sigma_z)
        two_site_terms = -1 * coupling_strength * kron(sigma_x,
                                                       kron(sigma_x, eye(2)))
        two_site_terms += -1 * coupling_strength * kron(eye(2),
                                                        kron(sigma_x, sigma_x))
        correct = single_site_terms + two_site_terms
        found = exact_ising_hamiltonian(coupling_strength,
                                        g,
                                        num_sites)
        self.assertTrue(allclose(correct, found))

    def test_four_sites(self):
        """
        Test the exact Ising Hamiltonian for four sites.
        """
        coupling_strength = 1.0
        g = 0.5
        num_sites = 4
        sigma_x, _, sigma_z = pauli_matrices()
        single_site_terms = -1 * g * kron(sigma_z, eye(8))
        single_site_terms += -1 * g * kron(eye(2), kron(sigma_z, eye(4)))
        single_site_terms += -1 * g * kron(eye(4), kron(sigma_z, eye(2)))
        single_site_terms += -1 * g * kron(eye(8), sigma_z)
        two_site_terms = -1 * coupling_strength * kron(sigma_x,
                                                       kron(sigma_x, eye(4)))
        two_site_terms += -1 * coupling_strength * kron(eye(2),
                                                        kron(sigma_x,
                                                             kron(sigma_x, eye(2))))
        two_site_terms += -1 * coupling_strength * kron(eye(4),
                                                        kron(sigma_x, sigma_x))
        correct = single_site_terms + two_site_terms
        found = exact_ising_hamiltonian(coupling_strength,
                                        g,
                                        num_sites)
        self.assertTrue(allclose(correct, found))

class TestExactFlippedIsingHamiltonian(TestCase):
    """
    Test the creation of an exact flipped Ising Hamiltonian.
    """

    def test_single_site(self):
        """
        Test the exact Ising Hamiltonian for a single site.
        """
        coupling_strength = 1.0
        g = 0.5
        num_sites = 1
        correct = -1* g * pauli_matrices()[0]
        found = flipped_exact_ising_hamiltonian(coupling_strength,
                                                g,
                                                num_sites)
        self.assertTrue(allclose(correct, found))

    def test_four_sites(self):
        """
        Test the exact Ising Hamiltonian for four sites.
        """
        coupling_strength = 1.0
        g = 0.5
        num_sites = 4
        sigma_x, _, sigma_z = pauli_matrices()
        single_site_terms = -1 * g * kron(sigma_x, eye(8))
        single_site_terms += -1 * g * kron(eye(2), kron(sigma_x, eye(4)))
        single_site_terms += -1 * g * kron(eye(4), kron(sigma_x, eye(2)))
        single_site_terms += -1 * g * kron(eye(8), sigma_x)
        two_site_terms = -1 * coupling_strength * kron(sigma_z,
                                                       kron(sigma_z, eye(4)))
        two_site_terms += -1 * coupling_strength * kron(eye(2),
                                                        kron(sigma_z,
                                                             kron(sigma_z, eye(2))))
        two_site_terms += -1 * coupling_strength * kron(eye(4),
                                                        kron(sigma_z,
                                                             sigma_z))
        correct = single_site_terms + two_site_terms
        found = flipped_exact_ising_hamiltonian(coupling_strength,
                                                g,
                                                num_sites)
        self.assertTrue(allclose(correct, found))

class TestExactLocalOperators(TestCase):
    """
    Test the creation of exact local operators.
    """

    def _site_id_creation(self,
                         num_sites: int
                         ) -> list[str]:
        """
        Creates a list of site ids.
        """
        return [f"site{i}" for i in range(num_sites)]

    def check_equality(self,
                      correct: dict[str, ndarray],
                      found: dict[str, ndarray]
                      ) -> None:
        """
        Check the equality of two dictionaries.
        """
        self.assertEqual(correct.keys(), found.keys())
        for key in correct:
            self.assertTrue(allclose(correct[key], found[key]))

    def test_single_site(self):
        """
        Test the creation of exact local operators for a single site.
        """
        site_ids = self._site_id_creation(1)
        local_operator = crandn(2,2)
        correct = {site_ids[0]: deepcopy(local_operator)}
        found = exact_local_operators(site_ids,
                                      local_operator)
        self.check_equality(correct, found)

    def test_two_sites(self):
        """
        Test the creation of exact local operators for two sites.
        """
        site_ids = self._site_id_creation(2)
        local_operator = crandn(2,2)
        correct = {site_ids[0]: kron(local_operator, eye(2)),
                   site_ids[1]: kron(eye(2), local_operator)}
        found = exact_local_operators(site_ids,
                                      local_operator)
        self.check_equality(correct, found)

    def test_three_sites(self):
        """
        Test the creation of exact local operators for three sites.
        """
        site_ids = self._site_id_creation(3)
        local_operator = crandn(2,2)
        correct = {site_ids[0]: kron(local_operator, eye(4)),
                   site_ids[1]: kron(eye(2), kron(local_operator, eye(2))),
                   site_ids[2]: kron(eye(4), local_operator)}
        found = exact_local_operators(site_ids,
                                      local_operator)
        self.check_equality(correct, found)

    def test_four_sites(self):
        """
        Test the creation of exact local operators for four sites.
        """
        site_ids = self._site_id_creation(4)
        local_operator = crandn(2,2)
        correct = {site_ids[0]: kron(local_operator, eye(8)),
                   site_ids[1]: kron(eye(2), kron(local_operator, eye(4))),
                   site_ids[2]: kron(eye(4), kron(local_operator, eye(2))),
                   site_ids[3]: kron(eye(8), local_operator)}
        found = exact_local_operators(site_ids,
                                      local_operator)
        self.check_equality(correct, found)

class TestExactZeroState(TestCase):
    """
    Test the creation of the zero state as a full vector.
    """

    def test_one_site_qubit(self):
        """
        Test the creation of the exact zero state for a single site.
        """
        num_sites = 1
        dim = 2
        correct = zeros(dim)
        correct[0] = 1
        found = exact_zero_state(num_sites,dim)
        self.assertTrue(allclose(correct, found))

    def test_two_sites_qubit(self):
        """
        Test the creation of the exact zero state for two sites.
        """
        num_sites = 2
        dim = 2
        correct = zeros(4)
        correct[0] = 1
        found = exact_zero_state(num_sites,dim)
        self.assertTrue(allclose(correct, found))

    def test_three_sites_qubit(self):
        """
        Test the creation of the exact zero state for three sites.
        """
        num_sites = 3
        dim = 2
        correct = zeros(8)
        correct[0] = 1
        found = exact_zero_state(num_sites,dim)
        self.assertTrue(allclose(correct, found))

    def test_one_site_qutrit(self):
        """
        Test the creation of the exact zero state for a single site.
        """
        num_sites = 1
        dim = 3
        correct = zeros(dim)
        correct[0] = 1
        found = exact_zero_state(num_sites,dim)
        self.assertTrue(allclose(correct, found))

    def test_two_sites_qutrit(self):
        """
        Test the creation of the exact zero state for two sites.
        """
        num_sites = 2
        dim = 3
        correct = zeros(9)
        correct[0] = 1
        found = exact_zero_state(num_sites,dim)
        self.assertTrue(allclose(correct, found))

    def test_three_sites_qutrit(self):
        """
        Teast the creation of the exact zero state for three sites.
        """
        num_sites = 3
        dim = 3
        correct = zeros(27)
        correct[0] = 1
        found = exact_zero_state(num_sites,dim)
        self.assertTrue(allclose(correct, found))

class TestExactVectorisedOperator(TestCase):
    """
    Test the vectorisation of an operator.
    """

    def test_exact_vectorised_operator_known(self):
        """
        Test the vectorisation for a known matrix.
        """
        matrix = asarray([[1, 2], [3, 4]])
        correct = asarray([[1, 2, 3, 4]])
        found = exact_vectorised_operator(matrix)
        self.assertTrue(allclose(correct, found))

    def test_exact_vectorised_operator_random(self):
        """
        Test the vectorisation for a random matrix.
        """
        matrix = crandn((2,2))
        correct = matrix.reshape(4)
        found = exact_vectorised_operator(matrix)
        self.assertTrue(allclose(correct, found))

class TestExactStateToDensityMatrix(TestCase):
    """
    Test the conversion of a state vector to a density matrix.
    """

    def test_zero_state(self):
        """
        Test the conversion of the zero state to a density matrix.
        """
        num_sites = 1
        dim = 2
        state = exact_zero_state(num_sites, dim)
        correct = asarray([[1, 0], [0, 0]])
        found = exact_state_to_density_matrix(state)
        self.assertTrue(allclose(correct, found))

    def test_superposition_state(self):
        """
        Test the conversion of a superposition state to a density matrix.
        """
        dim = 2
        state = asarray([1, 1])/dim**0.5
        correct = asarray([[1, 1], [1, 1]])/dim
        found = exact_state_to_density_matrix(state)
        self.assertTrue(allclose(correct, found))

class TestExactLindbladian(TestCase):
    """
    Test the construction of an exact Lindbladian superoperator.
    """

    def test_ham_part_known(self):
        """
        Test the Hamiltonian part of the Lindbladian for a known Hamiltonian.
        """
        hamiltonian = asarray([[1, 3+2j], [3-2j, 4]])
        correct = asarray([[0, -3+2j, 3+2j, 0],
                           [-3-2j, -3, 0, 3+2j],
                           [3-2j, 0, 3, -3+2j],
                           [0, 3-2j, -3-2j, 0]])
        found = _hamiltonian_part(hamiltonian)
        self.assertTrue(allclose(correct, found))

    def test_ham_part_random(self):
        """
        Test the Hamiltonian part of the Lindbladian for a random Hamiltonian.
        """
        hamiltonian = crandn((2,2))
        identity = eye(2)
        correct = kron(hamiltonian,identity) - kron(identity, hamiltonian.T)
        found = _hamiltonian_part(hamiltonian)
        self.assertTrue(allclose(correct, found))

    def test_jump_operator_terms_known(self):
        """
        Test the jump operator terms of the Lindbladian for a known jump operator.
        """
        jump_operator = asarray([[0,3+2j], [0,0]])
        correct = asarray([[0, 0, 0, 13j],
                           [0,13j / 2, 0, 0],
                           [0,0, -13j / 2,0],
                           [0,0,0,0]])
        found = _jump_operator_terms(jump_operator)
        self.assertTrue(allclose(correct, found))

    def test_jump_operator_terms_tuple(self):
        """
        Test the jump operator terms of the Lindbladian for a known jump operator
        given as a tuple.
        """
        jump_operator = asarray([[0,(3+2j) / 2], [0,0]])
        coeff = 2
        correct = asarray([[0, 0, 0, 13j],
                           [0,13j / 2, 0, 0],
                           [0,0, -13j / 2,0],
                           [0,0,0,0]])
        found = _jump_operator_terms((coeff,jump_operator))
        self.assertTrue(allclose(correct, found))

    def test_lindbladian_single_operator(self):
        """
        Test the Lindbladian with a single jump operator.
        """
        hamiltonian = asarray([[1, 3+2j], [3-2j, 4]])
        jump_operator = asarray([[0,3+2j], [0,0]])
        correct = _hamiltonian_part(hamiltonian) + _jump_operator_terms(jump_operator)
        found = exact_lindbladian(hamiltonian, [jump_operator])
        self.assertTrue(allclose(correct, found))

    def test_lindbladian_multiple_operators(self):
        """
        Test the Lindbladian construction with three operators.
        """
        hamiltonian = crandn((2,2))
        jump_operators = [crandn((2,2)) for _ in range(3)]
        correct = _hamiltonian_part(hamiltonian)
        correct += _jump_operator_terms(jump_operators[0])
        correct += _jump_operator_terms(jump_operators[1])
        correct += _jump_operator_terms(jump_operators[2])
        found = exact_lindbladian(hamiltonian, jump_operators)
        self.assertTrue(allclose(correct, found))

if __name__ == "__main__":
    unitmain()
