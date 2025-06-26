"""
This module contains unit tests for the TreeTensorNetworkState class.
"""
import unittest

from copy import deepcopy

import numpy as np

import pytreenet as ptn
from pytreenet.random import random_small_ttns, crandn

class TestTreeTensorNetworkStateSimple(unittest.TestCase):
    """
    Test methods on a TTNS with a simple tree structure.
    """
    def setUp(self):
        # Initialise initial state
        self.initial_state = random_small_ttns()
        # Operators
        single_site_operator = ptn.TensorProduct({"root": crandn((2,2))})
        two_site_operator = ptn.TensorProduct({"c1": crandn((3,3)),
                                               "c2": crandn((4,4))})
        three_site_operator = ptn.TensorProduct({"root": crandn((2,2)),
                                                 "c1": crandn((3,3)),
                                                 "c2": crandn((4,4))})
        self.operators = [single_site_operator,
                          two_site_operator,
                          three_site_operator]

    def test_scalar_product(self):
        """Test the scalar product of the TTNS with itself."""
        found_result = self.initial_state.scalar_product()

        state_vector = self.initial_state.completely_contract_tree(to_copy=True)[0]
        state_vector = state_vector.reshape(24)
        reference_result = state_vector.conj().T @ state_vector

        self.assertAlmostEqual(reference_result, found_result)

    def test_scalar_product_other(self):
        """Test the scalar product of the TTNS with another TTNS."""
        other = random_small_ttns()
        found_result = self.initial_state.scalar_product(other)

        state_vector = self.initial_state.completely_contract_tree(to_copy=True)[0]
        state_vector = state_vector.reshape(24)
        other_vector = other.completely_contract_tree(to_copy=True)[0]
        other_vector = other_vector.reshape(24)
        reference_result = other_vector.conj().T @ state_vector

        self.assertAlmostEqual(reference_result, found_result)

    def test_single_site_expectation_value_does_not_change_state(self):
        """Test that the single site expectation value does not change the state"""
        ref_ttns = deepcopy(self.initial_state)
        _ = self.initial_state.single_site_operator_expectation_value("root",
            self.operators[0]["root"])
        self.assertEqual(ref_ttns, self.initial_state)

    def test_single_site_expectation_value_no_canon_form(self):
        """Test the single site expectation value without canonical form."""
        found_result = self.initial_state.single_site_operator_expectation_value("root",
            self.operators[0]["root"])

        state_vector = self.initial_state.completely_contract_tree(to_copy=True)[0]
        state_vector = state_vector.reshape(24)
        op1 = np.kron(np.kron(self.operators[0]["root"], np.eye(3)), np.eye(4))
        reference_result = state_vector.conj().T @ op1 @ state_vector

        self.assertTrue(np.allclose(reference_result, found_result))

    def test_single_site_expectation_value_canon_form(self):
        """Test the single site expectation value with canonical form."""
        self.initial_state.canonical_form("root")
        found_result = self.initial_state.single_site_operator_expectation_value("root",
            self.operators[0]["root"])

        state_vector = self.initial_state.completely_contract_tree(to_copy=True)[0]
        state_vector = state_vector.reshape(24)
        op1 = np.kron(np.kron(self.operators[0]["root"], np.eye(3)), np.eye(4))
        reference_result = state_vector.conj().T @ op1 @ state_vector

        self.assertTrue(np.allclose(reference_result, found_result))

    def test_operator_expectation_value_single_site(self):
        """Test the operator expectation value for a single site."""
        found_result = self.initial_state.operator_expectation_value(self.operators[0])

        state_vector = self.initial_state.completely_contract_tree(to_copy=True)[0]
        state_vector = state_vector.reshape(24)
        op1 = np.kron(np.kron(self.operators[0]["root"], np.eye(3)), np.eye(4))
        reference_result = state_vector.conj().T @ op1 @ state_vector

        self.assertTrue(np.allclose(reference_result, found_result))

    def test_operator_expectation_value_full_tree(self):
        """
        Test the operator expectation value where the operator is non trivial
        on the full tree.
        """
        found_result = self.initial_state.operator_expectation_value(self.operators[2])

        state_vector = self.initial_state.completely_contract_tree(to_copy=True)[0]
        state_vector = state_vector.reshape(24)
        op1 = self.operators[2].into_operator().to_matrix().operator
        reference_result = state_vector.conj().T @ op1 @ state_vector

        self.assertAlmostEqual(reference_result, found_result)

    def test_operator_expectation_value_non_neighbour_sites(self):
        """
        Test the operator expectation value where the operator is non trivial
        on non-neighbour sites.
        """
        found_result = self.initial_state.operator_expectation_value(self.operators[1])

        state_vector = self.initial_state.completely_contract_tree(to_copy=True)[0]
        state_vector = state_vector.reshape(24)
        op1 = np.kron(np.eye(2), self.operators[1].into_operator().to_matrix().operator)
        reference_result = state_vector.conj().T @ op1 @ state_vector

        self.assertAlmostEqual(reference_result, found_result)

    def test_multi_single_site_expectation_value(self):
        """Test the multi single site expectation value."""
        found_result = self.initial_state.multi_single_site_expectation_value(
            self.operators[2])
        state_vector, _ = self.initial_state.to_vector(to_copy=True)
        op_root = np.kron(self.operators[2]["root"], np.eye(3 * 4))
        ref_root = state_vector.conj().T @ op_root @ state_vector
        self.assertTrue(np.allclose(ref_root, found_result["root"][0]))
        c1_op = np.kron(np.eye(2), np.kron(self.operators[2]["c1"], np.eye(4)))
        ref_c1 = state_vector.conj().T @ c1_op @ state_vector
        self.assertTrue(np.allclose(ref_c1, found_result["c1"][0]))
        c2_op = np.kron(np.eye(2 * 3), self.operators[2]["c2"])
        ref_c2 = state_vector.conj().T @ c2_op @ state_vector
        self.assertTrue(np.allclose(ref_c2, found_result["c2"][0]))

if __name__ == "__main__":
    unittest.main()
