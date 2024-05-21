import unittest

from copy import deepcopy

import numpy as np

import pytreenet as ptn

class TestTreeTensorNetworkStateSimple(unittest.TestCase):
    def setUp(self):
        # Initialise initial state
        self.initial_state = ptn.random_small_ttns()
        # Operators
        single_site_operator = ptn.TensorProduct({"root": ptn.crandn((2,2))})
        two_site_operator = ptn.TensorProduct({"c1": ptn.crandn((3,3)),
                                               "c2": ptn.crandn((4,4))})
        three_site_operator = ptn.TensorProduct({"root": ptn.crandn((2,2)),
                                                 "c1": ptn.crandn((3,3)),
                                                 "c2": ptn.crandn((4,4))})
        self.operators = [single_site_operator,
                          two_site_operator,
                          three_site_operator]

    def test_scalar_product(self):
        found_result = self.initial_state.scalar_product()

        state_vector = self.initial_state.completely_contract_tree(to_copy=True)[0]
        state_vector = state_vector.reshape(24)
        reference_result = state_vector.conj().T @ state_vector

        self.assertAlmostEqual(reference_result, found_result)

    def test_single_site_expectation_value_does_not_change_state(self):
        ref_ttns = deepcopy(self.initial_state)
        _ = self.initial_state.single_site_operator_expectation_value("root",
            self.operators[0]["root"])
        self.assertEqual(ref_ttns, self.initial_state)

    def test_single_site_expectation_value_no_canon_form(self):
        found_result = self.initial_state.single_site_operator_expectation_value("root",
            self.operators[0]["root"])

        state_vector = self.initial_state.completely_contract_tree(to_copy=True)[0]
        state_vector = state_vector.reshape(24)
        op1 = np.kron(np.kron(self.operators[0]["root"], np.eye(3)), np.eye(4))
        reference_result = state_vector.conj().T @ op1 @ state_vector

        self.assertTrue(np.allclose(reference_result, found_result))

    def test_single_site_expectation_value_canon_form(self):
        self.initial_state.canonical_form("root")
        found_result = self.initial_state.single_site_operator_expectation_value("root",
            self.operators[0]["root"])

        state_vector = self.initial_state.completely_contract_tree(to_copy=True)[0]
        state_vector = state_vector.reshape(24)
        op1 = np.kron(np.kron(self.operators[0]["root"], np.eye(3)), np.eye(4))
        reference_result = state_vector.conj().T @ op1.T @ state_vector

        self.assertTrue(np.allclose(reference_result, found_result))    

    def test_operator_expectation_value_single_site(self):
        found_result = self.initial_state.operator_expectation_value(self.operators[0])

        state_vector = self.initial_state.completely_contract_tree(to_copy=True)[0]
        state_vector = state_vector.reshape(24)
        op1 = np.kron(np.kron(self.operators[0]["root"], np.eye(3)), np.eye(4))
        reference_result = state_vector.conj().T @ op1 @ state_vector

        self.assertTrue(np.allclose(reference_result, found_result))

    def test_operator_expectation_value_full_tree(self):
        found_result = self.initial_state.operator_expectation_value(self.operators[2])

        state_vector = self.initial_state.completely_contract_tree(to_copy=True)[0]
        state_vector = state_vector.reshape(24)
        op1 = self.operators[2].into_operator().to_matrix().operator
        reference_result = state_vector.conj().T @ op1 @ state_vector

        self.assertAlmostEqual(reference_result, found_result)

    def test_operator_expectation_value_non_neighbour_sites(self):
        found_result = self.initial_state.operator_expectation_value(self.operators[1])

        state_vector = self.initial_state.completely_contract_tree(to_copy=True)[0]
        state_vector = state_vector.reshape(24)
        op1 = np.kron(np.eye(2), self.operators[1].into_operator().to_matrix().operator)
        reference_result = state_vector.conj().T @ op1 @ state_vector

        self.assertAlmostEqual(reference_result, found_result)


if __name__ == "__main__":
    unittest.main()
