"""
This module contains unit tests for common operators in PyTreeNet.
"""

import unittest

import numpy as np

import pytreenet as ptn
from pytreenet.random import random_hermitian_matrix

class TestCommonOperators(unittest.TestCase):

    def test_pauli_matrices_arrays(self):
        X, Y, Z = ptn.pauli_matrices()
        self.assertTrue(isinstance(X,np.ndarray))
        self.assertTrue(np.allclose(
                         np.asarray([[0,1],[1,0]], dtype=complex),
                         X))
        self.assertTrue(np.allclose(
                         np.asarray([[0,-1j],[1j,0]]),
                         Y))
        self.assertTrue(np.allclose(
                         np.asarray([[1,0],[0,-1]], dtype=complex),
                         Z))

    def test_spin_jumps_arrays(self):
        s_plus, s_minus = ptn.spin_jumps()
        self.assertTrue(isinstance(s_plus,np.ndarray))
        s_plus_correct = np.asarray([[0,1],[0,0]], dtype=complex)
        s_minus_correct = np.asarray([[0,0],[1,0]], dtype=complex)
        np.testing.assert_array_almost_equal(s_plus, s_plus_correct)
        np.testing.assert_array_almost_equal(s_minus, s_minus_correct)

    def test_bosonic_operators_errors(self):
        # Error for negative dimension
        self.assertRaises(ValueError, ptn.bosonic_operators, -3)

        # Error for zero dimension
        self.assertRaises(ValueError, ptn.bosonic_operators, 0)

    def test_bosonic_operators(self):
        # dim = 1
        creation_op, annihilation_op, number_op = ptn.bosonic_operators(1)
        self.assertTrue(np.allclose(np.asarray([0]),
                                    creation_op))
        self.assertTrue(np.allclose(np.asarray([0]),
                                    annihilation_op))
        self.assertTrue(np.allclose(np.asarray([0]),
                                    number_op))
        
        # dim = 2
        creation_op, annihilation_op, number_op = ptn.bosonic_operators(2)
        self.assertTrue(np.allclose(np.asarray([[0,0],
                                                [1,0]]),
                                    creation_op))
        self.assertTrue(np.allclose(np.asarray([[0,1],
                                                [0,0]]),
                                    annihilation_op))
        self.assertTrue(np.allclose(np.asarray([[0,0],
                                                [0,1]]),
                                    number_op))
        
        # dim = 3
        creation_op, annihilation_op, number_op = ptn.bosonic_operators(3)
        self.assertTrue(np.allclose(np.asarray([[0,0,0],
                                                [1,0,0],
                                                [0,np.sqrt(2),0]]),
                                    creation_op))
        self.assertTrue(np.allclose(np.asarray([[0,1,0],
                                                [0,0,np.sqrt(2)],
                                                [0,0,0]]),
                                    annihilation_op))
        self.assertTrue(np.allclose(np.asarray([[0,0,0],
                                                [0,1,0],
                                                [0,0,2]]),
                                    number_op))

    def test_swaps(self):
        # negative and zero dimension
        self.assertRaises(ValueError, ptn.swap_gate, -56)
        self.assertRaises(ValueError, ptn.swap_gate, 0)

        # dim = 1
        ref_swap = np.asarray([1], dtype=complex)
        self.assertTrue(np.allclose(ref_swap, ptn.swap_gate(dimension=1)))

        # dim = 2
        ref_swap = np.asarray([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1]], dtype=complex)
        self.assertTrue(np.allclose(ref_swap, ptn.swap_gate()))

    def test_hermitian_matrix(self):
        # negative and zero dimension
        self.assertRaises(ValueError, random_hermitian_matrix, -56)
        self.assertRaises(ValueError, random_hermitian_matrix, 0)

        # dim = 1
        number = random_hermitian_matrix(size=1)
        self.assertTrue(np.isreal(number))

        # dim = 2
        matrix = random_hermitian_matrix()
        self.assertTrue(np.allclose(matrix, matrix.T.conj()))

if __name__ == "__main__":
    unittest.main()
