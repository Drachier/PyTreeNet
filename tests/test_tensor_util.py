import unittest

from numpy import allclose, eye

from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.util.tensor_util import (tensor_matricization,
                                        compute_transfer_tensor,
                                        make_last_leg_first)
from pytreenet.random import crandn

class TestTensorUtilSimple(unittest.TestCase):
    def setUp(self):
        self.tensor1 = crandn((2, 3, 4, 5))
        self.output_legs = (1, 3)
        self.input_legs = (0, 2)

        self.tensor2 = crandn((32, 22, 14, 16))

    def test_matricization(self):
        matrix = tensor_matricization(self.tensor1,
                                          self.output_legs,
                                          self.input_legs)
        self.assertEqual(matrix.shape, (3*5, 2*4))

    def test_compute_transfer_tensor(self):
        X, _, _ = pauli_matrices()
        I = eye(2)

        transfer_tensor = compute_transfer_tensor(X, 0)
        self.assertTrue(allclose(I, transfer_tensor))
        transfer_tensor = compute_transfer_tensor(X, 1)
        self.assertTrue(allclose(I, transfer_tensor))

class Test_make_last_leg_first(unittest.TestCase):

    def test_make_last_leg_first_vector(self):
        vector = crandn(3)
        vector_reordered = make_last_leg_first(vector)
        self.assertTrue(allclose(vector_reordered, vector))

    def test_make_last_leg_first_matrix(self):
        matrix = crandn((3, 4))
        matrix_reordered = make_last_leg_first(matrix)
        self.assertTrue(allclose(matrix_reordered, matrix.T))

    def test_make_last_leg_first_3tensor(self):
        tensor = crandn((3, 4, 5))
        tensor_reordered = make_last_leg_first(tensor)
        self.assertTrue(allclose(tensor_reordered, tensor.transpose(2, 0, 1)))

    def test_make_last_leg_first_4tensor(self):
        tensor = crandn((3, 4, 5, 6))
        tensor_reordered = make_last_leg_first(tensor)
        self.assertTrue(allclose(tensor_reordered, tensor.transpose(3, 0, 1, 2)))

if __name__ == "__main__":
    unittest.main()
