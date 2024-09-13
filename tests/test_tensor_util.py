import unittest

from numpy import allclose, eye

from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.util import (tensor_matricization,
                            compute_transfer_tensor)
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

if __name__ == "__main__":
    unittest.main()
