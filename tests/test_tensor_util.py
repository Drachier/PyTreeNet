import unittest
import numpy as np

import pytreenet as ptn
from pytreenet.tensor_util import _determine_tensor_shape


class TestTreeTensorNetwork(unittest.TestCase):
    def setUp(self):
        self.tensor1 = ptn.crandn((2, 3, 4, 5))
        self.output_legs = (1, 3)
        self.input_legs = (0, 2)

        self.tensor2 = ptn.crandn((32, 22, 14, 16))

    def test_matricization(self):
        matrix = ptn.tensor_matricization(
            self.tensor1, self.output_legs, self.input_legs)

        self.assertEqual(matrix.shape, (3*5, 2*4))

    def test_determine_tensor_shape(self):
        matrix = ptn.tensor_matricization(
            self.tensor1, self.output_legs, self.input_legs)
        old_shape = self.tensor1.shape

        new_output_shape = _determine_tensor_shape(
            old_shape, matrix, self.output_legs)
        reference_output_shape = (3, 5, 8)
        self.assertEqual(new_output_shape, reference_output_shape)

        new_input_shape = _determine_tensor_shape(
            old_shape, matrix, self.input_legs, output=False)
        reference_input_shape = (15, 2, 4)
        self.assertEqual(new_input_shape, reference_input_shape)

    def test_compute_transfer_tensor(self):
        X, _, _ = ptn.pauli_matrices()
        I = np.eye(2)

        transfer_tensor = ptn.compute_transfer_tensor(X, 0)
        self.assertTrue(np.allclose(I, transfer_tensor))
        transfer_tensor = ptn.compute_transfer_tensor(X, 1)
        self.assertTrue(np.allclose(I, transfer_tensor))

    def test_tensor_qr_decomposition(self):
        q, r = ptn.tensor_qr_decomposition(
            self.tensor1, self.output_legs, self.input_legs)

        self.assertEqual(q.shape[-1], r.shape[0])
        tensor_shape = self.tensor1.shape
        self.assertEqual(q.shape[0:-1], (tensor_shape[1], tensor_shape[3]))
        self.assertEqual(r.shape[1:], (tensor_shape[0], tensor_shape[2]))

        recontracted_tensor = np.einsum("ijk,klm->limj", q, r)
        self.assertTrue(np.allclose(recontracted_tensor, self.tensor1))

        # q should be orthonormal
        connection_dimension = q.shape[-1]
        identity = np.eye(connection_dimension)
        transfer_tensor = ptn.compute_transfer_tensor(q, (0, 1))
        transfer_matrix = np.reshape(
            transfer_tensor, (connection_dimension, connection_dimension))
        self.assertTrue(np.allclose(identity, transfer_matrix))

    def test_tensor_svd(self):
        u, s, vh = ptn.tensor_svd(
            self.tensor1, self.output_legs, self.input_legs)

        self.assertEqual(u.shape[-1], len(s))
        self.assertEqual(vh.shape[0], len(s))
        tensor_shape = self.tensor1.shape
        self.assertEqual(u.shape[0:-1], (tensor_shape[1], tensor_shape[3]))
        self.assertEqual(vh.shape[1:], (tensor_shape[0], tensor_shape[2]))

        # We should be able to reconstruct the tensor.
        us = np.tensordot(u, np.diag(s), axes=(-1, 0))
        usvh = np.tensordot(us, vh, axes=(-1, 0))
        correct_tensor = self.tensor1.transpose([1, 3, 0, 2])

        self.assertTrue(np.allclose(correct_tensor, usvh))

    def test_check_truncation_parameters(self):
        self.assertRaises(TypeError, ptn.check_truncation_parameters,
                          max_bond_dim=1.3,  rel_tol=0.01, total_tol=1e-15)

        self.assertRaises(ValueError, ptn.check_truncation_parameters,
                          max_bond_dim=-100,  rel_tol=0.01, total_tol=1e-15)

        self.assertRaises(ValueError, ptn.check_truncation_parameters,
                          max_bond_dim=100,  rel_tol=-2.0, total_tol=1e-15)

        self.assertRaises(ValueError, ptn.check_truncation_parameters,
                          max_bond_dim=100,  rel_tol=0.01, total_tol=-1)

    def test_truncated_tensor_svd(self):
        truncation_parameter_list = (
            (15, 0.01, 1e-15), (200, 0.9, 1e-15), (200, 0.01, 35))

        for parameters in truncation_parameter_list:
            max_bond_dim = parameters[0]
            rel_tol = parameters[1]
            total_tol = parameters[2]

            u, s, vh = ptn.truncated_tensor_svd(self.tensor2,
                                                self.output_legs, self.input_legs,
                                                max_bond_dim=max_bond_dim,
                                                rel_tol=rel_tol, total_tol=total_tol)
            for singular_value in s:

                self.assertTrue(singular_value / s[0] >= rel_tol)
                self.assertTrue(singular_value >= total_tol)

            self.assertTrue(len(s) <= max_bond_dim)

            self.assertEqual(u.shape[-1], len(s))
            self.assertEqual(vh.shape[0], len(s))
            tensor_shape = self.tensor2.shape
            self.assertEqual(u.shape[0:-1], (tensor_shape[1], tensor_shape[3]))
            self.assertEqual(vh.shape[1:], (tensor_shape[0], tensor_shape[2]))


if __name__ == "__main__":
    unittest.main()
