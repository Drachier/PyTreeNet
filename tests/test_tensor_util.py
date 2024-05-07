import unittest
from copy import deepcopy

import numpy as np

import pytreenet as ptn
from pytreenet.util.tensor_splitting import _determine_tensor_shape


class TestTensorUtilSimple(unittest.TestCase):
    def setUp(self):
        self.tensor1 = ptn.crandn((2, 3, 4, 5))
        self.output_legs = (1, 3)
        self.input_legs = (0, 2)

        self.tensor2 = ptn.crandn((32, 22, 14, 16))

    def test_matricization(self):
        matrix = ptn.tensor_matricization(self.tensor1,
                                          self.output_legs,
                                          self.input_legs)
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
        u, s, vh = ptn.tensor_svd(self.tensor1,
                                  self.output_legs,
                                  self.input_legs)

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
        
        self.assertRaises(TypeError, ptn.SVDParameters,
                          max_bond_dim=1.3,  rel_tol=0.01, total_tol=1e-15)
        self.assertRaises(ValueError, ptn.SVDParameters,
                          max_bond_dim=-100,  rel_tol=0.01, total_tol=1e-15)
        self.assertRaises(ValueError, ptn.SVDParameters,
                          max_bond_dim=100,  rel_tol=-2.0, total_tol=1e-15)
        self.assertRaises(ValueError, ptn.SVDParameters,
                          max_bond_dim=100,  rel_tol=0.01, total_tol=-1)

    def test_truncated_tensor_svd(self):
        truncation_parameter_list = (
            (15, 0.01, 1e-15), (200, 0.9, 1e-15), (200, 0.01, 35))

        for parameters in truncation_parameter_list:
            svd_params = ptn.SVDParameters(*parameters)

            u, s, vh = ptn.truncated_tensor_svd(self.tensor2,
                                                self.output_legs, self.input_legs,
                                                svd_params=svd_params)
            for singular_value in s:

                self.assertTrue(singular_value / s[0] >= svd_params.rel_tol)
                self.assertTrue(singular_value >= svd_params.total_tol)

            self.assertTrue(len(s) <= svd_params.max_bond_dim)

            self.assertEqual(u.shape[-1], len(s))
            self.assertEqual(vh.shape[0], len(s))
            tensor_shape = self.tensor2.shape
            self.assertEqual(u.shape[0:-1], (tensor_shape[1], tensor_shape[3]))
            self.assertEqual(vh.shape[1:], (tensor_shape[0], tensor_shape[2]))

class TestTensorQRDecomp(unittest.TestCase):
    def setUp(self):
        self.tensor = ptn.crandn((4,5,2))

    def test_qr_reduced_q_legs_bigger(self):
        q, r = ptn.tensor_qr_decomposition(self.tensor, (0,2), (1, ))

        self.assertEqual((4,2,5),q.shape)
        self.assertEqual((5,5),r.shape)

        ref_tensor = np.transpose(self.tensor,(0,2,1))
        ref_tensor = np.reshape(ref_tensor,(8,5))
        ref_q, ref_r = np.linalg.qr(ref_tensor,mode="reduced")
        ref_q = np.reshape(ref_q, (4,2,5))

        self.assertTrue(np.allclose(ref_q, q))
        self.assertTrue(np.allclose(ref_r, r))

    def test_qr_reduced_r_legs_bigger(self):
        q, r = ptn.tensor_qr_decomposition(self.tensor, (0, ), (1,2))

        self.assertEqual((4,4),q.shape)
        self.assertEqual((4,5,2),r.shape)

        ref_tensor = np.reshape(self.tensor,(4,10))
        ref_q, ref_r = np.linalg.qr(ref_tensor,mode="reduced")
        ref_r = np.reshape(ref_r, (4,5,2))

        self.assertTrue(np.allclose(ref_q, q))
        self.assertTrue(np.allclose(ref_r, r))

    def test_qr_full_q_legs_bigger(self):
        q, r = ptn.tensor_qr_decomposition(self.tensor, (0,2), (1, ),
                                           ptn.SplitMode.FULL)

        self.assertEqual((4,2,8),q.shape)
        self.assertEqual((8,5),r.shape)

        ref_tensor = np.transpose(self.tensor,(0,2,1))
        ref_tensor = np.reshape(ref_tensor,(8,5))
        ref_q, ref_r = np.linalg.qr(ref_tensor,mode="complete")
        ref_q = np.reshape(ref_q, (4,2,8))

        self.assertTrue(np.allclose(ref_q, q))
        self.assertTrue(np.allclose(ref_r, r))

    def test_qr_full_r_legs_bigger(self):
        q, r = ptn.tensor_qr_decomposition(self.tensor, (1, ), (0,2),
                                           ptn.SplitMode.FULL)

        self.assertEqual((5,5),q.shape)
        self.assertEqual((5,4,2),r.shape)

        ref_tensor = np.transpose(self.tensor,(1,0,2))
        ref_tensor = np.reshape(ref_tensor,(5,8))
        ref_q, ref_r = np.linalg.qr(ref_tensor,mode="complete")
        ref_r = np.reshape(ref_r, (5,4,2))

        self.assertTrue(np.allclose(ref_q, q))
        self.assertTrue(np.allclose(ref_r, r))

    def test_qr_keep_q_legs_bigger(self):
        q, r = ptn.tensor_qr_decomposition(self.tensor, (0,2), (1, ),
                                           ptn.SplitMode.KEEP)

        self.assertEqual((4,2,5),q.shape)
        self.assertEqual((5,5),r.shape)

        ref_tensor = np.transpose(self.tensor,(0,2,1))
        ref_tensor = np.reshape(ref_tensor,(8,5))
        ref_q, ref_r = np.linalg.qr(ref_tensor,mode="reduced")
        ref_q = np.reshape(ref_q, (4,2,5))

        self.assertTrue(np.allclose(ref_q, q))
        self.assertTrue(np.allclose(ref_r, r))

    def test_qr_keep_r_legs_bigger(self):
        q, r = ptn.tensor_qr_decomposition(self.tensor, (1, ), (0,2),
                                           ptn.SplitMode.KEEP)

        self.assertEqual((5,8),q.shape)
        self.assertEqual((8,4,2),r.shape)

        ref_tensor = np.transpose(self.tensor,(1,0,2))
        ref_tensor = np.reshape(ref_tensor,(5,8))
        ref_q, ref_r = np.linalg.qr(ref_tensor,mode="reduced")
        ref_r = np.reshape(ref_r, (5,4,2))
        ref_q = np.pad(ref_q, [(0,0),(0,3)])
        ref_r = np.pad(ref_r, [(0,3),(0,0),(0,0)])

        self.assertTrue(np.allclose(ref_q, q))
        self.assertTrue(np.allclose(ref_r, r))

class TestSingularValueDecompositions(unittest.TestCase):
    def setUp(self):
        self.tensor = ptn.crandn((2,3,4,5))
        self.u_legs = (1,3)
        self.v_legs = (0,2)
        self.s_values = np.array([1.2,1,0.8,0.5,0.2,0.1,0.1,0.01])
        self.sum_s = np.sum(self.s_values)

    def test_svd_reduced(self):
        """
        Test the SVD of a tensor with REDUCED mode, i.e. the tensor legs
         of the resulting U and V tensors pointing to S have the minimal
         dimension possible.
        """
        reference_tensor = self.tensor.transpose((1,3,0,2))
        reference_tensor = reference_tensor.reshape((15,8))
        ref_u, ref_s, ref_v = np.linalg.svd(reference_tensor,
                                            full_matrices=False)
        ref_u = ref_u.reshape(3,5,8)
        ref_v = ref_v.reshape(8,2,4)
        u, s, vh = ptn.tensor_svd(self.tensor,self.u_legs,self.v_legs)
        self.assertTrue(np.allclose(ref_u,u))
        self.assertTrue(np.allclose(ref_s,s))
        self.assertTrue(np.allclose(ref_v,vh))

    def test_svd_full(self):
        """
        Test the SVD of a tensor in FULL mode, i.e. the tensor legs
         of the resulting U and V tensors pointing to S have the size of all
         other legs of the respective tensor taken together.
        """
        reference_tensor = self.tensor.transpose((1,3,0,2))
        reference_tensor = reference_tensor.reshape((15,8))
        ref_u, ref_s, ref_v = np.linalg.svd(reference_tensor,
                                            full_matrices=True)
        ref_u = ref_u.reshape(3,5,15)
        ref_v = ref_v.reshape(8,2,4)
        u, s, vh = ptn.tensor_svd(self.tensor,self.u_legs,self.v_legs,
                                  mode=ptn.SplitMode.FULL)
        self.assertTrue(np.allclose(ref_u,u))
        self.assertTrue(np.allclose(ref_s,s))
        self.assertTrue(np.allclose(ref_v,vh))

    def test_svd_keep(self):
        """
        Test the SVD of a tensor in KEEP mode, i.e. the tensor legs
         of the resulting U and V tensors pointing to S have the size of all
         other legs of the respective tensor taken together.
        """
        reference_tensor = self.tensor.transpose((1,3,0,2))
        reference_tensor = reference_tensor.reshape((15,8))
        ref_u, ref_s, ref_v = np.linalg.svd(reference_tensor,
                                            full_matrices=True)
        ref_u = ref_u.reshape(3,5,15)
        ref_v = ref_v.reshape(8,2,4)
        u, s, vh = ptn.tensor_svd(self.tensor,self.u_legs,self.v_legs,
                                  mode=ptn.SplitMode.KEEP)
        self.assertTrue(np.allclose(ref_u,u))
        self.assertTrue(np.allclose(ref_s,s))
        self.assertTrue(np.allclose(ref_v,vh))

    def test_renorm_singular_values(self):
        """
        Test the renormalisation of a random tensor of positive values which
         represent singular values against a truncated vector.
        """
        s = np.random.rand(10)
        s = np.array(list(reversed(np.sort(s))))
        s_new = s[:5]
        norm_old = np.sum(s)
        norm_new = np.sum(s_new)
        normed_s = s_new * (norm_old / norm_new)
        found_s = ptn.renormalise_singular_values(s,s_new)
        self.assertTrue(np.allclose(normed_s,found_s))

    def test_renorm_singular_value_equal_size(self):
        """
        Test the renormalisation of a random tensor of positive values which
         represent singular values against an equal vector.
        """
        s = np.random.rand(10)
        found_s = ptn.renormalise_singular_values(deepcopy(s),deepcopy(s))
        self.assertTrue(np.allclose(found_s,s))

    def test_truncate_singular_values_no_truncation(self):
        """
        Tests the truncation of singular values, however no truncation should
         happen given the parameters.
        """
        svd_params = ptn.SVDParameters(10,0,0)
        found_s, truncated_s = ptn.truncate_singular_values(deepcopy(self.s_values),
                                                            svd_params)
        self.assertTrue(np.allclose(self.s_values,found_s))
        self.assertTrue(np.allclose(np.array([]),truncated_s))

    def test_truncate_singular_values_max_bond_dim(self):
        """
        Tests the truncation of singular values caused by the maximum bond
         dimension given.
        """
        svd_params = ptn.SVDParameters(6,0,0)
        found_s, truncated_s = ptn.truncate_singular_values(deepcopy(self.s_values),
                                                            svd_params)
        correct_s = self.s_values[:6] * self.sum_s / np.sum(self.s_values[:6])
        self.assertTrue(np.allclose(correct_s, found_s))
        self.assertTrue(np.allclose(np.array([0.1,0.01]),truncated_s))

    def test_truncate_singular_values_abs_tol(self):
        """
        Tests the truncation of singular values caused by the absolute
         tolerance given.
        """
        abs_tol = 0.15
        svd_params = ptn.SVDParameters(10,0,abs_tol)
        found_s, truncated_s = ptn.truncate_singular_values(deepcopy(self.s_values),
                                                            svd_params)
        correct_s = self.s_values[:5] * self.sum_s / np.sum(self.s_values[:5])
        self.assertTrue(np.allclose(correct_s,found_s))
        correct_trunc = self.s_values[5:]
        self.assertTrue(np.allclose(correct_trunc,truncated_s))

    def test_truncate_singular_values_rel_tol(self):
        """
        Tests the truncation of singular values caused by the relative
         tolerance given.
        """
        rel_tol = 0.5
        svd_params = ptn.SVDParameters(10,rel_tol,0)
        found_s, truncated_s = ptn.truncate_singular_values(deepcopy(self.s_values),
                                                            svd_params)
        correct_s = self.s_values[:3]* self.sum_s / np.sum(self.s_values[:3])
        self.assertTrue(np.allclose(correct_s,found_s))
        correct_trunc = self.s_values[3:]
        self.assertTrue(np.allclose(correct_trunc,truncated_s))

    def test_truncate_singular_values_all_truncated(self):
        """
        Tests the truncation of singular values if all singular values were
            truncated.
        """
        svd_params = ptn.SVDParameters(10,0,3)
        found_s, truncated_s = ptn.truncate_singular_values(deepcopy(self.s_values),
                                                            svd_params)
        print("TEST SPEAKING: This warning is intended to be shown!")
        correct_s = np.array([self.s_values[0]]) * self.sum_s / self.s_values[0]
        self.assertTrue(np.allclose(correct_s,found_s))
        self.assertTrue(np.allclose(self.s_values[1:],truncated_s))
        self.assertWarns(UserWarning,ptn.truncate_singular_values,
                         deepcopy(self.s_values),
                         svd_params)

    def test_truncated_tensor_svd(self):
        """
        Tests the truncated SVD of a random tensor.
        """
        svd_params = ptn.SVDParameters(6,0,0)
        u, s, vh = ptn.truncated_tensor_svd(self.tensor,self.u_legs,self.v_legs,
                                            svd_params)
        u_ref, s_ref, vh_ref = ptn.tensor_svd(self.tensor,self.u_legs,self.v_legs,
                                              mode=ptn.SplitMode.FULL)
        u_ref = u_ref[:,:,:6]
        s_ref = s_ref[:6] * np.sum(s_ref) / np.sum(s_ref[:6])
        vh_ref = vh_ref[:6,:,:]
        self.assertTrue(np.allclose(u_ref,u))
        self.assertTrue(np.allclose(s_ref,s))
        self.assertTrue(np.allclose(vh_ref,vh))

    def test_contr_truncdated_tensor_svd_v_contr(self):
        """
        Tests the contracted truncated tensor svd, for which the singular
         values are contracted into the V-tensor.
        """
        svd_params = ptn.SVDParameters(6,0,0)
        u, vh = ptn.contr_truncated_svd_splitting(self.tensor,
                                                  self.u_legs,self.v_legs,
                                                  svd_params=svd_params)
        u_ref, s_ref, vh_ref = ptn.tensor_svd(self.tensor,self.u_legs,self.v_legs)
        u_ref = u_ref[:,:,:6]
        s_ref = s_ref[:6] * np.sum(s_ref) / np.sum(s_ref[:6])
        vh_ref = vh_ref[:6,:,:]
        vh_ref = np.tensordot(np.diag(s_ref),vh_ref,axes=(1,0))
        self.assertTrue(np.allclose(u_ref,u))
        self.assertTrue(np.allclose(vh_ref,vh))

    def test_contr_truncdated_tensor_svd_u_contr(self):
        """
        Tests the contracted truncated tensor svd, for which the singular
         values are contracted into the U-tensor.
        """
        svd_params = ptn.SVDParameters(6,0,0)
        u, vh = ptn.contr_truncated_svd_splitting(self.tensor,
                                                  self.u_legs,self.v_legs,
                                                  contr_mode=ptn.ContractionMode.UCONTR,
                                                  svd_params=svd_params)
        u_ref, s_ref, vh_ref = ptn.tensor_svd(self.tensor,self.u_legs,self.v_legs)
        u_ref = u_ref[:,:,:6]
        s_ref = s_ref[:6] * np.sum(s_ref) / np.sum(s_ref[:6])
        vh_ref = vh_ref[:6,:,:]
        u_ref = np.tensordot(u_ref,np.diag(s_ref),axes=(-1,0))
        self.assertTrue(np.allclose(u_ref,u))
        self.assertTrue(np.allclose(vh_ref,vh))

    def test_contr_truncdated_tensor_svd_equal_contr(self):
        """
        Tests the contracted truncated tensor svd, for which the singular
         values are distributed equally between U and V-tensors.
        """
        svd_params = ptn.SVDParameters(6,0,0)
        u, vh = ptn.contr_truncated_svd_splitting(self.tensor,
                                                  self.u_legs,self.v_legs,
                                                  contr_mode=ptn.ContractionMode.EQUAL,
                                                  svd_params=svd_params)
        u_ref, s_ref, vh_ref = ptn.tensor_svd(self.tensor,self.u_legs,self.v_legs)
        u_ref = u_ref[:,:,:6]
        s_ref = s_ref[:6] * np.sum(s_ref) / np.sum(s_ref[:6])
        vh_ref = vh_ref[:6,:,:]
        s_ref = np.sqrt(s_ref)
        u_ref = np.tensordot(u_ref,np.diag(s_ref),axes=(-1,0))
        vh_ref = np.tensordot(np.diag(s_ref),vh_ref,axes=(1,0))
        self.assertTrue(np.allclose(u_ref,u))
        self.assertTrue(np.allclose(vh_ref,vh))

if __name__ == "__main__":
    unittest.main()
