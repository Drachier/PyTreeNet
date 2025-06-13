
import unittest
from copy import deepcopy

from numpy import (transpose, reshape,
                   allclose, array, sum as sum_np,
                   sort, tensordot, diag,
                   einsum, eye, sqrt, pad)
from numpy.linalg import (qr, svd)
from numpy.random import rand

from pytreenet.util.tensor_util import (tensor_matricization,
                                        compute_transfer_tensor)
from pytreenet.util.tensor_splitting import (_determine_tensor_shape,
                                            truncated_tensor_svd,
                                            tensor_qr_decomposition,
                                            tensor_svd,
                                            renormalise_singular_values,
                                            value_truncation,
                                            sum_truncation,
                                            truncate_singular_values,
                                            contr_truncated_svd_splitting,
                                            idiots_splitting,
                                            SplitMode,
                                            ContractionMode,
                                            SVDParameters)
from pytreenet.random import crandn

class TestTensorUtilSimple(unittest.TestCase):
    def setUp(self):
        self.tensor1 = crandn((2, 3, 4, 5))
        self.output_legs = (1, 3)
        self.input_legs = (0, 2)

        self.tensor2 = crandn((32, 22, 14, 16))

    def test_determine_tensor_shape(self):
        matrix = tensor_matricization(
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

    def test_tensor_qr_decomposition(self):
        q, r = tensor_qr_decomposition(
            self.tensor1, self.output_legs, self.input_legs)
        self.assertEqual(q.shape[-1], r.shape[0])
        tensor_shape = self.tensor1.shape
        self.assertEqual(q.shape[0:-1], (tensor_shape[1], tensor_shape[3]))
        self.assertEqual(r.shape[1:], (tensor_shape[0], tensor_shape[2]))
        recontracted_tensor = einsum("ijk,klm->limj", q, r)
        self.assertTrue(allclose(recontracted_tensor, self.tensor1))
        # q should be orthonormal
        connection_dimension = q.shape[-1]
        identity = eye(connection_dimension)
        transfer_tensor = compute_transfer_tensor(q, (0, 1))
        transfer_matrix = reshape(
            transfer_tensor, (connection_dimension, connection_dimension))
        self.assertTrue(allclose(identity, transfer_matrix))

    def test_tensor_svd(self):
        u, s, vh = tensor_svd(self.tensor1,
                                  self.output_legs,
                                  self.input_legs)

        self.assertEqual(u.shape[-1], len(s))
        self.assertEqual(vh.shape[0], len(s))
        tensor_shape = self.tensor1.shape
        self.assertEqual(u.shape[0:-1], (tensor_shape[1], tensor_shape[3]))
        self.assertEqual(vh.shape[1:], (tensor_shape[0], tensor_shape[2]))

        # We should be able to reconstruct the tensor.
        us = tensordot(u, diag(s), axes=(-1, 0))
        usvh = tensordot(us, vh, axes=(-1, 0))
        correct_tensor = self.tensor1.transpose([1, 3, 0, 2])

        self.assertTrue(allclose(correct_tensor, usvh))

    def test_check_truncation_parameters(self):
        
        self.assertRaises(TypeError, SVDParameters,
                          max_bond_dim=1.3,  rel_tol=0.01, total_tol=1e-15)
        self.assertRaises(ValueError, SVDParameters,
                          max_bond_dim=-100,  rel_tol=0.01, total_tol=1e-15)
        self.assertRaises(ValueError, SVDParameters,
                          max_bond_dim=100,  rel_tol=-2.0, total_tol=1e-15)
        self.assertRaises(ValueError, SVDParameters,
                          max_bond_dim=100,  rel_tol=0.01, total_tol=-1)

    def test_truncated_tensor_svd(self):
        truncation_parameter_list = (
            (15, 0.01, 1e-15), (200, 0.9, 1e-15), (200, 0.01, 35))

        for parameters in truncation_parameter_list:
            svd_params = SVDParameters(*parameters)

            u, s, vh = truncated_tensor_svd(self.tensor2,
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
        self.tensor = crandn((4,5,2))

    def test_qr_reduced_q_legs_bigger(self):
        q, r = tensor_qr_decomposition(self.tensor, (0,2), (1, ))

        self.assertEqual((4,2,5),q.shape)
        self.assertEqual((5,5),r.shape)

        ref_tensor = transpose(self.tensor,(0,2,1))
        ref_tensor = reshape(ref_tensor,(8,5))
        ref_q, ref_r = qr(ref_tensor,mode="reduced")
        ref_q = reshape(ref_q, (4,2,5))

        self.assertTrue(allclose(ref_q, q))
        self.assertTrue(allclose(ref_r, r))

    def test_qr_reduced_r_legs_bigger(self):
        q, r = tensor_qr_decomposition(self.tensor, (0, ), (1,2))

        self.assertEqual((4,4),q.shape)
        self.assertEqual((4,5,2),r.shape)

        ref_tensor = reshape(self.tensor,(4,10))
        ref_q, ref_r = qr(ref_tensor,mode="reduced")
        ref_r = reshape(ref_r, (4,5,2))

        self.assertTrue(allclose(ref_q, q))
        self.assertTrue(allclose(ref_r, r))

    def test_qr_full_q_legs_bigger(self):
        q, r = tensor_qr_decomposition(self.tensor, (0,2), (1, ),
                                           SplitMode.FULL)

        self.assertEqual((4,2,8),q.shape)
        self.assertEqual((8,5),r.shape)

        ref_tensor = transpose(self.tensor,(0,2,1))
        ref_tensor = reshape(ref_tensor,(8,5))
        ref_q, ref_r = qr(ref_tensor,mode="complete")
        ref_q = reshape(ref_q, (4,2,8))

        self.assertTrue(allclose(ref_q, q))
        self.assertTrue(allclose(ref_r, r))

    def test_qr_full_r_legs_bigger(self):
        q, r = tensor_qr_decomposition(self.tensor, (1, ), (0,2),
                                           SplitMode.FULL)

        self.assertEqual((5,5),q.shape)
        self.assertEqual((5,4,2),r.shape)

        ref_tensor = transpose(self.tensor,(1,0,2))
        ref_tensor = reshape(ref_tensor,(5,8))
        ref_q, ref_r = qr(ref_tensor,mode="complete")
        ref_r = reshape(ref_r, (5,4,2))

        self.assertTrue(allclose(ref_q, q))
        self.assertTrue(allclose(ref_r, r))

    def test_qr_keep_q_legs_bigger(self):
        q, r = tensor_qr_decomposition(self.tensor, (0,2), (1, ),
                                           SplitMode.KEEP)

        self.assertEqual((4,2,5),q.shape)
        self.assertEqual((5,5),r.shape)

        ref_tensor = transpose(self.tensor,(0,2,1))
        ref_tensor = reshape(ref_tensor,(8,5))
        ref_q, ref_r = qr(ref_tensor,mode="reduced")
        ref_q = reshape(ref_q, (4,2,5))

        self.assertTrue(allclose(ref_q, q))
        self.assertTrue(allclose(ref_r, r))

    def test_qr_keep_r_legs_bigger(self):
        q, r = tensor_qr_decomposition(self.tensor, (1, ), (0,2),
                                           SplitMode.KEEP)

        self.assertEqual((5,8),q.shape)
        self.assertEqual((8,4,2),r.shape)

        ref_tensor = transpose(self.tensor,(1,0,2))
        ref_tensor = reshape(ref_tensor,(5,8))
        ref_q, ref_r = qr(ref_tensor,mode="reduced")
        ref_r = reshape(ref_r, (5,4,2))
        ref_q = pad(ref_q, [(0,0),(0,3)])
        ref_r = pad(ref_r, [(0,3),(0,0),(0,0)])

        self.assertTrue(allclose(ref_q, q))
        self.assertTrue(allclose(ref_r, r))

class TestSingularValueTruncation(unittest.TestCase):

    def setUp(self):
        self.s_values = array([1.2,1,0.8,0.5,0.2,0.1,0.1,0.01])

    def test_renorm_singular_values(self):
        """
        Test the renormalisation of a random tensor of positive values which
         represent singular values against a truncated vector.
        """
        s = rand(10)
        s = array(list(reversed(sort(s))))
        s_new = s[:5]
        norm_old = sum_np(s)
        norm_new = sum_np(s_new)
        normed_s = s_new * (norm_old / norm_new)
        found_s = renormalise_singular_values(s,s_new)
        self.assertTrue(allclose(normed_s,found_s))

    def test_renorm_singular_value_equal_size(self):
        """
        Test the renormalisation of a random tensor of positive values which
         represent singular values against an equal vector.
        """
        s = rand(10)
        found_s = renormalise_singular_values(deepcopy(s),deepcopy(s))
        self.assertTrue(allclose(found_s,s))

    def test_value_truncation_no_truncation(self):
        """
        Tests the truncation of singular values by threshold value, however
         no truncation should happen given the parameters.
        """
        s = deepcopy(self.s_values)
        total_tol = 0.001
        rel_tol = 0.0001
        found_s = value_truncation(s, total_tol, rel_tol)
        self.assertTrue(allclose(self.s_values,found_s))

    def test_value_truncation_total_tol(self):
        """
        Tests the truncation of singular values by threshold value, where the
         choice of total tolerances causes the truncation.
        """
        total_tol = 0.15
        rel_tol = 0
        found_s = value_truncation(deepcopy(self.s_values), total_tol, rel_tol)
        correct_s = self.s_values[:5]
        self.assertTrue(allclose(correct_s,found_s))

    def test_value_truncation_rel_tol(self):
        """
        Tests the truncation of singular values caused by the relative
         tolerance given.
        """
        rel_tol = 0.5
        total_tol = 0
        found_s = value_truncation(deepcopy(self.s_values), total_tol, rel_tol)
        correct_s = self.s_values[:3]
        self.assertTrue(allclose(correct_s,found_s))

    def test_value_truncation_all_values_truncated(self):
        """
        Tests the truncation of singular values when all singular values are
         truncated.
        """
        rel_tol = 0.5
        total_tol = 3
        found_s = value_truncation(deepcopy(self.s_values), total_tol, rel_tol)
        correct_s = []
        self.assertTrue(allclose(correct_s,found_s))

    def test_sum_truncation_no_truncation(self):
        """
        Tests the truncation of singular values by the sum of the singular
         values, however no truncation should happen given the parameters.
        """
        threshold = 0.001
        found_s = sum_truncation(deepcopy(self.s_values), threshold)
        print(found_s)
        self.assertTrue(allclose(self.s_values,found_s))

    def test_sum_truncation(self):
        """
        Tests the truncation of singular values caused by the sum of the
         singular values.
        """
        threshold = 0.1
        found_s = sum_truncation(deepcopy(self.s_values), threshold)
        correct_s = self.s_values[:5]
        self.assertTrue(allclose(correct_s,found_s))

    def test_sum_truncation_all_truncated(self):
        """
        Tests the truncation of singular values when all singular values are
         truncated.
        """
        threshold = 3
        found_s = sum_truncation(deepcopy(self.s_values), threshold)
        correct_s = []
        self.assertTrue(allclose(correct_s,found_s))

    def test_truncate_singular_values_no_truncation(self):
        """
        Tests the truncation of singular values, however no truncation should
         happen given the parameters.
        """
        svd_params = SVDParameters(10,0,0)
        found_s, truncated_s = truncate_singular_values(deepcopy(self.s_values),
                                                            svd_params)
        self.assertTrue(allclose(self.s_values,found_s))
        self.assertTrue(allclose(array([]),truncated_s))

    def test_truncate_singular_values_max_bond_dim(self):
        """
        Tests the truncation of singular values caused by the maximum bond
         dimension given.
        """
        svd_params = SVDParameters(6,0,0)
        found_s, truncated_s = truncate_singular_values(deepcopy(self.s_values),
                                                            svd_params)
        correct_s = self.s_values[:6]
        self.assertTrue(allclose(correct_s, found_s))
        self.assertTrue(allclose(array([0.1,0.01]),truncated_s))

    def test_truncate_singular_values_abs_tol(self):
        """
        Tests the truncation of singular values caused by the absolute
         tolerance given.
        """
        abs_tol = 0.15
        svd_params = SVDParameters(10,0,abs_tol)
        found_s, truncated_s = truncate_singular_values(deepcopy(self.s_values),
                                                            svd_params)
        correct_s = self.s_values[:5]
        self.assertTrue(allclose(correct_s,found_s))
        correct_trunc = self.s_values[5:]
        self.assertTrue(allclose(correct_trunc,truncated_s))

    def test_truncate_singular_values_rel_tol(self):
        """
        Tests the truncation of singular values caused by the relative
         tolerance given.
        """
        rel_tol = 0.5
        svd_params = SVDParameters(10,rel_tol,0)
        found_s, truncated_s = truncate_singular_values(deepcopy(self.s_values),
                                                            svd_params)
        correct_s = self.s_values[:3]
        self.assertTrue(allclose(correct_s,found_s))
        correct_trunc = self.s_values[3:]
        self.assertTrue(allclose(correct_trunc,truncated_s))

    def test_truncate_singular_values_all_truncated(self):
        """
        Tests the truncation of singular values if all singular values were
            truncated.
        """
        svd_params = SVDParameters(10,0,3)
        found_s, truncated_s = truncate_singular_values(deepcopy(self.s_values),
                                                            svd_params)
        print("TEST SPEAKING: This warning is intended to be shown!")
        correct_s = array([self.s_values[0]])
        self.assertTrue(allclose(correct_s,found_s))
        self.assertTrue(allclose(self.s_values[1:],truncated_s))
        self.assertWarns(UserWarning,truncate_singular_values,
                         deepcopy(self.s_values),
                         svd_params)

class TestSingularValueDecompositions(unittest.TestCase):
    def setUp(self):
        self.tensor = crandn((2,3,4,5))
        self.u_legs = (1,3)
        self.v_legs = (0,2)
        self.s_values = array([1.2,1,0.8,0.5,0.2,0.1,0.1,0.01])
        self.sum_s = sum_np(self.s_values)

    def test_svd_reduced(self):
        """
        Test the SVD of a tensor with REDUCED mode, i.e. the tensor legs
         of the resulting U and V tensors pointing to S have the minimal
         dimension possible.
        """
        reference_tensor = self.tensor.transpose((1,3,0,2))
        reference_tensor = reference_tensor.reshape((15,8))
        ref_u, ref_s, ref_v = svd(reference_tensor,
                                            full_matrices=False)
        ref_u = ref_u.reshape(3,5,8)
        ref_v = ref_v.reshape(8,2,4)
        u, s, vh = tensor_svd(self.tensor,self.u_legs,self.v_legs)
        self.assertTrue(allclose(ref_u,u))
        self.assertTrue(allclose(ref_s,s))
        self.assertTrue(allclose(ref_v,vh))

    def test_svd_full(self):
        """
        Test the SVD of a tensor in FULL mode, i.e. the tensor legs
         of the resulting U and V tensors pointing to S have the size of all
         other legs of the respective tensor taken together.
        """
        reference_tensor = self.tensor.transpose((1,3,0,2))
        reference_tensor = reference_tensor.reshape((15,8))
        ref_u, ref_s, ref_v = svd(reference_tensor,
                                            full_matrices=True)
        ref_u = ref_u.reshape(3,5,15)
        ref_v = ref_v.reshape(8,2,4)
        u, s, vh = tensor_svd(self.tensor,self.u_legs,self.v_legs,
                                  mode=SplitMode.FULL)
        self.assertTrue(allclose(ref_u,u))
        self.assertTrue(allclose(ref_s,s))
        self.assertTrue(allclose(ref_v,vh))

    def test_svd_keep(self):
        """
        Test the SVD of a tensor in KEEP mode, i.e. the tensor legs
         of the resulting U and V tensors pointing to S have the size of all
         other legs of the respective tensor taken together.
        """
        reference_tensor = self.tensor.transpose((1,3,0,2))
        reference_tensor = reference_tensor.reshape((15,8))
        ref_u, ref_s, ref_v = svd(reference_tensor,
                                            full_matrices=True)
        ref_u = ref_u.reshape(3,5,15)
        ref_v = ref_v.reshape(8,2,4)
        u, s, vh = tensor_svd(self.tensor,self.u_legs,self.v_legs,
                                  mode=SplitMode.KEEP)
        self.assertTrue(allclose(ref_u,u))
        self.assertTrue(allclose(ref_s,s))
        self.assertTrue(allclose(ref_v,vh))

    def test_truncated_tensor_svd(self):
        """
        Tests the truncated SVD of a random tensor.
        """
        svd_params = SVDParameters(6,0,0)
        u, s, vh = truncated_tensor_svd(self.tensor,self.u_legs,self.v_legs,
                                            svd_params)
        u_ref, s_ref, vh_ref = tensor_svd(self.tensor,self.u_legs,self.v_legs,
                                              mode=SplitMode.FULL)
        u_ref = u_ref[:,:,:6]
        s_ref = s_ref[:6]
        vh_ref = vh_ref[:6,:,:]
        self.assertTrue(allclose(u_ref,u))
        self.assertTrue(allclose(s_ref,s))
        self.assertTrue(allclose(vh_ref,vh))

    def test_contr_truncdated_tensor_svd_v_contr(self):
        """
        Tests the contracted truncated tensor svd, for which the singular
         values are contracted into the V-tensor.
        """
        svd_params = SVDParameters(6,0,0)
        u, vh = contr_truncated_svd_splitting(self.tensor,
                                                  self.u_legs,self.v_legs,
                                                  svd_params=svd_params)
        u_ref, s_ref, vh_ref = tensor_svd(self.tensor,self.u_legs,self.v_legs)
        u_ref = u_ref[:,:,:6]
        s_ref = s_ref[:6]
        vh_ref = vh_ref[:6,:,:]
        vh_ref = tensordot(diag(s_ref),vh_ref,axes=(1,0))
        self.assertTrue(allclose(u_ref,u))
        self.assertTrue(allclose(vh_ref,vh))

    def test_contr_truncdated_tensor_svd_u_contr(self):
        """
        Tests the contracted truncated tensor svd, for which the singular
         values are contracted into the U-tensor.
        """
        svd_params = SVDParameters(6,0,0)
        u, vh = contr_truncated_svd_splitting(self.tensor,
                                                  self.u_legs,self.v_legs,
                                                  contr_mode=ContractionMode.UCONTR,
                                                  svd_params=svd_params)
        u_ref, s_ref, vh_ref = tensor_svd(self.tensor,self.u_legs,self.v_legs)
        u_ref = u_ref[:,:,:6]
        s_ref = s_ref[:6]
        vh_ref = vh_ref[:6,:,:]
        u_ref = tensordot(u_ref,diag(s_ref),axes=(-1,0))
        self.assertTrue(allclose(u_ref,u))
        self.assertTrue(allclose(vh_ref,vh))

    def test_contr_truncdated_tensor_svd_equal_contr(self):
        """
        Tests the contracted truncated tensor svd, for which the singular
         values are distributed equally between U and V-tensors.
        """
        svd_params = SVDParameters(6,0,0)
        u, vh = contr_truncated_svd_splitting(self.tensor,
                                                  self.u_legs,self.v_legs,
                                                  contr_mode=ContractionMode.EQUAL,
                                                  svd_params=svd_params)
        u_ref, s_ref, vh_ref = tensor_svd(self.tensor,self.u_legs,self.v_legs)
        u_ref = u_ref[:,:,:6]
        s_ref = s_ref[:6]
        vh_ref = vh_ref[:6,:,:]
        s_ref = sqrt(s_ref)
        u_ref = tensordot(u_ref,diag(s_ref),axes=(-1,0))
        vh_ref = tensordot(diag(s_ref),vh_ref,axes=(1,0))
        self.assertTrue(allclose(u_ref,u))
        self.assertTrue(allclose(vh_ref,vh))

class TestIdiotsSplitting(unittest.TestCase):

    def test_idiots_splitting_valid(self):
        shape = (2,5,6,7,3,4)
        tensor = crandn(shape)
        a_tensor = crandn((2,6,4,8))
        b_tensor = crandn((8,5,7,3))
        legs_a = (0,2,5)
        legs_b = (1,3,4)
        a_copy, b_copy = deepcopy(a_tensor), deepcopy(b_tensor)
        a, b = idiots_splitting(tensor, legs_a, legs_b,
                                a_tensor=a_tensor,
                                b_tensor=b_tensor)
        self.assertTrue(allclose(a_copy,a))
        self.assertTrue(allclose(b_copy,b))

    def test_idiots_splitting_Nones(self):
        shape = (2,5,6,7,3,4)
        tensor = crandn(shape)
        a_tensor = None
        b_tensor = None
        legs_a = ()
        legs_b = ()
        self.assertRaises(ValueError,idiots_splitting,tensor,legs_a,legs_b,
                          a_tensor=a_tensor,b_tensor=b_tensor)

    def test_idiots_splitting_aNone(self):
        shape = (2,5,6,7,3,4)
        tensor = crandn(shape)
        a_tensor = None
        b_tensor = crandn((5,7,3))
        legs_a = ()
        legs_b = (1,3,4)
        self.assertRaises(ValueError,idiots_splitting,tensor,legs_a,legs_b,
                          a_tensor=a_tensor,b_tensor=b_tensor)

    def test_idiots_splitting_bNone(self):
        shape = (2,5,6,7,3,4)
        tensor = crandn(shape)
        a_tensor = crandn((2,6,4))
        b_tensor = None
        legs_a = (0,2,5)
        legs_b = ()
        self.assertRaises(ValueError,idiots_splitting,tensor,legs_a,legs_b,
                          a_tensor=a_tensor,b_tensor=b_tensor)

if __name__ == "__main__":
    unittest.main()
