import unittest
import pytreenet as ptn

from math import prod

class TestTreeTensorNetwork(unittest.TestCase):
    def setUp(self):
        self.tensor1 = ptn.crandn((2,3,4,5))
        self.output_legs = (1,3)
        self.input_legs = (0,2)


    def test_matricization(self):
        matrix = ptn.tensor_matricization(self.tensor1, self.output_legs, self.input_legs)

        self.assertEqual(matrix.shape, (3*5,2*4))

    def test_determine_tensor_shape(self):
        matrix = ptn.tensor_matricization(self.tensor1, self.output_legs, self.input_legs)
        old_shape = self.tensor1.shape

        new_output_shape = ptn.determine_tensor_shape(old_shape, matrix, self.output_legs)
        reference_output_shape = (3,5,8)
        self.assertEqual(new_output_shape, reference_output_shape)

        new_input_shape = ptn.determine_tensor_shape(old_shape, matrix, self.input_legs, output = False)
        reference_input_shape = (15,2,4)
        self.assertEqual(new_input_shape, reference_input_shape)

    def test_tensor_qr_decomposition(self):
        q, r = ptn.tensor_qr_decomposition(self.tensor1, self.output_legs, self.input_legs)

        self.assertEqual(q.shape[-1], r.shape[0])
        tensor_shape = self.tensor1.shape
        self.assertEqual(q.shape[0:-1],(tensor_shape[1],tensor_shape[3]))
        self.assertEqual(r.shape[1:],(tensor_shape[0],tensor_shape[2]))

    def test_tesnor_svd(self):
        u, s, vh = ptn.tensor_svd(self.tensor1, self.output_legs, self.input_legs)

        self.assertEqual(u.shape[-1], prod(u.shape[0:-1]))
        self.assertEqual(vh.shape[0], prod(vh.shape[1:]))
        tensor_shape = self.tensor1.shape
        self.assertEqual(u.shape[0:-1],(tensor_shape[1],tensor_shape[3]))
        self.assertEqual(vh.shape[1:],(tensor_shape[0],tensor_shape[2]))

if __name__ == "__main__":
    unittest.main()