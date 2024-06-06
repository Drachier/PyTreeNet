import unittest
import numpy as np
import copy

import pytreenet as ptn
from pytreenet.operators import NumericOperator
from pytreenet.random import crandn

class TestNumericOperator(unittest.TestCase):
    def setUp(self):
        self.identifiers = ["apple", "pear"]
        self.tensor = crandn((2,2,2,2))
        self.non_op_tensor = crandn((2,2,2))
        self.matrix = crandn((9,9))

    def test_init_correct_tensor_shape(self):
        operator = NumericOperator(self.tensor, self.identifiers)
        self.assertTrue(np.allclose(self.tensor, operator.operator))
        self.assertEqual(self.identifiers, operator.node_identifiers)

    def test_init_wrong_tensor_shape(self):
        self.assertRaises(AssertionError, NumericOperator,
            self.non_op_tensor, self.identifiers)

    def test_to_matrix_shape_2_2(self):
        correct_matrix = crandn((2,2))
        operator = NumericOperator(correct_matrix, self.identifiers[0])
        found_operator = operator.to_matrix()
        self.assertTrue(np.allclose(correct_matrix,
            found_operator.operator))

    def test_to_matrix_shape_2_2_2_2(self):
        operator = NumericOperator(self.tensor,
            self.identifiers)
        correct_matrix = np.reshape(self.tensor, (4,4))
        found_matrix = operator.to_matrix().operator
        self.assertTrue(np.allclose(correct_matrix,
            found_matrix))

    def test_to_tensor_no_dimension_information(self):
        operator = NumericOperator(self.matrix, self.identifiers)
        self.assertRaises(ValueError, operator.to_tensor)

    def test_to_tensor_dim_given(self):
        operator = NumericOperator(self.matrix, self.identifiers)
        tensor_operator = operator.to_tensor(dim=3)
        found_tensor = copy.deepcopy(tensor_operator.operator)
        correct_tensor = np.reshape(self.matrix, (3,3,3,3))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))

    def test_to_tensor_non_positive_dim(self):
        operator = NumericOperator(self.matrix, self.identifiers)
        self.assertRaises(ValueError, operator.to_tensor, -1)

    def test_to_tensor_ttn_given(self):
        # Building the reference TTN
        ttn = ptn.TreeTensorNetwork()
        ttn.add_root(ptn.Node(identifier="I"), crandn((2,2,3)))
        ttn.add_child_to_parent(ptn.Node(identifier=self.identifiers[0]),
            crandn((2,3)), 0, "I", 0)
        ttn.add_child_to_parent(ptn.Node(identifier=self.identifiers[1]),
            crandn((2,3)), 0, "I", 1)

        operator = NumericOperator(self.matrix, self.identifiers)
        tensor_operator = operator.to_tensor(ttn=ttn)
        found_tensor = copy.deepcopy(tensor_operator.operator)
        correct_tensor = np.reshape(self.matrix, (3,3,3,3))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))

    def test_is_unitary_matrix_2_2_true(self):
        # Building unitary matrix
        a = 1 / np.sqrt(2)
        b = 1j / np.sqrt(2)
        unitary = np.array([[a, b],
                            [b, a]])

        operator = NumericOperator(unitary, self.identifiers[0])
        self.assertTrue(operator.is_unitary())

    def test_is_unitary_matrix_2_2_false(self):
        matrix = np.array([[1,2],
                           [3,4]])
        operator = NumericOperator(matrix, self.identifiers[0])
        self.assertFalse(operator.is_unitary())

    def test_is_unitary_tensor_3_3_3_3_true(self):
        # Building unitary matrix
        unitary, _ = np.linalg.qr(crandn((9,9)))
        unitary_tensor = np.reshape(unitary, (3,3,3,3))
        operator = NumericOperator(unitary_tensor, self.identifiers)
        self.assertTrue(operator.is_unitary())

    def test_is_unitary_tensor_3_3_3_3_false(self):
        matrix = np.ones((3,3,3,3))
        operator = NumericOperator(matrix, self.identifiers)
        self.assertFalse(operator.is_unitary())

if __name__ == "__main__":
    unittest.main()
