import unittest
import numpy as np
import copy

import pytreenet as ptn
from pytreenet.operators import Operator, NumericOperator, SymbolicOperator

class TestOperator(unittest.TestCase):
    def setUp(self):
        self.identifiers = ["apple", "pear"]
        self.tensor = ptn.crandn((2,2))
        self.non_op_tensor = ptn.crandn((2,3))

    def test_init_symbolic_singleid(self):
        operator = Operator(self.string,
                                    self.identifiers[0])
        self.assertEqual(self.string, operator.operator)
        self.assertEqual([self.identifiers[0]], operator.node_identifiers)

    def test_init_symbolic_twoid(self):
        operator = Operator(self.string,
                                        self.identifiers)
        self.assertEqual(self.string, operator.operator)
        self.assertEqual(self.identifiers, operator.node_identifiers)

    def test_init_numeric_singleid(self):
        operator = Operator(self.tensor,
                                    self.identifiers[0])
        self.assertTrue(np.allclose(self.tensor, operator.operator))
        self.assertEqual([self.identifiers[0]], operator.node_identifiers)

    def test_init_numeric_twoid(self):
        operator = Operator(self.tensor,
                                    self.identifiers)
        self.assertTrue(np.allclose(self.tensor, operator.operator))
        self.assertEqual(self.identifiers, operator.node_identifiers)

class TestNumericOperator(unittest.TestCase):
    def setUp(self):
        self.identifiers = ["apple", "pear"]
        self.tensor = ptn.crandn((2,2,2,2))
        self.non_op_tensor = ptn.crandn((2,2,2))

    def test_init_correct_tensor_shape(self):
        operator = NumericOperator(self.tensor, self.identifiers)
        self.assertTrue(np.allclose(self.tensor, operator.operator))
        self.assertEqual(self.identifiers, operator.node_identifiers)

    def test_init_wrong_tensor_shape(self):
        self.assertRaises(AssertionError, NumericOperator,
            self.non_op_tensor, self.identifiers)

if __name__ == "__main__":
    unittest.main()
