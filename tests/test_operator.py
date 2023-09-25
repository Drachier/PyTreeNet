import unittest
import numpy as np
import copy
import pytreenet as ptn

class TestOperator(unittest.TestCase):
    def setUp(self):
        self.identifiers = ["apple", "pear"]
        self.string = "a^dagger"
        self.tensor = ptn.crandn((2,2))
        self.non_op_tensor = ptn.crandn((2,3))

    def test_init_symbolic_singleid(self):
        operator = ptn.operators.Operator(self.string,
                                    self.identifiers[0])
        self.assertEqual(self.string, operator.operator)
        self.assertEqual([self.identifiers[0]], operator.node_identifiers)

    def test_init_symbolic_twoid(self):
        operator = ptn.operators.Operator(self.string,
                                        self.identifiers)
        self.assertEqual(self.string, operator.operator)
        self.assertEqual(self.identifiers, operator.node_identifiers)

    def test_init_numeric_singleid(self):
        operator = ptn.operators.Operator(self.tensor,
                                    self.identifiers[0])
        self.assertTrue(np.allclose(self.tensor, operator.operator))
        self.assertEqual([self.identifiers[0]], operator.node_identifiers)

    def test_init_numeric_twoid(self):
        operator = ptn.operators.Operator(self.tensor,
                                    self.identifiers)
        self.assertTrue(np.allclose(self.tensor, operator.operator))
        self.assertEqual(self.identifiers, operator.node_identifiers)

if __name__ == "__main__":
    unittest.main()