import unittest

import numpy as np

import pytreenet as ptn

class TestTensorProductInit(unittest.TestCase):

    def test_init_empty(self):
        empty_tp = ptn.TensorProduct()
        self.assertEqual(0, len(empty_tp))

    def test_init_numerical(self):
        array_dict = {"site1": ptn.crandn((2,2)),
                      "site2": ptn.crandn((3,3))}
        tensor_prod = ptn.TensorProduct(array_dict)

        self.assertEqual(2, len(tensor_prod))
        self.assertTrue(np.allclose(array_dict["site1"], tensor_prod["site1"]))
        self.assertTrue(np.allclose(array_dict["site2"], tensor_prod["site2"]))

    def test_init_symbolic(self):
        str_dict = {"site1": "a^dagger",
                      "site2": "a"}
        tensor_prod = ptn.TensorProduct(str_dict)

        self.assertEqual(2, len(tensor_prod))
        self.assertEqual(str_dict["site1"], tensor_prod["site1"])
        self.assertEqual(str_dict["site2"], tensor_prod["site2"])

    def test_init_mixed(self):
        dictionary = {"site1": ptn.crandn((2,2)),
                      "site2": "a"}
        tensor_prod = ptn.TensorProduct(dictionary)

        self.assertEqual(2, len(tensor_prod))
        self.assertTrue(np.allclose(dictionary["site1"], tensor_prod["site1"]))
        self.assertEqual(dictionary["site2"], tensor_prod["site2"])

class TestTensorProduct(unittest.TestCase):

    def test_from_operator(self):
        rand = ptn.crandn((2,2))
        operators = [ptn.Operator("a", ["site1"]),
                     ptn.Operator(rand, ["site2"])]
        tensor_prod = ptn.TensorProduct.from_operators(operators)
        self.assertEqual(2, len(tensor_prod))
        self.assertEqual("a", tensor_prod["site1"])
        self.assertTrue(np.allclose(rand, tensor_prod["site2"]))

        # With non-single site operator
        operators.append(ptn.Operator("fail", ["site3", "site4"]))
        self.assertRaises(AssertionError, ptn.TensorProduct.from_operators,
                          operators)

    def test_into_operator_numeric(self):
        random_arrays = [ptn.crandn((2,2)),
                         ptn.crandn((3,3))]
        operators = [ptn.Operator(random_arrays[i], "site" + str(i))
                     for i in range(len(random_arrays))]
        tensor_prod = ptn.TensorProduct.from_operators(operators)
        new_operator = tensor_prod.into_operator()
        correct_array = np.kron(random_arrays[0], random_arrays[1])
        self.assertTrue(np.allclose(correct_array, new_operator.operator))
        self.assertEqual(["site0", "site1"], new_operator.node_identifiers)

    def test_into_operator_symbolic(self):
        conversion_dict = {"op0": ptn.crandn((2,2)),
                           "op1": ptn.crandn((3,3))}
        operators = [ptn.Operator("op" + str(i), "site" + str(i))
                     for i in range(len(conversion_dict))]
        tensor_prod = ptn.TensorProduct.from_operators(operators)
        new_operator = tensor_prod.into_operator(conversion_dict)
        correct_array = np.kron(conversion_dict["op0"],
                                conversion_dict["op1"])
        self.assertTrue(np.allclose(correct_array, new_operator.operator))
        self.assertEqual(["site0", "site1"], new_operator.node_identifiers)

        # Eror with no conversion dictionary
        self.assertRaises(TypeError, tensor_prod.into_operator)

if __name__ == "__main__":
    unittest.main()
