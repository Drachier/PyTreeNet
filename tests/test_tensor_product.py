import unittest

import numpy as np
from scipy.linalg import expm

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
        operators = [ptn.NumericOperator(rand, ["site1"]),
                     ptn.NumericOperator(rand, ["site2"])]
        tensor_prod = ptn.TensorProduct.from_operators(operators)
        self.assertEqual(2, len(tensor_prod))
        self.assertTrue(np.allclose(rand, tensor_prod["site1"]))
        self.assertTrue(np.allclose(rand, tensor_prod["site2"]))

        # With non-single site operator
        rand2 = ptn.crandn((2,2,2,2))
        operators.append(ptn.NumericOperator(rand2, ["site3", "site4"]))
        self.assertRaises(AssertionError, ptn.TensorProduct.from_operators,
                          operators)

    def test_into_operator_numeric(self):
        random_arrays = [ptn.crandn((2,2)),
                         ptn.crandn((3,3))]
        operators = [ptn.NumericOperator(random_arrays[i], "site" + str(i))
                     for i in range(len(random_arrays))]
        tensor_prod = ptn.TensorProduct.from_operators(operators)
        new_operator = tensor_prod.into_operator()
        correct_array = np.kron(random_arrays[0], random_arrays[1])
        self.assertTrue(np.allclose(correct_array, new_operator.operator))
        self.assertEqual(["site0", "site1"], new_operator.node_identifiers)

    def test_into_operator_symbolic(self):
        conversion_dict = {"op0": ptn.crandn((2,2)),
                           "op1": ptn.crandn((3,3))}
        tensor_prod = ptn.TensorProduct({"site0": "op0", "site1": "op1"})
        new_operator = tensor_prod.into_operator(conversion_dict)
        correct_array = np.kron(conversion_dict["op0"],
                                conversion_dict["op1"])
        self.assertTrue(np.allclose(correct_array, new_operator.operator))
        self.assertEqual(["site0", "site1"], new_operator.node_identifiers)

        # Eror with no conversion dictionary
        self.assertRaises(TypeError, tensor_prod.into_operator)

    def test_exp_matrix(self):
        simple_matrix = np.asarray([[1,2],
                                    [2,3]])
        identifier = "ID, Please"
        tensor_prod = ptn.TensorProduct({identifier: simple_matrix})
        
        # Without factor
        correct_matrix = expm(simple_matrix)
        found_operator = tensor_prod.exp()
        self.assertTrue(np.allclose(correct_matrix, found_operator.operator))
        self.assertEqual(identifier, found_operator.node_identifiers[0])

        # With factor
        factor = -1j * 0.3
        correct_matrix = expm(factor * simple_matrix)
        found_operator = tensor_prod.exp(factor)
        self.assertTrue(np.allclose(correct_matrix, found_operator.operator))
        self.assertEqual(identifier, found_operator.node_identifiers[0])

    def test_exp_two_factors(self):
        matrix1 = np.array([[1, 2],
                            [3, 4]])
        matrix2 = np.array([[0.1, 0.2, 0.3],
                            [0.6, 0.5, 0.4],
                            [0.7, 0.8, 0.9]])
        identifiers = ["HI", "HO"]
        tensor_product = ptn.TensorProduct({identifiers[0]: matrix1,
                                            identifiers[1]: matrix2})
        found_operator = tensor_product.exp()
        correct_result = expm(np.kron(matrix1, matrix2))
        self.assertTrue(np.allclose(correct_result, found_operator.operator))
        self.assertEqual(identifiers, found_operator.node_identifiers)

class TestTensorProductWithTTN(unittest.TestCase):
    def setUp(self):
        self.ttn = ptn.TreeTensorNetwork()
        self.ttn.add_root(ptn.Node(identifier="root"),
            ptn.crandn((2,2,3)))
        self.ttn.add_child_to_parent(ptn.Node(identifier="c1"),
            ptn.crandn((2,2)), 0, "root", 0)
        self.ttn.add_child_to_parent(ptn.Node(identifier="c2"),
            ptn.crandn((2,4)), 0, "root", 1)

        self.symbolic_dict = {"root": "X",
                         "c1": "a^dagger",
                         "c2": "H"}

    def test_pad_with_identity_no_pad_needed(self):
        ten_prod = ptn.TensorProduct(self.symbolic_dict)
        padded_tp = ten_prod.pad_with_identities(self.ttn)
        # Since no padding is needed, nothing should change
        self.assertEqual(ten_prod, padded_tp)

    def test_pad_with_identity_new_node_numeric(self):
        del self.symbolic_dict["c2"]
        ten_prod = ptn.TensorProduct(self.symbolic_dict)
        padded_tp = ten_prod.pad_with_identities(self.ttn)
        self.assertTrue("c2" in padded_tp.keys())
        self.assertTrue(np.allclose(np.eye(4), padded_tp["c2"]))

    def test_pad_with_identity_new_node_symbolic(self):
        del self.symbolic_dict["c2"]
        ten_prod = ptn.TensorProduct(self.symbolic_dict)
        padded_tp = ten_prod.pad_with_identities(self.ttn, symbolic=True)
        self.assertTrue("c2" in padded_tp.keys())
        self.assertEqual("I4", padded_tp["c2"])

    def test_pad_with_identity_node_not_in_ttn(self):
        self.symbolic_dict["wronged"] = "P"
        ten_prod = ptn.TensorProduct(self.symbolic_dict)
        self.assertRaises(KeyError, ten_prod.pad_with_identities, self.ttn)

if __name__ == "__main__":
    unittest.main()
