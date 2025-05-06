import unittest

import numpy as np
from scipy.linalg import expm

from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.operators.operator import NumericOperator
from pytreenet.core import Node, TreeTensorNetwork
from pytreenet.random import crandn

class TestTensorProductInit(unittest.TestCase):

    def test_init_empty(self):
        empty_tp = TensorProduct()
        self.assertEqual(0, len(empty_tp))

    def test_init_numerical(self):
        array_dict = {"site1": crandn((2,2)),
                      "site2": crandn((3,3))}
        tensor_prod = TensorProduct(array_dict)

        self.assertEqual(2, len(tensor_prod))
        self.assertTrue(np.allclose(array_dict["site1"], tensor_prod["site1"]))
        self.assertTrue(np.allclose(array_dict["site2"], tensor_prod["site2"]))

    def test_init_symbolic(self):
        str_dict = {"site1": "a^dagger",
                      "site2": "a"}
        tensor_prod = TensorProduct(str_dict)

        self.assertEqual(2, len(tensor_prod))
        self.assertEqual(str_dict["site1"], tensor_prod["site1"])
        self.assertEqual(str_dict["site2"], tensor_prod["site2"])

    def test_init_mixed(self):
        dictionary = {"site1": crandn((2,2)),
                      "site2": "a"}
        tensor_prod = TensorProduct(dictionary)

        self.assertEqual(2, len(tensor_prod))
        self.assertTrue(np.allclose(dictionary["site1"], tensor_prod["site1"]))
        self.assertEqual(dictionary["site2"], tensor_prod["site2"])

class TestTensorProduct(unittest.TestCase):

    def test_from_operator(self):
        rand = crandn((2,2))
        operators = [NumericOperator(rand, ["site1"]),
                     NumericOperator(rand, ["site2"])]
        tensor_prod = TensorProduct.from_operators(operators)
        self.assertEqual(2, len(tensor_prod))
        self.assertTrue(np.allclose(rand, tensor_prod["site1"]))
        self.assertTrue(np.allclose(rand, tensor_prod["site2"]))

        # With non-single site operator
        rand2 = crandn((2,2,2,2))
        operators.append(NumericOperator(rand2, ["site3", "site4"]))
        self.assertRaises(AssertionError, TensorProduct.from_operators,
                          operators)

    def test_into_operator_numeric(self):
        random_arrays = [crandn((2,2)),
                         crandn((3,3))]
        operators = [NumericOperator(random_arrays[i], "site" + str(i))
                     for i in range(len(random_arrays))]
        tensor_prod = TensorProduct.from_operators(operators)
        new_operator = tensor_prod.into_operator()
        correct_array = np.kron(random_arrays[0], random_arrays[1])
        self.assertTrue(np.allclose(correct_array, new_operator.operator))
        self.assertEqual(["site0", "site1"], new_operator.node_identifiers)

    def test_into_operator_symbolic(self):
        conversion_dict = {"op0": crandn((2,2)),
                           "op1": crandn((3,3))}
        tensor_prod = TensorProduct({"site0": "op0", "site1": "op1"})
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
        tensor_prod = TensorProduct({identifier: simple_matrix})
        
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
        tensor_product = TensorProduct({identifiers[0]: matrix1,
                                            identifiers[1]: matrix2})
        found_operator = tensor_product.exp()
        correct_result = expm(np.kron(matrix1, matrix2))
        self.assertTrue(np.allclose(correct_result, found_operator.operator))
        self.assertEqual(identifiers, found_operator.node_identifiers)

class TestTensorProductNaming(unittest.TestCase):
    """
    Tests functions that change the names of the nodes in the tensor product.
    """

    def test_add_suffix(self):
        """
        Test the add_suffix method of the TensorProduct class.
        """
        tp = TensorProduct({"node1": "A",
                            "node2": "B"})
        suff = "_suffix"
        tp_suffix = tp.add_suffix(suff)
        self.assertEqual(2, len(tp_suffix))
        self.assertEqual(tp["node1"], tp_suffix["node1"+suff])
        self.assertEqual(tp["node2"], tp_suffix["node2"+suff])

    def test_add_suffix_empty(self):
        """
        Test the add_suffix method of the TensorProduct class.
        """
        tp = TensorProduct({"node1": "A",
                            "node2": "B"})
        tp_suffix = tp.add_suffix("")
        self.assertEqual(tp, tp_suffix)

class TestTensorProductArithmetic(unittest.TestCase):

    def test_transpose(self):
        """
        Test the transpose method of the TensorProduct class.
        """
        tp = TensorProduct({"node1": "A",
                            "node2": "B"})
        sym_dict = {"A": False,
                    "B": True}
        tp_transpose = tp.transpose(sym_dict)
        self.assertEqual(2, len(tp_transpose))
        self.assertEqual("A_T", tp_transpose["node1"])
        self.assertEqual("B", tp_transpose["node2"])

    def test_conjugate(self):
        """
        Test the conjugate method of the TensorProduct class.
        """
        tp = TensorProduct({"node1": "A",
                            "node2": "B"})
        real_dict = {"A": False,
                     "B": True}
        tp_conj = tp.conjugate(real_dict)
        self.assertEqual(2, len(tp_conj))
        self.assertEqual("A_conj", tp_conj["node1"])
        self.assertEqual("B", tp_conj["node2"])

    def test_conjugate_transpose(self):
        """
        Test the conjugate_transpose method of the TensorProduct class.
        """
        tp = TensorProduct({"node1": "A",
                            "node2": "B"})
        herm_dict = {"A": False,
                     "B": True}
        tp_herm = tp.conjugate_transpose(herm_dict)
        self.assertEqual(2, len(tp_herm))
        self.assertEqual("A_H", tp_herm["node1"])
        self.assertEqual("B", tp_herm["node2"])

    def test_otimes(self):
        """
        Test the otimes method of the TensorProduct class.
        """
        tp1 = TensorProduct({"node1": "A",
                             "node2": "B"})
        tp2 = TensorProduct({"node3": "C",
                             "node4": "D"})
        tp_otimes = tp1.otimes(tp2)
        self.assertEqual(4, len(tp_otimes))
        self.assertEqual("A", tp_otimes["node1"])
        self.assertEqual("B", tp_otimes["node2"])
        self.assertEqual("C", tp_otimes["node3"])
        self.assertEqual("D", tp_otimes["node4"])
        self.assertIsNot(tp1, tp_otimes)

    def test_otimes_no_copy(self):
        """
        Test the otimes method of the TensorProduct class with a copied output.
        """
        tp1 = TensorProduct({"node1": "A",
                             "node2": "B"})
        tp2 = TensorProduct({"node3": "C",
                             "node4": "D"})
        tp_otimes = tp1.otimes(tp2, to_copy=False)
        self.assertEqual(4, len(tp_otimes))
        self.assertEqual("A", tp_otimes["node1"])
        self.assertEqual("B", tp_otimes["node2"])
        self.assertEqual("C", tp_otimes["node3"])
        self.assertEqual("D", tp_otimes["node4"])
        self.assertIs(tp1, tp_otimes)

    def test_otimes_same_nodeids(self):
        """
        Test the otimes method of the TensorProduct class with the same node
        identifiers.
        """
        tp1 = TensorProduct({"node1": "A",
                             "node2": "B"})
        tp2 = TensorProduct({"node1": "C",
                             "node2": "D"})
        self.assertRaises(ValueError, tp1.otimes, tp2)

    def test_multiply_no_identities(self):
        """
        Test the multiply method of the TensorProduct class with no identities.
        """
        tp1 = TensorProduct({"node1": "A",
                             "node2": "B"})
        tp2 = TensorProduct({"node1": "C",
                             "node2": "D"})
        tp_multiply = tp1.multiply(tp2)
        self.assertEqual(2, len(tp_multiply))
        self.assertEqual("A_mult_C", tp_multiply["node1"])
        self.assertEqual("B_mult_D", tp_multiply["node2"])

    def test_multiply_implicit_identities(self):
        """
        Test the multiply method of the TensorProduct class with implicit
        identities.
        """
        tp1 = TensorProduct({"node1": "A",
                             "node2": "B"})
        tp2 = TensorProduct({"node3": "C",
                             "node4": "D"})
        tp_multiply = tp1.multiply(tp2)
        self.assertEqual(4, len(tp_multiply))
        self.assertEqual("A", tp_multiply["node1"])
        self.assertEqual("B", tp_multiply["node2"])
        self.assertEqual("C", tp_multiply["node3"])
        self.assertEqual("D", tp_multiply["node4"])

    def test_multiply_explicit_identities(self):
        """
        Test the multiply method of the TensorProduct class with explicit
        identities.
        """
        tp1 = TensorProduct({"node1": "A",
                             "node2": "B"})
        tp2 = TensorProduct({"node1": "C",
                             "node2": "D"})
        id_dict = {"A": True,
                   "B": False,
                   "C": False,
                   "D": True}
        tp_multiply = tp1.multiply(tp2, id_dict)
        self.assertEqual(2, len(tp_multiply))
        self.assertEqual("C", tp_multiply["node1"])
        self.assertEqual("B", tp_multiply["node2"])

    def test_dunder_multiply(self):
        """
        Tests the use of @ for tensor products.
        """
        tp1 = TensorProduct({"node1": "A",
                             "node2": "B"})
        tp2 = TensorProduct({"node1": "C",
                             "node3": "D"})
        tp_multiply = tp1 @ tp2
        self.assertEqual(3, len(tp_multiply))
        self.assertEqual("A_mult_C", tp_multiply["node1"])
        self.assertEqual("B", tp_multiply["node2"])
        self.assertEqual("D", tp_multiply["node3"])

class TestTensorProductWithTTN(unittest.TestCase):
    def setUp(self):
        self.ttn = TreeTensorNetwork()
        self.ttn.add_root(Node(identifier="root"),
            crandn((2,2,3)))
        self.ttn.add_child_to_parent(Node(identifier="c1"),
            crandn((2,2)), 0, "root", 0)
        self.ttn.add_child_to_parent(Node(identifier="c2"),
            crandn((2,4)), 0, "root", 1)

        self.symbolic_dict = {"root": "X",
                         "c1": "a^dagger",
                         "c2": "H"}

    def test_pad_with_identity_no_pad_needed(self):
        ten_prod = TensorProduct(self.symbolic_dict)
        padded_tp = ten_prod.pad_with_identities(self.ttn)
        # Since no padding is needed, nothing should change
        self.assertEqual(ten_prod, padded_tp)

    def test_pad_with_identity_new_node_numeric(self):
        del self.symbolic_dict["c2"]
        ten_prod = TensorProduct(self.symbolic_dict)
        padded_tp = ten_prod.pad_with_identities(self.ttn)
        self.assertTrue("c2" in padded_tp.keys())
        self.assertTrue(np.allclose(np.eye(4), padded_tp["c2"]))

    def test_pad_with_identity_new_node_symbolic(self):
        del self.symbolic_dict["c2"]
        ten_prod = TensorProduct(self.symbolic_dict)
        padded_tp = ten_prod.pad_with_identities(self.ttn, symbolic=True)
        self.assertTrue("c2" in padded_tp.keys())
        self.assertEqual("I4", padded_tp["c2"])

    def test_pad_with_identity_node_not_in_ttn(self):
        self.symbolic_dict["wronged"] = "P"
        ten_prod = TensorProduct(self.symbolic_dict)
        self.assertRaises(KeyError, ten_prod.pad_with_identities, self.ttn)

if __name__ == "__main__":
    unittest.main()
