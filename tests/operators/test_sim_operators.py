import unittest
from copy import deepcopy
from fractions import Fraction

import numpy as np

from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.core.tree_structure import TreeStructure
from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.random import random_tensor_node
from pytreenet.operators.sim_operators import (single_site_operators,
                                               create_nearest_neighbour_hamiltonian,
                                               create_single_site_hamiltonian,
                                               create_multi_site_hamiltonian)

class TestSimOperators(unittest.TestCase):

    def test_single_site_operators_no_names(self):
        """
        Tests the function without operator names being given.
        """
        operator = "M"
        node_identifiers = ["A", "B", "C"]
        operators = single_site_operators(operator, node_identifiers)
        self.assertEqual(len(operators), 3)
        self.assertTrue(all(Fraction(1) == op[0]
                            for op in operators.values()))
        self.assertTrue(all("1" == op[1]
                            for op in operators.values()))
        self.assertTrue(all(node in operators
                            for node in node_identifiers))
        self.assertTrue(all(len(op[2]) == 1
                            for op in operators.values()))
        self.assertTrue(all(operators[node][2] == TensorProduct({node:operator})
                            for node in node_identifiers))

    def test_single_site_operators_with_names(self):
        """
        Tests the function with operator names being given.
        """
        operator = "M"
        node_identifiers = ["A", "B", "C"]
        operator_names = ["M1", "M2", "M3"]
        operators = single_site_operators(operator, node_identifiers,
                                          operator_names=operator_names)
        self.assertEqual(len(operators), 3)
        self.assertTrue(all(Fraction(1) == op[0]
                            for op in operators.values()))
        self.assertTrue(all("1" == op[1]
                            for op in operators.values()))
        self.assertTrue(all(op_name in operators
                            for op_name in operator_names))
        self.assertTrue(all(len(op[2]) == 1
                            for op in operators.values()))
        self.assertTrue(all(operators[op_name][2] == TensorProduct({node_identifiers[i]:operator})
                            for i, op_name in enumerate(operator_names)))

    def test_single_site_operators_with_factor(self):
        """
        Tests that the function works with one constant factor for all terms.
        """
        operator = "M"
        node_identifiers = ["A", "B", "C"]
        factor = (Fraction(2), "2")
        operators = single_site_operators(operator,
                                          node_identifiers,
                                          factor=factor)
        self.assertEqual(len(operators), 3)
        self.assertTrue(all(Fraction(2) == op[0]
                            for op in operators.values()))
        self.assertTrue(all("2" == op[1]
                            for op in operators.values()))
        self.assertTrue(all(len(op[2]) == 1
                            for op in operators.values()))
        self.assertTrue(all(operators[node][2] == TensorProduct({node:operator})
                            for node in node_identifiers))

    def test_single_site_operators_with_multiple_factors(self):
        """
        Tests that the function works with one factor for each term.
        """
        operator = "M"
        node_identifiers = ["A", "B", "C"]
        factor = [(Fraction(2),"2"),
                  (Fraction(3),"3"),
                  (Fraction(4),"4")]
        operators = single_site_operators(operator,
                                          node_identifiers,
                                          factor=factor)
        self.assertEqual(len(operators), 3)
        self.assertTrue(all(Fraction(i+2) == op[0]
                            for i,op in enumerate(operators.values())))
        self.assertTrue(all(str(i+2) == op[1]
                            for i, op in enumerate(operators.values())))
        self.assertTrue(all(len(op[2]) == 1
                            for op in operators.values()))
        self.assertTrue(all(operators[node][2] == TensorProduct({node:operator})
                            for node in node_identifiers))

    def test_single_site_operator_all_sites(self):
        """
        Tests, that a tree structure gets the operator assigned to every node.
        """
        operator = "M"
        tree = TreeTensorNetwork()
        node, tensor = random_tensor_node((2,2,2),identifier="A")
        tree.add_root(node, tensor)
        node, tensor = random_tensor_node((2,2,2),identifier="B")
        tree.add_child_to_parent(node,tensor,0,"A",0)
        node, tensor = random_tensor_node((2,2), identifier="C")
        tree.add_child_to_parent(node,tensor,0,"A",1)
        node, tensor = random_tensor_node((2,2,2), identifier="D")
        tree.add_child_to_parent(node,tensor,0,"B",1)
        node, tensor = random_tensor_node((2,2), identifier="E")
        tree.add_child_to_parent(node,tensor,0,"D",1)
        operators = single_site_operators(operator, tree)
        self.assertEqual(len(operators), 5)
        self.assertTrue(all(Fraction(1) == op[0]
                            for op in operators.values()))
        self.assertTrue(all("1" == op[1]
                            for op in operators.values()))
        self.assertTrue(all(node in operators
                            for node in tree.nodes))
        self.assertTrue(all(len(op[2]) == 1
                            for op in operators.values()))
        self.assertTrue(all(operators[node][2] == TensorProduct({node:operator})
                            for node in tree.nodes))

def mps_structure(n_sites: int) -> TreeStructure:
    """
    Generates an mps like tree structure with n_sites sites.

    Args:
        n_sites (int): The number of sites.
    
    Returns:
        TreeStructure: The tree structure.

    """
    tree = TreeTensorNetwork()
    node, tensor = random_tensor_node((2,2), identifier="A")
    tree.add_root(node, tensor)
    for i in range(1, n_sites-1):
        node, tensor = random_tensor_node((2,2,2), identifier=chr(65+i))
        tree.add_child_to_parent(node, tensor, 0, chr(65+i-1), 1)
    node, tensor = random_tensor_node((2,2), identifier=chr(65+n_sites-1))
    tree.add_child_to_parent(node, tensor, 0, chr(65+n_sites-2), 1)
    return tree

def complicated_tree_structure() -> TreeStructure:
    """
    Generates a complicated tree structure.

    Returns:
        TreeStructure: The tree structure.

            A    H
           / \\ /
          B    E---G
         / \\   \\
        C   D    F
    """
    tree = TreeTensorNetwork()
    node, tensor = random_tensor_node((2,2,2), identifier="A")
    tree.add_root(node,tensor)
    node, tensor = random_tensor_node((2,2,2,2), identifier="B")
    tree.add_child_to_parent(node,tensor,0,"A",0)
    node, tensor = random_tensor_node((2,2), identifier="C")
    tree.add_child_to_parent(node,tensor,0,"B",1)
    node, tensor = random_tensor_node((2,2), identifier="D")
    tree.add_child_to_parent(node,tensor,0,"B",2)
    node, tensor = random_tensor_node((2,2,2,2,2), identifier="E")
    tree.add_child_to_parent(node,tensor,0,"A",1)
    node, tensor = random_tensor_node((2,2), identifier="F")
    tree.add_child_to_parent(node,tensor,0,"E",1)
    node, tensor = random_tensor_node((2,2), identifier="G")
    tree.add_child_to_parent(node,tensor,0,"E",2)
    node, tensor = random_tensor_node((2,2), identifier="H")
    tree.add_child_to_parent(node,tensor,0,"E",3)
    return tree

class TestNearestNeighbourHamiltonian(unittest.TestCase):

    def test_nearest_neighbour_hamiltonian_mps_structure(self):
        """
        Tests the nearest neighbour Hamiltonian for an MPS like structure.
        """
        num_sites = 5
        tree = mps_structure(num_sites)
        local_operator = "op"
        found_ham = create_nearest_neighbour_hamiltonian(tree, local_operator)
        self.assertEqual(len(found_ham.terms), num_sites-1)
        # Reference Hamiltonian
        terms = [TensorProduct({"A": local_operator, "B": local_operator}),
                 TensorProduct({"B": local_operator, "C": local_operator}),
                 TensorProduct({"C": local_operator, "D": local_operator}),
                 TensorProduct({"D": local_operator, "E": local_operator})]
        ref_ham = Hamiltonian(terms)
        # Testing
        self.assertEqual(found_ham, ref_ham)
        self.assertEqual(found_ham.conversion_dictionary, {})

    def test_nearest_neighbour_hamiltonian_mps_structure_with_conversion(self):
        """
        Tests the nearest neighbour Hamiltonian for an MPS like structure with
        conversion dictionary.
        """
        num_sites = 5
        tree = mps_structure(num_sites)
        local_operator = "op"
        conversion_dict = {"op": [[1, 0], [0, 1]]}
        ref_conv = deepcopy(conversion_dict)
        found_ham = create_nearest_neighbour_hamiltonian(tree, local_operator,
                                                         conversion_dict=conversion_dict)
        self.assertEqual(len(found_ham.terms), num_sites-1)
        # Reference Hamiltonian
        terms = [TensorProduct({"A": local_operator, "B": local_operator}),
                 TensorProduct({"B": local_operator, "C": local_operator}),
                 TensorProduct({"C": local_operator, "D": local_operator}),
                 TensorProduct({"D": local_operator, "E": local_operator})]
        ref_ham = Hamiltonian(terms)
        # Testing
        self.assertEqual(found_ham, ref_ham)
        self.assertEqual(found_ham.conversion_dictionary,
                         ref_conv)

    def test_nearest_neighbour_hamiltonian_mps_structure_diff_ops(self):
        """
        Test the nearest neighbour Hamiltonian for an MPS like structure with
        different operators for each neighbour.
        """
        num_sites = 5
        local_operator = "op"
        local_operator2 = "op2"
        structure = [("A","B"),("B","C"),("C","D"),("D","E")]
        found_ham = create_nearest_neighbour_hamiltonian(structure,
                                                         local_operator,
                                                         local_operator2=local_operator2)
        self.assertEqual(len(found_ham.terms), num_sites-1)
        # Reference Hamiltonian
        terms = [TensorProduct({"A": local_operator, "B": local_operator2}),
                 TensorProduct({"B": local_operator, "C": local_operator2}),
                 TensorProduct({"C": local_operator, "D": local_operator2}),
                 TensorProduct({"D": local_operator, "E": local_operator2})]
        ref_ham = Hamiltonian(terms)
        # Testing
        self.assertEqual(found_ham, ref_ham)
        self.assertEqual(found_ham.conversion_dictionary, {})

    def test_nearest_neighbour_hamiltonian_complicated_structure(self):
        """
        Test the nearest neighbour Hamiltonian for a complicated tree structure.
        """
        tree = complicated_tree_structure()
        local_operator = "op"
        found_ham = create_nearest_neighbour_hamiltonian(tree, local_operator)
        self.assertEqual(len(found_ham.terms), 7)
        # Reference Hamiltonian
        terms = [TensorProduct({"A": local_operator, "B": local_operator}),
                 TensorProduct({"B": local_operator, "C": local_operator}),
                 TensorProduct({"B": local_operator, "D": local_operator}),
                 TensorProduct({"A": local_operator, "E": local_operator}),
                 TensorProduct({"E": local_operator, "F": local_operator}),
                 TensorProduct({"E": local_operator, "G": local_operator}),
                 TensorProduct({"E": local_operator, "H": local_operator})]
        ref_ham = Hamiltonian(terms)
        # Testing
        self.assertEqual(found_ham, ref_ham)
        self.assertEqual(found_ham.conversion_dictionary, {})

    def test_nearest_neighbour_hamiltonian_complicated_structure_with_conversion(self):
        """
        Test the nearest neighbour Hamiltonian for a complicated tree structure
        with conversion dictionary.
        """
        tree = complicated_tree_structure()
        local_operator = "op"
        conversion_dict = {"op": [[1, 0], [0, 1]]}
        ref_conv = deepcopy(conversion_dict)
        found_ham = create_nearest_neighbour_hamiltonian(tree, local_operator,
                                                         conversion_dict=conversion_dict)
        self.assertEqual(len(found_ham.terms), 7)
        # Reference Hamiltonian
        terms = [TensorProduct({"A": local_operator, "B": local_operator}),
                 TensorProduct({"B": local_operator, "C": local_operator}),
                 TensorProduct({"B": local_operator, "D": local_operator}),
                 TensorProduct({"A": local_operator, "E": local_operator}),
                 TensorProduct({"E": local_operator, "F": local_operator}),
                 TensorProduct({"E": local_operator, "G": local_operator}),
                 TensorProduct({"E": local_operator, "H": local_operator})]
        ref_ham = Hamiltonian(terms)
        # Testing
        self.assertEqual(found_ham, ref_ham)
        self.assertEqual(found_ham.conversion_dictionary,
                         ref_conv)

    def test_nearest_neighbour_hamiltonian_complicated_structure_diff_ops(self):
        """
        Test the nearest neighbour Hamiltonian for a complicated tree structure
        with different operators for each neighbour.
        """
        local_operator = "op"
        local_operator2 = "op2"
        structure = [("A","B"),("B","C"),
                     ("B","D"),("A","E"),
                     ("E","F"),("E","G"),
                     ("E","H")]
        found_ham = create_nearest_neighbour_hamiltonian(structure,
                                                         local_operator,
                                                         local_operator2=local_operator2)
        self.assertEqual(len(found_ham.terms), 7)
        # Reference Hamiltonian
        terms = [TensorProduct({"A": local_operator, "B": local_operator2}),
                 TensorProduct({"B": local_operator, "C": local_operator2}),
                 TensorProduct({"B": local_operator, "D": local_operator2}),
                 TensorProduct({"A": local_operator, "E": local_operator2}),
                 TensorProduct({"E": local_operator, "F": local_operator2}),
                 TensorProduct({"E": local_operator, "G": local_operator2}),
                 TensorProduct({"E": local_operator, "H": local_operator2})]
        ref_ham = Hamiltonian(terms)
        # Testing
        self.assertEqual(found_ham, ref_ham)
        self.assertEqual(found_ham.conversion_dictionary, {})

class TestSingleSiteHamiltonian(unittest.TestCase):

    def test_single_site_hamiltonian_mps_structure(self):
        """
        Tests the single site Hamiltonian for an MPS like structure.
        """
        num_sites = 5
        tree = mps_structure(num_sites)
        local_operator = "op"
        found_ham = create_single_site_hamiltonian(tree, local_operator)
        self.assertEqual(len(found_ham.terms), num_sites)
        # Reference Hamiltonian
        terms = [TensorProduct({"A": local_operator}),
                 TensorProduct({"B": local_operator}),
                 TensorProduct({"C": local_operator}),
                 TensorProduct({"D": local_operator}),
                 TensorProduct({"E": local_operator})]
        ref_ham = Hamiltonian(terms)
        # Testing
        self.assertEqual(found_ham, ref_ham)
        self.assertEqual(found_ham.conversion_dictionary, {})

    def test_single_site_hamiltonian_mps_structure_with_conversion(self):
        """
        Tests the single site Hamiltonian for an MPS like structure with
        conversion dictionary.
        """
        num_sites = 5
        tree = mps_structure(num_sites)
        local_operator = "op"
        conversion_dict = {"op": [[1, 0], [0, 1]]}
        ref_conv = deepcopy(conversion_dict)
        found_ham = create_single_site_hamiltonian(tree, local_operator,
                                                   conversion_dict=conversion_dict)
        self.assertEqual(len(found_ham.terms), num_sites)
        # Reference Hamiltonian
        terms = [TensorProduct({"A": local_operator}),
                 TensorProduct({"B": local_operator}),
                 TensorProduct({"C": local_operator}),
                 TensorProduct({"D": local_operator}),
                 TensorProduct({"E": local_operator})]
        ref_ham = Hamiltonian(terms)
        # Testing
        self.assertEqual(found_ham, ref_ham)
        self.assertEqual(found_ham.conversion_dictionary,
                         ref_conv)

    def test_single_site_hamiltonian_complicated_structure(self):
        """
        Test the single site Hamiltonian for a complicated tree structure.
        """
        tree = complicated_tree_structure()
        local_operator = "op"
        found_ham = create_single_site_hamiltonian(tree, local_operator)
        self.assertEqual(len(found_ham.terms), 8)
        # Reference Hamiltonian
        terms = [TensorProduct({"A": local_operator}),
                 TensorProduct({"B": local_operator}),
                 TensorProduct({"C": local_operator}),
                 TensorProduct({"D": local_operator}),
                 TensorProduct({"E": local_operator}),
                 TensorProduct({"F": local_operator}),
                 TensorProduct({"G": local_operator}),
                 TensorProduct({"H": local_operator})]
        ref_ham = Hamiltonian(terms)
        # Testing
        self.assertEqual(found_ham, ref_ham)
        self.assertEqual(found_ham.conversion_dictionary, {})

class TestMultiSiteHamiltonian(unittest.TestCase):
    """
    Tests the creation of multi-site Hamiltonians.
    """

    def setUp(self):
        self.node_ids = [["1","2","3"],
                         ["2","3","4"],
                         ["4","5","6"],
                         ["3","2","5"]]

    def test_constant_hamiltonian(self):
        """
        Tests the creation for only one operator.
        """
        op = "X"
        ham = create_multi_site_hamiltonian(self.node_ids,
                                            op)
        correct = [(1,"1",TensorProduct({nid[0]: op,
                                         nid[1]: op,
                                         nid[2]: op}))
                    for nid in self.node_ids]
        self.assertIn(correct[0], ham.terms)
        self.assertIn(correct[1], ham.terms)
        self.assertIn(correct[2], ham.terms)
        self.assertIn(correct[3], ham.terms)

    def test_non_constant_opstring(self):
        """
        Test the creation of a multi-site hamiltonian, where the operators are
        not all the same.
        """
        ops = ["X","Y","Z"]
        correct = [(1,"1",TensorProduct({nid[0]: ops[0],
                                         nid[1]: ops[1],
                                         nid[2]: ops[2]}))
                    for nid in self.node_ids]
        ham = create_multi_site_hamiltonian(self.node_ids,
                                            ops)
        self.assertIn(correct[0], ham.terms)
        self.assertIn(correct[1], ham.terms)
        self.assertIn(correct[2], ham.terms)
        self.assertIn(correct[3], ham.terms)

    def test_conv_dict_is_added(self):
        """
        Tests that a provided conversion dictionarty is included in the
        Hamiltonian.
        """
        conv_dict = {"X": np.zeros((2,2)),
                     "Y": np.ones((2,2))}
        ops = ["X","Y","Z"]
        ham = create_multi_site_hamiltonian(self.node_ids,
                                            ops,
                                            conversion_dict=deepcopy(conv_dict))
        self.assertEqual(len(conv_dict), len(ham.conversion_dictionary))
        for op, val in conv_dict.items():
            self.assertIn(op, ham.conversion_dictionary)
            np.testing.assert_array_equal(ham.conversion_dictionary[op],
                                          val)

    def test_coeff_map_is_added(self):
        """
        Tests that a provided coefficient map is added to the resulting
        Hamiltonian.
        """
        coeff_map = {"gamma": -1,
                     "L": 2+1j,
                     "1": 1} # Standard always there
        ops = ["X","Y","Z"]
        ham = create_multi_site_hamiltonian(self.node_ids,
                                            ops,
                                            coeffs_mapping=deepcopy(coeff_map))
        self.assertEqual(coeff_map, ham.coeffs_mapping)

    def test_invalid_node_combination(self):
        """
        If node identifiers combinations of different length are provided an
        error schould be raised.
        """
        node_ids = [["1","2","3"],
                         ["2","3"]]
        op = "X"
        self.assertRaises(ValueError,create_multi_site_hamiltonian,
                          node_ids,op)

    def test_invalid_operator_combination(self):
        """
        If the operator combination is of different length to the node
        identifiers an error should be raised.
        """
        ops = ["X","Y"]
        self.assertRaises(ValueError,create_multi_site_hamiltonian,
                          self.node_ids,ops)

if __name__ == "__main__":
    unittest.main()
