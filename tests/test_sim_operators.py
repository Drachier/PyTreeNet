import unittest
from copy import deepcopy
from fractions import Fraction

from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.core.tree_structure import TreeStructure
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.core.graph_node import GraphNode
from pytreenet.operators.sim_operators import (single_site_operators,
                                               create_nearest_neighbour_hamiltonian,
                                               create_single_site_hamiltonian)

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
        tree = TreeStructure()
        tree.add_root(GraphNode("A"))
        tree.add_child_to_parent(GraphNode("B"),"A")
        tree.add_child_to_parent(GraphNode("C"),"A")
        tree.add_child_to_parent(GraphNode("D"),"B")
        tree.add_child_to_parent(GraphNode("E"),"D")
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
    tree = TreeStructure()
    tree.add_root(GraphNode("A"))
    for i in range(1, n_sites):
        tree.add_child_to_parent(GraphNode(chr(65+i)), chr(65+i-1))
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
    tree = TreeStructure()
    tree.add_root(GraphNode("A"))
    tree.add_child_to_parent(GraphNode("B"),"A")
    tree.add_child_to_parent(GraphNode("C"),"B")
    tree.add_child_to_parent(GraphNode("D"),"B")
    tree.add_child_to_parent(GraphNode("E"),"A")
    tree.add_child_to_parent(GraphNode("F"),"E")
    tree.add_child_to_parent(GraphNode("G"),"E")
    tree.add_child_to_parent(GraphNode("H"),"E")
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
        print(found_ham)
        print(10*"-")
        print(ref_ham)
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
        tree = mps_structure(num_sites)
        local_operator = "op"
        local_operator2 = "op2"
        found_ham = create_nearest_neighbour_hamiltonian(tree, local_operator,
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
        tree = complicated_tree_structure()
        local_operator = "op"
        local_operator2 = "op2"
        found_ham = create_nearest_neighbour_hamiltonian(tree, local_operator,
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

if __name__ == "__main__":
    unittest.main()
