import unittest

from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.core.tree_structure import TreeStructure
from pytreenet.core.graph_node import GraphNode
from pytreenet.operators.sim_operators import (single_site_operators,
                                               single_site_operator_all_sites)

class TestSimOperators(unittest.TestCase):

    def test_single_site_operators_no_names(self):
        """
        Tests the function without operator names being given.
        """
        operator = "M"
        node_identifiers = ["A", "B", "C"]
        operators = single_site_operators(operator, node_identifiers)
        self.assertEqual(len(operators), 3)
        self.assertTrue(all(isinstance(op, TensorProduct)
                            for op in operators.values()))
        self.assertTrue(all(len(op) == 1
                            for op in operators.values()))
        self.assertTrue(all(node in operators
                            for node in node_identifiers))
        self.assertTrue(all(operators[node] == TensorProduct({node:operator})
                            for node in node_identifiers))

    def test_single_site_operators_with_names(self):
        """
        Tests the function with operator names being given.
        """
        operator = "M"
        node_identifiers = ["A", "B", "C"]
        operator_names = ["M1", "M2", "M3"]
        operators = single_site_operators(operator, node_identifiers, operator_names)
        self.assertEqual(len(operators), 3)
        self.assertTrue(all(isinstance(op, TensorProduct)
                            for op in operators.values()))
        self.assertTrue(all(len(op) == 1
                            for op in operators.values()))
        self.assertTrue(all(op_name in operators
                            for op_name in operator_names))
        self.assertTrue(all(operators[op_name] == TensorProduct({node_identifiers[i]:operator})
                            for i, op_name in enumerate(operator_names)))

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
        operators = single_site_operator_all_sites(operator, tree)
        self.assertEqual(len(operators), 5)
        self.assertTrue(all(isinstance(op, TensorProduct)
                            for op in operators.values()))
        self.assertTrue(all(len(op) == 1
                            for op in operators.values()))
        self.assertTrue(all(node in operators
                            for node in tree.nodes))
        self.assertTrue(all(operators[node] == TensorProduct({node:operator})
                            for node in tree.nodes))


if __name__ == "__main__":
    unittest.main()
