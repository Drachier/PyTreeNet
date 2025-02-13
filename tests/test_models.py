from unittest import TestCase, main as unitmain
from fractions import Fraction

from pytreenet.core.tree_structure import TreeStructure
from pytreenet.core.graph_node import GraphNode

from pytreenet.operators.models import (ising_model,
                                        flipped_ising_model)

class TestIsingModel(TestCase):

    def test_ising_for_list(self):
        """
        Test the creation of the ising model for a given list of nearest
        neighbours.
        """
        test_list = [("A", "B"), ("B", "C"), ("C", "D")]
        hamiltonian = ising_model(test_list, 3.0, factor=2.0)
        self.assertEqual(len(hamiltonian.terms), 7)
        single_terms = [term for term in hamiltonian.terms
                        if len(term[2]) == 1]
        self.assertEqual(len(single_terms), 4)
        for term in single_terms:
            self.assertEqual(term[0], Fraction(-1))
            self.assertEqual(term[1], "ext_magn")
            self.assertIn(list(term[2].keys())[0], ["A", "B", "C", "D"])
            self.assertEqual(list(term[2].values())[0], "Z" )
        nn_terms = [term for term in hamiltonian.terms
                        if len(term[2]) == 2]
        self.assertEqual(len(nn_terms), 3)
        for term in nn_terms:
            self.assertEqual(term[0], Fraction(-1))
            self.assertEqual(term[1], "coupling")
            self.assertTrue(all([node_id in ["A", "B", "C", "D"]
                                for node_id in term[2].keys()]))
            self.assertTrue(all([op == "X" for op in term[2].values()]))

class TestFlippedIsingModel(TestCase):

    def test_flipped_ising_for_list(self):
        """
        Test the creation of the flipped ising model for a given list of
        nearest neighbours.
        """
        test_list = [("A", "B"), ("B", "C"), ("C", "D")]
        hamiltonian = flipped_ising_model(test_list, 3.0, factor=2.0)
        self.assertEqual(len(hamiltonian.terms), 7)
        single_terms = [term for term in hamiltonian.terms
                        if len(term[2]) == 1]
        self.assertEqual(len(single_terms), 4)
        for term in single_terms:
            self.assertEqual(term[0], Fraction(-1))
            self.assertEqual(term[1], "ext_magn")
            self.assertIn(list(term[2].keys())[0], ["A", "B", "C", "D"])
            self.assertEqual(list(term[2].values())[0], "X" )
        nn_terms = [term for term in hamiltonian.terms
                        if len(term[2]) == 2]
        self.assertEqual(len(nn_terms), 3)
        for term in nn_terms:
            self.assertEqual(term[0], Fraction(-1))
            self.assertEqual(term[1], "coupling")
            self.assertTrue(all([node_id in ["A", "B", "C", "D"]
                                for node_id in term[2].keys()]))
            self.assertTrue(all([op == "Z" for op in term[2].values()]))

if __name__ == '__main__':
    unitmain()
