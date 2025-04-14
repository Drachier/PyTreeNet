from unittest import TestCase, main as unitmain
from fractions import Fraction

from numpy import eye

from pytreenet.operators.common_operators import pauli_matrices

from pytreenet.operators.models import (ising_model,
                                        flipped_ising_model,
                                        _grid_from_structure,
                                        _find_nn_pairs,
                                        ising_model_2D,
                                        flipped_ising_model_2D)

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

class Test2DModels(TestCase):
    """
    Test the generation of 2D ising models and associated utility functions.
    """

    def test_grid_from_structure_square(self):
        """
        Test if the correct grid of identifiers is generated.
        """
        prefix = "node"
        height = 2
        width = 2
        grid = _grid_from_structure(prefix, height, width)
        self.assertEqual(grid.shape, (2,2))
        self.assertEqual(grid[0,0], "node0_0")
        self.assertEqual(grid[0,1], "node0_1")
        self.assertEqual(grid[1,0], "node1_0")
        self.assertEqual(grid[1,1], "node1_1")

    def test_grid_from_structure_rectangular(self):
        """
        Test if the correct grid of identifiers is generated.
        """
        prefix = "node"
        height = 2
        width = 3
        grid = _grid_from_structure(prefix, height, width)
        self.assertEqual(grid.shape, (2,3))
        self.assertEqual(grid[0,0], "node0_0")
        self.assertEqual(grid[0,1], "node0_1")
        self.assertEqual(grid[1,0], "node1_0")
        self.assertEqual(grid[1,1], "node1_1")
        self.assertEqual(grid[1,2], "node1_2")
        self.assertEqual(grid[1,2], "node1_2")

    def test_find_nn_pairs(self):
        """
        Makes sure that for a given grid, the correct nn pairs are found.
        """
        grid = _grid_from_structure("node", 2, 3)
        pairs = _find_nn_pairs(grid)
        self.assertEqual(len(pairs), 7)
        self.assertIn(("node0_0", "node0_1"), pairs)
        self.assertIn(("node0_0", "node1_0"), pairs)
        self.assertIn(("node1_0", "node1_1"), pairs)
        self.assertIn(("node0_1", "node1_1"), pairs)
        self.assertIn(("node0_1", "node0_2"), pairs)
        self.assertIn(("node1_1", "node1_2"), pairs)
        self.assertIn(("node0_2", "node1_2"), pairs)

    def test_find_nn_pairs_1D(self):
        """
        Makes sure that for a 1D grid all pairs are found.
        """
        grid = _grid_from_structure("node", 1, 3)
        pairs = _find_nn_pairs(grid)
        self.assertEqual(len(pairs), 2)
        self.assertIn(("node0_0", "node0_1"), pairs)
        self.assertIn(("node0_1", "node0_2"), pairs)

    def test_find_nn_pairs_one_site(self):
        """
        Tests that for a grid of one site no pairs are found.
        """
        grid = _grid_from_structure("node", 1, 1)
        pairs = _find_nn_pairs(grid)
        self.assertEqual(len(pairs), 0)

    def test_ising_2D(self):
        """
        Tests the generation of a 2D ising model for a small grid.
        """
        is_ham = ising_model_2D(("node",2,3),
                                0.5,
                                1.3)
        self.assertEqual(len(is_ham.terms), 13)
        # Single terms
        single_terms = [term for term in is_ham.terms
                        if len(term[2]) == 1]
        self.assertEqual(len(single_terms), 6)
        node_ids = [f"node{i}_{j}"
                    for i in range(2) for j in range(3)]
        for term in single_terms:
            self.assertEqual(term[0], Fraction(-1))
            self.assertEqual(term[1], "ext_magn")
            self.assertIn(list(term[2].keys())[0], node_ids)
            self.assertEqual(list(term[2].values())[0], "Z")
            node_ids.remove(list(term[2].keys())[0])
        ## Assure every node has a term
        self.assertEqual(0, len(node_ids))
        # NN terms
        nn_terms = [term for term in is_ham.terms
                    if len(term[2]) == 2]
        self.assertEqual(len(nn_terms), 7)
        pairs = [("node0_0", "node0_1"),
                    ("node0_0", "node1_0"),
                    ("node1_0", "node1_1"),
                    ("node0_1", "node1_1"),
                    ("node0_1", "node0_2"),
                    ("node1_1", "node1_2"),
                    ("node0_2", "node1_2")]
        for term in nn_terms:
            self.assertEqual(term[0], Fraction(-1))
            self.assertEqual(term[1],"coupling")
            term_pair = tuple(term[2].keys())
            self.assertTrue((term_pair in pairs or (term_pair[1], term_pair[0]) in pairs))
            for node_id in term_pair:
                self.assertEqual(term[2][node_id], "X")
        # Conv dictionary
        self.assertEqual(4, len(is_ham.conversion_dictionary))
        conv_dict = {"X": pauli_matrices()[0],
                     "Z": pauli_matrices()[2],
                     "I1": eye(1),
                     "I2": eye(2)}
        for key, val in conv_dict.items():
            self.assertIn(key, is_ham.conversion_dictionary)
            self.assertTrue((val == is_ham.conversion_dictionary[key]).all())
        # Coeff Mapping
        self.assertEqual(3, len(is_ham.coeffs_mapping))
        coeff_dict = {"ext_magn": 0.5,
                      "coupling": 1.3,
                      "1": 1.0}
        for key, val in coeff_dict.items():
            self.assertIn(key, is_ham.coeffs_mapping)
            self.assertTrue(val == is_ham.coeffs_mapping[key])

    def test_flipped_ising_2D(self):
        """
        Tests the generation of a 2D ising model for a small grid.
        """
        is_ham = flipped_ising_model_2D(("node",2,3),
                                            0.5,
                                            1.3)
        self.assertEqual(len(is_ham.terms), 13)
        # Single terms
        single_terms = [term for term in is_ham.terms
                        if len(term[2]) == 1]
        self.assertEqual(len(single_terms), 6)
        node_ids = [f"node{i}_{j}"
                    for i in range(2) for j in range(3)]
        for term in single_terms:
            self.assertEqual(term[0], Fraction(-1))
            self.assertEqual(term[1], "ext_magn")
            self.assertIn(list(term[2].keys())[0], node_ids)
            self.assertEqual(list(term[2].values())[0], "X")
            node_ids.remove(list(term[2].keys())[0])
        ## Assure every node has a term
        self.assertEqual(0, len(node_ids))
        # NN terms
        nn_terms = [term for term in is_ham.terms
                    if len(term[2]) == 2]
        self.assertEqual(len(nn_terms), 7)
        pairs = [("node0_0", "node0_1"),
                    ("node0_0", "node1_0"),
                    ("node1_0", "node1_1"),
                    ("node0_1", "node1_1"),
                    ("node0_1", "node0_2"),
                    ("node1_1", "node1_2"),
                    ("node0_2", "node1_2")]
        for term in nn_terms:
            self.assertEqual(term[0], Fraction(-1))
            self.assertEqual(term[1],"coupling")
            term_pair = tuple(term[2].keys())
            self.assertTrue((term_pair in pairs or (term_pair[1], term_pair[0]) in pairs))
            for node_id in term_pair:
                self.assertEqual(term[2][node_id], "Z")
        # Conv dictionary
        self.assertEqual(4, len(is_ham.conversion_dictionary))
        conv_dict = {"X": pauli_matrices()[0],
                     "Z": pauli_matrices()[2],
                     "I1": eye(1),
                     "I2": eye(2)}
        for key, val in conv_dict.items():
            self.assertIn(key, is_ham.conversion_dictionary)
            self.assertTrue((val == is_ham.conversion_dictionary[key]).all())
        # Coeff Mapping
        self.assertEqual(3, len(is_ham.coeffs_mapping))
        coeff_dict = {"ext_magn": 0.5,
                      "coupling": 1.3,
                      "1": 1.0}
        for key, val in coeff_dict.items():
            self.assertIn(key, is_ham.coeffs_mapping)
            self.assertTrue(val == is_ham.coeffs_mapping[key])

if __name__ == '__main__':
    unitmain()
