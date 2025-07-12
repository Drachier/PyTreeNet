"""
This module contains unit tests for the Hamiltonian generation functions.
"""
from unittest import TestCase, main as unitmain
from fractions import Fraction

from numpy import eye
import numpy as np

from pytreenet.operators.common_operators import (pauli_matrices,
                                                  bosonic_operators)

from pytreenet.operators.models import (ising_model,
                                        flipped_ising_model,
                                        _grid_from_structure,
                                        _find_nn_pairs,
                                        ising_model_2D,
                                        flipped_ising_model_2D,
                                        bose_hubbard_model)

class TestIsingModel(TestCase):
    """
    Test the generation of the ising model and associated utility
    functions.
    """

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
    """
    Test the generation of the flipped ising model and associated utility
    functions.
    """

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

    def test_find_nn_pairs_1d(self):
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

    def test_ising_2d(self):
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

    def test_flipped_ising_2d(self):
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

class TestBoseHubbardModel(TestCase):
    """
    Tests the generation of the Bose-Hubbard model.
    """

    def setUp(self) -> None:
        self.pairs = [("node0_0", "node0_1"),
                      ("node0_0", "node1_0"),
                      ("node1_0", "node1_1"),
                      ("node0_1", "node1_1"),
                      ("node0_1", "node0_2"),
                      ("node1_1", "node1_2"),
                      ("node0_2", "node1_2")]

    def test_all_factors_zero(self):
        """
        Tests the generation of the Bose-Hubbard model with all factors set to
        zero.
        """
        found = bose_hubbard_model(self.pairs,
                                   hopping=0.0,
                                   on_site_int=0.0,
                                   chem_pot=0.0,)
        self.assertEqual(len(found.terms), 0)
        self.assertEqual(len(found.conversion_dictionary), 2)
        self.assertEqual(len(found.coeffs_mapping), 1)
        np.testing.assert_array_equal(found.conversion_dictionary["I2"],
                                      np.eye(2))

    def test_only_hopping(self):
        """
        Tests the generation of the Bose-Hubbard model with only the hopping
        factor set to a non-zero value.
        """
        found = bose_hubbard_model(self.pairs,
                                   hopping=1.0,
                                   on_site_int=0.0,
                                   chem_pot=0.0,)
        cr, an, _ = bosonic_operators()
        self.assertEqual(len(found.conversion_dictionary), 4)
        np.testing.assert_array_equal(found.conversion_dictionary["creation"],
                                      cr)
        np.testing.assert_array_equal(found.conversion_dictionary["annihilation"],
                                      an)
        self.assertEqual(found.coeffs_mapping["hopping"],1.0)
        self.assertEqual(len(found.terms), 14)
        for term in found.terms:
            self.assertEqual(term[0], Fraction(-1))
            self.assertEqual(term[1], "hopping")
            tp = term[2]
            self.assertTrue(len(tp) == 2)
            pair_ids = tuple(tp.keys())
            self.assertTrue((pair_ids in self.pairs or
                             (pair_ids[1], pair_ids[0]) in self.pairs))
            term_ops = tuple(tp.values())
            self.assertIn("creation", term_ops)
            self.assertIn("annihilation", term_ops)

    def test_only_hopping_all_combs(self):
        """
        Ensures that for the hopping factor, both combinations of
        annihilation and creation operators are generated.
        """
        # Very small example
        node_ids = [("node0","node1")]
        found = bose_hubbard_model(node_ids,
                                   hopping=1.0,
                                   on_site_int=0.0,
                                   chem_pot=0.0)
        self.assertEqual(len(found.terms), 2)
        combinations = [("node0","node1"),
                        ("node1","node0")]
        for term in found.terms:
            self.assertEqual(term[0], Fraction(-1))
            self.assertEqual(term[1], "hopping")
            tp = term[2]
            self.assertTrue(len(tp) == 2)
            comb = [None,None]
            for key, value in tp.items():
                if value == "creation":
                    comb[0] = key
                elif value == "annihilation":
                    comb[1] = key
            comb = tuple(comb)
            self.assertIn(comb, combinations)
            combinations.remove(comb)
        self.assertEqual(len(combinations), 0)

    def test_only_on_site_int(self):
        """
        Test the generation of the Bose-Hubbard model with only the on-site
        interaction factor set to a non-zero value.
        """
        found = bose_hubbard_model(self.pairs,
                                   hopping=0.0,
                                   on_site_int=1.0,
                                   chem_pot=0.0)
        self.assertEqual(len(found.terms), 6)
        _, _, num = bosonic_operators()
        on_site_op = num @ (num - eye(2))
        self.assertEqual(len(found.conversion_dictionary), 3)
        np.testing.assert_array_equal(found.conversion_dictionary["on_site_op"],
                                      on_site_op)
        self.assertEqual(found.coeffs_mapping["on_site_int"], 1.0)
        for term in found.terms:
            self.assertEqual(term[0], Fraction(-1, 2))
            self.assertEqual(term[1], "on_site_int")
            tp = term[2]
            self.assertTrue(len(tp) == 1)
            node_id = list(tp.keys())[0]
            self.assertIn(node_id, [pair[0] for pair in self.pairs] +
                          [pair[1] for pair in self.pairs])
            self.assertEqual(tp[node_id], "on_site_op")

    def test_only_chem_pot(self):
        """
        Test the generation of the Bose-Hubbard model with only the chemical
        potential factor set to non-zero.
        """
        found = bose_hubbard_model(self.pairs,
                                   hopping=0.0,
                                   on_site_int=0.0,
                                   chem_pot=1.0)
        self.assertEqual(len(found.terms), 6)
        _, _, num = bosonic_operators()
        self.assertEqual(len(found.conversion_dictionary), 3)
        np.testing.assert_array_equal(found.conversion_dictionary["number"],
                                      num)
        self.assertEqual(found.coeffs_mapping["chem_pot"], 1.0)
        for term in found.terms:
            self.assertEqual(term[0], Fraction(-1))
            self.assertEqual(term[1], "chem_pot")
            tp = term[2]
            self.assertTrue(len(tp) == 1)
            node_id = list(tp.keys())[0]
            self.assertIn(node_id, [pair[0] for pair in self.pairs] +
                          [pair[1] for pair in self.pairs])
            self.assertEqual(tp[node_id], "number")

if __name__ == '__main__':
    unitmain()
