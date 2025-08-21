"""
This model tests the creation of two site models.
"""
import unittest
from unittest import mock
from fractions import Fraction

import numpy as np

from pytreenet.operators.common_operators import (pauli_matrices,
                                                  bosonic_operators)
from pytreenet.operators.models.two_site_model import (TwoSiteModel,
                                                       HeisenbergModel,
                                                       IsingModel,
                                                       FlippedIsingModel,
                                                       BoseHubbardModel)
from pytreenet.operators.models.topology import Topology

class TestTwoSiteModel(unittest.TestCase):
    """
    Tests for the TwoSiteModels structure generation functions.
    """

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_chain_structure(self):
        """
        Test the generation of a chain structure.
        """
        model = TwoSiteModel()
        structure = ["site1", "site2", "site3", "site4", "site5"]
        pairs = model.generate_chain_structure(structure)
        correct = [("site1", "site2"),
                  ("site2", "site3"),
                  ("site3", "site4"),
                  ("site4", "site5")]
        self.assertEqual(pairs, correct)

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_chain_structure_intrange2(self):
        """
        Test the generation of a chain structure with interaction range 2.
        """
        model = TwoSiteModel(interaction_range=2)
        structure = ["site1", "site2", "site3", "site4", "site5"]
        pairs = model.generate_chain_structure(structure)
        correct = [("site1", "site3"),
                   ("site2", "site4"),
                   ("site3", "site5")]
        self.assertEqual(pairs, correct)

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_chain_structure_intrange3(self):
        """
        Test the generation of a chain structure with interaction range 3.
        """
        model = TwoSiteModel(interaction_range=3)
        structure = ["site1", "site2", "site3", "site4", "site5"]
        pairs = model.generate_chain_structure(structure)
        correct = [("site1", "site4"),
                   ("site2", "site5")]
        self.assertEqual(pairs, correct)

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_chain_structure_intrange4(self):
        """
        Test the generation of a chain structure with interaction range 4.
        """
        model = TwoSiteModel(interaction_range=4)
        structure = ["site1", "site2", "site3", "site4", "site5"]
        pairs = model.generate_chain_structure(structure)
        correct = [("site1", "site5")]
        self.assertEqual(pairs, correct)

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_chain_structure_intrange5(self):
        """
        Test the generation of a chain structure with interaction range 5.
        There should be no pairs generated.
        """
        model = TwoSiteModel(interaction_range=5)
        structure = ["site1", "site2", "site3", "site4", "site5"]
        pairs = model.generate_chain_structure(structure)
        correct = []
        self.assertEqual(pairs, correct)

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_t_topology_structure(self):
        """
        Test the generation of a T topology structure.
        """
        model = TwoSiteModel()
        structure = (["site0", "site1", "site2"],
                     ["site3", "site4", "site5"],
                     ["site6", "site7", "site8"])
        pairs = model.generate_t_topology_structure(structure)
        correct = [("site0", "site1"),
                   ("site1", "site2"),
                   ("site3", "site4"),
                   ("site4", "site5"),
                   ("site6", "site7"),
                   ("site7", "site8"),
                   ("site0", "site3"),
                   ("site0", "site6"),
                   ("site3", "site6")]
        self.assertEqual(len(pairs), len(correct))
        pairs_set = {frozenset(pair) for pair in pairs}
        correct_set = {frozenset(pair) for pair in correct}
        self.assertEqual(pairs_set, correct_set)

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_t_topology_structure_intrange2(self):
        """
        Test the generation of a T topology structure with interaction range 2.
        """
        model = TwoSiteModel(interaction_range=2)
        structure = (["site0", "site1", "site2"],
                     ["site3", "site4", "site5"],
                     ["site6", "site7", "site8"])
        pairs = model.generate_t_topology_structure(structure)
        correct = [("site0", "site2"),
                   ("site3", "site5"),
                   ("site6", "site8"),
                   ("site1", "site3"),
                   ("site1", "site6"),
                   ("site4", "site0"),
                   ("site4", "site6"),
                   ("site7", "site0"),
                   ("site7", "site3")]
        self.assertEqual(len(pairs), len(correct))
        pairs_set = {frozenset(pair) for pair in pairs}
        correct_set = {frozenset(pair) for pair in correct}
        self.assertEqual(pairs_set, correct_set)

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_t_topology_structure_intrange3(self):
        """
        Test the generation of a T topology structure with interaction range 3.
        """
        model = TwoSiteModel(interaction_range=3)
        structure = (["site0", "site1", "site2"],
                     ["site3", "site4", "site5"],
                     ["site6", "site7", "site8"])
        pairs = model.generate_t_topology_structure(structure)
        correct = [("site2", "site3"),
                   ("site2", "site6"),
                   ("site5", "site0"),
                   ("site5", "site6"),
                   ("site8", "site0"),
                   ("site8", "site3"),
                   ("site1", "site4"),
                   ("site1", "site7"),
                   ("site4", "site7")]
        self.assertEqual(len(pairs), len(correct))
        pairs_set = {frozenset(pair) for pair in pairs}
        correct_set = {frozenset(pair) for pair in correct}
        self.assertEqual(pairs_set, correct_set)

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_t_topology_structure_intrange4(self):
        """
        Test the generation of a T topology structure with interaction range 4.
        """
        model = TwoSiteModel(interaction_range=4)
        structure = (["site0", "site1", "site2"],
                     ["site3", "site4", "site5"],
                     ["site6", "site7", "site8"])
        pairs = model.generate_t_topology_structure(structure)
        correct = [("site2", "site7"),
                   ("site2", "site4"),
                   ("site5", "site1"),
                   ("site5", "site7"),
                   ("site8", "site1"),
                   ("site8", "site4")]
        self.assertEqual(len(pairs), len(correct))
        pairs_set = {frozenset(pair) for pair in pairs}
        correct_set = {frozenset(pair) for pair in correct}
        self.assertEqual(pairs_set, correct_set)

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_t_topology_structure_intrange5(self):
        """
        Test the generation of a T topology structure with interaction range 5.
        """
        model = TwoSiteModel(interaction_range=5)
        structure = (["site0", "site1", "site2"],
                     ["site3", "site4", "site5"],
                     ["site6", "site7", "site8"])
        pairs = model.generate_t_topology_structure(structure)
        correct = [("site2", "site8"),
                   ("site5", "site2"),
                   ("site8", "site5")]
        self.assertEqual(len(pairs), len(correct))
        pairs_set = {frozenset(pair) for pair in pairs}
        correct_set = {frozenset(pair) for pair in correct}
        self.assertEqual(pairs_set, correct_set)

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_t_topology_structure_intrange6(self):
        """
        Test the generation of a T topology structure with interaction range 6.
        There should be no pairs generated.
        """
        model = TwoSiteModel(interaction_range=6)
        structure = (["site0", "site1", "site2"],
                     ["site3", "site4", "site5"],
                     ["site6", "site7", "site8"])
        pairs = model.generate_t_topology_structure(structure)
        correct = []
        self.assertEqual(pairs, correct)

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_2d_structure(self):
        """
        Test the generation of a 2D structure.
        """
        model = TwoSiteModel()
        structure = (["site0", "site1", "site2"],
                     ["site3", "site4", "site5"],
                     ["site6", "site7", "site8"])
        pairs = model.generate_2d_structure(structure)
        correct = [("site0", "site1"),
                   ("site1", "site2"),
                   ("site3", "site4"),
                   ("site4", "site5"),
                   ("site6", "site7"),
                   ("site7", "site8"),
                   ("site0", "site3"),
                   ("site1", "site4"),
                   ("site2", "site5"),
                   ("site3", "site6"),
                   ("site4", "site7"),
                   ("site5", "site8")]
        self.assertEqual(len(pairs), len(correct))
        pairs_set = {frozenset(pair) for pair in pairs}
        correct_set = {frozenset(pair) for pair in correct}
        self.assertEqual(pairs_set, correct_set)

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_2d_structure_intrange2(self):
        """
        Test the generation of a 2D structure with interaction range 2.
        """
        model = TwoSiteModel(interaction_range=2)
        structure = (["site0", "site1", "site2"],
                     ["site3", "site4", "site5"],
                     ["site6", "site7", "site8"])
        pairs = model.generate_2d_structure(structure)
        correct = [("site0", "site2"),
                   ("site3", "site5"),
                   ("site6", "site8"),
                   ("site0", "site6"),
                   ("site1", "site7"),
                   ("site2", "site8")]
        self.assertEqual(len(pairs), len(correct))
        pairs_set = {frozenset(pair) for pair in pairs}
        correct_set = {frozenset(pair) for pair in correct}
        self.assertEqual(pairs_set, correct_set)

    @mock.patch.multiple(TwoSiteModel, __abstractmethods__=set())
    def test_generate_2d_structure_intrange3(self):
        """
        Test the generation of a 2D structure with interaction range 3.
        There should be no pairs generated.
        """
        model = TwoSiteModel(interaction_range=3)
        structure = (["site0", "site1", "site2"],
                     ["site3", "site4", "site5"],
                     ["site6", "site7", "site8"])
        pairs = model.generate_2d_structure(structure)
        correct = []
        self.assertEqual(pairs, correct)

class TestHeisenbergModel(unittest.TestCase):
    """
    Tests the hamiltonian generation of the Heisenberg model.
    """

    def test_heisenberg_only_xx(self):
        """
        Test the Heisenberg model with only XX interactions.
        """
        x_factor = 1.0
        model = HeisenbergModel(x_factor=x_factor)
        structure = [("A", "B"),
                     ("B", "C"),
                     ("C", "D"),
                     ("D", "E")]
        hamiltonian = model.generate_hamiltonian(structure)
        factor_symb = model.factor_prefix + "x"
        self.assertEqual(len(hamiltonian.terms), 4)
        for term in hamiltonian.terms:
            factor = Fraction(-1)
            op_symb = model.pauli_symbols[0]
            self.assertEqual(term[0], factor)
            self.assertEqual(term[1], factor_symb)
            keys = tuple(term[2].keys())
            self.assertEqual(len(keys), 2)
            self.assertTrue(keys in structure
                            or (keys[1],keys[0]) in structure)
            self.assertTrue(all([op == op_symb for op in term[2].values()]))
        self.assertEqual(3, len(hamiltonian.conversion_dictionary))
        self.assertIn(model.pauli_symbols[0],
                      hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[0]],
            pauli_matrices()[0])
        self.assertEqual(2, len(hamiltonian.coeffs_mapping))
        self.assertIn(factor_symb,
                      hamiltonian.coeffs_mapping)
        self.assertEqual(hamiltonian.coeffs_mapping[factor_symb],
                         x_factor)

    def test_heisenberg_only_yy(self):
        """
        Test the Heisenberg model with only YY interactions.
        """
        y_factor = 1.0
        model = HeisenbergModel(y_factor=y_factor,
                                x_factor=0.0)
        structure = [("A", "B"),
                     ("B", "C"),
                     ("C", "D"),
                     ("D", "E")]
        hamiltonian = model.generate_hamiltonian(structure)
        self.assertEqual(len(hamiltonian.terms), 4)
        factor_symb = model.factor_prefix + "y"
        for term in hamiltonian.terms:
            factor = Fraction(-1)
            op_symb = model.pauli_symbols[1]
            self.assertEqual(term[0], factor)
            self.assertEqual(term[1], factor_symb)
            keys = tuple(term[2].keys())
            self.assertTrue(keys in structure
                            or (keys[1],keys[0]) in structure)
            self.assertTrue(all([op == op_symb for op in term[2].values()]))
        self.assertEqual(3, len(hamiltonian.conversion_dictionary))
        self.assertIn(model.pauli_symbols[1],
                        hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[1]],
            pauli_matrices()[1])
        self.assertEqual(2, len(hamiltonian.coeffs_mapping))
        self.assertIn(factor_symb,
                      hamiltonian.coeffs_mapping)
        self.assertEqual(hamiltonian.coeffs_mapping[factor_symb],
                         y_factor)

    def test_heisenberg_only_zz(self):
        """
        Test the Heisenberg model with only ZZ interactions.
        """
        z_factor = 1.0
        model = HeisenbergModel(z_factor=z_factor,
                                x_factor=0.0)
        structure = [("A", "B"),
                     ("B", "C"),
                     ("C", "D"),
                     ("D", "E")]
        hamiltonian = model.generate_hamiltonian(structure)
        self.assertEqual(len(hamiltonian.terms), 4)
        factor_symb = model.factor_prefix + "z"
        for term in hamiltonian.terms:
            factor = Fraction(-1)
            op_symb = model.pauli_symbols[2]
            self.assertEqual(term[0], factor)
            self.assertEqual(term[1], factor_symb)
            keys = tuple(term[2].keys())
            self.assertTrue(keys in structure
                            or (keys[1],keys[0]) in structure)
            self.assertTrue(all([op == op_symb for op in term[2].values()]))
        self.assertEqual(3, len(hamiltonian.conversion_dictionary))
        self.assertIn(model.pauli_symbols[2],
                      hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[2]],
            pauli_matrices()[2])
        self.assertEqual(2, len(hamiltonian.coeffs_mapping))
        self.assertIn(factor_symb,
                      hamiltonian.coeffs_mapping)
        self.assertEqual(hamiltonian.coeffs_mapping[factor_symb],
                         z_factor)

    def test_x_and_y_same(self):
        """
        Tests the case where x and y factors are the same.
        """
        x_factor = 1.0
        model = HeisenbergModel(x_factor=x_factor,
                                y_factor=None)
        structure = [("A", "B"),
                     ("B", "C"),
                     ("C", "D"),
                     ("D", "E")]
        hamiltonian = model.generate_hamiltonian(structure)
        self.assertEqual(len(hamiltonian.terms), 8)
        factor_symb = model.factor_prefix + "x"
        for term in hamiltonian.terms:
            factor = Fraction(-1)
            op_symb = model.pauli_symbols[0:2]
            self.assertEqual(term[0], factor)
            self.assertEqual(term[1], factor_symb)
            keys = tuple(term[2].keys())
            self.assertTrue(keys in structure
                            or (keys[1],keys[0]) in structure)
            self.assertTrue(all([op in op_symb for op in term[2].values()]))
            self.assertTrue(term[2][keys[0]] == term[2][keys[1]])
        self.assertEqual(4, len(hamiltonian.conversion_dictionary))
        self.assertIn(model.pauli_symbols[0],
                      hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[0]],
            pauli_matrices()[0])
        self.assertIn(model.pauli_symbols[1],
                      hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[1]],
            pauli_matrices()[1])
        self.assertEqual(2, len(hamiltonian.coeffs_mapping))
        self.assertIn(factor_symb,
                      hamiltonian.coeffs_mapping)
        self.assertEqual(hamiltonian.coeffs_mapping[factor_symb],
                         x_factor)

    def test_x_and_z_same(self):
        """
        Tests the case where x and z factors are the same.
        """
        x_factor = 1.0
        model = HeisenbergModel(x_factor=x_factor,
                                z_factor=None)
        structure = [("A", "B"),
                     ("B", "C"),
                     ("C", "D"),
                     ("D", "E")]
        hamiltonian = model.generate_hamiltonian(structure)
        self.assertEqual(len(hamiltonian.terms), 8)
        factor_symb = model.factor_prefix + "x"
        for term in hamiltonian.terms:
            factor = Fraction(-1)
            op_symb = [model.pauli_symbols[i] for i in [0, 2]]
            self.assertEqual(term[0], factor)
            self.assertEqual(term[1], factor_symb)
            keys = tuple(term[2].keys())
            self.assertTrue(keys in structure
                            or (keys[1],keys[0]) in structure)
            self.assertTrue(all([op in op_symb for op in term[2].values()]))
            self.assertTrue(term[2][keys[0]] == term[2][keys[1]])
        self.assertEqual(4, len(hamiltonian.conversion_dictionary))
        self.assertIn(model.pauli_symbols[0],
                      hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[0]],
            pauli_matrices()[0])
        self.assertIn(model.pauli_symbols[2],
                      hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[2]],
            pauli_matrices()[2])
        self.assertEqual(2, len(hamiltonian.coeffs_mapping))
        self.assertIn(factor_symb,
                      hamiltonian.coeffs_mapping)
        self.assertEqual(hamiltonian.coeffs_mapping[factor_symb],
                         x_factor)

    def test_ext_z_field_only(self):
        """
        Tests the case where only an external z field is applied.
        """
        ext_z = 1.0
        model = HeisenbergModel(x_factor=0.0,
                                ext_z=ext_z)
        structure = [("A", "B"),
                     ("B", "C"),
                     ("C", "D"),
                     ("D", "E")]
        hamiltonian = model.generate_hamiltonian(structure)
        self.assertEqual(len(hamiltonian.terms), 5)
        factor_symb = model.ext_magn_prefix + "z"
        for term in hamiltonian.terms:
            factor = Fraction(-1)
            op_symb = model.pauli_symbols[2]
            self.assertEqual(term[0], factor)
            self.assertEqual(term[1], factor_symb)
            keys = tuple(term[2].keys())
            self.assertTrue(len(keys) == 1)
            self.assertTrue(keys[0] in ["A", "B", "C", "D", "E"])
            self.assertTrue(term[2][keys[0]] == op_symb)
        self.assertEqual(3, len(hamiltonian.conversion_dictionary))
        self.assertIn(model.pauli_symbols[2],
                      hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[2]],
            pauli_matrices()[2])
        self.assertEqual(2, len(hamiltonian.coeffs_mapping))
        self.assertIn(factor_symb,
                      hamiltonian.coeffs_mapping)
        self.assertEqual(hamiltonian.coeffs_mapping[factor_symb],
                         ext_z)

    def test_ext_x_field_only(self):
        """
        Test the case where only an external x field is applied.
        """
        ext_x = 1.0
        model = HeisenbergModel(x_factor=0.0,
                                ext_x=ext_x)
        structure = [("A", "B"),
                     ("B", "C"),
                     ("C", "D"),
                     ("D", "E")]
        hamiltonian = model.generate_hamiltonian(structure)
        self.assertEqual(len(hamiltonian.terms), 5)
        factor_symb = model.ext_magn_prefix + "x"
        for term in hamiltonian.terms:
            factor = Fraction(-1)
            op_symb = model.pauli_symbols[0]
            self.assertEqual(term[0], factor)
            self.assertEqual(term[1], factor_symb)
            keys = tuple(term[2].keys())
            self.assertTrue(len(keys) == 1)
            self.assertTrue(keys[0] in ["A", "B", "C", "D", "E"])
            self.assertTrue(term[2][keys[0]] == op_symb)
        self.assertEqual(3, len(hamiltonian.conversion_dictionary))
        self.assertIn(model.pauli_symbols[0],
                      hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[0]],
            pauli_matrices()[0])
        self.assertEqual(2, len(hamiltonian.coeffs_mapping))
        self.assertIn(factor_symb,
                      hamiltonian.coeffs_mapping)
        self.assertEqual(hamiltonian.coeffs_mapping[factor_symb],
                         ext_x)

    def test_ext_y_field_only(self):
        """
        Test the case where only an external y field is applied.
        """
        ext_y = 1.0
        model = HeisenbergModel(x_factor=0.0,
                                ext_y=ext_y)
        structure = [("A", "B"),
                     ("B", "C"),
                     ("C", "D"),
                     ("D", "E")]
        hamiltonian = model.generate_hamiltonian(structure)
        self.assertEqual(len(hamiltonian.terms), 5)
        factor_symb = model.ext_magn_prefix + "y"
        for term in hamiltonian.terms:
            factor = Fraction(-1)
            op_symb = model.pauli_symbols[1]
            self.assertEqual(term[0], factor)
            self.assertEqual(term[1], factor_symb)
            keys = tuple(term[2].keys())
            self.assertTrue(len(keys) == 1)
            self.assertTrue(keys[0] in ["A", "B", "C", "D", "E"])
            self.assertTrue(term[2][keys[0]] == op_symb)
        self.assertEqual(3, len(hamiltonian.conversion_dictionary))
        self.assertIn(model.pauli_symbols[1],
                      hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[1]],
            pauli_matrices()[1])
        self.assertEqual(2, len(hamiltonian.coeffs_mapping))
        self.assertIn(factor_symb,
                      hamiltonian.coeffs_mapping)
        self.assertEqual(hamiltonian.coeffs_mapping[factor_symb],
                         ext_y)
        
    def test_ext_z_and_x_same(self):
        """
        Tests the case where external z and x fields are the same.
        """
        ext_z = 1.0
        model = HeisenbergModel(x_factor=0.0,
                                ext_z=ext_z,
                                ext_x=None)
        structure = [("A", "B"),
                     ("B", "C"),
                     ("C", "D"),
                     ("D", "E")]
        hamiltonian = model.generate_hamiltonian(structure)
        self.assertEqual(len(hamiltonian.terms), 10)
        factor_symb = model.ext_magn_prefix + "z"
        for term in hamiltonian.terms:
            factor = Fraction(-1)
            op_symb = [model.pauli_symbols[i] for i in [0, 2]]
            self.assertEqual(term[0], factor)
            self.assertEqual(term[1], factor_symb)
            keys = tuple(term[2].keys())
            self.assertTrue(len(keys) == 1)
            self.assertTrue(keys[0] in ["A", "B", "C", "D", "E"])
            self.assertTrue(term[2][keys[0]] in op_symb)
        self.assertEqual(4, len(hamiltonian.conversion_dictionary))
        self.assertIn(model.pauli_symbols[0],
                      hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[0]],
            pauli_matrices()[0])
        self.assertIn(model.pauli_symbols[2],
                      hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[2]],
            pauli_matrices()[2])
        self.assertEqual(2, len(hamiltonian.coeffs_mapping))
        self.assertIn(factor_symb,
                      hamiltonian.coeffs_mapping)
        self.assertEqual(hamiltonian.coeffs_mapping[factor_symb],
                         ext_z)

    def test_ext_z_and_y_same(self):
        """
        Tests the case where external z and y fields are the same.
        """
        ext_z = 1.0
        model = HeisenbergModel(x_factor=0.0,
                                ext_z=ext_z,
                                ext_y=None)
        structure = [("A", "B"),
                     ("B", "C"),
                     ("C", "D"),
                     ("D", "E")]
        hamiltonian = model.generate_hamiltonian(structure)
        self.assertEqual(len(hamiltonian.terms), 10)
        factor_symb = model.ext_magn_prefix + "z"
        for term in hamiltonian.terms:
            factor = Fraction(-1)
            op_symb = [model.pauli_symbols[i] for i in [1, 2]]
            self.assertEqual(term[0], factor)
            self.assertEqual(term[1], factor_symb)
            keys = tuple(term[2].keys())
            self.assertTrue(len(keys) == 1)
            self.assertTrue(keys[0] in ["A", "B", "C", "D", "E"])
            self.assertTrue(term[2][keys[0]] in op_symb)
        self.assertEqual(4, len(hamiltonian.conversion_dictionary))
        self.assertIn(model.pauli_symbols[1],
                      hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[1]],
            pauli_matrices()[1])
        self.assertIn(model.pauli_symbols[2],
                      hamiltonian.conversion_dictionary)
        np.testing.assert_array_almost_equal(
            hamiltonian.conversion_dictionary[model.pauli_symbols[2]],
            pauli_matrices()[2])
        self.assertEqual(2, len(hamiltonian.coeffs_mapping))
        self.assertIn(factor_symb,
                      hamiltonian.coeffs_mapping)
        self.assertEqual(hamiltonian.coeffs_mapping[factor_symb],
                         ext_z)

    def build_for_chain(self,
                        num_sites: int,
                        factors: tuple[float,float,float],
                        exts: tuple[float,float,float]
                        ) -> np.ndarray:
        """
        Build the hamiltonian matrix exactly from numpy arrays.
        """
        ident = np.eye(2, dtype=complex)
        px, py, pz = pauli_matrices()
        ham = np.zeros((2**num_sites,2**num_sites), dtype=complex)
        for i in range(num_sites):
            term_tsx = np.eye(1)
            term_tsy = np.eye(1)
            term_tsz = np.eye(1)
            term_ssx = np.eye(1)
            term_ssy = np.eye(1)
            term_ssz = np.eye(1)
            for j in range(num_sites):
                if j == i:
                    term_tsx = -1 * factors[0] * np.kron(term_tsx,px)
                    term_tsy = -1 * factors[1] * np.kron(term_tsy,py)
                    term_tsz = -1 * factors[2] * np.kron(term_tsz,pz)
                    term_ssx = -1 * exts[0] * np.kron(term_ssx,px)
                    term_ssy = -1 * exts[1] * np.kron(term_ssy,py)
                    term_ssz = -1 * exts[2] * np.kron(term_ssz,pz)
                elif j == i + 1:
                    term_tsx = np.kron(term_tsx,px)
                    term_tsy = np.kron(term_tsy,py)
                    term_tsz = np.kron(term_tsz,pz)
                    term_ssx = np.kron(term_ssx,ident)
                    term_ssy = np.kron(term_ssy,ident)
                    term_ssz = np.kron(term_ssz,ident)
                else:
                    term_tsx = np.kron(term_tsx,ident)
                    term_tsy = np.kron(term_tsy,ident)
                    term_tsz = np.kron(term_tsz,ident)
                    term_ssx = np.kron(term_ssx,ident)
                    term_ssy = np.kron(term_ssy,ident)
                    term_ssz = np.kron(term_ssz,ident)
            if i != num_sites - 1:
                ham += term_tsx
                ham += term_tsy
                ham += term_tsz
            ham += term_ssx
            ham += term_ssy
            ham += term_ssz
        return ham

    def test_matrix_for_chain(self):
        """
        Tests that the correct matrix is produced for a chain topology.
        """
        num_sites = 5
        factors = (1,2,5)
        ext = (0.1,0.2,0.5)
        model = HeisenbergModel(x_factor=factors[0],
                                y_factor=factors[1],
                                z_factor=factors[2],
                                ext_x=ext[0],
                                ext_y=ext[1],
                                ext_z=ext[2])
        found = model.generate_matrix(Topology.CHAIN,num_sites)
        correct = self.build_for_chain(num_sites,
                                       factors,
                                       ext)
        np.testing.assert_allclose(correct, found)

    def test_matrix_for_t(self):
        """
        Tests the creation of the Hamiltonian matrix for a T-shape.
        """
        sys_size = 3
        factors = (1,2,5)
        ext = (0.1,0.2,0.5)
        model = HeisenbergModel(x_factor=factors[0],
                                y_factor=factors[1],
                                z_factor=factors[2],
                                ext_x=ext[0],
                                ext_y=ext[1],
                                ext_z=ext[2])
        found = model.generate_matrix(Topology.TTOPOLOGY,sys_size)
        # On each branch we get the same hamiltonian
        branch_ham = self.build_for_chain(sys_size,
                                          factors,
                                          ext)
        one_branch_ident = np.eye(2**sys_size,
                                   dtype=complex)
        two_branch_ident = np.kron(one_branch_ident,one_branch_ident)
        tot_dim = 2**Topology.TTOPOLOGY.num_sites(sys_size)
        total_ham = np.zeros((tot_dim,tot_dim),
                             dtype=complex)
        total_ham += np.kron(branch_ham,two_branch_ident)
        total_ham += np.kron(one_branch_ident,np.kron(branch_ham,one_branch_ident))
        total_ham += np.kron(two_branch_ident, branch_ham)
        # We need the term in the middle
        mid_adjacent = [sys_size*i for i in range(3)]
        mid_adjacent_pairs = [(mid_adjacent[0],mid_adjacent[1]),
                              (mid_adjacent[0],mid_adjacent[2]),
                              (mid_adjacent[1],mid_adjacent[2])]
        terms = [[np.eye(1) for _ in range(3)]
                 for _ in range(len(factors))]
        for i in range(Topology.TTOPOLOGY.num_sites(sys_size)):
            for pauli_index, pauli_terms in enumerate(terms):
                for pair_index, pair in enumerate(mid_adjacent_pairs):
                    if i == pair[0]:
                        factor = -1 * factors[pauli_index]
                        local_matrix = factor * pauli_matrices()[pauli_index]
                    elif i == pair[1]:
                        local_matrix = pauli_matrices()[pauli_index]
                    else:
                        local_matrix = np.eye(2)
                    pauli_terms[pair_index] = np.kron(pauli_terms[pair_index],
                                                        local_matrix)
        for termp in terms:
            for termfin in termp:
                total_ham += termfin
        np.testing.assert_allclose(total_ham, found)

class TestIsingModel(unittest.TestCase):
    """
    Tests the hamiltonian generation of the Ising model.
    """

    def test_ising_hamiltonian_generation(self):
        """
        The Ising model is a special case of the Heisenberg model
        """
        x_factor = 1.0
        ext_z = 2.0
        heis_model = HeisenbergModel(x_factor=x_factor,
                                     ext_z=ext_z)
        ising_model = IsingModel(factor=x_factor,
                                 ext_magn=ext_z)
        structure = [("A", "B"),
                     ("B", "C"),
                     ("C", "D"),
                     ("D", "E")]
        heis_hamiltonian = heis_model.generate_hamiltonian(structure)
        ising_hamiltonian = ising_model.generate_hamiltonian(structure)
        self.assertEqual(heis_hamiltonian.terms,
                         ising_hamiltonian.terms)

    def test_ising_chain(self):
        """
        Generate the Hamiltonian of an Ising chain.
        """
        x_factor = 1.0
        ext_z = 1.0
        num_sites = 5
        ising_model = IsingModel(factor=x_factor,
                                 ext_magn=ext_z)
        found = ising_model.generate_chain_model(num_sites)
        pairs = [("site0","site1"),
                 ("site1","site2"),
                 ("site2","site3"),
                 ("site3","site4")]
        correct = ising_model.generate_hamiltonian(pairs)
        self.assertEqual(correct,found)

    def test_ising_t_topology(self):
        """
        Test the Ising model creation on a T topology.
        """
        x_factor = 1.0
        ext_z = 1.0
        chain_length = 3
        ising_model = IsingModel(factor=x_factor,
                                 ext_magn=ext_z)
        found = ising_model.generate_t_topology_model(chain_length)
        pairs = [("site0", "site1"),
                   ("site1", "site2"),
                   ("site3", "site4"),
                   ("site4", "site5"),
                   ("site6", "site7"),
                   ("site7", "site8"),
                   ("site0", "site3"),
                   ("site0", "site6"),
                   ("site3", "site6")]
        correct = ising_model.generate_hamiltonian(pairs)
        self.assertEqual(correct, found)

    def test_ising_2d(self):
        """
        Test the generation of an Ising model on a 2d grid.
        """
        x_factor = 1.0
        ext_z = 2.0
        num_rows = 3
        model = IsingModel(factor=x_factor,
                           ext_magn=ext_z)
        found = model.generate_2d_model(num_rows)
        pairs = [("site0", "site1"),
                   ("site1", "site2"),
                   ("site3", "site4"),
                   ("site4", "site5"),
                   ("site6", "site7"),
                   ("site7", "site8"),
                   ("site0", "site3"),
                   ("site1", "site4"),
                   ("site2", "site5"),
                   ("site3", "site6"),
                   ("site4", "site7"),
                   ("site5", "site8")]
        correct = model.generate_hamiltonian(pairs)
        self.assertEqual(correct, found)

class TestFlippedIsingModel(unittest.TestCase):
    """
    Tests the hamiltonian generation of the Flipped Ising model.
    """

    def test_flipped_ising_hamiltonian_generation(self):
        """
        The Flipped Ising model is a special case of the Heisenberg model
        """
        z_factor = 1.0
        ext_x = 2.0
        heis_model = HeisenbergModel(z_factor=z_factor,
                                     ext_x=ext_x,
                                     x_factor=0.0)
        flipped_ising_model = FlippedIsingModel(factor=z_factor,
                                                 ext_magn=ext_x)
        structure = [("A", "B"),
                     ("B", "C"),
                     ("C", "D"),
                     ("D", "E")]
        heis_hamiltonian = heis_model.generate_hamiltonian(structure)
        flipped_ising_hamiltonian = flipped_ising_model.generate_hamiltonian(structure)
        self.assertEqual(heis_hamiltonian.terms,
                         flipped_ising_hamiltonian.terms)

class TestBoseHubbardModel(unittest.TestCase):
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
        model = BoseHubbardModel(hopping=0.0,
                                 on_site_int=0.0)
        found = model.generate_hamiltonian(self.pairs)
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
        model = BoseHubbardModel(hopping=1.0,
                                 on_site_int=0.0)
        found = model.generate_hamiltonian(self.pairs)
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
        model = BoseHubbardModel(hopping=1.0,
                                 on_site_int=0.0,
                                 chem_pot=0.0)
        found = model.generate_hamiltonian(node_ids)
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
        model = BoseHubbardModel(hopping=0.0,
                                 on_site_int=1.0,
                                 chem_pot=0.0)
        found = model.generate_hamiltonian(self.pairs)
        self.assertEqual(len(found.terms), 6)
        _, _, num = bosonic_operators()
        on_site_op = num @ (num - np.eye(2))
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
        model = BoseHubbardModel(hopping=0.0,
                                 on_site_int=0.0,
                                 chem_pot=1.0)
        found = model.generate_hamiltonian(self.pairs)
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


if __name__ == "__main__":
    unittest.main()
