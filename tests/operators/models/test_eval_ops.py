"""
Unittests for the evaluation operator creation functions.
"""
import unittest

from pytreenet.operators.models.eval_ops import (local_magnetisation,
                                                 local_magnetisation_from_topology)
from pytreenet.operators.models.topology import Topology
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.operators.common_operators import pauli_matrices

class TestLocalMagnetisation(unittest.TestCase):
    """
    Tests the functions generating local magnetisation.
    """

    def test_loc_magn_empty_list(self):
        """
        Tests the generation of the local magnetisation for an empty list.
        """
        found = local_magnetisation([])
        correct = {}
        self.assertEqual(correct, found)

    def test_loc_magn_single_id(self):
        """
        Tests the function for a list with a single identifier.
        """
        idents = ["site"]
        found = local_magnetisation(idents)
        correct = {"site": TensorProduct({"site": pauli_matrices()[2]})}
        self.assertIn("site", idents)
        self.assertEqual(1, len(found))
        self.assertTrue(correct["site"].allclose(found["site"]))

    def compare_results(self,
                        correct: dict[str, TensorProduct],
                        found: dict[str, TensorProduct],
                        idents: list[str]):
        """
        Asseses wether the found dictionary is correct.
        """
        for ident in idents:
            self.assertIn(ident, found)
        self.assertEqual(len(idents), len(found))
        for key, found_tp in found.items():
            self.assertTrue(found_tp.allclose(correct[key]))

    def test_loc_magn_three_ids(self):
        """
        Test the function for a list with three identifiers.
        """
        idents = ["site" + str(i) for i in range(3)]
        found  = local_magnetisation(idents)
        correct = {ident: TensorProduct({ident: pauli_matrices()[2]})
                   for ident in idents}
        self.compare_results(correct, found, idents)

    def test_from_topology_chain(self):
        """
        Test getting the loc magn for a chain topology.
        """
        sys_size = 4
        found = local_magnetisation_from_topology(Topology.CHAIN,
                                                  sys_size)
        idents = ["site" + str(i) for i in range(sys_size)]
        correct = {ident: TensorProduct({ident: pauli_matrices()[2]})
                   for ident in idents}
        self.compare_results(correct, found, idents)

    def test_from_topology_tstar(self):
        """
        Test getting the local magn for a t-topology.
        """
        sys_size = 3
        topology = Topology.TTOPOLOGY
        found = local_magnetisation_from_topology(topology,
                                                  sys_size)
        idents = ["site" + str(i)
                  for i in range(topology.num_sites(sys_size))]
        correct = {ident: TensorProduct({ident: pauli_matrices()[2]})
                   for ident in idents}
        self.compare_results(correct, found, idents)
