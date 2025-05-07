"""
This module provides unit tests for the contraction of two TTNS.
"""
import unittest
from copy import deepcopy

import numpy as np

import pytreenet as ptn
from pytreenet.random import (random_small_ttns,
                              random_big_ttns_two_root_children,
                              RandomTTNSMode)

class TestContractTwoTTNsSimple(unittest.TestCase):
    """
    Tests the contraction of two simple TTNS.
    """

    def test_contract_two_ttns_simple(self):
        """
        Tests the contraction of two simple TTNS.
        """
        ttns = random_small_ttns()
        ttns2 = random_small_ttns()

        # Saved to check, the TTN are not affected
        refs = [deepcopy(ttns), deepcopy(ttns2)]

        # Contracting the two TTNs
        result = ptn.contract_two_ttns(ttns, ttns2)

        # Check that the original TTNs are not affected
        self.assertEqual(refs[0], ttns)
        self.assertEqual(refs[1], ttns2)

        # Reference Computation
        tensor1 = ttns.completely_contract_tree(to_copy=True)[0]
        tensor2 = ttns2.completely_contract_tree(to_copy=True)[0]
        legs = tuple(range(len(ttns)))
        ref = np.tensordot(tensor1, tensor2, axes=(legs, legs))
        self.assertTrue(np.allclose(ref, result))

    def test_contract_two_ttns_simple_diff_ids(self):
        """
        Tests the contraction of two simple TTNS with different IDs.
        """
        ttns = random_small_ttns()
        ttns2 = random_small_ttns(ids=["r", "b", "c"])
        def id_trafo(node_id: str) -> str:
            if node_id == "root":
                return "r"
            if node_id == "c1":
                return "b"
            if node_id == "c2":
                return "c"
            raise ValueError(f"Unknown node ID: {node_id}")

        # Saved to check, the TTN are not affected
        refs = [deepcopy(ttns), deepcopy(ttns2)]

        # Contracting the two TTNs
        result = ptn.contract_two_ttns(ttns, ttns2,
                                       id_trafo=id_trafo)

        # Check that the original TTNs are not affected
        self.assertEqual(refs[0], ttns)
        self.assertEqual(refs[1], ttns2)

        # Reference Computation
        tensor1 = ttns.completely_contract_tree(to_copy=True)[0]
        tensor2 = ttns2.completely_contract_tree(to_copy=True)[0]
        legs = tuple(range(len(ttns)))
        ref = np.tensordot(tensor1, tensor2, axes=(legs, legs))
        self.assertTrue(np.allclose(ref, result))

class TestContractTwoTTNsComplicated(unittest.TestCase):
    """
    Tests the contraction of two complicated TTNS.
    """
    def setUp(self):
        mode = RandomTTNSMode.DIFFVIRT
        self.ttns1 = random_big_ttns_two_root_children(mode)
        self.ttns2 = random_big_ttns_two_root_children(mode)

        # Save the original TTNs
        self.refs = [deepcopy(self.ttns1), deepcopy(self.ttns2)]

    def test_contract_two_ttns_complicated(self):
        """
        Test the contraction of two complicated TTNS.
        """
        # Contracting the two TTNs
        result = ptn.contract_two_ttns(self.ttns1, self.ttns2)

        # Check that the original TTNs are not affected
        self.assertEqual(self.refs[0], self.ttns1)
        self.assertEqual(self.refs[1], self.ttns2)

        # Reference Computation
        tensor1 = self.ttns1.completely_contract_tree(to_copy=True)[0]
        tensor2 = self.ttns2.completely_contract_tree(to_copy=True)[0]
        legs = tuple(range(len(self.ttns1)))
        ref = np.tensordot(tensor1, tensor2, axes=(legs, legs))
        self.assertTrue(np.allclose(ref, result))

if __name__ == "__main__":
    unittest.main()
