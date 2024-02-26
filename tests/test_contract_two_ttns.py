import unittest
from copy import deepcopy

import numpy as np

import pytreenet as ptn

class TestContractTwoTTNsSimple(unittest.TestCase):

    def test_contract_two_ttns_simple(self):
        ttns = ptn.random_small_ttns()
        ttns2 = ptn.random_small_ttns()

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

class TestContractTwoTTNsComplicated(unittest.TestCase):
    def setUp(self):
        mode = ptn.RandomTTNSMode.DIFFVIRT
        self.ttns1 = ptn.random_big_ttns_two_root_children(mode)
        self.ttns2 = ptn.random_big_ttns_two_root_children(mode)

        # Save the original TTNs
        self.refs = [deepcopy(self.ttns1), deepcopy(self.ttns2)]

    def test_contract_two_ttns_complicated(self):
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
