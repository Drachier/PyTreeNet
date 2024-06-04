import unittest
from copy import deepcopy

import numpy as np

import pytreenet as ptn
from pytreenet.random import (random_small_ttns,
                              random_big_ttns_two_root_children,
                              RandomTTNSMode)

class TestTreeContractionSimple(unittest.TestCase):
    def setUp(self):
        self.ttn = random_small_ttns()

    def test_completely_contract_tree(self):
        # Reference Computation
        ref = deepcopy(self.ttn)
        axes = (ref.nodes["root"].neighbour_index("c1"),
                ref.nodes["c1"].neighbour_index("root"))
        ref_tensor = np.tensordot(ref.tensors["root"],
                                  ref.tensors["c1"],
                                  axes=axes)
        axes = (0,ref.nodes["c2"].neighbour_index("root"))
        ref_tensor = np.tensordot(ref_tensor,
                                  ref.tensors["c2"],
                                  axes=axes)
        correct_order = ["root","c1","c2"]

        # Actual Computation
        found_tensor, found_order = ptn.completely_contract_tree(self.ttn)
        self.assertEqual(correct_order,found_order)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_completely_contract_to_copy(self):
        ref = deepcopy(self.ttn)
        found_tensor, found_order = ptn.completely_contract_tree(self.ttn,
                                                                 to_copy=True)
        self.assertEqual(ref,self.ttn)
        second_tensor, second_order = ptn.completely_contract_tree(self.ttn)
        self.assertEqual(second_order,found_order)
        self.assertTrue(np.allclose(second_tensor,found_tensor))

class TestTreeContractionComplicated(unittest.TestCase):
    def setUp(self):
        self.ttn = random_big_ttns_two_root_children(mode=RandomTTNSMode.DIFFVIRT)

    def test_completely_contract_tree(self):
        # Reference Computation
        ref = deepcopy(self.ttn)
        axes = (ref.nodes["site0"].neighbour_index("site1"),
                ref.nodes["site1"].neighbour_index("site0"))
        ref_tensor = np.tensordot(ref.tensors["site0"],
                                  ref.tensors["site1"],
                                  axes=axes)
        ref_tensor = np.tensordot(ref_tensor,
                                  ref.tensors["site2"],
                                  axes=(2,ref.nodes["site2"].neighbour_index("site1")))
        ref_tensor = np.tensordot(ref_tensor,
                                  ref.tensors["site3"],
                                  axes=(2,ref.nodes["site3"].neighbour_index("site1")))
        ref_tensor = np.tensordot(ref_tensor,
                                  ref.tensors["site4"],
                                  axes=(4,ref.nodes["site4"].neighbour_index("site3")))
        ref_tensor = np.tensordot(ref_tensor,
                                  ref.tensors["site5"],
                                  axes=(4,ref.nodes["site5"].neighbour_index("site3")))
        ref_tensor = np.tensordot(ref_tensor,
                                  ref.tensors["site6"],
                                  axes=(0,ref.nodes["site6"].neighbour_index("site0")))
        ref_tensor = np.tensordot(ref_tensor,
                                  ref.tensors["site7"],
                                  axes=(6,ref.nodes["site7"].neighbour_index("site6")))
        correct_order = ["site0","site1","site2","site3",
                               "site4","site5","site6","site7"]

        # Actual Computation
        found_tensor, found_order = ptn.completely_contract_tree(self.ttn)
        self.assertEqual(correct_order,found_order)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_completely_contract_to_copy(self):
        ref = deepcopy(self.ttn)
        found_tensor, found_order = ptn.completely_contract_tree(self.ttn,
                                                                 to_copy=True)
        self.assertEqual(ref,self.ttn)
        second_tensor, second_order = ptn.completely_contract_tree(self.ttn)
        self.assertEqual(second_order,found_order)
        self.assertTrue(np.allclose(second_tensor,found_tensor))

if __name__ == "__main__":
    unittest.main()
