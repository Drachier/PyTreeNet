"""
This module provides tests for the local contraction class if the contracted
tensor are leafs, i.e. have a single neighbour.
"""
from __future__ import annotations
import unittest

import numpy as np
import numpy.testing as npt

from pytreenet.random.random_node import random_tensor_node
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict
from pytreenet.contractions.local_contr import (LocalContraction,
                                                TensorKind)

SEED = 686984451

class TestLocalContractionLeaf(unittest.TestCase):
    """
    Test contractions for leaf nodes (single neighbour).
    """

    def setUp(self):
        self.ket = random_tensor_node((3,2), seed=SEED)
        self.bra = random_tensor_node((4,2), seed=SEED+1)
        self.op1 = random_tensor_node((5,2,2), seed=SEED+2)
        self.op2 = random_tensor_node((6,2,2), seed=SEED+3)
        self.ign_leg_id = "ignore"
        for node in (self.ket, self.bra, self.op1, self.op2):
            node[0].open_leg_to_parent(self.ign_leg_id,0)

    def test_ket(self):
        """
        Test contraction of a ket leaf node. This should return the same
        tensor as the original ket node.
        """
        nodes_tensors = [self.ket]
        contr = LocalContraction(nodes_tensors,
                                 PartialTreeCachDict(),
                                 ignored_leg=self.ign_leg_id)
        result = contr.contract_all()
        npt.assert_allclose(result, self.ket[1])

    def test_operator(self):
        """
        Test contraction of an operator leaf node. This should return the same
        tensor as the original operator node.
        """
        nodes_tensors = [self.op1]
        contr = LocalContraction(nodes_tensors,
                                 PartialTreeCachDict(),
                                 ignored_leg=self.ign_leg_id,
                                 connection_index=1)
        result = contr.contract_all()
        npt.assert_allclose(result, self.op1[1])

    def test_bra(self):
        """
        Test contraction of a bra leaf node. This should return the same
        tensor as the original bra node.
        """
        nodes_tensors = [self.bra]
        contr = LocalContraction(nodes_tensors,
                                 PartialTreeCachDict(),
                                 ignored_leg=self.ign_leg_id,
                                 connection_index=1,
                                 highest_tensor=TensorKind.BRA)
        result = contr.contract_all()
        npt.assert_allclose(result, self.bra[1])

    def test_ket_operator(self):
        """
        Test contraction of a ket and an operator leaf node.
        """
        nodes_tensors = [self.ket, self.op1]
        contr = LocalContraction(nodes_tensors,
                                 PartialTreeCachDict(),
                                 ignored_leg=self.ign_leg_id)
        # Reference result
        ref = np.tensordot(self.ket[1], self.op1[1], axes=([1],[2]))
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_ket_operator_rev_contr_order(self):
        """
        Test contraction of a ket and an operator leaf node with reversed
        contraction order.
        """
        nodes_tensors = [self.ket, self.op1]
        contr = LocalContraction(nodes_tensors,
                                 PartialTreeCachDict(),
                                 ignored_leg=self.ign_leg_id,
                                 contraction_order=[1,0])
        # Reference result
        ref = np.tensordot(self.ket[1], self.op1[1], axes=([1],[2]))
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_op_op(self):
        """
        Test contraction of two operator leaf nodes.
        """
        nodes_tensors = [self.op1, self.op2]
        contr = LocalContraction(nodes_tensors,
                                 PartialTreeCachDict(),
                                 ignored_leg=self.ign_leg_id,
                                 connection_index=1)
        # Reference result
        ref = np.tensordot(self.op1[1], self.op2[1], axes=([1],[2]))
        # Note that the out leg should be before the in leg to conform with
        # standard matrix notation
        ref = ref.transpose(0,2,3,1)
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_op_op_rev_contr_order(self):
        """
        Test contraction of two operator leaf nodes with reversed contraction
        order.
        """
        nodes_tensors = [self.op1, self.op2]
        contr = LocalContraction(nodes_tensors,
                                 PartialTreeCachDict(),
                                 ignored_leg=self.ign_leg_id,
                                 connection_index=1,
                                 contraction_order=[1,0])
        # Reference result
        ref = np.tensordot(self.op1[1], self.op2[1], axes=([1],[2]))
        # Note that the out leg should be before the in leg to conform with
        # standard matrix notation
        ref = ref.transpose(0,2,3,1)
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

if __name__ == "__main__":
    unittest.main()
