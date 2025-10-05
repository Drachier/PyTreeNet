"""
This module implements unittests for the local contraction of a node with two
neighbours.
"""

from __future__ import annotations
import unittest

import numpy as np
import numpy.testing as npt

from pytreenet.random.random_node import random_tensor_node
from pytreenet.random.random_matrices import crandn
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict
from pytreenet.contractions.local_contr import (LocalContraction,
                                                TensorKind)

SEED = 686984451

class TestLocalContractionTwoNeighOneIgnoreOneSubtree(unittest.TestCase):
    """
    Test contractions for a node with two neighbours, where one is not
    contracted and one is contracted to a subtree tensor.
    """

    def setUp(self):
        self.node_id = "node_id"
        self.ket = random_tensor_node((3,4,2),
                                      identifier=self.node_id,
                                      seed=SEED)
        self.bra = random_tensor_node((5,6,2),
                                      identifier=self.node_id,
                                      seed=SEED+1)
        self.op1 = random_tensor_node((7,8,2,2),
                                      identifier=self.node_id,
                                      seed=SEED+2)
        self.op2 = random_tensor_node((9,10,2,2),
                                      identifier=self.node_id,
                                      seed=SEED+3)
        self.ign_leg_id = "ignore"
        self.neigh_id = "neigh_id"
        for node in (self.ket, self.bra, self.op1, self.op2):
            node[0].open_leg_to_parent(self.ign_leg_id,0)
            node[0].open_leg_to_child(self.neigh_id,1)
        self.subtree_dict = PartialTreeCachDict()

    def _add_subtree_tensor(self, shape):
        subtree_tensor = crandn(shape, seed=SEED+10)
        self.subtree_dict.add_entry(self.neigh_id,
                                    self.node_id,
                                    subtree_tensor)

    def test_ket_node(self):
        """
        Test the contraction of a ket node to one subtree with one remaining
        open neighbour leg.
        """
        nodes_tensors = [self.ket]
        self._add_subtree_tensor((4,8,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 ignored_leg=self.ign_leg_id
                                 )
        result = contr.contract_all()
        # Reference
        subtree_tensor = self.subtree_dict.get_entry(self.neigh_id,
                                                     self.node_id)
        ref = np.tensordot(self.ket[1],
                           subtree_tensor,
                           axes=(1,0))
        ref = ref.transpose(0,2,3,1)
        npt.assert_allclose(ref,result)

    def test_operator(self):
        """
        Test contraction of an operator leaf node with one subtree tensor.
        """
        nodes_tensors = [self.op1]
        self._add_subtree_tensor((4,8,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 ignored_leg=self.ign_leg_id)
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        ref = np.tensordot(self.op1[1],
                            subtree_tens,
                            axes=([1],[1]))
        ref = ref.transpose(0,3,4,1,2)
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_bra(self):
        """
        Test contraction of a bra leaf node with one subtree tensor.
        """
        nodes_tensors = [self.bra]
        self._add_subtree_tensor((4,8,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=2,
                                 ignored_leg=self.ign_leg_id,
                                 highest_tensor=TensorKind.BRA)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        ref = np.tensordot(self.bra[1],
                           subtree_tens,
                            axes=([1],[2]))
        ref = ref.transpose(0,2,3,1)
        npt.assert_allclose(result, ref)

    def test_ket_operator(self):
        """
        Test contraction of a ket and an operator leaf node with one subtree
        tensor.
        """
        nodes_tensors = [self.ket, self.op1]
        self._add_subtree_tensor((4,8,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 ignored_leg=self.ign_leg_id)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                   self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([1],[0]))
        ref = np.tensordot(intermed,
                           self.op1[1],
                            axes=([1,2],[3,1]))
        ref = ref.transpose(0,2,1,3)
        npt.assert_allclose(result, ref)

    def test_ket_operator_rev_contr_order(self):
        """
        Test contraction of a ket and an operator leaf node with one subtree
        tensor and reversed contraction order.
        """
        nodes_tensors = [self.ket, self.op1]
        self._add_subtree_tensor((4,8,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 ignored_leg=self.ign_leg_id,
                                 contraction_order=[1,0])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                   self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([1],[0]))
        ref = np.tensordot(intermed,
                           self.op1[1],
                            axes=([1,2],[3,1]))
        ref = ref.transpose(0,2,1,3)
        npt.assert_allclose(result, ref)

    def test_op_op(self):
        """
        Test contraction of two operator leaf nodes with one subtree tensor.
        """
        nodes_tensors = [self.op1, self.op2]
        self._add_subtree_tensor((4,8,10,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 ignored_leg=self.ign_leg_id,
                                 connection_index=1)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([1],[1]))
        ref = np.tensordot(intermed,
                           self.op2[1],
                            axes=([1,4],[3,1]))
        ref = ref.transpose(0,4,2,3,5,1)
        npt.assert_allclose(result, ref)

    def test_op_op_rev_contr_order(self):
        """
        Test contraction of two operator leaf nodes with one subtree tensor
        and reversed contraction order.
        """
        nodes_tensors = [self.op1, self.op2]
        self._add_subtree_tensor((4,8,10,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 ignored_leg=self.ign_leg_id,
                                 connection_index=1,
                                 contraction_order=[1,0])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([1],[1]))
        ref = np.tensordot(intermed,
                           self.op2[1],
                            axes=([1,4],[3,1]))
        ref = ref.transpose(0,4,2,3,5,1)
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node(self):
        """
        Test contraction of an operator and a bra node with one subtree tensor
        and one ignored neighbour leg.
        """
        nodes_tensors = [self.op2, self.bra]
        self._add_subtree_tensor((4,8,10,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 ignored_leg=self.ign_leg_id,
                                 connection_index=2)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op2[1],
                                subtree_tens,
                                axes=([1],[2]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                            axes=([1,5],[2,1]))
        ref = ref.transpose(0,4,2,3,1)
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node_rev_contr_order(self):
        """
        Test contraction of an operator and a bra  node with one subtree tensor
        and one ignored neighbour leg and non-trivial contraction order.
        """
        nodes_tensors = [self.op2, self.bra]
        self._add_subtree_tensor((4,8,10,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 ignored_leg=self.ign_leg_id,
                                 connection_index=2,
                                 contraction_order=[1,0])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op2[1],
                                subtree_tens,
                                axes=([1],[2]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                            axes=([1,5],[2,1]))
        ref = ref.transpose(0,4,2,3,1)
        npt.assert_allclose(result, ref)

    def test_ket_op_op(self):
        """
        Test contraction of a ket and two operator nodes with one subtree
        and on uncontracted neighbour leg.
        """
        valid_contr_orders = [
            [0, 1, 2],
            [1, 0, 2],
            [1, 2, 0],
            [2, 1, 0],
        ]
        self._add_subtree_tensor((4,8,10,6))
        # Reference
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([1],[0]))
        intermed = np.tensordot(intermed,
                                self.op1[1],
                                axes=([1,2],[3,1]))
        intermed = np.tensordot(intermed,
                                 self.op2[1],
                                 axes=([1,4],[1,3]))
        ref = intermed.transpose(0,2,3,1,4)
        for contr_order in valid_contr_orders:
            lcocal_contr = LocalContraction([self.ket, self.op1, self.op2],
                                             self.subtree_dict,
                                             ignored_leg=self.ign_leg_id,
                                             contraction_order=contr_order)
            result = lcocal_contr.contract_all()
            npt.assert_allclose(result, ref)

    def test_ket_op_op_bra(self):
        """
        Test contraction of a ket, two operator and a bra leaf nodes with one
        subtree tensor.
        """
        valid_contr_orders = [
            [0,1,2,3],
            [1,0,2,3],
            [1,2,0,3],
            [1,2,3,0],
            [2,3,1,0],
            [2,1,0,3],
            [2,1,3,0],
            [3,2,1,0],
        ]
        self._add_subtree_tensor((4,8,10,6))
        # Reference
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([1],[0]))
        intermed = np.tensordot(intermed,
                                self.op1[1],
                                axes=([1,2],[3,1]))
        intermed = np.tensordot(intermed,
                                 self.op2[1],
                                 axes=([1,4],[1,3]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                           axes=([4,1],[2,1]))
        for contr_order in valid_contr_orders:
            lcocal_contr = LocalContraction([self.ket, self.op1, self.op2,
                                             self.bra],
                                             self.subtree_dict,
                                             ignored_leg=self.ign_leg_id,
                                             contraction_order=contr_order)
            result = lcocal_contr.contract_all()
            npt.assert_allclose(result, ref)

class TestLocalContractionTwoNeighTwoSubtree(unittest.TestCase):
    """
    Test contractions for a node with two neighbours, where one is not
    contracted and one is contracted to a subtree tensor.
    """

    def setUp(self):
        self.node_id = "node_id"
        self.ket = random_tensor_node((3,4,2),
                                      identifier=self.node_id,
                                      seed=SEED)
        self.bra = random_tensor_node((5,6,2),
                                      identifier=self.node_id,
                                      seed=SEED+1)
        self.op1 = random_tensor_node((7,8,2,2),
                                      identifier=self.node_id,
                                      seed=SEED+2)
        self.op2 = random_tensor_node((9,10,2,2),
                                      identifier=self.node_id,
                                      seed=SEED+3)
        self.neigh_ids = [f"neig{i}" for i in range(2)]
        for node in (self.ket, self.bra, self.op1, self.op2):
            node[0].open_leg_to_parent(self.neigh_ids[0],0)
            node[0].open_leg_to_child(self.neigh_ids[1],1)
        self.subtree_dict = PartialTreeCachDict()

    def _add_subtree_tensors(self, shape0, shape1):
        subtree_tensor = crandn(shape0, seed=SEED+10)
        self.subtree_dict.add_entry(self.neigh_ids[0],
                                    self.node_id,
                                    subtree_tensor)
        subtree_tensor = crandn(shape1, seed=SEED+11)
        self.subtree_dict.add_entry(self.neigh_ids[1],
                                    self.node_id,
                                    subtree_tensor)

    def test_ket_node(self):
        """
        Test the contraction of a ket node to one subtree with one remaining
        open neighbour leg.
        """
        nodes_tensors = [self.ket]
        self._add_subtree_tensors((3,7,9,5),(4,8,10,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict
                                 )
        result = contr.contract_all()
        # Reference
        subtree_tensor = self.subtree_dict.get_entry(self.neigh_ids[0],
                                                     self.node_id)
        ref = np.tensordot(self.ket[1],
                           subtree_tensor,
                           axes=(0,0))
        subtree_tensor = self.subtree_dict.get_entry(self.neigh_ids[1],
                                                     self.node_id)
        ref = np.tensordot(ref,
                           subtree_tensor,
                           axes=(0,0))
        ref = ref.transpose(1,2,3,4,5,6,0)
        npt.assert_allclose(ref,result)

    def test_operator(self):
        """
        Test contraction of an operator leaf node with one subtree tensor.
        """
        nodes_tensors = [self.op1]
        self._add_subtree_tensors((3,7,9,5),(4,8,10,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1)
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_ids[0],
                                                    self.node_id)
        ref = np.tensordot(self.op1[1],
                            subtree_tens,
                            axes=([0],[1]))
        subtree_tens = self.subtree_dict.get_entry(self.neigh_ids[1],
                                                   self.node_id)
        ref = np.tensordot(ref,
                           subtree_tens,
                           axes=(0,1))
        ref = ref.transpose(2,3,4,5,6,7,0,1)
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_bra(self):
        """
        Test contraction of a bra leaf node with one subtree tensor.
        """
        nodes_tensors = [self.bra]
        self._add_subtree_tensors((3,7,9,5),(4,8,10,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=3,
                                 highest_tensor=TensorKind.BRA)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_ids[0],
                                                    self.node_id)
        ref = np.tensordot(self.bra[1],
                           subtree_tens,
                            axes=(0,3))
        subtree_tens = self.subtree_dict.get_entry(self.neigh_ids[1],
                                                   self.node_id)
        ref = np.tensordot(ref,
                           subtree_tens,
                           axes=(0,3))
        ref = ref.transpose(1,2,3,4,5,6,0)
        npt.assert_allclose(result, ref)

    def test_ket_operator(self):
        """
        Test contraction of a ket and an operator leaf node with one subtree
        tensor.
        """
        nodes_tensors = [self.ket, self.op1]
        self._add_subtree_tensors((3,7,9,5),(4,8,10,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict)
        result = contr.contract_all()
        # Reference result
        subtree_tensor = self.subtree_dict.get_entry(self.neigh_ids[0],
                                                     self.node_id)
        ref = np.tensordot(self.ket[1],
                           subtree_tensor,
                           axes=(0,0))
        subtree_tensor = self.subtree_dict.get_entry(self.neigh_ids[1],
                                                     self.node_id)
        ref = np.tensordot(ref,
                           subtree_tensor,
                           axes=(0,0))
        ref = np.tensordot(ref,
                           self.op1[1],
                           axes=([0,1,4],[3,0,1]))
        npt.assert_allclose(result, ref)

    def test_ket_operator_rev_contr_order(self):
        """
        Test contraction of a ket and an operator leaf node with one subtree
        tensor and reversed contraction order.
        """
        nodes_tensors = [self.ket, self.op1]
        self._add_subtree_tensors((3,7,9,5),(4,8,10,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 contraction_order=[1,0])
        result = contr.contract_all()
        # Reference result
        subtree_tensor = self.subtree_dict.get_entry(self.neigh_ids[0],
                                                     self.node_id)
        ref = np.tensordot(self.ket[1],
                           subtree_tensor,
                           axes=(0,0))
        subtree_tensor = self.subtree_dict.get_entry(self.neigh_ids[1],
                                                     self.node_id)
        ref = np.tensordot(ref,
                           subtree_tensor,
                           axes=(0,0))
        ref = np.tensordot(ref,
                           self.op1[1],
                           axes=([0,1,4],[3,0,1]))
        npt.assert_allclose(result, ref)

    def test_op_op(self):
        """
        Test contraction of two operator leaf nodes with one subtree tensor.
        """
        nodes_tensors = [self.op1, self.op2]
        self._add_subtree_tensors((3,7,9,5),(4,8,10,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_ids[0],
                                                    self.node_id)
        ref = np.tensordot(self.op1[1],
                            subtree_tens,
                            axes=([0],[1]))
        subtree_tens = self.subtree_dict.get_entry(self.neigh_ids[1],
                                                   self.node_id)
        ref = np.tensordot(ref,
                           subtree_tens,
                           axes=(0,1))
        ref = np.tensordot(ref,
                           self.op2[1],
                           axes=([0,3,6],[3,0,1]))
        ref = ref.transpose(1,2,3,4,5,0)
        npt.assert_allclose(result, ref)

    def test_op_op_rev_contr_order(self):
        """
        Test contraction of two operator leaf nodes with one subtree tensor
        and reversed contraction order.
        """
        nodes_tensors = [self.op1, self.op2]
        self._add_subtree_tensors((3,7,9,5),(4,8,10,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 contraction_order=[1,0])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_ids[0],
                                                    self.node_id)
        ref = np.tensordot(self.op1[1],
                            subtree_tens,
                            axes=([0],[1]))
        subtree_tens = self.subtree_dict.get_entry(self.neigh_ids[1],
                                                   self.node_id)
        ref = np.tensordot(ref,
                           subtree_tens,
                           axes=(0,1))
        ref = np.tensordot(ref,
                           self.op2[1],
                           axes=([0,3,6],[3,0,1]))
        ref = ref.transpose(1,2,3,4,5,0)
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node(self):
        """
        Test contraction of an operator and a bra node with one subtree tensor
        and one ignored neighbour leg.
        """
        nodes_tensors = [self.op2, self.bra]
        self._add_subtree_tensors((3,7,9,5),(4,8,10,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=2)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_ids[0],
                                                    self.node_id)
        intermed = np.tensordot(self.op2[1],
                                subtree_tens,
                                axes=([0],[2]))
        subtree_tens = self.subtree_dict.get_entry(self.neigh_ids[1],
                                                    self.node_id)
        intermed = np.tensordot(intermed,
                                subtree_tens,
                                axes=([0],[2]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                            axes=([0,4,7],[2,0,1]))
        ref = ref.transpose(1,2,3,4,0)
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node_rev_contr_order(self):
        """
        Test contraction of an operator and a bra  node with one subtree tensor
        and one ignored neighbour leg and non-trivial contraction order.
        """
        nodes_tensors = [self.op2, self.bra]
        self._add_subtree_tensors((3,7,9,5),(4,8,10,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=2,
                                 contraction_order=[1,0])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_ids[0],
                                                    self.node_id)
        intermed = np.tensordot(self.op2[1],
                                subtree_tens,
                                axes=([0],[2]))
        subtree_tens = self.subtree_dict.get_entry(self.neigh_ids[1],
                                                    self.node_id)
        intermed = np.tensordot(intermed,
                                subtree_tens,
                                axes=([0],[2]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                            axes=([0,4,7],[2,0,1]))
        ref = ref.transpose(1,2,3,4,0)
        npt.assert_allclose(result, ref)

    def test_ket_op_op(self):
        """
        Test contraction of a ket and two operator nodes with one subtree
        and on uncontracted neighbour leg.
        """
        valid_contr_orders = [
            [0, 1, 2],
            [1, 0, 2],
            [1, 2, 0],
            [2, 1, 0],
        ]
        self._add_subtree_tensors((3,7,9,5),(4,8,10,6))
        # Reference result
        subtree_tensor = self.subtree_dict.get_entry(self.neigh_ids[0],
                                                     self.node_id)
        ref = np.tensordot(self.ket[1],
                           subtree_tensor,
                           axes=(0,0))
        subtree_tensor = self.subtree_dict.get_entry(self.neigh_ids[1],
                                                     self.node_id)
        ref = np.tensordot(ref,
                           subtree_tensor,
                           axes=(0,0))
        ref = np.tensordot(ref,
                           self.op1[1],
                           axes=([0,1,4],[3,0,1]))
        ref = np.tensordot(ref,
                           self.op2[1],
                           axes=([0,2,4],[0,1,3]))
        # No Transpose needed
        for contr_order in valid_contr_orders:
            lcocal_contr = LocalContraction([self.ket, self.op1, self.op2],
                                             self.subtree_dict,
                                             contraction_order=contr_order)
            result = lcocal_contr.contract_all()
            npt.assert_allclose(result, ref)

    def test_ket_op_op_bra(self):
        """
        Test contraction of a ket, two operator and a bra leaf nodes with one
        subtree tensor.
        """
        valid_contr_orders = [
            [0,1,2,3],
            [1,0,2,3],
            [1,2,0,3],
            [1,2,3,0],
            [2,3,1,0],
            [2,1,0,3],
            [2,1,3,0],
            [3,2,1,0],
        ]
        self._add_subtree_tensors((3,7,9,5),(4,8,10,6))
        # Reference

        # Reference result
        subtree_tensor = self.subtree_dict.get_entry(self.neigh_ids[0],
                                                     self.node_id)
        ref = np.tensordot(self.ket[1],
                           subtree_tensor,
                           axes=(0,0))
        subtree_tensor = self.subtree_dict.get_entry(self.neigh_ids[1],
                                                     self.node_id)
        ref = np.tensordot(ref,
                           subtree_tensor,
                           axes=(0,0))
        ref = np.tensordot(ref,
                           self.op1[1],
                           axes=([0,1,4],[3,0,1]))
        ref = np.tensordot(ref,
                           self.op2[1],
                           axes=([0,2,4],[0,1,3]))
        ref = np.tensordot(ref,
                           self.bra[1],
                           axes=([0,1,2],[0,1,2]))
        for contr_order in valid_contr_orders:
            lcocal_contr = LocalContraction([self.ket, self.op1, self.op2,
                                             self.bra],
                                             self.subtree_dict,
                                             contraction_order=contr_order)
            result = lcocal_contr.contract_all()
            npt.assert_allclose(result, ref)
