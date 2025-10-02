"""
This module provides tests for the local contraction class if the contracted
tensor are leafs, i.e. have a single neighbour.
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

class TestLocalContractionLeafNoNeigh(unittest.TestCase):
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

    def test_operator_and_bra_node(self):
        """
        Test contraction of an operator and a bra leaf node with one subtree
        tensor.
        """
        nodes_tensors = [self.op1, self.bra]
        contr = LocalContraction(nodes_tensors,
                                 PartialTreeCachDict(),
                                 connection_index=1,
                                 ignored_leg=self.ign_leg_id)
        result = contr.contract_all()
        # Reference result
        ref = np.tensordot(self.op1[1],
                           self.bra[1],
                            axes=(1,1))
        ref = ref.transpose(0,2,1)
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node_rev_contr_order(self):
        """
        Test contraction of an operator and a bra leaf node with one subtree
        tensor and reversed contraction order.
        """
        nodes_tensors = [self.op1, self.bra]
        contr = LocalContraction(nodes_tensors,
                                 PartialTreeCachDict(),
                                 ignored_leg=self.ign_leg_id,
                                 connection_index=1,
                                 contraction_order=[1,0])
        result = contr.contract_all()
        # Reference result
        ref = np.tensordot(self.op1[1],
                           self.bra[1],
                            axes=(1,1))
        ref = ref.transpose(0,2,1)
        npt.assert_allclose(result, ref)

    def test_ket_op_op(self):
        """
        Test contraction of a ket and two operator leaf nodes with one subtree
        tensor.
        """
        valid_contr_orders = [
            [0, 1, 2],
            [1, 0, 2],
            [1, 2, 0],
            [2, 1, 0],
        ]
        # Reference
        intermed = np.tensordot(self.ket[1],
                                self.op1[1],
                                axes=(1,2))
        ref = np.tensordot(intermed,
                           self.op2[1],
                            axes=(2,2))
        # No transposition needed
        for contr_order in valid_contr_orders:
            lcocal_contr = LocalContraction([self.ket, self.op1, self.op2],
                                             PartialTreeCachDict(),
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
        # Reference
        intermed = np.tensordot(self.ket[1],
                                self.op1[1],
                                axes=(1,2))
        intermed = np.tensordot(intermed,
                                self.op2[1],
                                axes=(2,2))
        ref = np.tensordot(intermed,
                           self.bra[1],
                           axes=(3,1))
        for contr_order in valid_contr_orders:
            lcocal_contr = LocalContraction([self.ket, self.op1, self.op2,
                                             self.bra],
                                             PartialTreeCachDict(),
                                             ignored_leg=self.ign_leg_id,
                                             contraction_order=contr_order)
            result = lcocal_contr.contract_all()
            npt.assert_allclose(result, ref)

class TestLocalContractionLeafNeigh(unittest.TestCase):
    """
    Test contractions for leaf nodes (single neighbour).
    """

    def setUp(self):
        self.node_id = "node"
        self.ket = random_tensor_node((3,2), seed=SEED,
                                      identifier=self.node_id)
        self.bra = random_tensor_node((4,2), seed=SEED+1,
                                      identifier=self.node_id)
        self.op1 = random_tensor_node((5,2,2), seed=SEED+2,
                                      identifier=self.node_id)
        self.op2 = random_tensor_node((6,2,2), seed=SEED+3,
                                      identifier=self.node_id)
        self.neigh_id = "neigh"
        for node in (self.ket, self.bra, self.op1, self.op2):
            node[0].open_leg_to_parent(self.neigh_id,0)
        self.subtree_dict = PartialTreeCachDict()

    def _add_subtree_tensor(self, shape):
        subtree_tensor = crandn(shape, seed=SEED+10)
        self.subtree_dict.add_entry(self.neigh_id,
                                    self.node_id,
                                    subtree_tensor)

    def test_ket(self):
        """
        Test contraction of a ket leaf node with one subtree tensor.
        """
        nodes_tensors = [self.ket]
        self._add_subtree_tensor((3,5,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 )
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        ref = np.tensordot(self.ket[1],
                           subtree_tens,
                           axes=([0],[0]))
        ref = ref.transpose(1,2,0)
        npt.assert_allclose(result, ref)

    def test_operator(self):
        """
        Test contraction of an operator leaf node with one subtree tensor.
        """
        nodes_tensors = [self.op1]
        self._add_subtree_tensor((3,5,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        ref = np.tensordot(self.op1[1],
                            subtree_tens,
                            axes=([0],[1]))
        ref = ref.transpose(2,3,0,1)
        npt.assert_allclose(result, ref)

    def test_bra(self):
        """
        Test contraction of a bra leaf node with one subtree tensor.
        """
        nodes_tensors = [self.bra]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=2,
                                 highest_tensor=TensorKind.BRA)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        ref = np.tensordot(self.bra[1],
                           subtree_tens,
                            axes=([0],[2]))
        ref = ref.transpose(1,2,0)
        npt.assert_allclose(result, ref)

    def test_ket_operator(self):
        """
        Test contraction of a ket and an operator leaf node with one subtree
        tensor.
        """
        nodes_tensors = [self.ket, self.op1]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([0],[0]))
        ref = np.tensordot(intermed,
                           self.op1[1],
                            axes=([0,1],[2,0]))
        # Not transpose needed
        npt.assert_allclose(result, ref)

    def test_ket_operator_rev_contr_order(self):
        """
        Test contraction of a ket and an operator leaf node with one subtree
        tensor and reversed contraction order.
        """
        nodes_tensors = [self.ket, self.op1]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 contraction_order=[1,0])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([0],[0]))
        ref = np.tensordot(intermed,
                           self.op1[1],
                            axes=([0,1],[2,0]))
        # Not transpose needed
        npt.assert_allclose(result, ref)

    def test_op_op(self):
        """
        Test contraction of two operator leaf nodes with one subtree tensor.
        """
        nodes_tensors = [self.op1, self.op2]
        self._add_subtree_tensor((3,5,6,7))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([0],[1]))
        ref = np.tensordot(intermed,
                           self.op2[1],
                            axes=([0,3],[2,0]))
        ref = ref.transpose(1,2,3,0)
        npt.assert_allclose(result, ref)

    def test_op_op_rev_contr_order(self):
        """
        Test contraction of two operator leaf nodes with one subtree tensor
        and reversed contraction order.
        """
        nodes_tensors = [self.op1, self.op2]
        self._add_subtree_tensor((3,5,6,7))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 contraction_order=[1,0])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([0],[1]))
        ref = np.tensordot(intermed,
                           self.op2[1],
                            axes=([0,3],[2,0]))
        ref = ref.transpose(1,2,3,0)
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node(self):
        """
        Test contraction of an operator and a bra leaf node with one subtree
        tensor.
        """
        nodes_tensors = [self.op1, self.bra]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([0],[1]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                            axes=([0,3],[1,0]))
        ref = ref.transpose(1,0)
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node_rev_contr_order(self):
        """
        Test contraction of an operator and a bra leaf node with one subtree
        tensor and reversed contraction order.
        """
        nodes_tensors = [self.op1, self.bra]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 contraction_order=[1,0])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([0],[1]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                            axes=([0,3],[1,0]))
        ref = ref.transpose(1,0)
        npt.assert_allclose(result, ref)

    def test_ket_op_op(self):
        """
        Test contraction of a ket and two operator leaf nodes with one subtree
        tensor.
        """
        valid_contr_orders = [
            [0, 1, 2],
            [1, 0, 2],
            [1, 2, 0],
            [2, 1, 0],
        ]
        self._add_subtree_tensor((3,5,6,4))
        # Reference
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([0],[0]))
        intermed = np.tensordot(intermed,
                                self.op1[1],
                                axes=([0,1],[2,0]))
        ref = np.tensordot(intermed,
                           self.op2[1],
                            axes=([0,2],[0,2]))
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
        self._add_subtree_tensor((3,5,6,4))
        # Reference
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([0],[0]))
        intermed = np.tensordot(intermed,
                                self.op1[1],
                                axes=([0,1],[2,0]))
        intermed = np.tensordot(intermed,
                                self.op2[1],
                                axes=([0,2],[0,2]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                           axes=([0,1],[0,1]))
        for contr_order in valid_contr_orders:
            lcocal_contr = LocalContraction([self.ket, self.op1, self.op2,
                                             self.bra],
                                             self.subtree_dict,
                                             contraction_order=contr_order)
            result = lcocal_contr.contract_all()
            npt.assert_allclose(result, ref)

class TestLocalContractionLeafNeighDiffIdentifiers(unittest.TestCase):
    """
    Test contractions for leaf nodes (single neighbour).
    """

    def setUp(self):
        self.node_id = "node"
        self.ket = random_tensor_node((3,2), seed=SEED,
                                      identifier=self.node_id)
        ids = ["op1","op2","bra"]
        self.id_maps = [lambda x: x]
        self.id_maps += [lambda x: x+ids[0]]
        self.id_maps += [lambda x: x+ids[1]]
        self.id_maps += [lambda x: x+ids[2]]
        self.bra = random_tensor_node((4,2), seed=SEED+1,
                                      identifier=self.id_maps[3](self.node_id))
        self.op1 = random_tensor_node((5,2,2), seed=SEED+2,
                                      identifier=self.id_maps[1](self.node_id))
        self.op2 = random_tensor_node((6,2,2), seed=SEED+3,
                                      identifier=self.id_maps[2](self.node_id))
        self.neigh_id = "neigh"
        for node, mapping in zip((self.ket, self.op1, self.op2, self.bra),self.id_maps):
            neigh_id = mapping(self.neigh_id)
            node[0].open_leg_to_parent(neigh_id,0)
        self.subtree_dict = PartialTreeCachDict()

    def _add_subtree_tensor(self, shape):
        subtree_tensor = crandn(shape, seed=SEED+10)
        self.subtree_dict.add_entry(self.neigh_id,
                                    self.node_id,
                                    subtree_tensor)

    def test_operator(self):
        """
        Test contraction of an operator leaf node with one subtree tensor.
        """
        nodes_tensors = [self.op1]
        self._add_subtree_tensor((3,5,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 id_trafos=[self.id_maps[1]],
                                 node_identifier=self.node_id,
                                 neighbour_order=[self.neigh_id])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        ref = np.tensordot(self.op1[1],
                            subtree_tens,
                            axes=([0],[1]))
        ref = ref.transpose(2,3,0,1)
        npt.assert_allclose(result, ref)

    def test_bra(self):
        """
        Test contraction of a bra leaf node with one subtree tensor.
        """
        nodes_tensors = [self.bra]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=2,
                                 highest_tensor=TensorKind.BRA,
                                 id_trafos=[self.id_maps[3]],
                                 node_identifier=self.node_id,
                                 neighbour_order=[self.neigh_id])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        ref = np.tensordot(self.bra[1],
                           subtree_tens,
                            axes=([0],[2]))
        ref = ref.transpose(1,2,0)
        npt.assert_allclose(result, ref)

    def test_ket_operator(self):
        """
        Test contraction of a ket and an operator leaf node with one subtree
        tensor.
        """
        nodes_tensors = [self.ket, self.op1]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 id_trafos=self.id_maps[:2])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([0],[0]))
        ref = np.tensordot(intermed,
                           self.op1[1],
                            axes=([0,1],[2,0]))
        # Not transpose needed
        npt.assert_allclose(result, ref)

    def test_ket_operator_rev_contr_order(self):
        """
        Test contraction of a ket and an operator leaf node with one subtree
        tensor and reversed contraction order.
        """
        nodes_tensors = [self.ket, self.op1]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 contraction_order=[1,0],
                                 id_trafos=self.id_maps[:2],
                                 node_identifier=self.node_id)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([0],[0]))
        ref = np.tensordot(intermed,
                           self.op1[1],
                            axes=([0,1],[2,0]))
        # Not transpose needed
        npt.assert_allclose(result, ref)

    def test_op_op(self):
        """
        Test contraction of two operator leaf nodes with one subtree tensor.
        """
        nodes_tensors = [self.op1, self.op2]
        self._add_subtree_tensor((3,5,6,7))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 id_trafos=self.id_maps[1:3],
                                 neighbour_order=[self.neigh_id],
                                 node_identifier=self.node_id)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([0],[1]))
        ref = np.tensordot(intermed,
                           self.op2[1],
                            axes=([0,3],[2,0]))
        ref = ref.transpose(1,2,3,0)
        npt.assert_allclose(result, ref)

    def test_op_op_rev_contr_order(self):
        """
        Test contraction of two operator leaf nodes with one subtree tensor
        and reversed contraction order.
        """
        nodes_tensors = [self.op1, self.op2]
        self._add_subtree_tensor((3,5,6,7))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 contraction_order=[1,0],
                                 id_trafos=self.id_maps[1:3],
                                 neighbour_order=[self.neigh_id],
                                 node_identifier=self.node_id)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([0],[1]))
        ref = np.tensordot(intermed,
                           self.op2[1],
                            axes=([0,3],[2,0]))
        ref = ref.transpose(1,2,3,0)
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node(self):
        """
        Test contraction of an operator and a bra leaf node with one subtree
        tensor.
        """
        nodes_tensors = [self.op1, self.bra]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 node_identifier=self.node_id,
                                 neighbour_order=[self.neigh_id],
                                 id_trafos=[self.id_maps[1],self.id_maps[-1]])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([0],[1]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                            axes=([0,3],[1,0]))
        ref = ref.transpose(1,0)
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node_rev_contr_order(self):
        """
        Test contraction of an operator and a bra leaf node with one subtree
        tensor and reversed contraction order.
        """
        nodes_tensors = [self.op1, self.bra]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 contraction_order=[1,0],
                                 node_identifier=self.node_id,
                                 neighbour_order=[self.neigh_id],
                                 id_trafos=[self.id_maps[1],self.id_maps[-1]])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([0],[1]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                            axes=([0,3],[1,0]))
        ref = ref.transpose(1,0)
        npt.assert_allclose(result, ref)

    def test_ket_op_op(self):
        """
        Test contraction of a ket and two operator leaf nodes with one subtree
        tensor.
        """
        valid_contr_orders = [
            [0, 1, 2],
            [1, 0, 2],
            [1, 2, 0],
            [2, 1, 0],
        ]
        self._add_subtree_tensor((3,5,6,4))
        # Reference
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([0],[0]))
        intermed = np.tensordot(intermed,
                                self.op1[1],
                                axes=([0,1],[2,0]))
        ref = np.tensordot(intermed,
                           self.op2[1],
                            axes=([0,2],[0,2]))
        for contr_order in valid_contr_orders:
            lcocal_contr = LocalContraction([self.ket, self.op1, self.op2],
                                             self.subtree_dict,
                                             contraction_order=contr_order,
                                             id_trafos=self.id_maps[:3])
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
        self._add_subtree_tensor((3,5,6,4))
        # Reference
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([0],[0]))
        intermed = np.tensordot(intermed,
                                self.op1[1],
                                axes=([0,1],[2,0]))
        intermed = np.tensordot(intermed,
                                self.op2[1],
                                axes=([0,2],[0,2]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                           axes=([0,1],[0,1]))
        for contr_order in valid_contr_orders:
            lcocal_contr = LocalContraction([self.ket, self.op1, self.op2,
                                             self.bra],
                                             self.subtree_dict,
                                             contraction_order=contr_order,
                                             id_trafos=self.id_maps)
            result = lcocal_contr.contract_all()
            npt.assert_allclose(result, ref)

class TestLocalContractionLeafNeighDiffIdentifiersTwoOpen(unittest.TestCase):
    """
    Test contractions for leaf nodes (single neighbour) where the different
    node in the stack have different identifiers and the nodes have two
    open legs.
    """

    def setUp(self):
        self.node_id = "node"
        self.ket = random_tensor_node((3,2,7), seed=SEED,
                                      identifier=self.node_id)
        ids = ["op1","op2","bra"]
        self.id_maps = [lambda x: x]
        self.id_maps += [lambda x: x+ids[0]]
        self.id_maps += [lambda x: x+ids[1]]
        self.id_maps += [lambda x: x+ids[2]]
        self.bra = random_tensor_node((4,2,7), seed=SEED+1,
                                      identifier=self.id_maps[3](self.node_id))
        self.op1 = random_tensor_node((5,2,7,2,7), seed=SEED+2,
                                      identifier=self.id_maps[1](self.node_id))
        self.op2 = random_tensor_node((6,2,7,2,7), seed=SEED+3,
                                      identifier=self.id_maps[2](self.node_id))
        self.neigh_id = "neigh"
        for node, mapping in zip((self.ket, self.op1, self.op2, self.bra),self.id_maps):
            neigh_id = mapping(self.neigh_id)
            node[0].open_leg_to_parent(neigh_id,0)
        self.subtree_dict = PartialTreeCachDict()

    def _add_subtree_tensor(self, shape):
        subtree_tensor = crandn(shape, seed=SEED+10)
        self.subtree_dict.add_entry(self.neigh_id,
                                    self.node_id,
                                    subtree_tensor)

    def test_operator(self):
        """
        Test contraction of an operator leaf node with one subtree tensor.
        """
        nodes_tensors = [self.op1]
        self._add_subtree_tensor((3,5,6))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 id_trafos=[self.id_maps[1]],
                                 node_identifier=self.node_id,
                                 neighbour_order=[self.neigh_id])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        ref = np.tensordot(self.op1[1],
                            subtree_tens,
                            axes=([0],[1]))
        ref = ref.transpose(4,5,0,1,2,3)
        npt.assert_allclose(result, ref)

    def test_bra(self):
        """
        Test contraction of a bra leaf node with one subtree tensor.
        """
        nodes_tensors = [self.bra]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=2,
                                 highest_tensor=TensorKind.BRA,
                                 id_trafos=[self.id_maps[3]],
                                 node_identifier=self.node_id,
                                 neighbour_order=[self.neigh_id])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        ref = np.tensordot(self.bra[1],
                           subtree_tens,
                            axes=([0],[2]))
        ref = ref.transpose(2,3,0,1)
        npt.assert_allclose(result, ref)

    def test_ket_operator(self):
        """
        Test contraction of a ket and an operator leaf node with one subtree
        tensor.
        """
        nodes_tensors = [self.ket, self.op1]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 id_trafos=self.id_maps[:2])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([0],[0]))
        ref = np.tensordot(intermed,
                           self.op1[1],
                            axes=([0,1,2],[3,4,0]))
        # No transpose needed
        npt.assert_allclose(result, ref)

    def test_ket_operator_rev_contr_order(self):
        """
        Test contraction of a ket and an operator leaf node with one subtree
        tensor and reversed contraction order.
        """
        nodes_tensors = [self.ket, self.op1]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 contraction_order=[1,0],
                                 id_trafos=self.id_maps[:2],
                                 node_identifier=self.node_id)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([0],[0]))
        ref = np.tensordot(intermed,
                           self.op1[1],
                            axes=([0,1,2],[3,4,0]))
        # Not transpose needed
        npt.assert_allclose(result, ref)

    def test_op_op(self):
        """
        Test contraction of two operator leaf nodes with one subtree tensor.
        """
        nodes_tensors = [self.op1, self.op2]
        self._add_subtree_tensor((3,5,6,7))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 id_trafos=self.id_maps[1:3],
                                 neighbour_order=[self.neigh_id],
                                 node_identifier=self.node_id)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([0],[1]))
        ref = np.tensordot(intermed,
                           self.op2[1],
                           axes=([0,1,5],[3,4,0]))
        ref = ref.transpose(2,3,4,5,0,1)
        npt.assert_allclose(result, ref)

    def test_op_op_rev_contr_order(self):
        """
        Test contraction of two operator leaf nodes with one subtree tensor
        and reversed contraction order.
        """
        nodes_tensors = [self.op1, self.op2]
        self._add_subtree_tensor((3,5,6,7))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 contraction_order=[1,0],
                                 id_trafos=self.id_maps[1:3],
                                 neighbour_order=[self.neigh_id],
                                 node_identifier=self.node_id)
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([0],[1]))
        ref = np.tensordot(intermed,
                           self.op2[1],
                           axes=([0,1,5],[3,4,0]))
        ref = ref.transpose(2,3,4,5,0,1)
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node(self):
        """
        Test contraction of an operator and a bra leaf node with one subtree
        tensor.
        """
        nodes_tensors = [self.op1, self.bra]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 node_identifier=self.node_id,
                                 neighbour_order=[self.neigh_id],
                                 id_trafos=[self.id_maps[1],self.id_maps[-1]])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([0],[1]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                            axes=([0,1,5],[1,2,0]))
        ref = ref.transpose(2,0,1)
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node_rev_contr_order(self):
        """
        Test contraction of an operator and a bra leaf node with one subtree
        tensor and reversed contraction order.
        """
        nodes_tensors = [self.op1, self.bra]
        self._add_subtree_tensor((3,5,4))
        contr = LocalContraction(nodes_tensors,
                                 self.subtree_dict,
                                 connection_index=1,
                                 contraction_order=[1,0],
                                 node_identifier=self.node_id,
                                 neighbour_order=[self.neigh_id],
                                 id_trafos=[self.id_maps[1],self.id_maps[-1]])
        result = contr.contract_all()
        # Reference result
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.op1[1],
                                subtree_tens,
                                axes=([0],[1]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                            axes=([0,1,5],[1,2,0]))
        ref = ref.transpose(2,0,1)
        npt.assert_allclose(result, ref)

    def test_ket_op_op(self):
        """
        Test contraction of a ket and two operator leaf nodes with one subtree
        tensor.
        """
        valid_contr_orders = [
            [0, 1, 2],
            [1, 0, 2],
            [1, 2, 0],
            [2, 1, 0],
        ]
        self._add_subtree_tensor((3,5,6,4))
        # Reference
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([0],[0]))
        intermed = np.tensordot(intermed,
                                self.op1[1],
                                axes=([0,1,2],[3,4,0]))
        ref = np.tensordot(intermed,
                           self.op2[1],
                            axes=([0,2,3],[0,3,4]))
        for contr_order in valid_contr_orders:
            lcocal_contr = LocalContraction([self.ket, self.op1, self.op2],
                                             self.subtree_dict,
                                             contraction_order=contr_order,
                                             id_trafos=self.id_maps[:3])
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
        self._add_subtree_tensor((3,5,6,4))
        # Reference
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        subtree_tens = self.subtree_dict.get_entry(self.neigh_id,
                                                    self.node_id)
        intermed = np.tensordot(self.ket[1],
                                subtree_tens,
                                axes=([0],[0]))
        intermed = np.tensordot(intermed,
                                self.op1[1],
                                axes=([0,1,2],[3,4,0]))
        intermed = np.tensordot(intermed,
                           self.op2[1],
                            axes=([0,2,3],[0,3,4]))
        ref = np.tensordot(intermed,
                           self.bra[1],
                           axes=([0,1,2],[0,1,2]))
        for contr_order in valid_contr_orders:
            lcocal_contr = LocalContraction([self.ket, self.op1, self.op2,
                                             self.bra],
                                             self.subtree_dict,
                                             contraction_order=contr_order,
                                             id_trafos=self.id_maps)
            result = lcocal_contr.contract_all()
            npt.assert_allclose(result, ref)

if __name__ == "__main__":
    unittest.main()
