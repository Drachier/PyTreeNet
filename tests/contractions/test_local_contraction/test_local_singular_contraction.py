"""
This module provides tests for the local contraction class if the contracted
tensor have no neighbours (singular contraction).
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

class TestSingularContractions(unittest.TestCase):
    """
    Test contractions for nodes without a neighbour.
    """

    def test_ket_node(self):
        """
        A singular ket node should just give itself back.
        """
        node, tensor = random_tensor_node((3,), identifier="A",
                                  seed=SEED)
        contr = LocalContraction([(node, tensor)],
                                 PartialTreeCachDict()
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, tensor)

    def test_operator_node(self):
        """
        A singular operator node should just give itself back.
        """
        node, tensor = random_tensor_node((3, 3), identifier="A",
                                  seed=SEED)
        contr = LocalContraction([(node, tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=1
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, tensor)

    def test_bra_node(self):
        """
        A singular bra node should just give itself back.
        """
        node, tensor = random_tensor_node((3,), identifier="A",
                                  seed=SEED)
        contr = LocalContraction([(node, tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=1,
                                 highest_tensor=TensorKind.BRA
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, tensor)

    def test_ket_and_operator_node(self):
        """
        A ket and operator node should be contracted at the 
        correct index.
        """
        ket_node, ket_tensor = random_tensor_node((3,), identifier="A",
                                          seed=SEED)
        op_node, op_tensor = random_tensor_node((3, 3), identifier="B",
                                        seed=SEED+1)
        # Ref contraction (Just a matrix-vector product)
        ref = op_tensor @ ket_tensor
        # Local contraction
        contr = LocalContraction([(ket_node, ket_tensor),
                                  (op_node, op_tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=0
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_ket_and_operator_node_reverse_contr_order(self):
        """
        A ket and operator node should be contracted at the 
        correct index, even if the contraction order is reversed.
        """
        ket_node, ket_tensor = random_tensor_node((3,), identifier="A",
                                          seed=SEED)
        op_node, op_tensor = random_tensor_node((3, 3), identifier="B",
                                        seed=SEED+1)
        # Ref contraction (Just a matrix-vector product)
        ref = op_tensor @ ket_tensor
        # Local contraction
        contr = LocalContraction([(ket_node, ket_tensor),
                                  (op_node, op_tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=0,
                                 contraction_order=[1, 0]
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_two_operator_nodes(self):
        """
        Two operator nodes should be contracted at the 
        correct index.
        """
        op1_node, op1_tensor = random_tensor_node((3, 3), identifier="A",
                                          seed=SEED)
        op2_node, op2_tensor = random_tensor_node((3, 3), identifier="B",
                                        seed=SEED+1)
        # Ref contraction (Just a matrix-matrix product)
        ref = op2_tensor @ op1_tensor
        # Local contraction
        contr = LocalContraction([(op1_node, op1_tensor),
                                  (op2_node, op2_tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=1
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_two_operator_nodes_reverse_contr_order(self):
        """
        Two operator nodes should be contracted at the 
        correct index, even if the contraction order is reversed.
        """
        op1_node, op1_tensor = random_tensor_node((3, 3), identifier="A",
                                          seed=SEED)
        op2_node, op2_tensor = random_tensor_node((3, 3), identifier="B",
                                        seed=SEED+1)
        # Ref contraction (Just a matrix-matrix product)
        ref = op2_tensor @ op1_tensor
        # Local contraction
        contr = LocalContraction([(op1_node, op1_tensor),
                                  (op2_node, op2_tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=1,
                                 contraction_order=[1, 0]
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node(self):
        """
        An operator and bra node should be contracted at the 
        correct index.
        """
        bra_node, bra_tensor = random_tensor_node((3,), identifier="A",
                                          seed=SEED)
        op_node, op_tensor = random_tensor_node((3, 3), identifier="B",
                                        seed=SEED+1)
        # Ref contraction (Just a vector-matrix product)
        ref = bra_tensor @ op_tensor
        # Local contraction
        contr = LocalContraction([(op_node, op_tensor),
                                  (bra_node, bra_tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=1
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node_reverse_contr_order(self):
        """
        An operator and bra node should be contracted at the 
        correct index, even if the contraction order is reversed.
        """
        bra_node, bra_tensor = random_tensor_node((3,), identifier="A",
                                          seed=SEED)
        op_node, op_tensor = random_tensor_node((3, 3), identifier="B",
                                        seed=SEED+1)
        # Ref contraction (Just a vector-matrix product)
        ref = bra_tensor @ op_tensor
        # Local contraction
        contr = LocalContraction([(op_node, op_tensor),
                                  (bra_node, bra_tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=1,
                                 contraction_order=[1, 0]
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_ket_and_two_operator_nodes_valid_order(self):
        """
        A ket and two operator nodes should be contracted at the 
        correct index.
        """
        ket_node, ket_tensor = random_tensor_node((3,), identifier="A",
                                          seed=SEED)
        op1_node, op1_tensor = random_tensor_node((3, 3), identifier="B",
                                        seed=SEED+1)
        op2_node, op2_tensor = random_tensor_node((3, 3), identifier="C",
                                        seed=SEED+2)
        # Ref contraction (Just a matrix-matrix-vector product)
        ref = op2_tensor @ (op1_tensor @ ket_tensor)
        valid_contr_orders = [
            [0, 1, 2],
            [1, 0, 2],
            [1, 2, 0],
            [2, 1, 0],
        ]
        for contr_order in valid_contr_orders:
            # Local contraction
            contr = LocalContraction([(ket_node, ket_tensor),
                                      (op1_node, op1_tensor),
                                      (op2_node, op2_tensor)],
                                     PartialTreeCachDict(),
                                     connection_index=0,
                                     contraction_order=contr_order
                                     )
            result = contr.contract_all()
            npt.assert_allclose(result, ref)

    def test_ket_and_two_operator_nodes_invalid_order(self):
        """
        A ket and two operator nodes should raise an error if the
        contraction order is invalid.
        """
        ket_node, ket_tensor = random_tensor_node((3,), identifier="A",
                                          seed=SEED)
        op1_node, op1_tensor = random_tensor_node((3, 3), identifier="B",
                                        seed=SEED+1)
        op2_node, op2_tensor = random_tensor_node((3, 3), identifier="C",
                                        seed=SEED+2)
        # Invalid contraction orders
        invalid_contr_orders = [
            [0, 2, 1],
            [2, 0, 1],
        ]
        for contr_order in invalid_contr_orders:
            # Local contraction
            with self.assertRaises(ValueError):
                _ = LocalContraction([(ket_node, ket_tensor),
                                      (op1_node, op1_tensor),
                                      (op2_node, op2_tensor)],
                                     PartialTreeCachDict(),
                                     connection_index=0,
                                     contraction_order=contr_order
                                     )

    def test_contract_ket_op_op_bra(self):
        """
        A ket, two operators and a bra should be contracted at the 
        correct index.
        """
        ket_node, ket_tensor = random_tensor_node((3,), identifier="A",
                                          seed=SEED)
        op1_node, op1_tensor = random_tensor_node((3, 3), identifier="B",
                                        seed=SEED+1)
        op2_node, op2_tensor = random_tensor_node((3, 3), identifier="C",
                                        seed=SEED+2)
        bra_node, bra_tensor = random_tensor_node((3,), identifier="D",
                                        seed=SEED+3)
        # Ref contraction (Just a bra-matrix-matrix-vector product)
        ref = bra_tensor @ (op2_tensor @ (op1_tensor @ ket_tensor))
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
        for contr_order in valid_contr_orders:
            # Local contraction
            contr = LocalContraction([(ket_node, ket_tensor),
                                      (op1_node, op1_tensor),
                                      (op2_node, op2_tensor),
                                      (bra_node, bra_tensor)],
                                     PartialTreeCachDict(),
                                     connection_index=0,
                                     contraction_order=contr_order
                                     )
            result = contr.contract_all()
            npt.assert_allclose(result, ref)

    # Now we repeat this with two open legs everywhere
    def test_ket_node_two_legs(self):
        """
        A singular ket node with two legs should just give itself back.
        """
        node, tensor = random_tensor_node((3, 3), identifier="A",
                                  seed=SEED)
        contr = LocalContraction([(node, tensor)],
                                 PartialTreeCachDict()
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, tensor)

    def test_bra_node_two_legs(self):
        """
        A singular bra node with two legs should just give itself back.
        """
        node, tensor = random_tensor_node((3, 3), identifier="A",
                                  seed=SEED)
        contr = LocalContraction([(node, tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=1,
                                 highest_tensor=TensorKind.BRA
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, tensor)

    def test_operator_node_two_legs(self):
        """
        A singular operator node with two legs should just give itself back.
        """
        node, tensor = random_tensor_node((3, 3, 3, 3), identifier="A",
                                  seed=SEED)
        contr = LocalContraction([(node, tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=1
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, tensor)

    def test_ket_and_operator_node_two_legs(self):
        """
        A ket and operator node with two legs should be contracted at the 
        correct index.
        """
        ket_node, ket_tensor = random_tensor_node((3, 3), identifier="A",
                                          seed=SEED)
        op_node, op_tensor = random_tensor_node((3, 3, 3, 3), identifier="B",
                                        seed=SEED+1)
        # Ref contraction (Just a matrix-vector product on the last leg)
        ref = np.tensordot(op_tensor, ket_tensor, axes=((2,3),(0,1)))
        # Local contraction
        contr = LocalContraction([(ket_node, ket_tensor),
                                  (op_node, op_tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=0
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_ket_and_operator_node_two_legs_reverse_contr_order(self):
        """
        A ket and operator node with two legs should be contracted at the 
        correct index, even if the contraction order is reversed.
        """
        ket_node, ket_tensor = random_tensor_node((3, 3), identifier="A",
                                          seed=SEED)
        op_node, op_tensor = random_tensor_node((3, 3, 3, 3), identifier="B",
                                        seed=SEED+1)
        # Ref contraction (Just a matrix-vector product on the last leg)
        ref = np.tensordot(op_tensor, ket_tensor, axes=((2,3),(0,1)))
        # Local contraction
        contr = LocalContraction([(ket_node, ket_tensor),
                                  (op_node, op_tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=0,
                                 contraction_order=[1, 0]
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_two_operator_nodes_two_legs(self):
        """
        Two operator nodes with two legs should be contracted at the 
        correct index.
        """
        op1_node, op1_tensor = random_tensor_node((3, 3, 3, 3), identifier="A",
                                          seed=SEED)
        op2_node, op2_tensor = random_tensor_node((3, 3, 3, 3), identifier="B",
                                        seed=SEED+1)
        # Ref contraction (Just a matrix-matrix product on the last leg)
        ref = np.tensordot(op2_tensor, op1_tensor, axes=((2,3),(0,1)))
        # Local contraction
        contr = LocalContraction([(op1_node, op1_tensor),
                                  (op2_node, op2_tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=1
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_two_operator_nodes_two_legs_reverse_contr_order(self):
        """
        Two operator nodes with two legs should be contracted at the 
        correct index, even if the contraction order is reversed.
        """
        op1_node, op1_tensor = random_tensor_node((3, 3, 3, 3), identifier="A",
                                          seed=SEED)
        op2_node, op2_tensor = random_tensor_node((3, 3, 3, 3), identifier="B",
                                        seed=SEED+1)
        # Ref contraction (Just a matrix-matrix product on the last leg)
        ref = np.tensordot(op2_tensor, op1_tensor, axes=((2,3),(0,1)))
        # Local contraction
        contr = LocalContraction([(op1_node, op1_tensor),
                                  (op2_node, op2_tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=1,
                                 contraction_order=[1, 0]
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node_two_legs(self):
        """
        An operator and bra node with two legs should be contracted at the 
        correct index.
        """
        bra_node, bra_tensor = random_tensor_node((3, 3), identifier="A",
                                          seed=SEED)
        op_node, op_tensor = random_tensor_node((3, 3, 3, 3), identifier="B",
                                        seed=SEED+1)
        # Ref contraction (Just a vector-matrix product on the last leg)
        ref = np.tensordot(bra_tensor, op_tensor, axes=((0,1),(0,1)))
        # Local contraction
        contr = LocalContraction([(op_node, op_tensor),
                                  (bra_node, bra_tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=1
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_operator_and_bra_node_two_legs_reverse_contr_order(self):
        """
        An operator and bra node with two legs should be contracted at the 
        correct index, even if the contraction order is reversed.
        """
        bra_node, bra_tensor = random_tensor_node((3, 3), identifier="A",
                                          seed=SEED)
        op_node, op_tensor = random_tensor_node((3, 3, 3, 3), identifier="B",
                                        seed=SEED+1)
        # Ref contraction (Just a vector-matrix product on the last leg)
        ref = np.tensordot(bra_tensor, op_tensor, axes=((0,1),(0,1)))
        # Local contraction
        contr = LocalContraction([(op_node, op_tensor),
                                  (bra_node, bra_tensor)],
                                 PartialTreeCachDict(),
                                 connection_index=1,
                                 contraction_order=[1, 0]
                                 )
        result = contr.contract_all()
        npt.assert_allclose(result, ref)

    def test_ket_and_two_operator_nodes(self):
        """
        A ket and two operator nodes with two legs should be contracted at the 
        correct index.
        """
        ket_node, ket_tensor = random_tensor_node((3, 3), identifier="A",
                                          seed=SEED)
        op1_node, op1_tensor = random_tensor_node((3, 3, 3, 3), identifier="B",
                                        seed=SEED+1)
        op2_node, op2_tensor = random_tensor_node((3, 3, 3, 3), identifier="C",
                                        seed=SEED+2)
        # Ref contraction (Just a matrix-matrix-vector product on the last leg)
        ref = np.tensordot(op2_tensor,op1_tensor,
                           axes=((2,3),(0,1)))
        ref = np.tensordot(ref, ket_tensor, axes=((2,3),(0,1)))
        valid_contr_orders = [
            [0,1,2],
            [1,0,2],
            [1,2,0],
            [2,1,0],
        ]
        for contr_order in valid_contr_orders:
            # Local contraction
            contr = LocalContraction([(ket_node, ket_tensor),
                                      (op1_node, op1_tensor),
                                      (op2_node, op2_tensor)],
                                     PartialTreeCachDict(),
                                     connection_index=0,
                                     contraction_order=contr_order)
            result = contr.contract_all()
            npt.assert_allclose(result, ref)

    def test_contract_ket_op_op_bra_two_legs(self):
        """
        A ket, two operators and a bra should be contracted at the 
        correct index, even with two legs each.
        """
        ket_node, ket_tensor = random_tensor_node((3,3), identifier="A",
                                          seed=SEED)
        op1_node, op1_tensor = random_tensor_node((3, 3, 3, 3), identifier="B",
                                        seed=SEED+1)
        op2_node, op2_tensor = random_tensor_node((3, 3, 3, 3), identifier="C",
                                        seed=SEED+2)
        bra_node, bra_tensor = random_tensor_node((3, 3), identifier="D",
                                        seed=SEED+3)
        # Ref contraction (Just a bra-matrix-matrix-vector product)
        ref = np.tensordot(op1_tensor, ket_tensor, axes=((2,3),(0,1)))
        ref = np.tensordot(op2_tensor, ref, axes=((2,3),(0,1)))
        ref = np.tensordot(bra_tensor, ref, axes=((0,1),(0,1)))
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
        for contr_order in valid_contr_orders:
            # Local contraction
            contr = LocalContraction([(ket_node, ket_tensor),
                                      (op1_node, op1_tensor),
                                      (op2_node, op2_tensor),
                                      (bra_node, bra_tensor)],
                                     PartialTreeCachDict(),
                                     connection_index=0,
                                     contraction_order=contr_order
                                     )
            result = contr.contract_all()
            npt.assert_allclose(result, ref)

if __name__ == "__main__":
    unittest.main()
