import unittest

import numpy as np

import pytreenet as ptn

from pytreenet.contractions.state_operator_contraction import (contract_leaf,
                                                               contract_operator_tensor_ignoring_one_leg,
                                                               contract_bra_tensor_ignore_one_leg)
from pytreenet.contractions.contraction_util import (contract_all_but_one_neighbour_block_to_ket)

class TestStateOperatorContraction(unittest.TestCase):
    def setUp(self):
        self.state = ptn.random_small_ttns()
        self.conversion_dict = {"root_op1": ptn.crandn((2,2)),
                                "root_op2": ptn.crandn((2,2)),
                                "I2": np.eye(2),
                                "c1_op": ptn.crandn((3,3)),
                                "I3": np.eye(3),
                                "c2_op": ptn.crandn((4,4)),
                                "I4": np.eye(4)}
        self.state = ptn.random_small_ttns()
        tensor_prod = [ptn.TensorProduct({"c1": "I3", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op2", "c2": "c2_op"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "I2", "c2": "c2_op"})
                       ]
        ham = ptn.Hamiltonian(tensor_prod, self.conversion_dict)
        # All bond dim are 2
        self.operator = ptn.TTNO.from_hamiltonian(ham, self.state)

        self.tensor_dict = ptn.PartialTreeCachDict()
        self.tensor_dict.add_entry("c1","root",ptn.crandn((5,2,5)))
        self.tensor_dict.add_entry("c2","root",ptn.crandn((6,2,6)))

    def test_contract_leaf_c1(self):
        """
        Contract the bra operator and ket for the leaf c1
        """
        node_id = "c1"
        # Reference Contraction
        ket_tensor = self.state.tensors[node_id]
        op_tensor = self.operator.tensors[node_id]
        ref_tensor = np.tensordot(ket_tensor,op_tensor,
                                  axes=(1,2))
        bra_tensor = ket_tensor.conj()
        ref_tensor = np.tensordot(ref_tensor,bra_tensor,
                                  axes=(2,1))
        found_tensor = contract_leaf(node_id,
                                     self.state,
                                     self.operator)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_leaf_c2(self):
        """
        Contract the bra operator and ket for the leaf c2
        """
        node_id = "c2"
        # Reference Contraction
        ket_tensor = self.state.tensors[node_id]
        op_tensor = self.operator.tensors[node_id]
        ref_tensor = np.tensordot(ket_tensor,op_tensor,
                                  axes=(1,2))
        bra_tensor = ket_tensor.conj()
        ref_tensor = np.tensordot(ref_tensor,bra_tensor,
                                  axes=(2,1))
        found_tensor = contract_leaf(node_id,
                                     self.state,
                                     self.operator)
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_operator_tensor_ignoring_leg_to_c2(self):
        """
        Contract

                                    ______
                                   |      |
                            _______|      |
                                   |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  H  |    |  c1  |
                        |_____|    |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|

        where the open legs point towards c2.
        """
        ket_node, ket_tensor = self.state["root"]
        ketblock_tensor = contract_all_but_one_neighbour_block_to_ket(ket_tensor,
                                                                      ket_node,
                                                                      "c2",
                                                                      self.tensor_dict)
        op_node, op_tensor = self.operator["root"]
        # Reference Contraction
        ref_tensor = np.tensordot(ketblock_tensor,
                                  op_tensor,
                                  axes=((1,2),(3,0)))
        found_tensor = contract_operator_tensor_ignoring_one_leg(ketblock_tensor,
                                                                 ket_node,
                                                                 op_tensor,
                                                                 op_node,
                                                                 "c2")
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_operator_tensor_ignoring_leg_to_c1(self):
        """
        Contract

                                    ______
                                   |      |
                            _______|      |
                                   |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  H  |    |  c2  |
                        |_____|    |      |
                           |       |      |
                           |       |      |
                         __|__     |      |
                    ____|     |____|      |
                        |  A  |    |      |
                        |_____|    |______|

        where the open legs point towards c1.
        """
        ket_node, ket_tensor = self.state["root"]
        ketblock_tensor = contract_all_but_one_neighbour_block_to_ket(ket_tensor,
                                                                      ket_node,
                                                                      "c1",
                                                                      self.tensor_dict)
        op_node, op_tensor = self.operator["root"]
        # Reference Contraction
        ref_tensor = np.tensordot(ketblock_tensor,
                                  op_tensor,
                                  axes=((1,2),(3,1)))
        found_tensor = contract_operator_tensor_ignoring_one_leg(ketblock_tensor,
                                                                 ket_node,
                                                                 op_tensor,
                                                                 op_node,
                                                                 "c1")
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_bra_tensor_ignore_leg_to_c2(self):
        """
        Contract
                     _____      ______
                ____|     |____|      |
                    |  A* |    |      |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
                ____|     |____|      |
                    |  H  |    |  C   |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
                ____|     |____|      |
                    |  A  |    |      |
                    |_____|    |______|
        where the open legs point towards c2.
        """
        ket_node, ket_tensor = self.state["root"]
        ketblock_tensor = contract_all_but_one_neighbour_block_to_ket(ket_tensor,
                                                                      ket_node,
                                                                      "c2",
                                                                      self.tensor_dict)
        op_node, op_tensor = self.operator["root"]
        ketblockop_tensor = contract_operator_tensor_ignoring_one_leg(ketblock_tensor,
                                                                      ket_node,
                                                                      op_tensor,
                                                                      op_node,
                                                                      "c2")
        # Reference Contraction
        ref_tensor = np.tensordot(ketblockop_tensor,
                                  ket_tensor.conj(),
                                  axes=((3,1),(2,0)))
        found_tensor = contract_bra_tensor_ignore_one_leg(ket_tensor.conj(),
                                                          ketblockop_tensor,
                                                          ket_node,
                                                          "c2")
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

    def test_contract_bra_tensor_ignore_leg_to_c1(self):
        """
        Contract
                     _____      ______
                ____|     |____|      |
                    |  A* |    |      |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
                ____|     |____|      |
                    |  H  |    |  C   |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
                ____|     |____|      |
                    |  A  |    |      |
                    |_____|    |______|
        where the open legs point towards c1.
        """
        ket_node, ket_tensor = self.state["root"]
        ketblock_tensor = contract_all_but_one_neighbour_block_to_ket(ket_tensor,
                                                                      ket_node,
                                                                      "c1",
                                                                      self.tensor_dict)
        op_node, op_tensor = self.operator["root"]
        ketblockop_tensor = contract_operator_tensor_ignoring_one_leg(ketblock_tensor,
                                                                      ket_node,
                                                                      op_tensor,
                                                                      op_node,
                                                                      "c1")
        # Reference Contraction
        ref_tensor = np.tensordot(ketblockop_tensor,
                                  ket_tensor.conj(),
                                  axes=((3,1),(2,1)))
        found_tensor = contract_bra_tensor_ignore_one_leg(ket_tensor.conj(),
                                                          ketblockop_tensor,
                                                          ket_node,
                                                          "c1")
        self.assertTrue(np.allclose(ref_tensor,found_tensor))

# class TestPartialTreeCacheComplicated(unittest.TestCase):
#     def setUp(self):
#         self.ref_state = ptn.random_big_ttns_two_root_children()
#         self.hamiltonian = ptn.TTNO.from_hamiltonian(ptn.random_hamiltonian_compatible(),
#                                                      self.ref_state)
#         self.partial_tree_cache = ptn.PartialTreeCachDict()
#         self.partial_tree_cache.add_entry("site2","site1",
#                                            ptn.PartialTreeCache.for_leaf("site2",
#                                                                          self.ref_state,
#                                                                          self.hamiltonian))
#         self.partial_tree_cache.add_entry("site5","site3",
#                                            ptn.PartialTreeCache.for_leaf("site5",
#                                                                          self.ref_state,
#                                                                          self.hamiltonian))
#         self.partial_tree_cache.add_entry("site7","site6",
#                                            ptn.PartialTreeCache.for_leaf("site7",
#                                                                          self.ref_state,
#                                                                          self.hamiltonian))
#         self.partial_tree_cache.add_entry("site6","site0",
#                                            ptn.PartialTreeCache.with_existing_cache("site6","site0",
#                                                                                     self.partial_tree_cache,
#                                                                                     self.ref_state,
#                                                                                     self.hamiltonian))
#         self.partial_tree_cache.add_entry("site0","site1",
#                                            ptn.PartialTreeCache.with_existing_cache("site0","site1",
#                                                                                     self.partial_tree_cache,
#                                                                                     self.ref_state,
#                                                                                     self.hamiltonian))

if __name__ == "__main__":
    unittest.main()
