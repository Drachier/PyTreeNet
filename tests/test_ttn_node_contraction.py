from __future__ import annotations
from typing import List
import unittest
from copy import deepcopy

import numpy as np

import pytreenet as ptn
from pytreenet.random import crandn

class TestTensorContration(unittest.TestCase):

    def setUp(self) -> None:
        shapes = [(5,6,7),(4,5,7),(6, ),(4,5),(5, ),
                  (2,3,4),(2, ),(3, ),(2,3,4,5),(3, ),
                  (2, ),(5, )]
        self.tensors = {"site"+str(i): crandn(shape)
                        for i, shape in enumerate(shapes)}

        self.ttn = ptn.TreeTensorNetwork()
        self.ttn.add_root(ptn.Node(identifier="site0"), self.tensors["site0"])
        self.ttn.add_child_to_parent(ptn.Node(identifier="site1"),
                                     self.tensors["site1"], 2, "site0", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site2"),
                                     self.tensors["site2"], 0, "site0", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site3"),
                                     self.tensors["site3"], 1, "site0", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site4"),
                                     self.tensors["site4"], 0, "site1", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site5"),
                                     self.tensors["site5"], 2, "site1", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site6"),
                                     self.tensors["site6"], 0, "site5", 1)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site7"),
                                        self.tensors["site7"], 0, "site5", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site8"),
                                     self.tensors["site8"], 2, "site3", 1)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site9"),
                                     self.tensors["site9"], 0, "site8", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site10"),
                                        self.tensors["site10"], 0, "site8", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site11"),
                                     self.tensors["site11"], 0, "site8", 3)

    def check_children(self, node_id: str, child_ids: List[str]):
        node = self.ttn.nodes[node_id]
        for child_id in child_ids:
            self.assertTrue(child_id in self.ttn.nodes)
            self.assertTrue(child_id in self.ttn.tensors)
            child_node = self.ttn.nodes[child_id]
            self.assertTrue(child_node.is_child_of(node_id))
            self.assertTrue(node.is_parent_of(child_id))
        if len(child_ids) == 0:
            self.assertTrue(node.is_leaf())

    def check_parent(self, node_id: str, parent_id: str):
        node = self.ttn.nodes[node_id]
        self.assertTrue(parent_id in self.ttn.nodes)
        self.assertTrue(parent_id in self.ttn.tensors)
        parent_node = self.ttn.nodes[parent_id]
        self.assertTrue(node.is_child_of(parent_id))
        self.assertTrue(parent_node.is_parent_of(node_id))

    def check_root(self, node_id: str):
        node = self.ttn.nodes[node_id]
        self.assertTrue(node.is_root())
        self.assertEqual(node_id, self.ttn.root_id)

    def test_contraction_root_w_child_with_mult_children(self):
        """
        Contract a root node with one of its children that has multiple children.
        """
        self.ttn.contract_nodes("site0", "site1",new_identifier="01")
        found_tensor = self.ttn.tensors["01"]
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.transpose(correct_tensor, (1,0,3,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("01", ["site2", "site3", "site4", "site5"])
        self.check_root("01")

    def test_contraction_root_w_child_with_no_children(self):
        """
        Contract the root node with one of its children that has no children.
        """
        self.ttn.contract_nodes("site0", "site2",new_identifier="02")
        found_tensor = self.ttn.tensors["02"]
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site2"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("02", ["site1", "site3"])
        self.check_root("02")

    def test_contraction_root_w_child_with_one_child(self):
        """
        Contract the root node with one of its children that has one child.
        """
        self.ttn.contract_nodes("site0", "site3",new_identifier="03")
        found_tensor = self.ttn.tensors["03"]
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site3"], axes=(0,1))
        correct_tensor = np.transpose(correct_tensor, (1,0,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("03", ["site1", "site2", "site8"])
        self.check_root("03")

    def test_contraction_node_w_child_with_parent_and_other_child_with_children(self):
        """
        Contract a node with one of its children. The node itself has a parent
         and another child, which also has children.
        """
        self.ttn.contract_nodes("site1", "site4",new_identifier="14")
        found_tensor = self.ttn.tensors["14"]
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site4"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("14", ["site5"])
        self.check_parent("14", "site0")

    def test_contraction_node_w_child_with_parent_and_other_leaf_child(self):
        """
        Contract a node with one of its children. The node itself has a parent
         and another child, which is a leaf.
        """
        self.ttn.contract_nodes("site1", "site5",new_identifier="15")
        found_tensor = self.ttn.tensors["15"]
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site5"], axes=(0,2))
        correct_tensor = np.transpose(correct_tensor, (1,0,2,3))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("15", ["site4", "site6", "site7"])
        self.check_parent("15", "site0")

    def test_contraction_node_w_child_with_parent_and_no_other_child(self):
        """
        Contract a node with one of its children. The node itself has a parent
         and no other children.
        """
        self.ttn.contract_nodes("site3", "site8",new_identifier="38")
        found_tensor = self.ttn.tensors["38"]
        correct_tensor = np.tensordot(self.tensors["site3"], self.tensors["site8"], axes=(0,2))
        correct_tensor = np.transpose(correct_tensor, (0,2,1,3))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("38", ["site9", "site10", "site11"])
        self.check_parent("38", "site0")

    def contract_leaf_with_parent_and_one_other_leaf(self):
        """
        Contract a node with a child that is a leaf. The node has one other
         child that is also a leaf.
        """
        self.ttn.contract_nodes("site5", "site6",new_identifier="56")
        found_tensor = self.ttn.tensors["56"]
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site6"], axes=(0,0))
        correct_tensor = np.transpose(correct_tensor, (1,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("56", ["site7"])
        self.check_parent("56", "site1")

        self.setUp()
        self.ttn.contract_nodes("site5", "site7",new_identifier="57")
        found_tensor = self.ttn.tensors["57"]
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site7"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("57", ["site6"])
        self.check_parent("57", "site1")

    def test_contract_leaf_with_parent_and_two_other_leafs(self):
        """
        Contract a node with a child that is a leaf. The node has two other
         children that are also leaves.
        """
        self.ttn.contract_nodes("site8", "site9",new_identifier="89")
        found_tensor = self.ttn.tensors["89"]
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site9"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("89", ["site10", "site11"])
        self.check_parent("89", "site3")

        self.setUp()
        self.ttn.contract_nodes("site8", "site10",new_identifier="810")
        found_tensor = self.ttn.tensors["810"]
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site10"], axes=(0,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("810", ["site9", "site11"])
        self.check_parent("810", "site3")

        self.setUp()
        self.ttn.contract_nodes("site8", "site11",new_identifier="811")
        found_tensor = self.ttn.tensors["811"]
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site11"], axes=(3,0))
        correct_tensor = np.transpose(correct_tensor, (2,1,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("811", ["site9", "site10"])
        self.check_parent("811", "site3")

    def test_contraction_root_w_child_with_mult_children_other_way(self):
        """
        Contract a root node with one of its children that has multiple children.
        """
        self.ttn.contract_nodes("site1", "site0",new_identifier="01")
        found_tensor = self.ttn.tensors["01"]
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.transpose(correct_tensor, (3,2,1,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("01", ["site2", "site3", "site4", "site5"])
        self.check_root("01")

    def test_contraction_root_w_child_with_no_children_other_way(self):
        """
        Contract the root node with one of its children that has no children.
        """
        self.ttn.contract_nodes("site2", "site0",new_identifier="02")
        found_tensor = self.ttn.tensors["02"]
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site2"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("02", ["site1", "site3"])
        self.check_root("02")

    def test_contraction_root_w_child_with_one_child_other_way(self):
        """
        Contract the root node with one of its children that has one child.
        """
        self.ttn.contract_nodes("site3", "site0",new_identifier="03")
        found_tensor = self.ttn.tensors["03"]
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site3"], axes=(0,1))
        correct_tensor = np.transpose(correct_tensor, (2,1,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("03", ["site1", "site2", "site8"])
        self.check_root("03")

    def test_contraction_node_w_child_with_parent_and_other_child_with_children_other_way(self):
        """
        Contract a node with one of its children. The node itself has a parent
         and another child, which also has children.
        """
        self.ttn.contract_nodes("site4", "site1",new_identifier="14")
        found_tensor = self.ttn.tensors["14"]
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site4"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("14", ["site5"])
        self.check_parent("14", "site0")

    def test_contraction_node_w_child_with_parent_and_other_leaf_child_other_way(self):
        """
        Contract a node with one of its children. The node itself has a parent
         and another child, which is a leaf.
        """
        self.ttn.contract_nodes("site5", "site1",new_identifier="15")
        found_tensor = self.ttn.tensors["15"]
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site5"], axes=(0,2))
        correct_tensor = np.transpose(correct_tensor, (1,2,3,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("15", ["site4", "site6", "site7"])
        self.check_parent("15", "site0")

    def test_contraction_node_w_child_with_parent_and_no_other_child_other_way(self):
        """
        Contract a node with one of its children. The node itself has a parent
         and no other children.
        """
        self.ttn.contract_nodes("site8", "site3",new_identifier="38")
        found_tensor = self.ttn.tensors["38"]
        correct_tensor = np.tensordot(self.tensors["site3"], self.tensors["site8"], axes=(0,2))
        correct_tensor = np.transpose(correct_tensor, (0,2,1,3))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("38", ["site9", "site10", "site11"])
        self.check_parent("38", "site0")

    def test_contract_leaf_with_parent_and_one_other_leaf_other_way(self):
        """
        Contract a node with a child that is a leaf. The node has one other
         child that is also a leaf.
        """
        self.ttn.contract_nodes("site6", "site5",new_identifier="56")
        found_tensor = self.ttn.tensors["56"]
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site6"], axes=(0,0))
        correct_tensor = np.transpose(correct_tensor, (1,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("56", ["site7"])
        self.check_parent("56", "site1")

        self.setUp()
        self.ttn.contract_nodes("site7", "site5",new_identifier="57")
        found_tensor = self.ttn.tensors["57"]
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site7"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("57", ["site6"])
        self.check_parent("57", "site1")

    def test_contract_leaf_with_parent_and_two_other_leafs_other_way(self):
        """
        Contract a node with a child that is a leaf. The node has two other
         children that are also leaves.
        """
        self.ttn.contract_nodes("site9", "site8",new_identifier="89")
        found_tensor = self.ttn.tensors["89"]
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site9"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("89", ["site10", "site11"])
        self.check_parent("89", "site3")

        self.setUp()
        self.ttn.contract_nodes("site10", "site8",new_identifier="810")
        found_tensor = self.ttn.tensors["810"]
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site10"], axes=(0,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("810", ["site9", "site11"])
        self.check_parent("810", "site3")

        self.setUp()
        self.ttn.contract_nodes("site11", "site8",new_identifier="811")
        found_tensor = self.ttn.tensors["811"]
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site11"], axes=(3,0))
        correct_tensor = np.transpose(correct_tensor, (2,1,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("811", ["site9", "site10"])
        self.check_parent("811", "site3")

    def test_contract_node_with_two_leaf_children(self):
        """
        Contract a node with two leaf children.
            |
            5
           / \\
          6   7
        """
        # Contraction order ((5,6),7)
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site6"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site7"], axes=(0,0))
        self.ttn.contract_nodes("site5", "site6",new_identifier="56")
        self.ttn.contract_nodes("56", "site7",new_identifier="567")
        found_tensor = self.ttn.tensors["567"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("567", [])
        self.check_parent("567", "site1")

        # Contraction order ((5,7),6)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site6"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site7"], axes=(0,0))
        self.ttn.contract_nodes("site5", "site7",new_identifier="57")
        self.ttn.contract_nodes("57", "site6",new_identifier="567")
        found_tensor = self.ttn.tensors["567"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("567", [])
        self.check_parent("567", "site1")

        # Contraction order ((6,5),7)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site6"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site7"], axes=(0,0))
        self.ttn.contract_nodes("site6", "site5",new_identifier="56")
        self.ttn.contract_nodes("56", "site7",new_identifier="567")
        found_tensor = self.ttn.tensors["567"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("567", [])
        self.check_parent("567", "site1")

        # Contraction order ((7,5),6)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site6"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site7"], axes=(0,0))
        self.ttn.contract_nodes("site7", "site5",new_identifier="57")
        self.ttn.contract_nodes("57", "site6",new_identifier="567")
        found_tensor = self.ttn.tensors["567"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("567", [])
        self.check_parent("567", "site1")

        # Contraction order ((6,7),5)
        self.setUp()
        self.assertRaises(ptn.NoConnectionException, self.ttn.contract_nodes, "site6", "site7")

    def test_contract_node_with_three_leaf_children(self):
        """
        Contract a node with three leaf children.
            |
            8----11
           / \\
          9   10
        """
        # Contraction order (((8,9),10),11)
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site9"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site10"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site11"], axes=(1,0))
        self.ttn.contract_nodes("site8", "site9",new_identifier="89")
        self.ttn.contract_nodes("89", "site10",new_identifier="8910")
        self.ttn.contract_nodes("8910", "site11",new_identifier="891011")
        found_tensor = self.ttn.tensors["891011"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("891011", [])
        self.check_parent("891011", "site3")

        # Contraction order (((8,10),9),11)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site9"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site10"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site11"], axes=(1,0))
        self.ttn.contract_nodes("site8", "site10",new_identifier="810")
        self.ttn.contract_nodes("810", "site9",new_identifier="8109")
        self.ttn.contract_nodes("8109", "site11",new_identifier="810911")
        found_tensor = self.ttn.tensors["810911"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("810911", [])
        self.check_parent("810911", "site3")

        # Contraction order (((8,9),11),10)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site9"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site10"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site11"], axes=(1,0))
        self.ttn.contract_nodes("site8", "site9",new_identifier="89")
        self.ttn.contract_nodes("89", "site11",new_identifier="8911")
        self.ttn.contract_nodes("8911", "site10",new_identifier="891110")
        found_tensor = self.ttn.tensors["891110"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("891110", [])
        self.check_parent("891110", "site3")

        # Contraction order (((11,8),9),10)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site9"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site10"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site11"], axes=(1,0))
        self.ttn.contract_nodes("site11", "site8",new_identifier="118")
        self.ttn.contract_nodes("118", "site9",new_identifier="1189")
        self.ttn.contract_nodes("1189", "site10",new_identifier="118910")
        found_tensor = self.ttn.tensors["118910"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("118910", [])
        self.check_parent("118910", "site3")

    def test_contract_node_with_two_non_leaf_children(self):
        """
        Contract a node with two non-leaf children.
            |
            1
            / \\
            4   5---
                /
        """
        # Contraction order ((1,4),5)
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site4"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site5"], axes=(0,2))
        self.ttn.contract_nodes("site1", "site4",new_identifier="14")
        self.ttn.contract_nodes("14", "site5",new_identifier="145")
        found_tensor = self.ttn.tensors["145"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("145", ["site6", "site7"])
        self.check_parent("145", "site0")

        # Contraction order ((1,5),4)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site4"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site5"], axes=(0,2))
        self.ttn.contract_nodes("site1", "site5",new_identifier="15")
        self.ttn.contract_nodes("15", "site4",new_identifier="145")
        found_tensor = self.ttn.tensors["145"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("145", ["site6", "site7"])
        self.check_parent("145", "site0")

        # Contraction order ((4,1),5)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site4"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site5"], axes=(0,2))
        self.ttn.contract_nodes("site4", "site1",new_identifier="14")
        self.ttn.contract_nodes("14", "site5",new_identifier="145")
        found_tensor = self.ttn.tensors["145"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("145", ["site6", "site7"])
        self.check_parent("145", "site0")

        # Contraction order ((5,1),4)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site4"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site5"], axes=(0,2))
        self.ttn.contract_nodes("site5", "site1",new_identifier="15")
        self.ttn.contract_nodes("15", "site4",new_identifier="145")
        found_tensor = self.ttn.tensors["145"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("145", ["site6", "site7"])
        self.check_parent("145", "site0")

        # Contraction order ((4,5),1)
        self.setUp()
        self.assertRaises(ptn.NoConnectionException, self.ttn.contract_nodes, "site4", "site5")

    def test_contract_root_with_children(self):
        """
        Contract the root node with its children.
            
            0----3
           / \\
          1   2
        """
        # Contraction order (((0,1),2),3)
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site2"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site3"], axes=(0,1))
        self.ttn.contract_nodes("site0", "site1",new_identifier="01")
        self.ttn.contract_nodes("01", "site2",new_identifier="012")
        self.ttn.contract_nodes("012", "site3",new_identifier="0123")
        found_tensor = self.ttn.tensors["0123"].transpose((1,0,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("0123", ["site4", "site5", "site8"])
        self.check_root("0123")

        # Contraction order (((0,2),1),3)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site2"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site3"], axes=(0,1))
        self.ttn.contract_nodes("site0", "site2",new_identifier="02")
        self.ttn.contract_nodes("02", "site1",new_identifier="021")
        self.ttn.contract_nodes("021", "site3",new_identifier="0213")
        found_tensor = self.ttn.tensors["0213"].transpose((1,0,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("0213", ["site4", "site5", "site8"])
        self.check_root("0213")

        # Contraction order (((0,3),1),2)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site2"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site3"], axes=(0,1))
        self.ttn.contract_nodes("site0", "site3",new_identifier="03")
        self.ttn.contract_nodes("03", "site1",new_identifier="031")
        self.ttn.contract_nodes("031", "site2",new_identifier="0312")
        found_tensor = self.ttn.tensors["0312"].transpose((2,1,0))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("0312", ["site4", "site5", "site8"])
        self.check_root("0312")

        # Contraction order (((1,0),3),2)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site2"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site3"], axes=(0,1))
        self.ttn.contract_nodes("site1", "site0",new_identifier="10")
        self.ttn.contract_nodes("10", "site3",new_identifier="103")
        self.ttn.contract_nodes("103", "site2",new_identifier="1032")
        found_tensor = self.ttn.tensors["1032"].transpose((1,0,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("1032", ["site4", "site5", "site8"])
        self.check_root("1032")

    def test_contraction_contract_all(self):
        """
        Contract all nodes.
        """
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site2"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site3"], axes=(0,1))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site4"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site5"], axes=(0,2))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site6"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site7"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site8"], axes=(0,2))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site9"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site10"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site11"], axes=(0,0))

        temp_ttn = deepcopy(self.ttn)
        temp_ttn.contract_nodes("site0", "site1",new_identifier="01")
        temp_ttn.contract_nodes("01", "site2",new_identifier="012")
        temp_ttn.contract_nodes("012", "site3",new_identifier="0123")
        temp_ttn.contract_nodes("0123", "site4",new_identifier="01234")
        temp_ttn.contract_nodes("01234", "site5",new_identifier="012345")
        temp_ttn.contract_nodes("012345", "site6",new_identifier="0123456")
        temp_ttn.contract_nodes("0123456", "site7",new_identifier="01234567")
        temp_ttn.contract_nodes("01234567", "site8",new_identifier="012345678")
        temp_ttn.contract_nodes("012345678", "site9",new_identifier="0123456789")
        temp_ttn.contract_nodes("0123456789", "site10",new_identifier="012345678910")
        temp_ttn.contract_nodes("012345678910", "site11",new_identifier="01234567891011")
        found_tensor = temp_ttn.tensors["01234567891011"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.assertTrue(temp_ttn.root_id == "01234567891011")
        self.assertTrue(len(temp_ttn.nodes) == 1)
        self.assertTrue(len(temp_ttn.tensors) == 1)
        self.assertTrue(temp_ttn.nodes["01234567891011"].is_root())
        self.assertTrue(temp_ttn.nodes["01234567891011"].is_leaf())

        temp_ttn = deepcopy(self.ttn)
        temp_ttn.contract_nodes("site3", "site0",new_identifier="03")
        temp_ttn.contract_nodes("03", "site2",new_identifier="032")
        temp_ttn.contract_nodes("032", "site1",new_identifier="0321")
        temp_ttn.contract_nodes("0321", "site4",new_identifier="03214")
        temp_ttn.contract_nodes("03214", "site5",new_identifier="032145")
        temp_ttn.contract_nodes("032145", "site6",new_identifier="0321456")
        temp_ttn.contract_nodes("0321456", "site7",new_identifier="03214567")
        temp_ttn.contract_nodes("03214567", "site8",new_identifier="032145678")
        temp_ttn.contract_nodes("032145678", "site9",new_identifier="0321456789")
        temp_ttn.contract_nodes("0321456789", "site10",new_identifier="032145678910")
        temp_ttn.contract_nodes("032145678910", "site11",new_identifier="03214567891011")
        found_tensor = temp_ttn.tensors["03214567891011"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.assertTrue(temp_ttn.root_id == "03214567891011")
        self.assertTrue(len(temp_ttn.nodes) == 1)
        self.assertTrue(len(temp_ttn.tensors) == 1)
        self.assertTrue(temp_ttn.nodes["03214567891011"].is_root())
        self.assertTrue(temp_ttn.nodes["03214567891011"].is_leaf())

        temp_ttn = deepcopy(self.ttn)
        temp_ttn.contract_nodes("site5", "site6",new_identifier="56")
        temp_ttn.contract_nodes("56", "site7",new_identifier="567")
        temp_ttn.contract_nodes("567", "site1",new_identifier="5671")
        temp_ttn.contract_nodes("5671", "site4",new_identifier="56714")
        temp_ttn.contract_nodes("56714", "site0",new_identifier="567140")
        temp_ttn.contract_nodes("567140", "site2",new_identifier="5671402")
        temp_ttn.contract_nodes("5671402", "site3",new_identifier="56714023")
        temp_ttn.contract_nodes("56714023", "site8",new_identifier="567140238")
        temp_ttn.contract_nodes("567140238", "site9",new_identifier="5671402389")
        temp_ttn.contract_nodes("5671402389", "site10",new_identifier="567140238910")
        temp_ttn.contract_nodes("567140238910", "site11",new_identifier="56714023891011")
        found_tensor = temp_ttn.tensors["56714023891011"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.assertTrue(temp_ttn.root_id == "56714023891011")
        self.assertTrue(len(temp_ttn.nodes) == 1)
        self.assertTrue(len(temp_ttn.tensors) == 1)
        self.assertTrue(temp_ttn.nodes["56714023891011"].is_root())
        self.assertTrue(temp_ttn.nodes["56714023891011"].is_leaf())

class TestTensorContractionOpenLegs(unittest.TestCase):
    def setUp(self) -> None:
        shapes = [(5,6,7,2),(4,5,7,2),(6,2),(4,5,2),(5,2),
                  (2,3,4,2),(2,2),(3,2),(2,3,2,4,5),(2,3),
                  (2,2),(5,2)]
        self.tensors = {"site"+str(i): crandn(shape)
                        for i, shape in enumerate(shapes)}

        self.ttn = ptn.TreeTensorNetwork()
        self.ttn.add_root(ptn.Node(identifier="site0"), self.tensors["site0"])
        self.ttn.add_child_to_parent(ptn.Node(identifier="site1"),
                                     self.tensors["site1"], 2, "site0", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site2"),
                                     self.tensors["site2"], 0, "site0", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site3"),
                                     self.tensors["site3"], 1, "site0", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site4"),
                                     self.tensors["site4"], 0, "site1", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site5"),
                                     self.tensors["site5"], 2, "site1", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site6"),
                                     self.tensors["site6"], 0, "site5", 1)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site7"),
                                        self.tensors["site7"], 0, "site5", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site8"),
                                     self.tensors["site8"], 3, "site3", 1)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site9"),
                                     self.tensors["site9"], 1, "site8", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site10"),
                                        self.tensors["site10"], 0, "site8", 2)
        self.ttn.add_child_to_parent(ptn.Node(identifier="site11"),
                                     self.tensors["site11"], 0, "site8", 4)

    def check_children(self, node_id: str, child_ids: List[str]):
        node = self.ttn.nodes[node_id]
        for child_id in child_ids:
            self.assertTrue(child_id in self.ttn.nodes)
            self.assertTrue(child_id in self.ttn.tensors)
            child_node = self.ttn.nodes[child_id]
            self.assertTrue(child_node.is_child_of(node_id))
            self.assertTrue(node.is_parent_of(child_id))
        if len(child_ids) == 0:
            self.assertTrue(node.is_leaf())

    def check_parent(self, node_id: str, parent_id: str):
        node = self.ttn.nodes[node_id]
        self.assertTrue(parent_id in self.ttn.nodes)
        self.assertTrue(parent_id in self.ttn.tensors)
        parent_node = self.ttn.nodes[parent_id]
        self.assertTrue(node.is_child_of(parent_id))
        self.assertTrue(parent_node.is_parent_of(node_id))

    def check_root(self, node_id: str):
        node = self.ttn.nodes[node_id]
        self.assertTrue(node.is_root())
        self.assertEqual(node_id, self.ttn.root_id)

    def test_contraction_root_w_child_with_mult_children(self):
        """
        Contract a root node with one of its children that has multiple children.
        """
        self.ttn.contract_nodes("site0", "site1",new_identifier="01")
        found_tensor = self.ttn.tensors["01"]
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.transpose(correct_tensor, (1,0,4,3,2,5))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("01", ["site2", "site3", "site4", "site5"])
        self.check_root("01")

    def test_contraction_root_w_child_with_no_children(self):
        """
        Contract the root node with one of its children that has no children.
        """
        self.ttn.contract_nodes("site0", "site2",new_identifier="02")
        found_tensor = self.ttn.tensors["02"]
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site2"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,2,3))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("02", ["site1", "site3"])
        self.check_root("02")

    def test_contraction_root_w_child_with_one_child(self):
        """
        Contract the root node with one of its children that has one child.
        """
        self.ttn.contract_nodes("site0", "site3",new_identifier="03")
        found_tensor = self.ttn.tensors["03"]
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site3"], axes=(0,1))
        correct_tensor = np.transpose(correct_tensor, (1,0,3,2,4))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("03", ["site1", "site2", "site8"])
        self.check_root("03")

    def test_contraction_node_w_child_with_parent_and_other_child_with_children(self):
        """
        Contract a node with one of its children. The node itself has a parent
         and another child, which also has children.
        """
        self.ttn.contract_nodes("site1", "site4",new_identifier="14")
        found_tensor = self.ttn.tensors["14"]
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site4"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,2,3))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("14", ["site5"])
        self.check_parent("14", "site0")

    def test_contraction_node_w_child_with_parent_and_other_leaf_child(self):
        """
        Contract a node with one of its children. The node itself has a parent
         and another child, which is a leaf.
        """
        self.ttn.contract_nodes("site1", "site5",new_identifier="15")
        found_tensor = self.ttn.tensors["15"]
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site5"], axes=(0,2))
        correct_tensor = np.transpose(correct_tensor, (1,0,3,4,2,5))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("15", ["site4", "site6", "site7"])
        self.check_parent("15", "site0")

    def test_contraction_node_w_child_with_parent_and_no_other_child(self):
        """
        Contract a node with one of its children. The node itself has a parent
         and no other children.
        """
        self.ttn.contract_nodes("site3", "site8",new_identifier="38")
        found_tensor = self.ttn.tensors["38"]
        correct_tensor = np.tensordot(self.tensors["site3"], self.tensors["site8"], axes=(0,3))
        correct_tensor = np.transpose(correct_tensor, (0,3,2,5,1,4))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("38", ["site9", "site10", "site11"])
        self.check_parent("38", "site0")

    def test_contract_leaf_with_parent_and_one_other_leaf(self):
        """
        Contract a node with a child that is a leaf. The node has one other
         child that is also a leaf.
        """
        self.ttn.contract_nodes("site5", "site6",new_identifier="56")
        found_tensor = self.ttn.tensors["56"]
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site6"], axes=(0,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,2,3))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("56", ["site7"])
        self.check_parent("56", "site1")

        self.setUp()
        self.ttn.contract_nodes("site5", "site7",new_identifier="57")
        found_tensor = self.ttn.tensors["57"]
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site7"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,2,3))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("57", ["site6"])
        self.check_parent("57", "site1")

    def test_contract_leaf_with_parent_and_two_other_leafs(self):
        """
        Contract a node with a child that is a leaf. The node has two other
         children that are also leaves.
        """
        self.ttn.contract_nodes("site8", "site9",new_identifier="89")
        found_tensor = self.ttn.tensors["89"]
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site9"], axes=(1,1))
        correct_tensor = np.transpose(correct_tensor, (2,0,3,1,4))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("89", ["site10", "site11"])
        self.check_parent("89", "site3")

        self.setUp()
        self.ttn.contract_nodes("site8", "site10",new_identifier="810")
        found_tensor = self.ttn.tensors["810"]
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site10"], axes=(0,0))
        correct_tensor = np.transpose(correct_tensor, (2,0,3,1,4))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("810", ["site9", "site11"])
        self.check_parent("810", "site3")

        self.setUp()
        self.ttn.contract_nodes("site8", "site11",new_identifier="811")
        found_tensor = self.ttn.tensors["811"]
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site11"], axes=(4,0))
        correct_tensor = np.transpose(correct_tensor, (3,1,0,2,4))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("811", ["site9", "site10"])
        self.check_parent("811", "site3")

    def test_contraction_root_w_child_with_mult_children_other_way(self):
        """
        Contract a root node with one of its children that has multiple children.
        """
        self.ttn.contract_nodes("site1", "site0",new_identifier="01")
        found_tensor = self.ttn.tensors["01"]
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.transpose(correct_tensor, (4,3,1,0,5,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("01", ["site2", "site3", "site4", "site5"])
        self.check_root("01")

    def test_contraction_root_w_child_with_no_children_other_way(self):
        """
        Contract the root node with one of its children that has no children.
        """
        self.ttn.contract_nodes("site2", "site0",new_identifier="02")
        found_tensor = self.ttn.tensors["02"]
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site2"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,3,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("02", ["site1", "site3"])
        self.check_root("02")

    def test_contraction_root_w_child_with_one_child_other_way(self):
        """
        Contract the root node with one of its children that has one child.
        """
        self.ttn.contract_nodes("site3", "site0",new_identifier="03")
        found_tensor = self.ttn.tensors["03"]
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site3"], axes=(0,1))
        correct_tensor = np.transpose(correct_tensor, (3,1,0,4,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("03", ["site1", "site2", "site8"])
        self.check_root("03")

    def test_contraction_node_w_child_with_parent_and_other_child_with_children_other_way(self):
        """
        Contract a node with one of its children. The node itself has a parent
         and another child, which also has children.
        """
        self.ttn.contract_nodes("site4", "site1",new_identifier="14")
        found_tensor = self.ttn.tensors["14"]
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site4"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,3,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("14", ["site5"])
        self.check_parent("14", "site0")

    def test_contraction_node_w_child_with_parent_and_other_leaf_child_other_way(self):
        """
        Contract a node with one of its children. The node itself has a parent
         and another child, which is a leaf.
        """
        self.ttn.contract_nodes("site5", "site1",new_identifier="15")
        found_tensor = self.ttn.tensors["15"]
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site5"], axes=(0,2))
        correct_tensor = np.transpose(correct_tensor, (1,3,4,0,5,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("15", ["site4", "site6", "site7"])
        self.check_parent("15", "site0")

    def test_contraction_node_w_child_with_parent_and_no_other_child_other_way(self):
        """
        Contract a node with one of its children. The node itself has a parent
         and no other children.
        """
        self.ttn.contract_nodes("site8", "site3",new_identifier="38")
        found_tensor = self.ttn.tensors["38"]
        correct_tensor = np.tensordot(self.tensors["site3"], self.tensors["site8"], axes=(0,3))
        correct_tensor = np.transpose(correct_tensor, (0,3,2,5,4,1))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("38", ["site9", "site10", "site11"])
        self.check_parent("38", "site0")

    def test_contract_leaf_with_parent_and_one_other_leaf_other_way(self):
        """
        Contract a node with a child that is a leaf. The node has one other
         child that is also a leaf.
        """
        self.ttn.contract_nodes("site6", "site5",new_identifier="56")
        found_tensor = self.ttn.tensors["56"]
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site6"], axes=(0,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,3,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("56", ["site7"])
        self.check_parent("56", "site1")

        self.setUp()
        self.ttn.contract_nodes("site7", "site5",new_identifier="57")
        found_tensor = self.ttn.tensors["57"]
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site7"], axes=(1,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,3,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("57", ["site6"])
        self.check_parent("57", "site1")

    def test_contract_leaf_with_parent_and_two_other_leafs_other_way(self):
        """
        Contract a node with a child that is a leaf. The node has two other
         children that are also leaves.
        """
        self.ttn.contract_nodes("site9", "site8",new_identifier="89")
        found_tensor = self.ttn.tensors["89"]
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site9"], axes=(1,1))
        correct_tensor = np.transpose(correct_tensor, (2,0,3,4,1))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("89", ["site10", "site11"])
        self.check_parent("89", "site3")

        self.setUp()
        self.ttn.contract_nodes("site10", "site8",new_identifier="810")
        found_tensor = self.ttn.tensors["810"]
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site10"], axes=(0,0))
        correct_tensor = np.transpose(correct_tensor, (2,0,3,4,1))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("810", ["site9", "site11"])
        self.check_parent("810", "site3")

        self.setUp()
        self.ttn.contract_nodes("site11", "site8",new_identifier="811")
        found_tensor = self.ttn.tensors["811"]
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site11"], axes=(4,0))
        correct_tensor = np.transpose(correct_tensor, (3,1,0,4,2))
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("811", ["site9", "site10"])
        self.check_parent("811", "site3")

    def test_contract_node_with_two_leaf_children(self):
        """
        Contract a node with two leaf children.
            |
            5
           / \\
          6   7
        """
        # Contraction order ((5,6),7)
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site6"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site7"], axes=(0,0))
        self.ttn.contract_nodes("site5", "site6",new_identifier="56")
        self.ttn.contract_nodes("56", "site7",new_identifier="567")
        found_tensor = self.ttn.tensors["567"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("567", [])
        self.check_parent("567", "site1")

        # Contraction order ((5,7),6)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site6"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site7"], axes=(0,0))
        correct_tensor = np.transpose(correct_tensor, (0,1,3,2))
        self.ttn.contract_nodes("site5", "site7",new_identifier="57")
        self.ttn.contract_nodes("57", "site6",new_identifier="567")
        found_tensor = self.ttn.tensors["567"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("567", [])
        self.check_parent("567", "site1")

        # Contraction order ((6,5),7)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site6"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site7"], axes=(0,0))
        correct_tensor = np.transpose(correct_tensor, (0,2,1,3))
        self.ttn.contract_nodes("site6", "site5",new_identifier="56")
        self.ttn.contract_nodes("56", "site7",new_identifier="567")
        found_tensor = self.ttn.tensors["567"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("567", [])
        self.check_parent("567", "site1")

        # Contraction order ((7,5),6)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site5"], self.tensors["site6"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site7"], axes=(0,0))
        correct_tensor = np.transpose(correct_tensor, (0,3,1,2))
        self.ttn.contract_nodes("site7", "site5",new_identifier="57")
        self.ttn.contract_nodes("57", "site6",new_identifier="567")
        found_tensor = self.ttn.tensors["567"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("567", [])
        self.check_parent("567", "site1")

        # Contraction order ((6,7),5)
        self.setUp()
        self.assertRaises(ptn.NoConnectionException, self.ttn.contract_nodes, "site6", "site7")

    def test_contract_node_with_three_leaf_children(self):
        """
        Contract a node with three leaf children.
            |
            8----11
           / \\
          9   10
        """
        # Contraction order (((8,9),10),11)
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site9"], axes=(1,1))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site10"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site11"], axes=(2,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,2,3,4))
        self.ttn.contract_nodes("site8", "site9",new_identifier="89")
        self.ttn.contract_nodes("89", "site10",new_identifier="8910")
        self.ttn.contract_nodes("8910", "site11",new_identifier="891011")
        found_tensor = self.ttn.tensors["891011"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("891011", [])
        self.check_parent("891011", "site3")

        # Contraction order (((8,10),9),11)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site9"], axes=(1,1))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site10"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site11"], axes=(2,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,3,2,4))
        self.ttn.contract_nodes("site8", "site10",new_identifier="810")
        self.ttn.contract_nodes("810", "site9",new_identifier="8109")
        self.ttn.contract_nodes("8109", "site11",new_identifier="810911")
        found_tensor = self.ttn.tensors["810911"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("810911", [])
        self.check_parent("810911", "site3")

        # Contraction order (((8,9),11),10)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site9"], axes=(1,1))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site10"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site11"], axes=(2,0))
        correct_tensor = np.transpose(correct_tensor, (1,0,2,4,3))
        self.ttn.contract_nodes("site8", "site9",new_identifier="89")
        self.ttn.contract_nodes("89", "site11",new_identifier="8911")
        self.ttn.contract_nodes("8911", "site10",new_identifier="891110")
        found_tensor = self.ttn.tensors["891110"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("891110", [])
        self.check_parent("891110", "site3")

        # Contraction order (((11,8),9),10)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site8"], self.tensors["site9"], axes=(1,1))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site10"], axes=(0,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site11"], axes=(2,0))
        correct_tensor = np.transpose(correct_tensor, (1,4,0,2,3))
        self.ttn.contract_nodes("site11", "site8",new_identifier="118")
        self.ttn.contract_nodes("118", "site9",new_identifier="1189")
        self.ttn.contract_nodes("1189", "site10",new_identifier="118910")
        found_tensor = self.ttn.tensors["118910"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("118910", [])
        self.check_parent("118910", "site3")

    def test_contract_node_with_two_non_leaf_children(self):
        """
        Contract a node with two non-leaf children.
            |
            1
            / \\
            4   5---
                /
        """
        # Contraction order ((1,4),5)
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site4"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site5"], axes=(0,2))
        correct_tensor = np.transpose(correct_tensor, (0,3,4,1,2,5))
        self.ttn.contract_nodes("site1", "site4",new_identifier="14")
        self.ttn.contract_nodes("14", "site5",new_identifier="145")
        found_tensor = self.ttn.tensors["145"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("145", ["site6", "site7"])
        self.check_parent("145", "site0")

        # Contraction order ((1,5),4)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site4"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site5"], axes=(0,2))
        correct_tensor = np.transpose(correct_tensor, (0,3,4,1,5,2))
        self.ttn.contract_nodes("site1", "site5",new_identifier="15")
        self.ttn.contract_nodes("15", "site4",new_identifier="145")
        found_tensor = self.ttn.tensors["145"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("145", ["site6", "site7"])
        self.check_parent("145", "site0")

        # Contraction order ((4,1),5)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site4"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site5"], axes=(0,2))
        correct_tensor = np.transpose(correct_tensor, (0,3,4,2,1,5))
        self.ttn.contract_nodes("site4", "site1",new_identifier="14")
        self.ttn.contract_nodes("14", "site5",new_identifier="145")
        found_tensor = self.ttn.tensors["145"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("145", ["site6", "site7"])
        self.check_parent("145", "site0")

        # Contraction order ((5,1),4)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site1"], self.tensors["site4"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site5"], axes=(0,2))
        correct_tensor = np.transpose(correct_tensor, (0,3,4,5,1,2))
        self.ttn.contract_nodes("site5", "site1",new_identifier="15")
        self.ttn.contract_nodes("15", "site4",new_identifier="145")
        found_tensor = self.ttn.tensors["145"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("145", ["site6", "site7"])
        self.check_parent("145", "site0")

        # Contraction order ((4,5),1)
        self.setUp()
        self.assertRaises(ptn.NoConnectionException, self.ttn.contract_nodes, "site4", "site5")

    def test_contract_root_with_children(self):
        """
        Contract the root node with its children.
            
            0----3
           / \\
          1   2
        """
        # Contraction order (((0,1),2),3)
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site2"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site3"], axes=(0,1))
        correct_tensor = np.transpose(correct_tensor, (2,1,5,0,3,4,6))
        self.ttn.contract_nodes("site0", "site1",new_identifier="01")
        self.ttn.contract_nodes("01", "site2",new_identifier="012")
        self.ttn.contract_nodes("012", "site3",new_identifier="0123")
        found_tensor = self.ttn.tensors["0123"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("0123", ["site4", "site5", "site8"])
        self.check_root("0123")

        # Contraction order (((0,2),1),3)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site2"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site3"], axes=(0,1))
        correct_tensor = np.transpose(correct_tensor, (2,1,5,0,4,3,6))
        self.ttn.contract_nodes("site0", "site2",new_identifier="02")
        self.ttn.contract_nodes("02", "site1",new_identifier="021")
        self.ttn.contract_nodes("021", "site3",new_identifier="0213")
        found_tensor = self.ttn.tensors["0213"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("0213", ["site4", "site5", "site8"])
        self.check_root("0213")

        # Contraction order (((0,3),1),2)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site2"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site3"], axes=(0,1))
        correct_tensor = np.transpose(correct_tensor, (5,2,1,0,6,3,4))
        self.ttn.contract_nodes("site0", "site3",new_identifier="03")
        self.ttn.contract_nodes("03", "site1",new_identifier="031")
        self.ttn.contract_nodes("031", "site2",new_identifier="0312")
        found_tensor = self.ttn.tensors["0312"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("0312", ["site4", "site5", "site8"])
        self.check_root("0312")

        # Contraction order (((1,0),3),2)
        self.setUp()
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site2"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site3"], axes=(0,1))
        correct_tensor = np.transpose(correct_tensor, (2,1,5,3,0,6,4))
        self.ttn.contract_nodes("site1", "site0",new_identifier="10")
        self.ttn.contract_nodes("10", "site3",new_identifier="103")
        self.ttn.contract_nodes("103", "site2",new_identifier="1032")
        found_tensor = self.ttn.tensors["1032"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.check_children("1032", ["site4", "site5", "site8"])
        self.check_root("1032")

    def test_contraction_contract_all(self):
        """
        Contract all nodes.
        """
        correct_tensor = np.tensordot(self.tensors["site0"], self.tensors["site1"], axes=(2,2))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site2"], axes=(1,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site3"], axes=(0,1))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site4"], axes=(2,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site5"], axes=(1,2))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site6"], axes=(6,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site7"], axes=(6,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site8"], axes=(3,3))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site9"], axes=(9,1))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site10"], axes=(8,0))
        correct_tensor = np.tensordot(correct_tensor, self.tensors["site11"], axes=(9,0))

        temp_ttn = deepcopy(self.ttn)
        temp_ttn.contract_nodes("site0", "site1",new_identifier="01")
        temp_ttn.contract_nodes("01", "site2",new_identifier="012")
        temp_ttn.contract_nodes("012", "site3",new_identifier="0123")
        temp_ttn.contract_nodes("0123", "site4",new_identifier="01234")
        temp_ttn.contract_nodes("01234", "site5",new_identifier="012345")
        temp_ttn.contract_nodes("012345", "site6",new_identifier="0123456")
        temp_ttn.contract_nodes("0123456", "site7",new_identifier="01234567")
        temp_ttn.contract_nodes("01234567", "site8",new_identifier="012345678")
        temp_ttn.contract_nodes("012345678", "site9",new_identifier="0123456789")
        temp_ttn.contract_nodes("0123456789", "site10",new_identifier="012345678910")
        temp_ttn.contract_nodes("012345678910", "site11",new_identifier="01234567891011")
        found_tensor = temp_ttn.tensors["01234567891011"]
        self.assertTrue(np.allclose(correct_tensor, found_tensor))
        self.assertTrue(temp_ttn.root_id == "01234567891011")
        self.assertTrue(len(temp_ttn.nodes) == 1)
        self.assertTrue(len(temp_ttn.tensors) == 1)
        self.assertTrue(temp_ttn.nodes["01234567891011"].is_root())
        self.assertTrue(temp_ttn.nodes["01234567891011"].is_leaf())

        temp_ttn = deepcopy(self.ttn)
        temp_ttn.contract_nodes("site3", "site0",new_identifier="03")
        temp_ttn.contract_nodes("03", "site2",new_identifier="032")
        temp_ttn.contract_nodes("032", "site1",new_identifier="0321")
        temp_ttn.contract_nodes("0321", "site4",new_identifier="03214")
        temp_ttn.contract_nodes("03214", "site5",new_identifier="032145")
        temp_ttn.contract_nodes("032145", "site6",new_identifier="0321456")
        temp_ttn.contract_nodes("0321456", "site7",new_identifier="03214567")
        temp_ttn.contract_nodes("03214567", "site8",new_identifier="032145678")
        temp_ttn.contract_nodes("032145678", "site9",new_identifier="0321456789")
        temp_ttn.contract_nodes("0321456789", "site10",new_identifier="032145678910")
        temp_ttn.contract_nodes("032145678910", "site11",new_identifier="03214567891011")
        found_tensor = temp_ttn.tensors["03214567891011"]
        self.assertTrue(np.allclose(correct_tensor.transpose((3,0,2,1,4,5,6,7,8,9,10,11)),
                                    found_tensor))
        self.assertTrue(temp_ttn.root_id == "03214567891011")
        self.assertTrue(len(temp_ttn.nodes) == 1)
        self.assertTrue(len(temp_ttn.tensors) == 1)
        self.assertTrue(temp_ttn.nodes["03214567891011"].is_root())
        self.assertTrue(temp_ttn.nodes["03214567891011"].is_leaf())

        temp_ttn = deepcopy(self.ttn)
        temp_ttn.contract_nodes("site5", "site6",new_identifier="56")
        temp_ttn.contract_nodes("56", "site7",new_identifier="567")
        temp_ttn.contract_nodes("567", "site1",new_identifier="5671")
        temp_ttn.contract_nodes("5671", "site4",new_identifier="56714")
        temp_ttn.contract_nodes("56714", "site0",new_identifier="567140")
        temp_ttn.contract_nodes("567140", "site2",new_identifier="5671402")
        temp_ttn.contract_nodes("5671402", "site3",new_identifier="56714023")
        temp_ttn.contract_nodes("56714023", "site8",new_identifier="567140238")
        temp_ttn.contract_nodes("567140238", "site9",new_identifier="5671402389")
        temp_ttn.contract_nodes("5671402389", "site10",new_identifier="567140238910")
        temp_ttn.contract_nodes("567140238910", "site11",new_identifier="56714023891011")
        found_tensor = temp_ttn.tensors["56714023891011"]
        self.assertTrue(np.allclose(correct_tensor.transpose((5,6,7,1,4,0,2,3,8,9,10,11)),
                                    found_tensor))
        self.assertTrue(temp_ttn.root_id == "56714023891011")
        self.assertTrue(len(temp_ttn.nodes) == 1)
        self.assertTrue(len(temp_ttn.tensors) == 1)
        self.assertTrue(temp_ttn.nodes["56714023891011"].is_root())
        self.assertTrue(temp_ttn.nodes["56714023891011"].is_leaf())

if __name__ == "__main__":
    unittest.main()
