import unittest

import numpy as np

import pytreenet as ptn

class TestTreeContraction(unittest.TestCase):

    def setUp(self):
        self.tree_tensor_network = ptn.TreeTensorNetwork()

        # Constructing a tree for tests
        self.node1 = ptn.random_tensor_node((2,2), identifier="node1")
        self.node2 = ptn.random_tensor_node((2,3,3), identifier="node2")
        self.node3 = ptn.random_tensor_node((3,), identifier="node3")
        self.node4 = ptn.random_tensor_node((3,4), identifier="node4")
        self.node5 = ptn.random_tensor_node((2,3,3), identifier="node5")
        self.node6 = ptn.random_tensor_node((3,4,4), identifier="node6")
        self.node7 = ptn.random_tensor_node((4,2,2), identifier="node7")

        self.tree_tensor_network.add_root(self.node1)
        self.tree_tensor_network.add_child_to_parent(self.node2, 0, "node1", 0)
        self.tree_tensor_network.add_child_to_parent(self.node3, 0, "node2", 1)
        self.tree_tensor_network.add_child_to_parent(self.node4, 0, "node2", 2)
        self.tree_tensor_network.add_child_to_parent(self.node5, 0, "node1", 1)
        self.tree_tensor_network.add_child_to_parent(self.node6, 0, "node5", 2)
        self.tree_tensor_network.add_child_to_parent(self.node7, 0, "node6", 1)

    def test_contract_nodes_in_tree(self):
        self.assertRaises(ptn.NoConnectionException ,
                          ptn.contract_nodes_in_tree, self.tree_tensor_network,
                          "node1", "node3")

        nodes = self.tree_tensor_network.nodes

        ptn.contract_nodes_in_tree(self.tree_tensor_network, "node2", "node4")
        idcontr24 = "node2_contr_node4"
        self.assertTrue(idcontr24 in nodes)
        self.assertEqual(len(nodes), 6)
        self.assertFalse("node2" in nodes)
        self.assertFalse("node4" in nodes)

        contracted_node = nodes[idcontr24]
        self.assertEqual(contracted_node.open_legs, [2])
        self.assertEqual(contracted_node.parent_leg, ["node1", 0])
        self.assertEqual(contracted_node.children_legs, {"node3":1})

        self.assertFalse(self.node2 == self.node4)
        self.assertEqual(nodes["node3"].parent_leg, [idcontr24,0])
        self.assertEqual(nodes["node1"].children_legs, {idcontr24: 0, "node5": 1})

        ref_tensor = np.tensordot(self.node2.tensor, self.node4.tensor, axes=(2,0))
        self.assertTrue(np.allclose(contracted_node.tensor, ref_tensor))

        ptn.contract_nodes_in_tree(self.tree_tensor_network, "node1", idcontr24)
        idcontr124 = "node1_contr_node2_contr_node4"
        self.assertTrue(idcontr124 in nodes)
        self.assertEqual(self.tree_tensor_network.root_id, idcontr124)
        self.assertTrue(nodes[idcontr124].is_root())
        self.assertEqual(nodes[idcontr124].tensor.shape, (2,3,4))

        ptn.contract_nodes_in_tree(self.tree_tensor_network, idcontr124, "node3")
        idcontr1243 = idcontr124 + "_contr_node3"
        self.assertTrue(idcontr1243 in nodes)
        self.assertEqual(nodes[idcontr1243].children_legs, {"node5": 0})
        self.assertEqual(nodes[idcontr1243].open_legs, [1])
        self.assertTrue(nodes[idcontr1243].is_root())

        ptn.contract_nodes_in_tree(self.tree_tensor_network,
                          self.tree_tensor_network.root_id, "node5")
        ptn.contract_nodes_in_tree(self.tree_tensor_network,
                          self.tree_tensor_network.root_id, "node6")
        ptn.contract_nodes_in_tree(self.tree_tensor_network,
                          self.tree_tensor_network.root_id, "node7")

        id_all_contracted = self.tree_tensor_network.root_id
        self.assertEqual(len(nodes),1)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[id_all_contracted].children_legs, dict())
        self.assertEqual(nodes[id_all_contracted].open_legs, [0,1,2,3,4])
        
    def test_completely_contract_tree(self):
        result_ttn = ptn.completely_contract_tree(self.tree_tensor_network, to_copy=True)
        result_nodes = result_ttn.nodes
        result_node = result_nodes[result_ttn.root_id]
        result_tensor = result_node.tensor
        
        # The contraction order should not matter for the final result
        # Except for leg_order
        ptn.contract_nodes_in_tree(self.tree_tensor_network, "node2", "node1")
        ptn.contract_nodes_in_tree(self.tree_tensor_network, "node3", "node2_contr_node1")
        ptn.contract_nodes_in_tree(self.tree_tensor_network, "node4", "node3_contr_node2_contr_node1")
        ptn.contract_nodes_in_tree(self.tree_tensor_network, "node5", "node4_contr_node3_contr_node2_contr_node1")
        ptn.contract_nodes_in_tree(self.tree_tensor_network, "node6", "node5_contr_node4_contr_node3_contr_node2_contr_node1")
        ptn.contract_nodes_in_tree(self.tree_tensor_network, "node7", "node6_contr_node5_contr_node4_contr_node3_contr_node2_contr_node1"
                          , new_identifier="final_node")
        ref_nodes = self.tree_tensor_network.nodes
        ref_node = ref_nodes[self.tree_tensor_network.root_id]
        ref_tensor = ref_node.tensor
        ref_tensor = np.transpose(ref_tensor, axes=(3,4,2,0,1))
        
        self.assertEqual(len(result_nodes), 1)
        self.assertTrue(np.allclose(result_tensor, ref_tensor))
        self.assertEqual(result_node.open_legs, ref_node.open_legs)
        self.assertEqual(result_node.children_legs, dict())
        self.assertTrue(result_node.is_root())
        
    

if __name__ == "__main__":
    unittest.main()