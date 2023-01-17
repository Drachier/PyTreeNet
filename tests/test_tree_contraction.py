import unittest

import numpy as np

import pytreenet as ptn

class TestTreeContraction(unittest.TestCase):

    def setUp(self):
        self.tree_tensor_network = ptn.TreeTensorNetwork()

        # Constructing a tree for tests
        self.node1 = ptn.random_tensor_node((2,3), identifier="node1")
        self.node2 = ptn.random_tensor_node((2,4,5), identifier="node2")
        self.node3 = ptn.random_tensor_node((4,), identifier="node3")
        self.node4 = ptn.random_tensor_node((5,8), identifier="node4")
        self.node5 = ptn.random_tensor_node((3,6,7), identifier="node5")
        self.node6 = ptn.random_tensor_node((7,10,9), identifier="node6")
        self.node7 = ptn.random_tensor_node((10,11,12), identifier="node7")

        self.tree_tensor_network.add_root(self.node1)
        self.tree_tensor_network.add_child_to_parent(self.node2, 0, "node1", 0)
        self.tree_tensor_network.add_child_to_parent(self.node3, 0, "node2", 1)
        self.tree_tensor_network.add_child_to_parent(self.node4, 0, "node2", 2)
        self.tree_tensor_network.add_child_to_parent(self.node5, 0, "node1", 1)
        self.tree_tensor_network.add_child_to_parent(self.node6, 0, "node5", 2)
        self.tree_tensor_network.add_child_to_parent(self.node7, 0, "node6", 1)

        # Generate a 2nd TTN with the same structure but different tensors
        self.tree_tensor_network2 = ptn.TreeTensorNetwork(original_tree=self.tree_tensor_network,
                                                          deep = True)
        
        for node_id in self.tree_tensor_network2.nodes:
            node = self.tree_tensor_network2.nodes[node_id]
            shape = node.tensor.shape
            node.tensor = ptn.crandn(shape)
        
        # Setting up a simple TTN
        self.simple_ttn1 = ptn.TreeTensorNetwork()
        
        node1 = ptn.random_tensor_node((2,3), identifier="node1")
        node2 = ptn.random_tensor_node((2,4,5), identifier="node2")
        node3 = ptn.random_tensor_node((3,6), identifier="node3")
        
        self.simple_ttn1.add_root(node1)
        self.simple_ttn1.add_child_to_parent(node2, 0, "node1", 0)
        self.simple_ttn1.add_child_to_parent(node3, 0, "node1", 1)
        
        # Generate a 2nd TTN with the same structure but different tensors
        self.simple_ttn2 = ptn.TreeTensorNetwork(original_tree=self.simple_ttn1,
                                                 deep = True)
        
        for node_id in self.simple_ttn2.nodes:
            node = self.simple_ttn2.nodes[node_id]
            shape = node.tensor.shape
            node.tensor = ptn.crandn(shape)
        

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
        self.assertEqual(nodes[idcontr124].tensor.shape, (3,4,8))

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
        
    def test_contract_two_ttn_simple(self):        
        # We get the correct result via manual contraction
        
        identifiers = ["node2", "node3"]
        axs = [[1,2], [1]]
        transfer_tensors = []
        
        for i, identifier in enumerate(identifiers):                        
            tensor1 = self.simple_ttn1.nodes[identifier].tensor
            tensor2 = self.simple_ttn2.nodes[identifier].tensor
            
            result_tensor = np.tensordot(tensor1, tensor2, axes=(axs[i], axs[i]))
            
            transfer_tensors.append(result_tensor)
            
        tensor1 = self.simple_ttn1["node1"].tensor
        tensor2 = self.simple_ttn2["node1"].tensor
            
        result = np.tensordot(tensor1, tensor2, axes=0)
        
        result = np.tensordot(result, transfer_tensors[0], axes=([0,2],[0,1]))
        correct_result = np.tensordot(result, transfer_tensors[1], axes=([0,1],[0,1]))
        
        found_result = ptn.contract_two_ttn(self.simple_ttn1, self.simple_ttn2)
        
        self.assertAlmostEqual(correct_result, found_result)
        
    def test_contract_two_ttn_complicated(self):
        
        found_result = ptn.contract_two_ttn(self.tree_tensor_network, 
                                            self.tree_tensor_network2)
        
        # Since both have the same structure, we can completely contract them and take the scalar product
        ttn1_contr = ptn.completely_contract_tree(self.tree_tensor_network,
                                                to_copy=True)
        ttn1_tensor = ttn1_contr.nodes[ttn1_contr.root_id].tensor
        
        ttn2_contr = ptn.completely_contract_tree(self.tree_tensor_network2,
                                                to_copy=True)
        ttn2_tensor = ttn2_contr.nodes[ttn2_contr.root_id].tensor
        
        all_axes = range(ttn1_tensor.ndim)
        
        correct_result = np.tensordot(ttn1_tensor, ttn2_tensor,
                                      axes=(all_axes, all_axes))
               
        self.assertAlmostEqual(correct_result.item(), found_result.item())
        
    def test_single_site_operator_expectation_value(self):        
        
        node_ids = ["node4", "node5", "node6"]
        dim = [8, 6, 9]
        
        for i, node_id in enumerate(node_ids):
            operator = ptn.random_hermitian_matrix(dim[i])
            
            ttn1 = ptn.TreeTensorNetwork(original_tree=self.tree_tensor_network,
                                          deep=True)
            
            found_result = ptn.single_site_operator_expectation_value(ttn1,
                                                                      node_id,
                                                                      operator)
            
            ttn1z = ptn.TreeTensorNetwork(original_tree=self.tree_tensor_network,
                                          deep=True)
            
            ttn1z_conj = ttn1z.conjugate()
            
            # Apply Operator locally
            open_leg = ttn1z.nodes[node_id].open_legs[0]
            ttn1z.nodes[node_id].absorb_tensor(operator, [1], [open_leg])
            
            correct_result = ptn.contract_two_ttn(ttn1z, ttn1z_conj).flatten()
            correct_result = correct_result[0]
        
            self.assertTrue(np.allclose(correct_result, found_result))
            
    def test_operator_expectation_value(self):
        
        node_ids = ["node4", "node5", "node6"]
        dims = [8, 6, 9]

        operator_dict = {node_ids[i]: ptn.random_hermitian_matrix(dims[i])
                     for i in range(len(dims))}
        
        ttn1  = ptn.TreeTensorNetwork(original_tree=self.tree_tensor_network,
                                      deep=True)
        
        found_result = ptn.operator_expectation_value(ttn1, operator_dict)
        
        ttn1op = ptn.TreeTensorNetwork(original_tree=self.tree_tensor_network,
                                      deep=True)
        
        ttn1op_conj = ttn1op.conjugate()
        
        # Manually contract operators
        tensor4 = ttn1op.nodes["node4"].tensor
        ttn1op.nodes["node4"].tensor = np.tensordot(tensor4, operator_dict["node4"],
                                                    axes=(1,1))
        
        tensor5 = ttn1op.nodes["node5"].tensor
        tensor5 = np.tensordot(tensor5, operator_dict["node5"],
                                                    axes=(1,1))
        ttn1op.nodes["node5"].tensor = tensor5.transpose([0,2,1])
        
        tensor6 = ttn1op.nodes["node6"].tensor
        ttn1op.nodes["node6"].tensor = np.tensordot(tensor6, operator_dict["node6"],
                                                    axes=(2,1))
        
        correct_result = ptn.contract_two_ttn(ttn1op, ttn1op_conj).flatten()
        correct_result = correct_result[0]

        self.assertTrue(np.allclose(correct_result, found_result))
        

if __name__ == "__main__":
    unittest.main()