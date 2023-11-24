import unittest

import numpy as np

import pytreenet as ptn

class TestMPT_Attaching(unittest.TestCase):
    def setUp(self) -> None:
        self.mpt = ptn.MatrixProductTree()
        self.mpt.add_root(ptn.Node(identifier="site2"), ptn.crandn((4,5,2)))
    
        self.nodes = [ptn.Node(identifier="site"+str(i)) for i in (0,1,3,4)]
        shapes = [(3,2),(3,4,2),(5,3,2),(3,2)]
        self.tensors = [ptn.crandn(shape) for shape in shapes]

    def test_add_left_first(self):
        self.mpt.attach_node_left_end(self.nodes[1],self.tensors[1])
        self.mpt.attach_node_right_end(self.nodes[2],self.tensors[2])

        self.assertEqual({"site1": self.nodes[1]}, self.mpt.left_nodes)
        self.assertEqual({"site3": self.nodes[2]}, self.mpt.right_nodes)

        self.assertTrue("site1" in self.mpt)
        self.assertTrue("site3" in self.mpt)

        ref_mpt = np.tensordot(self.mpt.root[1],
                               self.tensors[1],
                               axes=(self.mpt.root[0].neighbour_index("site1"),1))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[2],
                               axes=(0,0))
        
        found_mpt = self.mpt.completely_contract_tree().root[1]
        self.assertTrue(np.allclose(ref_mpt,found_mpt))

    def test_add_right_first(self):
        self.mpt.attach_node_right_end(self.nodes[2],self.tensors[2])
        self.mpt.attach_node_left_end(self.nodes[1],self.tensors[1])

        self.assertEqual({"site1": self.nodes[1]}, self.mpt.left_nodes)
        self.assertEqual({"site3": self.nodes[2]}, self.mpt.right_nodes)

        self.assertTrue("site1" in self.mpt)
        self.assertTrue("site3" in self.mpt)

        ref_mpt = np.tensordot(self.mpt.root[1],
                               self.tensors[1],
                               axes=(self.mpt.root[0].neighbour_index("site1"),1))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[2],
                               axes=(0,0))
        ref_mpt = np.transpose(ref_mpt,
                               (0,3,4,1,2))

        found_mpt = self.mpt.completely_contract_tree().root[1]
        self.assertTrue(np.allclose(ref_mpt,found_mpt))

    def test_add_two_left_nodes(self):
        self.mpt.attach_node_left_end(self.nodes[1],self.tensors[1])
        self.mpt.attach_node_left_end(self.nodes[0],self.tensors[0],final=True)

        self.assertTrue("site0" in self.mpt.left_nodes)
        self.assertTrue("site1" in self.mpt.left_nodes)

        self.assertTrue("site0" in self.mpt)
        self.assertTrue("site1" in self.mpt)

        ref_mpt = np.tensordot(self.mpt.root[1],
                               self.tensors[1],
                               axes=(self.mpt.root[0].neighbour_index("site1"),1))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[0],
                               axes=(2,0))

        found_mpt = self.mpt.completely_contract_tree().root[1]
        self.assertTrue(np.allclose(ref_mpt,found_mpt))

    def test_add_two_right_nodes(self):
        self.mpt.attach_node_right_end(self.nodes[2],self.tensors[2])
        self.mpt.attach_node_right_end(self.nodes[3],self.tensors[3])

        self.assertTrue("site3" in self.mpt.right_nodes)
        self.assertTrue("site4" in self.mpt.right_nodes)

        self.assertTrue("site3" in self.mpt)
        self.assertTrue("site4" in self.mpt)

        ref_mpt = np.tensordot(self.mpt.root[1],
                               self.tensors[2],
                               axes=(self.mpt.root[0].neighbour_index("site3"),0))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[3],
                               axes=(2,0))

        found_mpt = self.mpt.completely_contract_tree().root[1]
        self.assertTrue(np.allclose(ref_mpt,found_mpt))

    def test_add_four_nodes(self):
        self.mpt.attach_node_left_end(self.nodes[1],self.tensors[1])
        self.mpt.attach_node_left_end(self.nodes[0],self.tensors[0],final=True)
        self.mpt.attach_node_right_end(self.nodes[2],self.tensors[2])
        self.mpt.attach_node_right_end(self.nodes[3],self.tensors[3])

        self.assertTrue("site0" in self.mpt.left_nodes)
        self.assertTrue("site1" in self.mpt.left_nodes)
        self.assertTrue("site3" in self.mpt.right_nodes)
        self.assertTrue("site4" in self.mpt.right_nodes)
        self.assertTrue("site0" in self.mpt)
        self.assertTrue("site1" in self.mpt)
        self.assertTrue("site3" in self.mpt)
        self.assertTrue("site4" in self.mpt)

        ref_mpt = np.tensordot(self.mpt.root[1],
                               self.tensors[1],
                               axes=(self.mpt.root[0].neighbour_index("site1"),1))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[0],
                               axes=(2,0))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[2],
                               axes=(0,0))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[3],
                               axes=(3,0))

        found_mpt = self.mpt.completely_contract_tree().root[1]
        self.assertTrue(np.allclose(ref_mpt,found_mpt))

class TestMPT_classmethods(unittest.TestCase):
    def setUp(self) -> None:
        shapes = [(3,2),(3,4,2),(4,5,2),(5,3,2),(3,2)]
        self.tensors = [ptn.crandn(shape) for shape in shapes]

    def test_from_tensor_list_root_at_2(self):
        mpt = ptn.MatrixProductTree.from_tensor_list(self.tensors,
                                                     root_site=2)

        ref_mpt = np.tensordot(self.tensors[2],
                               self.tensors[1],
                               axes=(0,1))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[0],
                               axes=(2,0))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[3],
                               axes=(0,0))
        ref_mpt = np.tensordot(ref_mpt,
                               self.tensors[4],
                               axes=(3,0))

        found_mpt = mpt.completely_contract_tree().root[1]
        self.assertTrue(np.allclose(ref_mpt,found_mpt))

if __name__ == "__main__":
    unittest.main()
