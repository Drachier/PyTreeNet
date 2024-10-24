from unittest import TestCase, main
from copy import deepcopy

from numpy import allclose

from pytreenet.special_ttn.star import StarTreeTensorNetwork
from pytreenet.random.random_matrices import crandn

class TestStarTreeTensorNetwork(TestCase):

    def test_init(self):
        """
        Test a standard initilisation of an STTN.
        """
        sttn = StarTreeTensorNetwork()
        self.assertEqual([], sttn.chains)
        self.assertEqual("center",sttn.central_node_identifier)
        self.assertEqual("node",sttn.non_center_prefix)

    def test_custom_init(self):
        """
        Test the initilisation with custom identifiers.
        """
        sttn = StarTreeTensorNetwork(central_node_identifier="A",
                                     non_center_prefix="B")
        self.assertEqual([],sttn.chains)
        self.assertEqual("A",sttn.central_node_identifier)
        self.assertEqual("B",sttn.non_center_prefix)

    def test_central_node_id_property(self):
        """
        Test the central id property.
        """
        sttn = StarTreeTensorNetwork()
        self.assertEqual("center",sttn.central_node_id)

    def test_chain_id(self):
        """
        Test that the chain id is returned correctly.
        """
        sttn = StarTreeTensorNetwork()
        self.assertEqual("node0_0", sttn.chain_id(0,0))

    def test_chain_id_custom(self):
        """
        Test that the chain id is returned correctly with a custom prefix.
        """
        sttn = StarTreeTensorNetwork(non_center_prefix="A")
        self.assertEqual("A0_0", sttn.chain_id(0,0))

    def test_num_chains_empty(self):
        """
        Test that the number of chains is 0 when there are no chains.
        """
        sttn = StarTreeTensorNetwork()
        self.assertEqual(0, sttn.num_chains())

    def test_num_chains(self):
        """
        Test that the number of chains is correct.
        """
        sttn = StarTreeTensorNetwork()
        sttn.chains.append([])
        sttn.chains.append([])
        self.assertEqual(2, sttn.num_chains())

    def test_chain_length(self):
        """
        Test that the chain length is returned correctly.
        """
        sttn = StarTreeTensorNetwork()
        sttn.chains.append([1,2,3])
        self.assertEqual(3, sttn.chain_length(0))
        sttn.chains.append([1,2,3,4])
        self.assertEqual(4, sttn.chain_length(1))

    def test_add_center_node(self):
        """
        Test the adding of the center node.
        """
        sttn = StarTreeTensorNetwork()
        tensor = crandn((2,3,4,5))
        ref_tensor = deepcopy(tensor)
        sttn.add_center_node(tensor)
        center_id = sttn.central_node_id
        self.assertTrue(center_id in sttn.tensors)
        self.assertTrue(center_id in sttn.nodes)
        self.assertTrue(allclose(ref_tensor,sttn.tensors[center_id]))

    def test_add_chain_node_first(self):
        """
        Test the adding of a first chain node.
        """
        sttn = StarTreeTensorNetwork()
        sttn.add_center_node(crandn((2,3,4,5)))
        tensor = crandn((2,6,7))
        ref_tensor = deepcopy(tensor)
        sttn.add_chain_node(tensor,0)
        self.assertEqual(1, sttn.num_chains())
        node_id = sttn.chain_id(0,0)
        self.assertTrue(node_id in sttn.tensors)
        self.assertTrue(node_id in sttn.nodes)
        self.assertTrue(allclose(ref_tensor,sttn.tensors[node_id]))

    def test_add_chain_node_second_node_same_chain(self):
        """
        Test the adding of a second tensor to the same chain.
        """
        sttn = StarTreeTensorNetwork()
        sttn.add_center_node(crandn((2,3,4,5)))
        sttn.add_chain_node(crandn((2,6,7)),0)
        tensor = crandn((6,3,8))
        ref_tensor = deepcopy(tensor)
        sttn.add_chain_node(tensor,0)
        self.assertEqual(1, sttn.num_chains())
        self.assertEqual(2, sttn.chain_length(0))
        node_id = sttn.chain_id(0,1)
        self.assertTrue(node_id in sttn.tensors)
        self.assertTrue(node_id in sttn.nodes)
        self.assertTrue(allclose(ref_tensor,sttn.tensors[node_id]))

if __name__ == "__main__":
    main()
