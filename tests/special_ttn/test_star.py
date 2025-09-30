from unittest import TestCase, main
from copy import deepcopy

from numpy import allclose
import numpy as np

from pytreenet.operators.common_operators import ket_i
from pytreenet.special_ttn.star import StarTreeTensorNetwork, StarTreeTensorState
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

class TestStarTreeTensorNetworkFromTensorLists(TestCase):
    """
    Test the from_tensor_lists method of the StarTreeTensorNetwork.
    """

    def test_empty_tensor_listlist(self):
        """
        Test that an empty list of lists causes a root-only STTN.
        """
        centre_tensor = crandn((2,3,4,5))
        sttn = StarTreeTensorNetwork.from_tensor_lists(centre_tensor,
                                                       tensors=[])
        self.assertTrue(sttn.root_id is not None)
        np.testing.assert_array_equal(sttn.tensors[sttn.root_id], centre_tensor)

    def test_emtpy_tensorlists(self):
        """
        Test that a list of empty lists causes a root-only STTN.
        """
        centre_tensor = crandn((2,3,4,5))
        sttn = StarTreeTensorNetwork.from_tensor_lists(centre_tensor,
                                                       tensors=[[],[],[]])
        self.assertTrue(sttn.root_id is not None)
        np.testing.assert_array_equal(sttn.tensors[sttn.root_id], centre_tensor)

    def test_single_chain(self):
        """
        Test that a single chain is created correctly.
        """
        centre_tensor = crandn((2,3,4,5))
        chain_tensors = [crandn((2,6,7)), crandn((6,3,8))]
        identifiers = ["node0", "node1"]
        sttn = StarTreeTensorNetwork.from_tensor_lists(centre_tensor,
                                                       tensors=[chain_tensors],
                                                       identifiers=[identifiers])
        self.assertEqual(1, sttn.num_chains())
        self.assertEqual(2, sttn.chain_length(0))
        np.testing.assert_array_equal(sttn.tensors[sttn.root_id], centre_tensor)
        for i, tensor in enumerate(chain_tensors):
            node_id = identifiers[i]
            np.testing.assert_array_equal(sttn.tensors[node_id], tensor)

    def test_two_chains(self):
        """
        Test that two chains are created correctly.
        """
        centre_tensor = crandn((2,3,4,5))
        chain1_tensors = [crandn((2,6,7)), crandn((6,3,8))]
        chain2_tensors = [crandn((3,9,10)), crandn((9,7,11))]
        identifiers1 = ["node0_0", "node0_1"]
        identifiers2 = ["node1_0", "node1_1"]
        sttn = StarTreeTensorNetwork.from_tensor_lists(centre_tensor,
                                                       tensors=[chain1_tensors, chain2_tensors],
                                                       identifiers=[identifiers1, identifiers2])
        self.assertEqual(2, sttn.num_chains())
        self.assertEqual(2, sttn.chain_length(0))
        self.assertEqual(2, sttn.chain_length(1))
        np.testing.assert_array_equal(sttn.tensors[sttn.root_id], centre_tensor)
        # Test tensors in chain 1
        for i, tensor in enumerate(chain1_tensors):
            node_id = identifiers1[i]
            np.testing.assert_array_equal(sttn.tensors[node_id], tensor)
        # Test tensors in chain 2
        for i, tensor in enumerate(chain2_tensors):
            node_id = identifiers2[i]
            np.testing.assert_array_equal(sttn.tensors[node_id], tensor)

    def test_three_chain(self):
        """
        Tests that three chains are created correctly.
        """
        centre_tensor = crandn((2,3,4,5))
        chain1_tensors = [crandn((2,6,7)), crandn((6,3,8))]
        chain2_tensors = [crandn((3,9,10))]
        chain3_tensors = [crandn((4,12,13)), crandn((12,8,14)), crandn(8,5,3)]
        identifiers1 = ["node0_0", "node0_1"]
        identifiers2 = ["node1_0"]
        identifiers3 = ["node2_0", "node2_1", "node2_2"]
        tensors = [chain1_tensors, chain2_tensors, chain3_tensors]
        identifiers = [identifiers1, identifiers2, identifiers3]
        sttn = StarTreeTensorNetwork.from_tensor_lists(centre_tensor,
                                                       tensors=tensors,
                                                       identifiers=identifiers)
        self.assertEqual(3, sttn.num_chains())
        self.assertEqual(2, sttn.chain_length(0))
        self.assertEqual(1, sttn.chain_length(1))
        self.assertEqual(3, sttn.chain_length(2))
        np.testing.assert_array_equal(sttn.tensors[sttn.root_id], centre_tensor)
        # Test tensors in chain 1
        for i, tensor in enumerate(chain1_tensors):
            node_id = identifiers1[i]
            np.testing.assert_array_equal(sttn.tensors[node_id], tensor)
        # Test tensors in chain 2
        for i, tensor in enumerate(chain2_tensors):
            node_id = identifiers2[i]
            np.testing.assert_array_equal(sttn.tensors[node_id], tensor)
        # Test tensors in chain 3
        for i, tensor in enumerate(chain3_tensors):
            node_id = identifiers3[i]
            np.testing.assert_array_equal(sttn.tensors[node_id], tensor)

    def test_identifiers_mismatch(self):
        """
        Test that a ValueError is raised if the number of identifiers does not
        match the number of tensors.
        """
        centre_tensor = crandn((2,3,4,5))
        chain_tensors = [crandn((2,6,7)), crandn((6,3,8))]
        identifiers = ["node0"]
        self.assertRaises(ValueError,
                        StarTreeTensorNetwork.from_tensor_lists,
                        centre_tensor,
                        tensors=[chain_tensors],
                        identifiers=[identifiers])

class TestStartTreeTensorNetworkSpecialStates(TestCase):
    """
    Tests the creation of some special states that came up in simulations and
    thus may just as well be used for tests.
    """

    def test_tstar_qubit(self):
        """
        Creates a T of qubits, where the first tensor on the first chain is
        a |+> state and all other tensors are |0> states.
        """
        main_state = np.array([1,1], dtype=complex) / np.sqrt(2)
        ttns = StarTreeTensorState()
        centre_tensor = np.ones((1,1,1,1), dtype=complex)
        ttns.add_center_node(centre_tensor)
        main_tensor = main_state.reshape((1,1,2))
        ttns.add_chain_node(main_tensor, 0, identifier="qmain")
        mid_tensor = np.reshape(ket_i(0,2).astype(complex), (1,1,2))
        end_tensor = np.reshape(ket_i(0,2).astype(complex), (1,2))
        ttns.add_chain_node(deepcopy(mid_tensor), 1, identifier="q2")
        ttns.add_chain_node(deepcopy(mid_tensor), 2, identifier="q5")
        for i in range(3):
            ttns.add_chain_node(deepcopy(mid_tensor), i, identifier=f"q{3*i}")
            ttns.add_chain_node(deepcopy(end_tensor), i, identifier=f"q{3*i+1}")
        ttns.pad_bond_dimensions(2)
        ref = np.kron(main_state, ket_i(0,2**8))
        found, _ = ttns.completely_contract_tree(to_copy=True)
        np.testing.assert_array_equal(np.reshape(found, ref.shape), ref)

if __name__ == "__main__":
    main()
