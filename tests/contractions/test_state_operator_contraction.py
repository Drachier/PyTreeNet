from unittest import TestCase, main as unitmain

from numpy import allclose, tensordot, eye

from pytreenet.random.random_matrices import crandn
from pytreenet.core.node import Node
from pytreenet.random.random_node import random_tensor_node
from pytreenet.random.random_ttns import (random_small_ttns)
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict

from pytreenet.contractions.state_operator_contraction import (single_node_expectation_value,
                                                               contract_operator_tensor_ignoring_one_leg,
                                                               contract_bra_tensor_ignore_one_leg,
                                                               contract_single_site_operator_env)

class TestSingleNodeExpectationValue(TestCase):
    """
    Tests the function single_node_expectation_value.
    """

    def test_standard(self):
        """
        Test the expectation value of a single node.
        """
        ket_tensor = crandn((4))
        operator_tensor = crandn((4,4))
        node_id = "node"
        node = Node(identifier=node_id,
                    tensor=ket_tensor)
        ref_result = ket_tensor.conj().T @ operator_tensor @ ket_tensor
        found_result = single_node_expectation_value(node,
                                                     ket_tensor,
                                                     operator_tensor)
        self.assertTrue(allclose(ref_result, found_result))

    def test_with_separate_bra_tensor(self):
        """
        Test the expectation value of a single node with a separate bra tensor.
        """
        ket_tensor = crandn((4))
        bra_tensor = crandn((4))
        operator_tensor = crandn((4,4))
        node_id = "node"
        node = Node(identifier=node_id,
                    tensor=ket_tensor)
        ref_result = bra_tensor @ operator_tensor @ ket_tensor
        found_result = single_node_expectation_value(node,
                                                     ket_tensor,
                                                     operator_tensor,
                                                     bra_tensor)
        self.assertTrue(allclose(ref_result, found_result))

class TestContractOperatorTensorIgnoringOneLeg(TestCase):
    """
    Tests the function contract_operator_tensor_ignoring_one_leg.
    """

    def test_same_structure(self):
        """
        Tests the contraction, where the ket and operator nodes have the
        neighbours in the exact same order.
        """
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,4,3,2),
                                         identifier=ket_id)
        operator_id = "operator"
        op_node, op_tensor = random_tensor_node((6,5,4,2,2),
                                                identifier=operator_id)
        # Add other nodes
        neighbours_ids = ["neighbour"+str(i) for i in range(3)]
        ket_node.open_leg_to_child(neighbours_ids[0], 0)
        ket_node.open_leg_to_child(neighbours_ids[1], 1)
        ket_node.open_leg_to_child(neighbours_ids[2], 2)
        op_node.open_leg_to_child(neighbours_ids[0], 0)
        op_node.open_leg_to_child(neighbours_ids[1], 1)
        op_node.open_leg_to_child(neighbours_ids[2], 2)
        # Environment Tensor
        current_tensor = crandn((5,2,5,4,4,3))
        # Reference
        ref_result = tensordot(current_tensor,
                               op_tensor,
                               axes=([1,2,4],[4,1,2]))
        # Found
        found_result = contract_operator_tensor_ignoring_one_leg(current_tensor,
                                                                 ket_node,
                                                                 op_tensor,
                                                                 op_node,
                                                                 neighbours_ids[0])
        self.assertTrue(allclose(ref_result, found_result))

    def test_diff_order(self):
        """
        Tests the contraction, where the ket and operator nodes have the
        neighbours in a different order.
        """
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,4,3,2),
                                         identifier=ket_id)
        operator_id = "operator"
        op_node, op_tensor = random_tensor_node((6,5,4,2,2),
                                                identifier=operator_id)
        # Add other nodes
        neighbours_ids = ["neighbour"+str(i) for i in range(3)]
        ket_node.open_leg_to_child(neighbours_ids[0], 0)
        ket_node.open_leg_to_child(neighbours_ids[1], 1)
        ket_node.open_leg_to_child(neighbours_ids[2], 2)
        op_node.open_leg_to_child(neighbours_ids[1], 1)
        op_node.open_leg_to_child(neighbours_ids[2], 2)
        op_node.open_leg_to_child(neighbours_ids[0], 2)
        op_tensor = op_tensor.transpose((1,2,0,3,4))
        # Environment Tensor
        current_tensor = crandn((5,2,5,4,4,3))
        # Reference
        ref_result = tensordot(current_tensor,
                               op_tensor,
                               axes=([1,2,4],[4,0,1]))
        # Found
        found_result = contract_operator_tensor_ignoring_one_leg(current_tensor,
                                                                 ket_node,
                                                                 op_tensor,
                                                                 op_node,
                                                                 neighbours_ids[0])
        self.assertTrue(allclose(ref_result, found_result))

    def test_diff_neighbour_ids(self):
        """
        Tests the contraction, where the ket and operator nodes have the
        neighbours with different ids.
        """
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,4,3,2),
                                         identifier=ket_id)
        operator_id = "operator"
        op_node, op_tensor = random_tensor_node((6,5,4,2,2),
                                                identifier=operator_id)
        # Add other nodes
        neighbours_ids = ["neighbour"+str(i) for i in range(3)]
        ket_node.open_leg_to_child(neighbours_ids[0], 0)
        ket_node.open_leg_to_child(neighbours_ids[1], 1)
        ket_node.open_leg_to_child(neighbours_ids[2], 2)
        add_str = "_other"
        op_node.open_leg_to_child(neighbours_ids[0]+add_str, 0)
        op_node.open_leg_to_child(neighbours_ids[1]+add_str, 1)
        op_node.open_leg_to_child(neighbours_ids[2]+add_str, 2)
        # Environment Tensor
        current_tensor = crandn((5,2,5,4,4,3))
        # Reference
        ref_result = tensordot(current_tensor,
                               op_tensor,
                               axes=([1,2,4],[4,1,2]))
        # Found
        def id_trafo(node_id):
            return node_id+add_str
        found_result = contract_operator_tensor_ignoring_one_leg(current_tensor,
                                                                 ket_node,
                                                                 op_tensor,
                                                                 op_node,
                                                                 neighbours_ids[0],
                                                                 id_trafo=id_trafo)
        self.assertTrue(allclose(ref_result, found_result))

    def test_diff_order_diff_neighbour_ids(self):
        """
        Tests the contraction, where the ket and operator nodes have the
        neighbours with different ids and are in a different order.
        """
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,4,3,2),
                                         identifier=ket_id)
        operator_id = "operator"
        op_node, op_tensor = random_tensor_node((6,5,4,2,2),
                                                identifier=operator_id)
        # Add other nodes
        neighbours_ids = ["neighbour"+str(i) for i in range(3)]
        ket_node.open_leg_to_child(neighbours_ids[0], 0)
        ket_node.open_leg_to_child(neighbours_ids[1], 1)
        ket_node.open_leg_to_child(neighbours_ids[2], 2)
        add_str = "_other"
        op_node.open_leg_to_child(neighbours_ids[1]+add_str, 1)
        op_node.open_leg_to_child(neighbours_ids[2]+add_str, 2)
        op_node.open_leg_to_child(neighbours_ids[0]+add_str, 2)
        op_tensor = op_tensor.transpose((1,2,0,3,4))
        # Environment Tensor
        current_tensor = crandn((5,2,5,4,4,3))
        # Reference
        ref_result = tensordot(current_tensor,
                               op_tensor,
                               axes=([1,2,4],[4,0,1]))
        # Found
        def id_trafo(node_id):
            return node_id+add_str
        found_result = contract_operator_tensor_ignoring_one_leg(current_tensor,
                                                                 ket_node,
                                                                 op_tensor,
                                                                 op_node,
                                                                 neighbours_ids[0],
                                                                 id_trafo=id_trafo)
        self.assertTrue(allclose(ref_result, found_result))

class TestContractBraTensorIgnoreOneLeg(TestCase):
    """
    Test the function contract_bra_tensor_ignore_one_leg.
    """

    def test_same_structure(self):
        """
        Tests the contraction, where the bra and ket nodes have the
        neighbours in the exact same order.
        """
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,4,3,2),
                                         identifier=ket_id)
        bra_id = "bra"
        bra_node, bra_tensor = random_tensor_node((6,5,4,2),
                                                  identifier=bra_id)
        # Add other nodes
        neighbour_ids = ["neighbour"+str(i) for i in range(3)]
        ket_node.open_leg_to_child(neighbour_ids[0], 0)
        ket_node.open_leg_to_child(neighbour_ids[1], 1)
        ket_node.open_leg_to_child(neighbour_ids[2], 2)
        bra_node.open_leg_to_child(neighbour_ids[0], 0)
        bra_node.open_leg_to_child(neighbour_ids[1], 1)
        bra_node.open_leg_to_child(neighbour_ids[2], 2)
        # Environment Tensor
        current_tensor = crandn((5,5,4,7,2))
        # Reference
        ref_result = tensordot(current_tensor,
                               bra_tensor,
                               axes=([1,2,4],[1,2,3]))
        # Found
        found_result = contract_bra_tensor_ignore_one_leg(bra_tensor,
                                                          bra_node,
                                                          current_tensor,
                                                          ket_node,
                                                          neighbour_ids[0])
        self.assertTrue(allclose(ref_result, found_result))

    def test_diff_order(self):
        """
        Tests the contraction, where the bra and ket nodes have the
        neighbours in a different order.
        """
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,4,3,2),
                                         identifier=ket_id)
        bra_id = "bra"
        bra_node, bra_tensor = random_tensor_node((6,5,4,2),
                                                  identifier=bra_id)
        # Add other nodes
        neighbour_ids = ["neighbour"+str(i) for i in range(3)]
        ket_node.open_leg_to_child(neighbour_ids[0], 0)
        ket_node.open_leg_to_child(neighbour_ids[1], 1)
        ket_node.open_leg_to_child(neighbour_ids[2], 2)
        bra_node.open_leg_to_child(neighbour_ids[1], 1)
        bra_node.open_leg_to_child(neighbour_ids[2], 2)
        bra_node.open_leg_to_child(neighbour_ids[0], 2)
        bra_tensor = bra_tensor.transpose((1,2,0,3))
        # Environment Tensor
        current_tensor = crandn((5,5,4,7,2))
        # Reference
        ref_result = tensordot(current_tensor,
                               bra_tensor,
                               axes=([1,2,4],[0,1,3]))
        # Found
        found_result = contract_bra_tensor_ignore_one_leg(bra_tensor,
                                                          bra_node,
                                                          current_tensor,
                                                          ket_node,
                                                          neighbour_ids[0])
        self.assertTrue(allclose(ref_result, found_result))

    def test_diff_ids(self):
        """
        Tests the contraction, where the bra and ket nodes have the
        neighbours with different ids.
        """
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,4,3,2),
                                         identifier=ket_id)
        bra_id = "bra"
        bra_node, bra_tensor = random_tensor_node((6,5,4,2),
                                                  identifier=bra_id)
        # Add other nodes
        neighbour_ids = ["neighbour"+str(i) for i in range(3)]
        ket_node.open_leg_to_child(neighbour_ids[0], 0)
        ket_node.open_leg_to_child(neighbour_ids[1], 1)
        ket_node.open_leg_to_child(neighbour_ids[2], 2)
        add_str = "_other"
        bra_node.open_leg_to_child(neighbour_ids[0]+add_str, 0)
        bra_node.open_leg_to_child(neighbour_ids[1]+add_str, 1)
        bra_node.open_leg_to_child(neighbour_ids[2]+add_str, 2)
        # Environment Tensor
        current_tensor = crandn((5,5,4,7,2))
        # Reference
        ref_result = tensordot(current_tensor,
                               bra_tensor,
                               axes=([1,2,4],[1,2,3]))
        # Found
        def id_trafo(node_id):
            return node_id+add_str
        found_result = contract_bra_tensor_ignore_one_leg(bra_tensor,
                                                          bra_node,
                                                          current_tensor,
                                                          ket_node,
                                                          neighbour_ids[0],
                                                          id_trafo=id_trafo)
        self.assertTrue(allclose(ref_result, found_result))

    def test_diff_order_diff_ids(self):
        """
        Tests the contraction, where the bra and ket nodes have the
        neighbours with different ids and are in a different order.
        """
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,4,3,2),
                                         identifier=ket_id)
        bra_id = "bra"
        bra_node, bra_tensor = random_tensor_node((6,5,4,2),
                                                  identifier=bra_id)
        # Add other nodes
        neighbour_ids = ["neighbour"+str(i) for i in range(3)]
        ket_node.open_leg_to_child(neighbour_ids[0], 0)
        ket_node.open_leg_to_child(neighbour_ids[1], 1)
        ket_node.open_leg_to_child(neighbour_ids[2], 2)
        add_str = "_other"
        bra_node.open_leg_to_child(neighbour_ids[1]+add_str, 1)
        bra_node.open_leg_to_child(neighbour_ids[2]+add_str, 2)
        bra_node.open_leg_to_child(neighbour_ids[0]+add_str, 2)
        bra_tensor = bra_tensor.transpose((1,2,0,3))
        # Environment Tensor
        current_tensor = crandn((5,5,4,7,2))
        # Reference
        ref_result = tensordot(current_tensor,
                               bra_tensor,
                               axes=([1,2,4],[0,1,3]))
        # Found
        def id_trafo(node_id):
            return node_id+add_str
        found_result = contract_bra_tensor_ignore_one_leg(bra_tensor,
                                                          bra_node,
                                                          current_tensor,
                                                          ket_node,
                                                          neighbour_ids[0],
                                                          id_trafo=id_trafo)
        self.assertTrue(allclose(ref_result, found_result))

class TestContractSingleSiteOperatorEnv(TestCase):
    """
    Tests the function contract_single_site_operator_env.
    """

    def test_simple_root(self):
        """
        Tests the contraction of a single site operator on a simple TTNS
        structure.
        """
        ttns = random_small_ttns()
        root_id = ttns.root_id
        ttns.canonical_form(root_id)
        # Create a single site operator
        op = crandn((2,2))
        # We know that the environments are the identity
        env_dict = PartialTreeCachDict()
        root_node, root_tensor = ttns[root_id]
        c1_id, c2_id = tuple(root_node.children)
        dims = [root_node.neighbour_dim(c1_id),
                root_node.neighbour_dim(c2_id)]
        env_dict.add_entry(c1_id,
                           root_id,
                            eye(dims[0], dtype=complex))     
        env_dict.add_entry(c2_id,
                            root_id,
                             eye(dims[1], dtype=complex))
        found = contract_single_site_operator_env(root_node,
                                                  root_tensor,
                                                  root_node,
                                                  root_tensor.conj(),
                                                  op,
                                                  env_dict)
        root_open_leg = root_node.open_legs[0]
        ref = tensordot(root_tensor,
                          op,
                          axes=(root_open_leg, 1))
        ref = tensordot(ref,
                          root_tensor.conj(),
                          axes=([0,1,2],
                                 [0,1,2]))
        # Compare the results
        assert allclose(ref, found)

    def test_simple_non_root(self):
        """
        Test the contraction of a single site operator on a simple TTNS
        structure, where the operator is not applied to the root node.
        """
        ttns = random_small_ttns()
        node_id = "c1"
        ttns.canonical_form(node_id)
        # Create a single site operator
        op = crandn((3,3))
        # We know that the environments are the identity
        env_dict = PartialTreeCachDict()
        node, tensor = ttns[node_id]
        parent_id = node.parent
        dim = node.parent_leg_dim()
        env_dict.add_entry(parent_id,
                           node_id,
                            eye(dim, dtype=complex))
        # Contract the operator
        found = contract_single_site_operator_env(node,
                                                  tensor,
                                                  node,
                                                  tensor.conj(),
                                                  op,
                                                  env_dict)
        # Reference
        node_open_leg = node.open_legs[0]
        ref = tensordot(tensor,
                          op,
                          axes=(node_open_leg, 1))
        ref = tensordot(ref,
                        tensor.conj(),
                        axes=([0,1],
                               [0,1]))
        # Compare the results
        assert allclose(ref, found)

if __name__ == "__main__":
    unitmain()
