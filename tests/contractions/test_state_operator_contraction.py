from unittest import TestCase, main as unitmain

from numpy import allclose, tensordot, eye, prod

from pytreenet.random.random_matrices import crandn
from pytreenet.core.node import Node
from pytreenet.random.random_node import random_tensor_node
from pytreenet.random.random_ttns import (random_small_ttns)
from pytreenet.contractions.tree_cach_dict import PartialTreeCachDict
from pytreenet.ttns.ttns import TTNS
from pytreenet.ttno.ttno_class import TTNO

from pytreenet.contractions.state_operator_contraction import (get_matrix_element,
                                                               single_node_expectation_value,
                                                               contract_operator_tensor_ignoring_one_leg,
                                                               contract_bra_tensor_ignore_one_leg,
                                                               contract_single_site_operator_env,
                                                               contract_ket_ham_with_envs,
                                                               _neighbours_before_ham,
                                                               contract_leaf)

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

class TestLargeMatrixElement(TestCase):
    """
    Tests the computation to obtain the matrix element <bra|O|ket>.
    Here bra and ket are TTNS and O is a TTNO.
    """

    def test_full_case(self):
        """
        Test the function, where the nodes in ket, bra, and op have different
        identifiers and are in a different order.
        """
        # Build ket
        ket = TTNS()
        root_id = "root"
        root_node, root_tensor = random_tensor_node((7,8),
                                                    identifier=root_id)
        ket.add_root(root_node, root_tensor)
        n10_id = "n10"
        n10_node, n10_tensor = random_tensor_node((7,1,4), 
                                                  identifier=n10_id)
        ket.add_child_to_parent(n10_node, n10_tensor, 0, root_id, 0)
        n11_id = "n11"
        n11_node, n11_tensor = random_tensor_node((8,6,2),
                                                    identifier=n11_id)
        ket.add_child_to_parent(n11_node, n11_tensor, 0, root_id, 1)
        n20_id = "n20"
        n20_node, n20_tensor = random_tensor_node((6,4,1),
                                                  identifier=n20_id)
        ket.add_child_to_parent(n20_node, n20_tensor, 0, n11_id, 1)
        nl1_id = "nl1"
        nl1_node, nl1_tensor = random_tensor_node((1,2),
                                                    identifier=nl1_id)
        ket.add_child_to_parent(nl1_node, nl1_tensor, 0, n10_id, 1)
        nl2_id = "nl2"
        nl2_node, nl2_tensor = random_tensor_node((4,2,2),
                                                   identifier=nl2_id)
        ket.add_child_to_parent(nl2_node, nl2_tensor, 0, n10_id, 2)
        nl3_id = "nl3"
        nl3_node, nl3_tensor = random_tensor_node((4,2,2),
                                                   identifier=nl3_id)
        ket.add_child_to_parent(nl3_node, nl3_tensor, 0, n20_id, 1)
        nl4_id = "nl4"
        nl4_node, nl4_tensor = random_tensor_node((1,2),
                                                    identifier=nl4_id)
        ket.add_child_to_parent(nl4_node, nl4_tensor, 0, n20_id, 2)
        # Build bra
        def bra_id_trafo(node_id):
            return node_id + "_bra"
        bra = TTNS()
        bra_root_node, bra_root_tensor = random_tensor_node((7,8),
                                                            identifier=bra_id_trafo(root_id))
        bra.add_root(bra_root_node, bra_root_tensor)
        bra_n11_node, bra_n11_tensor = random_tensor_node((8,6,2),
                                                            identifier=bra_id_trafo(n11_id))
        bra.add_child_to_parent(bra_n11_node, bra_n11_tensor, 0, bra_root_node.identifier, 1)
        bra_n10_node, bra_n10_tensor = random_tensor_node((7,1,4),
                                                            identifier=bra_id_trafo(n10_id))
        bra.add_child_to_parent(bra_n10_node, bra_n10_tensor, 0, bra_root_node.identifier, 1)
        bra_n20_node, bra_n20_tensor = random_tensor_node((6,4,1),
                                                            identifier=bra_id_trafo(n20_id))
        bra.add_child_to_parent(bra_n20_node, bra_n20_tensor, 0, bra_n11_node.identifier, 1)
        bra_nl2_node, bra_nl2_tensor = random_tensor_node((4,2,2),
                                                            identifier=bra_id_trafo(nl2_id))
        bra.add_child_to_parent(bra_nl2_node, bra_nl2_tensor, 0, bra_n10_node.identifier, 2)
        bra_nl1_node, bra_nl1_tensor = random_tensor_node((1,2),
                                                            identifier=bra_id_trafo(nl1_id))
        bra.add_child_to_parent(bra_nl1_node, bra_nl1_tensor, 0, bra_n10_node.identifier, 2)
        bra_nl4_node, bra_nl4_tensor = random_tensor_node((1,2),
                                                            identifier=bra_id_trafo(nl4_id))
        bra.add_child_to_parent(bra_nl4_node, bra_nl4_tensor, 0, bra_n20_node.identifier, 2)
        bra_nl3_node, bra_nl3_tensor = random_tensor_node((4,2,2),
                                                            identifier=bra_id_trafo(nl3_id))
        bra.add_child_to_parent(bra_nl3_node, bra_nl3_tensor, 0, bra_n20_node.identifier, 2)
        # Build operator
        def op_id_trafo(node_id):
            return node_id + "_op"
        op = TTNO()
        op_root_node, op_root_tensor = random_tensor_node((8,6),
                                                        identifier=op_id_trafo(root_id))
        op.add_root(op_root_node, op_root_tensor)
        op_n11_node, op_n11_tensor = random_tensor_node((6,8),
                                                        identifier=op_id_trafo(n11_id))
        op.add_child_to_parent(op_n11_node, op_n11_tensor, 1, op_root_node.identifier, 0)
        op_n10_node, op_n10_tensor = random_tensor_node((6,3,5),
                                                        identifier=op_id_trafo(n10_id))
        op.add_child_to_parent(op_n10_node, op_n10_tensor, 0, op_root_node.identifier, 1)
        op_n20_node, op_n20_tensor = random_tensor_node((6,5,3),
                                                        identifier=op_id_trafo(n20_id))
        op.add_child_to_parent(op_n20_node, op_n20_tensor, 0, op_n11_node.identifier, 1)
        op_nl2_node, op_nl2_tensor = random_tensor_node((5,2,2),
                                                        identifier=op_id_trafo(nl2_id))
        op.add_child_to_parent(op_nl2_node, op_nl2_tensor, 0, op_n10_node.identifier, 2)
        op_nl1_node, op_nl1_tensor = random_tensor_node((3,2,2),
                                                        identifier=op_id_trafo(nl1_id))
        op.add_child_to_parent(op_nl1_node, op_nl1_tensor, 0, op_n10_node.identifier, 2)
        op_nl4_node, op_nl4_tensor = random_tensor_node((3,2,2),
                                                        identifier=op_id_trafo(nl4_id))
        op.add_child_to_parent(op_nl4_node, op_nl4_tensor, 0, op_n20_node.identifier, 2)
        op_nl3_node, op_nl3_tensor = random_tensor_node((5,2,2),
                                                        identifier=op_id_trafo(nl3_id))
        op.add_child_to_parent(op_nl3_node, op_nl3_tensor, 0, op_n20_node.identifier, 2)
        # Found
        found_result = get_matrix_element(bra, op, ket)
        # Reference
        ket_vec, _ = ket.completely_contract_tree(to_copy=True)
        bra_vec, _ = bra.completely_contract_tree(to_copy=True)
        op_mat, _ = op.as_matrix()

class TestContractLeaf(TestCase):
    """
    Test the sandwich contraction of a leaf node with an operator.
    """

    def test_standard(self):
        """
        Tests the contraction of a leaf node with an operator.
        """
        node_id = "node"
        ket_node, ket_tensor = random_tensor_node((4,2),
                                          identifier=node_id)
        ham_node, ham_tensor = random_tensor_node((5,3,2),
                                          identifier=node_id)
        bra_node, bra_tensor = random_tensor_node((6,3),
                                          identifier=node_id)
        nodes = [ket_node, ham_node, bra_node]
        for node in nodes:
            node.open_leg_to_parent("parent", 0)
        # Reference
        ref_result = tensordot(ket_tensor,
                               ham_tensor,
                               axes=(1,2))
        ref_result = tensordot(ref_result,
                                 bra_tensor,
                                 axes=(2,1))
        # Found
        found_result = contract_leaf(ket_node,
                                     ket_tensor,
                                     ham_node,
                                     ham_tensor,
                                     bra_node=bra_node,
                                     bra_tensor=bra_tensor)
        self.assertTrue(allclose(ref_result, found_result))

    def test_no_bra(self):
        """
        Tests the contraction of a leaf node with an operator, where there
        is no explicit bra tensor.
        """
        node_id = "node"
        ket_node, ket_tensor = random_tensor_node((4,2),
                                          identifier=node_id)
        ham_node, ham_tensor = random_tensor_node((5,2,2),
                                          identifier=node_id)
        nodes = [ket_node, ham_node]
        for node in nodes:
            node.open_leg_to_parent("parent", 0)
        # Reference
        ref_result = tensordot(ket_tensor,
                               ham_tensor,
                               axes=(1,2))
        ref_result = tensordot(ref_result,
                                 ket_tensor.conj(),
                                 axes=(2,1))
        # Found
        found_result = contract_leaf(ket_node,
                                     ket_tensor,
                                     ham_node,
                                     ham_tensor)
        self.assertTrue(allclose(ref_result, found_result))

    def test_multiple_open_legs(self):
        """
        Tests the contraction of a leaf node with an operator, where the
        nodes have multiple open legs.
        """
        node_id = "node"
        ket_node, ket_tensor = random_tensor_node((6,2,3),
                                          identifier=node_id)
        ham_node, ham_tensor = random_tensor_node((7,4,5,2,3),
                                          identifier=node_id)
        bra_node, bra_tensor = random_tensor_node((8,4,5),
                                          identifier=node_id)
        nodes = [ket_node, ham_node, bra_node]
        for node in nodes:
            node.open_leg_to_parent("parent", 0)
        # Reference
        ref_result = tensordot(ket_tensor,
                               ham_tensor,
                               axes=([1,2],[3,4]))
        ref_result = tensordot(ref_result,
                                 bra_tensor,
                                 axes=([2,3],[1,2]))
        # Found
        found_result = contract_leaf(ket_node,
                                     ket_tensor,
                                     ham_node,
                                     ham_tensor,
                                     bra_node=bra_node,
                                     bra_tensor=bra_tensor)
        self.assertTrue(allclose(ref_result, found_result))

    def test_only_one_of_bra_given(self):
        """
        An exception is raised, if only one of the bra node or tensor is given.
        """
        node_id = "node"
        ket_node, ket_tensor = random_tensor_node((6,2,3),
                                          identifier=node_id)
        ham_node, ham_tensor = random_tensor_node((7,4,5,2,3),
                                          identifier=node_id)
        bra_node, bra_tensor = random_tensor_node((8,4,5),
                                          identifier=node_id)
        self.assertRaises(ValueError,
                          contract_leaf,
                            ket_node,
                            ket_tensor,
                            ham_node,
                            ham_tensor,
                            bra_node=bra_node)
        self.assertRaises(ValueError,
                          contract_leaf,
                            ket_node,
                            ket_tensor,
                            ham_node,
                            ham_tensor,
                            bra_tensor=bra_tensor)

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

    def test_multiple_open_legs(self):
        """
        Test the contraction, where the nodes have multiple open legs.
        """
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,6,7,2,3,4),
                                         identifier=ket_id)
        operator_id = "operator"
        op_node, op_tensor = random_tensor_node((10,6,8,2,3,4,2,3,4),
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
        current_tensor = crandn((5,2,3,4,6,9,8,7))
        # Reference
        ref_result = tensordot(current_tensor,
                                 op_tensor,
                                 axes=([1,2,3,4,6],[6,7,8,1,2]))
        # Found
        found_result = contract_operator_tensor_ignoring_one_leg(current_tensor,
                                                                    ket_node,
                                                                    op_tensor,
                                                                    op_node,
                                                                    neighbours_ids[0])
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

    def test_multiple_open_legs(self):
        """
        Test the contraction, where the nodes have multiple open legs.
        """
        ket_id = "ket"
        ket_node, _ = random_tensor_node((5,6,7,2,3,4),
                                         identifier=ket_id)
        bra_id = "bra"
        bra_node, bra_tensor = random_tensor_node((9,7,8,2,3,4),
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
        current_tensor = crandn((5,7,8,6,2,3,4))
        # Reference
        ref_result = tensordot(current_tensor,
                               bra_tensor,
                               axes=([4,5,6,1,2],[3,4,5,1,2]))
        # Found
        found_result = contract_bra_tensor_ignore_one_leg(bra_tensor,
                                                          bra_node,
                                                          current_tensor,
                                                          ket_node,
                                                          neighbour_ids[0])
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

class TestContractKetHamWithEnvs(TestCase):
    """
    Tests the function contract_ket_ham_with_envs.
    """

    def test_no_neighbours(self):
        """
        Tests the contraction of a ket with a Hamiltonian and the environments
        when there are no neighbours, i.e. no environments.
        """
        ket_node, ket_tensor = random_tensor_node((2, ), identifier="ket")
        ham_node, ham_tensor = random_tensor_node((3, 2), identifier="ham")
        # Reference
        ref = tensordot(ket_tensor,
                        ham_tensor,
                        axes=(0, 1))
        # Found
        found = contract_ket_ham_with_envs(ket_node,
                                           ket_tensor,
                                           ham_node,
                                           ham_tensor,
                                           PartialTreeCachDict())
        self.assertTrue(allclose(ref, found))

    def test_one_neighbour(self):
        """
        Tests the contraction of a ket with a Hamiltonian and the environments
        when there is one neighbour, i.e. the node is a leaf.
        """
        node_id = "node"
        ket_node, ket_tensor = random_tensor_node((6,2, ), identifier=node_id)
        ham_node, ham_tensor = random_tensor_node((5,3,2), identifier=node_id)
        neighbour_id = "neighbour"
        ket_node.open_leg_to_parent(neighbour_id, 0)
        ham_node.open_leg_to_child(neighbour_id, 0)
        neighbour_tensor = crandn((6,5,4))
        cache = PartialTreeCachDict()
        cache.add_entry(neighbour_id,
                        node_id,
                        neighbour_tensor)
        # Reference
        ref = tensordot(ket_tensor,
                        ham_tensor,
                        axes=(1,2))
        ref = tensordot(ref,
                        neighbour_tensor,
                        axes = ([0,1],[0,1]))
        ref = ref.T # We want the result in the same order as the ket tensor
        # Found
        found = contract_ket_ham_with_envs(ket_node,
                                           ket_tensor,
                                           ham_node,
                                           ham_tensor,
                                           cache)
        self.assertTrue(allclose(ref, found))
        
    def test_two_neighbours(self):
        """
        Tests the contraction of a ket with a Hamiltonian and the environments
        when there are two neighbours, i.e. as in an MPS.
        """
        node_id = "node"
        ket_node, ket_tensor = random_tensor_node((7,8,2, ), identifier=node_id)
        ham_node, ham_tensor = random_tensor_node((5,6,2,2), identifier=node_id)
        neighbour1_id = "neighbour1"
        neighbour1_tensor = crandn((7,5,3))
        neighbour2_id = "neighbour2"
        neighbour2_tensor = crandn((8,6,4))
        ket_node.open_leg_to_parent(neighbour1_id, 0)
        ket_node.open_leg_to_child(neighbour2_id, 1)
        ham_node.open_leg_to_child(neighbour1_id, 0)
        ham_node.open_leg_to_child(neighbour2_id, 1)
        cache = PartialTreeCachDict()
        cache.add_entry(neighbour1_id,
                        node_id,
                        neighbour1_tensor)
        cache.add_entry(neighbour2_id,
                        node_id,
                        neighbour2_tensor)
        # Reference
        ref = tensordot(ket_tensor,
                        ham_tensor,
                        axes=(2,3))
        ref = tensordot(ref,
                        neighbour1_tensor,
                        axes=([0,2],[0,1]))
        ref = tensordot(ref,
                        neighbour2_tensor,
                        axes=([0,1],[0,1]))
        ref = ref.transpose([1,2,0])
        # Found
        found = contract_ket_ham_with_envs(ket_node,
                                           ket_tensor,
                                           ham_node,
                                           ham_tensor,
                                           cache)
        self.assertTrue(allclose(ref, found))

    def test_two_neighbours_mixed(self):
        """
        Test the contraction for a node with two neighbours, where they are
        in a different order on the ket and Hamiltonian node.
        """
        node_id = "node"
        ket_node, ket_tensor = random_tensor_node((7,8,2, ), identifier=node_id)
        ham_node, ham_tensor = random_tensor_node((6,5,2,2), identifier=node_id)
        neighbour1_id = "neighbour1"
        neighbour1_tensor = crandn((7,5,3))
        neighbour2_id = "neighbour2"
        neighbour2_tensor = crandn((8,6,4))
        ket_node.open_leg_to_parent(neighbour1_id, 0)
        ket_node.open_leg_to_child(neighbour2_id, 1)
        ham_node.open_leg_to_child(neighbour2_id, 0)
        ham_node.open_leg_to_child(neighbour1_id, 1)
        cache = PartialTreeCachDict()
        cache.add_entry(neighbour1_id,
                        node_id,
                        neighbour1_tensor)
        cache.add_entry(neighbour2_id,
                        node_id,
                        neighbour2_tensor)
        # Reference
        ref = tensordot(ket_tensor,
                        ham_tensor,
                        axes=(2,3))
        ref = tensordot(ref,
                        neighbour1_tensor,
                        axes=([0,3],[0,1]))
        ref = tensordot(ref,
                        neighbour2_tensor,
                        axes=([0,1],[0,1]))
        ref = ref.transpose([1,2,0])
        # Found
        found = contract_ket_ham_with_envs(ket_node,
                                           ket_tensor,
                                           ham_node,
                                           ham_tensor,
                                           cache)
        self.assertTrue(allclose(ref, found))

    def test_three_neighbours(self):
        """
        Tests the contraction of a ket with a Hamiltonian and the environments
        when there are three neighbours, i.e. as in a proper TTNS.
        """
        node_id = "node"
        ket_node, ket_tensor = random_tensor_node((9,10,11,2, ),
                                                  identifier=node_id)
        ham_node, ham_tensor = random_tensor_node((6,7,8,2,2),
                                                  identifier=node_id)
        neighbour1_id = "neighbour1"
        neighbour1_tensor = crandn((9,6,3))
        neighbour2_id = "neighbour2"
        neighbour2_tensor = crandn((10,7,4))
        neighbour3_id = "neighbour3"
        neighbour3_tensor = crandn((11,8,5))
        ket_node.open_leg_to_parent(neighbour1_id, 0)
        ket_node.open_leg_to_child(neighbour2_id, 1)
        ket_node.open_leg_to_child(neighbour3_id, 2)
        ham_node.open_leg_to_parent(neighbour1_id, 0)
        ham_node.open_leg_to_child(neighbour2_id, 1)
        ham_node.open_leg_to_child(neighbour3_id, 2)
        cache = PartialTreeCachDict()
        cache.add_entry(neighbour1_id,
                        node_id,
                        neighbour1_tensor)
        cache.add_entry(neighbour2_id,
                        node_id,
                        neighbour2_tensor)
        cache.add_entry(neighbour3_id,
                        node_id,
                        neighbour3_tensor)
        # Reference
        ref = tensordot(ket_tensor,
                        ham_tensor,
                        axes=(3,4))
        ref = tensordot(ref,
                        neighbour1_tensor,
                        axes=([0,3],[0,1]))
        ref = tensordot(ref,
                        neighbour2_tensor,
                        axes=([0,2],[0,1]))
        ref = tensordot(ref,
                        neighbour3_tensor,
                        axes=([0,1],[0,1]))
        ref = ref.transpose([1,2,3,0])
        # Found
        found = contract_ket_ham_with_envs(ket_node,
                                           ket_tensor,
                                           ham_node,
                                           ham_tensor,
                                           cache)
        self.assertTrue(allclose(ref, found))

    def test_four_neighbours_mixed(self):
        """
        Tests the contraction for a node of degree four with two open legs,
        as obtained by merging two degree-three nodes for a two-site update,
        where the neighbours are in a different order on the ket and the
        Hamiltonian node.
        """
        node_id = "node"
        chis = (5, 6, 7, 8)
        ham_dims = (3, 4, 3, 4)
        ham_order = (2, 0, 3, 1)
        ket_node, ket_tensor = random_tensor_node(chis + (2, 2),
                                                  identifier=node_id)
        ham_shape = tuple(ham_dims[index] for index in ham_order)
        ham_node, ham_tensor = random_tensor_node(ham_shape + (2, 2, 2, 2),
                                                  identifier=node_id)
        neighbour_ids = [f"neighbour{index}" for index in range(4)]
        blocks = [crandn((chis[index], ham_dims[index], chis[index]))
                  for index in range(4)]
        cache = PartialTreeCachDict()
        for index, neighbour_id in enumerate(neighbour_ids):
            if index == 0:
                ket_node.open_leg_to_parent(neighbour_id, 0)
            else:
                ket_node.open_leg_to_child(neighbour_id, index)
            cache.add_entry(neighbour_id, node_id, blocks[index])
        for position, index in enumerate(ham_order):
            if position == 0:
                ham_node.open_leg_to_parent(neighbour_ids[index], 0)
            else:
                ham_node.open_leg_to_child(neighbour_ids[index], position)
        # Reference. The open legs of the ket are contracted with the input
        # legs of the Hamiltonian, leaving its neighbour legs in its own order.
        ref = tensordot(ket_tensor, ham_tensor,
                        axes=([4, 5], [6, 7]))
        # ref legs: chi0 chi1 chi2 chi3 | m2 m0 m3 m1 | out0 out1
        ref = tensordot(ref, blocks[0], axes=([0, 5], [0, 1]))
        ref = tensordot(ref, blocks[1], axes=([0, 5], [0, 1]))
        ref = tensordot(ref, blocks[2], axes=([0, 2], [0, 1]))
        ref = tensordot(ref, blocks[3], axes=([0, 1], [0, 1]))
        # ref legs: out0 out1 | bra0 bra1 bra2 bra3
        ref = ref.transpose([2, 3, 4, 5, 0, 1])
        # Found
        found = contract_ket_ham_with_envs(ket_node,
                                           ket_tensor,
                                           ham_node,
                                           ham_tensor,
                                           cache)
        self.assertEqual(ref.shape, found.shape)
        self.assertTrue(allclose(ref, found))

class TestNeighboursBeforeHam(TestCase):
    """
    Tests the function _neighbours_before_ham.
    """

    def _peak(self, ham_dims, before_ham):
        """
        The larger of the two partial products the split trades off.

        Args:
            ham_dims (tuple[int, ...]): The Hamiltonian bond dimensions.
            before_ham (tuple[int, ...]): The indices contracted first.

        Returns:
            int: The factor by which the largest intermediate exceeds the
                result.
        """
        before = prod([ham_dims[index] for index in before_ham])
        after = prod([dim for index, dim in enumerate(ham_dims)
                      if index not in before_ham])
        return max(before, after)

    def _minimal_peak(self, ham_dims):
        """
        The minimal peak over all subsets, found by brute force.

        Args:
            ham_dims (tuple[int, ...]): The Hamiltonian bond dimensions.

        Returns:
            int: The smallest peak any split of these dimensions attains.
        """
        num = len(ham_dims)
        return min(self._peak(ham_dims,
                              [index for index in range(num)
                               if subset >> index & 1])
                   for subset in range(1 << num))

    def test_split_is_optimal(self):
        """
        Tests that the split attains the minimal peak.

        The indices have to be a valid ascending subset of the neighbours and
        the peak they produce has to be the smallest any subset attains.
        """
        cases = [(3, 4, 3, 4),
                 (2, 2, 2),
                 (6, 2, 3, 5),
                 (3, 4, 3),
                 (7, 7, 7, 7, 7),
                 (1, 5, 1, 5),
                 (2, 3, 5, 7, 11, 13)]
        for ham_dims in cases:
            with self.subTest(ham_dims=ham_dims):
                before_ham = _neighbours_before_ham(ham_dims)
                self.assertEqual(sorted(set(before_ham)), list(before_ham))
                self.assertTrue(set(before_ham).issubset(
                    range(len(ham_dims))))
                self.assertEqual(self._minimal_peak(ham_dims),
                                 self._peak(ham_dims, before_ham))

    def test_plain_order_without_a_gain(self):
        """
        Tests that a node whose peak cannot be lowered is left alone.

        With no neighbour, a single neighbour or trivial Hamiltonian bonds
        there is nothing to trade off, so every block stays in front of the
        Hamiltonian, as in the plain sequence of contractions.
        """
        self.assertEqual((), _neighbours_before_ham(()))
        self.assertEqual((0, ), _neighbours_before_ham((5, )))
        self.assertEqual((0, 1, 2), _neighbours_before_ham((1, 1, 1)))

    def test_above_the_exact_cutoff(self):
        """
        Tests the approximate split used above the exact search cutoff.

        Every neighbour still ends up on exactly one of the two sides and the
        peak stays within one dimension of the square root of the total.
        """
        ham_dims = (2, ) * 13
        total = prod(ham_dims)
        before_ham = _neighbours_before_ham(ham_dims)
        before = prod([ham_dims[index] for index in before_ham])
        after = prod([dim for index, dim in enumerate(ham_dims)
                      if index not in before_ham])
        self.assertEqual(total, before * after)
        self.assertLessEqual(max(before, after),
                             max(ham_dims) * total ** 0.5)

    def test_exact_cutoff_is_adjustable(self):
        """
        Tests that max_exact_degree selects between the two searches.

        These dimensions are a case where the cheaper search does not attain
        the minimal peak, so the two branches are told apart by their result.
        """
        ham_dims = (6, 2, 3, 5)
        exact = _neighbours_before_ham(ham_dims, max_exact_degree=4)
        self.assertEqual(self._minimal_peak(ham_dims),
                         self._peak(ham_dims, exact))
        approximate = _neighbours_before_ham(ham_dims, max_exact_degree=3)
        self.assertGreater(self._peak(ham_dims, approximate),
                           self._minimal_peak(ham_dims))

if __name__ == "__main__":
    unitmain()
