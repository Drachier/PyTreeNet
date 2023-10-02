from __future__ import annotations

from copy import deepcopy
import numpy as np

from .ttn import TreeTensorNetwork
from .operators.tensorproduct import TensorProduct
from .node import random_tensor_node
from util import crandn

class TreeTensorNetworkState(TreeTensorNetwork):
    """
    This class holds methods commonly used with tree tensor networks representing a state.
    """

    def __init__(self):
        """
        Initialises in the same way as a normal TTN.
        """
        super().__init__()

    def scalar_product(self) -> complex:
        """
        Computes the scalar product of this TTNS

        Returns:
            complex: The resulting scalar product <TTNS|TTNS>
        """
        # Very inefficient, fix later without copy
        ttn = deepcopy(self)
        return ttn.contract_two_ttn(ttn.conjugate())

    def single_site_operator_expectation_value(self, node_id: str, operator: np.ndarray,
                                               canon: bool=False) -> complex:
        """
        Find the expectation value of this TTNS given the single-site operator acting on
         the node specified.
        Assumes the node has only one open leg.

        Args:
            node_id (str): The identifier of the node, the operator is applied to.
            operator (np.ndarray): The operator of which we determine the expectation value.
             Note that the state will be contracted with axis/leg 0 of this operator.
            canon (bool, optional): Whether the node is the orthogonality center of the TTNS.
                                     Defaults to False.

        Returns:
            complex: The resulting expectation value < TTNS| Operator| TTN >
        """
        if canon:
            tensor = deepcopy(self.tensors[node_id])
            tensor_op = np.tensordot(tensor, operator, axes=(-1,0))
            tensor_conj = tensor.conj()
            legs = tuple(range(tensor.ndim))
            return complex(np.tensordot(tensor_op, tensor_conj, axes=(legs,legs)))

        tensor_product = TensorProduct({node_id: operator})
        return self.operator_expectation_value(tensor_product)

    def operator_expectation_value(self, operator: TensorProduct) -> complex:
        """
        Finds the expectation value of the operator specified, given this TTNS.

        Args:
            operator (TensorProduct): A TensorProduct representing the operator
             as many single site operators.

        Returns:
            complex: The resulting expectation value < TTNS | operator | TTNS>
        """
        # Very inefficient, fix later without copy
        ttn = deepcopy(self)
        conj_ttn = ttn.conjugate()
        for node_id, single_site_operator in operator.items():
            ttn.absorb_into_open_legs(node_id, single_site_operator)
        return ttn.contract_two_ttn(conj_ttn)

def random_small_ttns() -> TreeTensorNetworkState:
    """
    Generates a small TreeTensorNetworkState consisting of three nodes:
     The root with identifier `"root"` and its two children with identifiers
     `"c1"`and `"c2"`. The associated tensors are random, but its dimensions are set.
    """
    random_ttns = TreeTensorNetworkState()
    random_ttns.add_root(Node(identifier="root"), crandn((5,6,2)))
    random_ttns.add_child_to_parent(Node(identifier="c1"),
        crandn((5,3)), 0, "root", 0)
    random_ttns.add_child_to_parent(Node(identifier="c2"),
        crandn((6,4)), 0, "root", 1)
    return random_ttns

def random_big_ttns(option: str) -> TreeTensorNetworkState:
    """
    Generates a big TTNS with identifiers of the form `"site" + int`.
     The identifiers and dimensions are set, but the associated tensors
     are random.

    Args:
        option (str): A parameter to choose between different topologies and
         dimensions.
    """

    if option == "same_dimension":
        # All dimensions virtual and physical are initially the same
        # We need a ttn to work on.
        node1, tensor1 = random_tensor_node((2,2,2,2), identifier="site1")
        node2, tensor2 = random_tensor_node((2,2,2), identifier="site2")
        node3, tensor3 = random_tensor_node((2,2), identifier="site3")
        node4, tensor4 = random_tensor_node((2,2,2), identifier="site4")
        node5, tensor5 = random_tensor_node((2,2), identifier="site5")
        node6, tensor6 = random_tensor_node((2,2,2,2), identifier="site6")
        node7, tensor7 = random_tensor_node((2,2), identifier="site7")
        node8, tensor8 = random_tensor_node((2,2), identifier="site8")

        random_ttns = TreeTensorNetworkState()

        random_ttns.add_root(node1, tensor1)
        random_ttns.add_child_to_parent(node2, tensor2, 0, "site1", 0)
        random_ttns.add_child_to_parent(node3, tensor3, 0, "site2", 1)
        random_ttns.add_child_to_parent(node4, tensor4, 0, "site1", 1)
        random_ttns.add_child_to_parent(node5, tensor5, 0, "site4", 1)
        random_ttns.add_child_to_parent(node6, tensor6, 0, "site1", 2)
        random_ttns.add_child_to_parent(node7, tensor7, 0, "site6", 1)
        random_ttns.add_child_to_parent(node8, tensor8, 0, "site6", 2)

    return random_ttn
