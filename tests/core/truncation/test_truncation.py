from unittest import TestCase, main

from typing import Dict

from numpy import ndarray, diag, allclose, eye
from numpy.linalg import qr

from pytreenet.random.random_matrices import random_unitary_matrix, crandn
from pytreenet.util.tensor_splitting import SVDParameters
from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.core.node import Node
from pytreenet.core.truncation.svd_truncation import svd_truncation

def singular_value_matrix() -> ndarray:
    """
    Returns a matrix with singular values on the diagonal.
    """
    return diag([1,0.5,0.1,0.05])

def node_id(site: int) -> str:
    """
    Generates a node id for a given site.
    """
    return f"site_{site}"

def generate_tensors() -> Dict[str,ndarray]:
    """
    Generates isometric tensors for every site.
    """
    tensors = {}
    tensor = crandn((4**3,4))
    q, _ = qr(tensor)
    # The last leg is the special leg
    tensors[node_id(0)] = q.reshape((4,4,4,4))
    tensors[node_id(1)] = random_unitary_matrix(4)
    tensor = crandn((4**2,4))
    q, _ = qr(tensor)
    # The last leg is the special leg
    tensors[node_id(2)] = q.reshape((4,4,4))
    tensors[node_id(3)] = random_unitary_matrix(4)
    tensors[node_id(4)] = random_unitary_matrix(4)
    return tensors

class TestSVDSplitting(TestCase):

    def test_unitarity_of_tensors(self):
        """
        Test if the tensors are isometric.
        """
        dim = 4
        tensors = generate_tensors()
        identity = eye(dim)
        for node_ide, tensor in tensors.items():
            if node_ide not in [node_id(0),node_id(2)]:
                matrix = tensor
            elif node_ide == node_id(0):
                matrix = tensor.reshape(dim**3,dim)
            else:
                matrix = tensor.reshape(dim**2,dim)
            result = matrix.T.conj()@matrix
            self.assertTrue(allclose(result,identity))

    def test_svd_splitting_0_1(self):
        """
        Test if the SVD splitting works for the first and second site.
        """
        # Build reference tree
        ref_tree = TreeTensorNetwork()
        tensors = generate_tensors()
        ref_tree.add_root(Node(identifier=node_id(0)), tensors[node_id(0)])
        ref_tree.add_child_to_parent(Node(identifier="svals"), singular_value_matrix(),
                                     0, node_id(0), 3)
        ref_tree.add_child_to_parent(Node(identifier=node_id(1)), tensors[node_id(1)],
                                     0, "svals", 1)
        ref_tree.add_child_to_parent(Node(identifier=node_id(2)), tensors[node_id(2)],
                                        2, node_id(0), 1)
        ref_tree.add_child_to_parent(Node(identifier=node_id(3)), tensors[node_id(3)],
                                        0, node_id(0), 2)
        ref_tree.add_child_to_parent(Node(identifier=node_id(4)), tensors[node_id(4)],
                                        0, node_id(2), 1)
        ref_tree.contract_nodes("svals", node_id(0), new_identifier=node_id(0))
        ref_tree.canonical_form(node_id(0))
        svd_params = SVDParameters(rel_tol=0.15)
        ref_tree = svd_truncation(ref_tree, svd_params)
        self.assertEqual(set([2,4]), set(ref_tree.nodes[node_id(0)].shape))
        self.assertEqual(set([2,4]), set(ref_tree.nodes[node_id(1)].shape))
        for node_ident, nodes in ref_tree.nodes.items():
            if node_ident not in [node_id(0),node_id(1)]:
                self.assertEqual(set([4]), set(nodes.shape))



if __name__ == "__main__":
    main()
