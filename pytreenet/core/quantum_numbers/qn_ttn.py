"""
Implements quantum numbers for tree tensor networks.
"""
from itertools import product

from numpy import ndarray, sum as npsum

from ..ttn import TreeTensorNetwork
from .qn_node import QNNode

class QNTTN(TreeTensorNetwork):
    """
    Quantum number tree tensor network class.
    Inherits from TreeTensorNetwork and uses QNNode for nodes.
    
    Note: To use the construction functions always use a tuple of array and
    quantum numbers corresponding to that tensor, instead of only the array.
    """

    def __init__(self):
        super().__init__()
        self._nodes: dict[int, QNNode]

    def ensure_shape_matching(self,
                              new_tensor, tensor_leg,
                              old_node, old_leg,
                              new_node_id = None):
        # Since QN nodes are always attached using a tuple of tensor and 
        # quantum numbers, we need to hand over only the tensor to the
        # super class.
        return super().ensure_shape_matching(new_tensor[0], tensor_leg,
                                             old_node, old_leg,
                                             new_node_id)
        # TODO: Implement check for compatible QN

    def find_open_qn(self,
                     node_ids: list[str]
                     ) -> list[ndarray]:
        """
        Find the quantum numbers corresponding to the contraction.

        Args:
            node_ids (list[str]): List of node identifiers defining a sub tree
                tensor network.
        
        Returns:
            list[numpy.ndarray]: List of quantum numbers for the node that
                corresponds to the contraction of the given nodes.
        """
        open_qns = []
        for node_id in node_ids:
            node = self.nodes[node_id]
            for neighbour_id in node.neighbouring_nodes():
                if neighbour_id not in node_ids:
                    open_qns.append(node.get_neighbour_qn(neighbour_id))
            for leg_ind in node.open_legs:
                open_qns.append(node.qn[leg_ind])
        return open_qns

def num_nonzero(quantum_numbers: list[ndarray]) -> int:
    """
    The number of non-zero elements to which these quantum numbers correspond.
    
    Args:
        quantum_numbers (list[numpy.ndarray]): List of quantum numbers.
        
    Returns:
        int: Number of non-zero elements.
    """
    count = 0
    for element_qn in product(*quantum_numbers):
        if npsum(element_qn) == 0:
            count += 1
    return count
