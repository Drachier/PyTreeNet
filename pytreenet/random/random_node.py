"""
Module to generate a random node.
"""
import numpy as np
from ..core.node import Node
from ..core.graph_node import GraphNode
from .random_matrices import crandn

def random_graph_node() -> GraphNode:
    """
    Create a graph node with a random identifier.
    """
    return GraphNode()

def random_tensor_node(shape, identifier: str = "", dtype: np.dtype = np.complex128):
    """
    Creates a tensor node with an a random associated tensor with shape=shape.
    """
    rand_tensor = crandn(shape)
    return (Node(tensor=rand_tensor.astype(dtype), identifier=identifier), rand_tensor.astype(dtype))
