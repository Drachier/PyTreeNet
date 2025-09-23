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

def random_tensor_node(shape,
                       identifier: str = "",
                       seed = None,
                       dtype: np.dtype = np.complex128):
    """
    Creates a tensor node with an a random associated tensor with shape=shape.
    """
    rand_tensor = crandn(shape, seed=seed).astype(dtype)
    return (Node(tensor=rand_tensor, identifier=identifier), rand_tensor)
