"""
Module to generate a random node.
"""

from ..core.node import Node
from ..core.graph_node import GraphNode
from ..util.std_utils import crandn

def random_graph_node() -> GraphNode:
    """
    Create a graph node with a random identifier.
    """
    return GraphNode()

def random_tensor_node(shape, identifier: str = ""):
    """
    Creates a tensor node with an a random associated tensor with shape=shape.
    """
    rand_tensor = crandn(shape)
    return (Node(tensor=rand_tensor, identifier=identifier), rand_tensor)
