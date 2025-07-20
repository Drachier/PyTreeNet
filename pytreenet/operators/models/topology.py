"""
Implementation of the topology aspects of a quantum system.

Note, this is the actual physical topology of the system, not the
representation in the tree tensor network.
"""
from enum import Enum

class Topology(Enum):
    """
    Enumeration for different types of topologies.
    """
    CHAIN = "chain"
    TTOPOLOGY = "t_topology"
    CALEY = "caley_tree"
    SQUARE = "square_lattice"