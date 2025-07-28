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

    def num_sites(self,
                  system_size: int) -> int:
        """
        Returns the number of sites in the topology.

        Args:
            system_size (int): The characteristic size of the system.
        
        Returns:
            int: The number of sites in the topology.
        """
        if self == Topology.CHAIN:
            return system_size
        if self == Topology.TTOPOLOGY:
            return system_size * 3
        if self == Topology.CALEY:
            return sum(2 ** i for i in range(system_size))
        if self == Topology.SQUARE:
            return system_size ** 2
        raise ValueError(f"Unknown topology: {self}")
