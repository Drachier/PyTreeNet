"""
Implements an enumeration for different special types of tensor network
structures.
"""
from enum import Enum

class TTNStructure(Enum):
    """
    Enumeration for different types of tensor network structures.
    """
    MPS = "mps"
    BINARY = "binary"
    TSTAR = "tstar"
    
