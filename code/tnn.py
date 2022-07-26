import numpy as np
import treelib as tl

class tnn(tl.Tree):
    """
    A tree tensor network is made up of a tree. Each node contains a tensor,
    that is part of the network.
    """
    
    