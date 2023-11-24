from __future__ import annotations
from copy import deepcopy

import numpy as np

from ..ttn import TreeTensorNetwork
from ..ttns import TreeTensorNetworkState
from ..node import Node
from ..ttno.ttno import TTNO

class MatrixProductTree(TreeTensorNetwork):
    """
    A tree tensor network in the form of a chain.
     In principle every node can have an arbitrary number
     of legs. Important special cases are the MPS and MPO.

     Mostly used for testing.
    """

    def __init__(self, length: int, tensor: np.ndarray, 
        node_prefix: str = "site", first_site: int = 0):
        """Intialises a constant matrix product tree/chain

        Args:
            lenght (int): Number of nodes in the chain
            tensor (np.ndarray): A tensor to which every node is initialised.
                The leg order is `(left_leg, right_leg, open_legs)`.
            node_prefix (str, optional): Nodes will have an identifier of the
             form `"node_prefix "+ number`. Defaults to `"site"`.
            first_site (int, optional): The number that should be associated to
             the leftmost site. Defaults to `0`.
        """
        super().__init__()
        self._node_prefix = node_prefix
        self._first_site = first_site
        
        for site in range(first_site, first_site + length):
            identifier = self._node_prefix + str(site)
            local_state = deepcopy(tensor)
            # A node is only linked to an actual tensor, once it is added to a TTN.
            node = Node(identifier = identifier)
            if site == first_site:
                # Open boundary conditions on first site        
                self.add_root(node, local_state[0])
            else:
                parent_id = self._node_prefix + str(site - 1)
                if site == first_site + length - 1:
                    # Open boundary conditions on last site
                    self.add_child_to_parent(local_state, tensor[:,0], 0, parent_id, 2)
                else:
                    if site == 1:
                        # Due to boundary condition on first site
                        self.add_child_to_parent(node, local_state, 0, parent_id, 1)
                    else:
                        self.add_child_to_parent(node, local_state, 0, parent_id, 2)

class MatrixProductState(MatrixProductTree, TreeTensorNetworkState):
    def __init__(self,  length: int, tensor: np.ndarray, 
        node_prefix: str = "site", first_site: int = 0):
        if tensor.ndim != 3:
            errstr = "The generating tensor of an MPS must have exactly 3 legs!\n"
            errstr += f" {tensor.ndim} != 3"
            raise ValueError(errstr)
        super().__init__(length, tensor, node_prefix=node_prefix, first_site=first_site)

class MatrixProductOperator(MatrixProductTree, TTNO):
    def __init__(self,  length: int, tensor: np.ndarray,
        node_prefix: str = "site", first_site: int = 0):
        if tensor.ndim != 4:
            errstr = "The generating tensor of an MPO must have exactly 4 legs!\n"
            errstr += f" {tensor.ndim} != 4"
            raise ValueError(errstr)
        super().__init__(length, tensor, node_prefix=node_prefix, first_site=first_site)

def constant_product_state(state_value: int,
                           dimension: int,
                           num_sites: int) -> MatrixProductState:
    """
    Generates an MPS that corresponds to a product state with the same value
     at every site.

    Args:
        state_value (int): The state's value at every site.
        dimension (int): The local dimension of the MPS.
        num_sites (int): The number of sites in the MPS.

    Raises:
        ValueError: If state_value is negative or dimension non-positive.
         Also raised if state value is larger than dimension.

    Returns:
        MatrixProductState: MPS representing a product state of the form
         |psi> = |value> otimes |value> otimes ... |value>
    """
    if dimension < 1:
        errstr = f"Dimension of a state must be positive not {dimension}!"
        raise ValueError(errstr)
    if state_value >= dimension:
        errstr = "State value cannot be larger than the state's dimension!"
        raise ValueError(errstr)
    elif state_value < 0:
        errstr = f"State value must be non-negative not {state_value}!"
        raise ValueError(errstr)
    single_site_tensor = np.zeros(dimension)
    single_site_tensor[state_value] = 1
    single_site_tensor = np.reshape(single_site_tensor, (1,1,dimension))
    return MatrixProductState(num_sites, single_site_tensor)
