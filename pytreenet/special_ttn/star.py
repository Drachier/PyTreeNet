
from typing import Union, Self

from numpy import ndarray, zeros

from ..core.ttn import TreeTensorNetwork
from ..core.node import Node
from ..ttns.ttns import TreeTensorNetworkState
from ..ttno.ttno_class import TreeTensorNetworkOperator
from ..util.ttn_exceptions import non_negativity_check
from .util import check_product_state_parameters

class StarTreeTensorNetwork(TreeTensorNetwork):
    """
    A tree in the form of a star.

    A central node is connected to multiple lines of MPS-like tensor chains.
    A common example is the T-shaped tree:

        C02 -- C01 -- C00 -- R -- C10 -- C11 -- C12
                             |
                             |-- C20 -- C21 -- C22
    """

    def __init__(self,
                 central_node_identifier: str = "center",
                 non_center_prefix: str = "node"):
        """
        Initialise a StarTreeTensorNetwork object.

        Args:
            central_node_identifier (str): Identifier of the central node.
                Defaults to "center".
            non_center_prefix (str): Prefix of the nodes not in the central chain.
                The identifiers will have the form `prefix + chain index + _ + position`.
                Defaults to "node".

        """
        super().__init__()
        self.central_node_identifier = central_node_identifier
        self.non_center_prefix = non_center_prefix
        self.chains = []

    @property
    def central_node_id(self):
        """
        Returns the central node identifier.
        """
        return self.central_node_identifier

    @property
    def central_node(self) -> Node:
        """
        Returns the central node.
        """
        return self.nodes[self.central_node_id]

    def chain_id(self,
                 chain_index: int,
                 index_on_chain: int) -> str:
        """
        Creates the identifier of the node at the specified coordinates.
        """
        return f"{self.non_center_prefix}{chain_index}_{index_on_chain}"

    def node_id(self,
                chain_index: int | None,
                index_on_chain: int | None
                ) -> str:
        """
        Returns the identifiier of the node at the specified coordinates.

        Args:
            chain_index (int | None): The index of the chain. If None, the
                central node identifier will be returned.
            index_on_chain (int | None): The index of the node on the chain.
                If None, the central node identifier will be returned.

        Returns:
            str: The identifier of the node.
        
        Raises:
            IndexError: If either index is out of bounds.
        """
        if chain_index is None or index_on_chain is None:
            return self.central_node_id
        if chain_index >= self.num_chains():
            raise IndexError("Chain index is out of bounds!")
        if index_on_chain >= self.chain_length(chain_index):
            raise IndexError("Index on chain is out of bounds!")
        return self.chains[chain_index][index_on_chain].identifier

    def num_chains(self) -> int:
        """
        Returns the current number of chains.
        """
        return len(self.chains)

    def chain_length(self, chain_index: int) -> int:
        """
        Returns the length of the specified chain.
        """
        return len(self.chains[chain_index])

    def add_center_node(self, tensor: ndarray):
        """
        Add a new node to the central chain.
        """
        center_node = Node(identifier=self.central_node_identifier)
        self.add_root(center_node, tensor)

    def _add_chain(self, tensor: ndarray,
                   parent_leg: Union[int,None] = None,
                   identifier: str | None = None):
        """
        Adds a new chain to the star and directly adds the first node.
        
        Args:
            tensor (ndarray): The tensor to be added.
            parent_leg (Union[int,None]): The leg of the parent node to connect
                to. If None, the node will be connected to the first open leg
                of the parent.
            identifier (str): Identifier for the first node in the new chain.
                If None, it will be generated automatically based on the chain
                index.

        """
        parent_id = self.central_node_id
        chain_index = self.num_chains()
        chain = []
        if identifier is None:
            identifier = self.chain_id(chain_index, 0)
        new_node = Node(identifier=identifier)
        if parent_leg is None:
            parent_id = self.central_node_id
            parent_node = self.nodes[parent_id]
            parent_leg = parent_node.nvirt_legs()
        self.add_child_to_parent(new_node,tensor,0,
                                 parent_id,parent_leg)
        chain.append(new_node)
        self.chains.append(chain)

    def add_chain_node(self, tensor: ndarray,
                       chain_index: int,
                       parent_leg: Union[int,None] = None,
                       identifier: str | None = None):
        """
        Adds a node to one of the chains.

        Args:
            tensor (ndarray): The tensor to be added.
            chain_index (int): The index of the chain. If the chain does not
                exist, it will be created.
            parent_leg (Union[int,None]): The leg of the parent node to connect
                to. If None, the node will be connected to the first open leg
                of the parent.
            identifier (str): Identifier for the new node. If None, it will be
                generated automatically based on the chain index and the
                position on the chain.

        """
        if chain_index > self.central_node.nlegs():
            raise ValueError("Chain index is too high!")
        if chain_index > self.num_chains():
            raise ValueError("This is not the next chain index!")
        if chain_index == self.num_chains():
            self._add_chain(tensor,parent_leg,identifier=identifier)
        else:
            parent_node = self.chains[chain_index][-1]
            parent_id = parent_node.identifier
            if identifier is None:
                index_on_chain = self.chain_length(chain_index)
                identifier = self.chain_id(chain_index,index_on_chain)
            new_node = Node(identifier=identifier)
            if parent_leg is None:
                parent_leg = parent_node.nvirt_legs()
            self.add_child_to_parent(new_node,tensor,0,
                                     parent_id,parent_leg)
            self.chains[chain_index].append(new_node)

    @classmethod
    def from_tensor_lists(cls,
                          center_tensor: ndarray,
                          tensors: list[list[ndarray]],
                          central_node_identifier: str = "center",
                          identifiers: list[list[str]] | None = None,
                          non_center_prefix: str = "node"
                          ) -> Self:
        """
        Creates a StarTreeTensorNetwork from lists of tensors.

        Args:
            center_tensor (ndarray): The tensor for the central node.
            tensors (list[list[ndarray]]): A list of lists of tensors.
                Each list corresponds to one of the chains. The first
                leg will always be attached towards the central node,
                while the second leg is attached to the next node
                in the chain. All other legs remain open.
            central_node_identifier (str): Identifier of the central node.
                Defaults to "center".
            identifiers (list[list[str]]): A list of lists of identifiers.
                Each list corresponds to the list of tensors in the same
                position in the `tensors` list. If None, the identifiers will
                be generated automatically.
            non_center_prefix (str): Prefix of the nodes not in the central chain.
                The identifiers will have the form `prefix + chain index + _ + position`.
                Defaults to "node".
            
        Returns:
            StarTreeTensorNetwork: The created star tree tensor network.

        Raises:
            ValueError: If the number of identifiers does not match the number
                of tensors.
        """
        ttn = cls(central_node_identifier=central_node_identifier,
                   non_center_prefix=non_center_prefix)
        ttn.add_center_node(center_tensor)
        for chain_index, chain_tensors in enumerate(tensors):
            if identifiers is not None:
                chain_identifiers = identifiers[chain_index]
                if len(chain_identifiers) != len(chain_tensors):
                    errstr = "Number of identifiers does not match number of tenors!"
                    raise ValueError(errstr)
            for position, tensor in enumerate(chain_tensors):
                if identifiers is not None:
                    identifier = chain_identifiers[position]
                else:
                    # The method will take care of generating the identifier.
                    identifier = None
                ttn.add_chain_node(tensor, chain_index,
                                    identifier=identifier)
        return ttn

class StarTreeTensorState(StarTreeTensorNetwork,TreeTensorNetworkState):
    """
    A state with the star topology.
    """

    def __init__(self,
                 central_node_identifier: str = "central",
                 non_center_prefix: str = "node"):
        """
        Initialise a StarTreeTensorNetworkState object.

        Args:
            central_node_identifier (str): Identifier of the central node.
                Defaults to "central".
            non_center_prefix (str): Prefix of the nodes not in the central chain.
                The identifiers will have the form `prefix + chain index + _ + position`.
                Defaults to "node".

        """
        super().__init__(central_node_identifier,non_center_prefix)

    @classmethod
    def constant_product_state(cls,
                               state_value: int,
                               dimension: int,
                               chain_length: int,
                               num_chains: int,
                               node_prefix: str = "site",
                               ) -> Self:
        """
        Generates a STTS that corresponds to a product state with the same
        value on all sites.

        Args:
            state_value (int): The value of the state.
            dimension (int): The dimension of the state.
            chain_length (int): The length of the chains.
            num_chains (int): The number of chains.
            node_prefix (str): The prefix of the node identifiers.
                Defaults to "site".
        
        Returns:
            StarTreeTensorState: The product state.

        """
        check_product_state_parameters(state_value,dimension)
        non_negativity_check(num_chains,"number of chains")
        non_negativity_check(chain_length,"chain length")
        state = cls(non_center_prefix=node_prefix)
        local_state = zeros(dimension, dtype=complex)
        local_state[state_value] = 1
        shape = tuple([1]*num_chains+[dimension])
        center_tensor = local_state.reshape(shape)
        state.add_center_node(center_tensor)
        for i in range(num_chains):
            for j in range(chain_length):
                if j == chain_length - 1:
                    tensor = local_state.reshape(1,2)
                else:
                    tensor = local_state.reshape((1,1,2))
                state.add_chain_node(tensor,i)
        return state

class StarTreeOperator(StarTreeTensorNetwork,TreeTensorNetworkOperator):
    """
    An operator with the star topology.
    """

    def __init__(self,
                 central_node_identifier: str = "central",
                 non_center_prefix: str = "node"):
        super().__init__(central_node_identifier,non_center_prefix)
