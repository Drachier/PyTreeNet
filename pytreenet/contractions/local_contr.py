"""
This module implements the LocalContraction class, which abstracts the
contraction of tensors on one node with their subtree tensors.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Iterator
from enum import Enum
from itertools import product

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..core.node import Node
    from .sandwich_caching import PartialTreeCachDict

class TensorKind(Enum):
    """
    An enumeration of the different kinds of tensors.
    """
    KET = 0
    BRA = 1
    OPERATOR = 2

class LocalContraction:

    def __init__(self,
                 nodes_tensors: list[tuple[Node,npt.NDArray]],
                 subtree_dict: PartialTreeCachDict,
                 cache_id_trafo: Callable = lambda x: x,
                 neighbour_order: list[str] | None = None,
                 ignored_leg: str = "",
                 id_trafos: None | list[Callable] = None,
                 connection_index: int = 0
                 ):
        """
        Initialize a LocalContraction object.

        Args:
        node_tensors (list[tuple[Node,npt.NDArray]]): Pairs of nodes and
            tensors that represent the stack of tensors on one site. The
            stack order of tensors corresponds to the order in this list.
            May be less than the degree of the subtree tensors.
        subtree_dict (PartialTreeCachDict): Contains the subtree tensors
            to be used. Subtree identifiers should correspond to the lowest
            node in the list, possibly after transformin the node's
            identifiers.
        cache_id_trafo (Callable): Maps the identifiers of the lowest node's
            neighbours to the identifiers used in the subtree cache.
        neighbour_order (list[str] | None): The order in which the neighbour
            legs should be in the end. If not supplied, the order of the first
            tensor is used.
        ignored_leg (str): A leg that is not to be contracted. Should
            correspond to the first node in the node list. If `""` no leg is
            ignored.
        id_trafos (None | list[Callable]): A list mapping the neighbour
            identifiers in `neighbour_order` to the neighbours identifiers of
            the node in this position. If None, defaults to the identity.
        connection_index(int): The index of the subtree tensors to which the
            first node in the node list should be connected.

        """
        if len(nodes_tensors) == 0:
            errstr = "No tensors given to contract!"
            raise ValueError(errstr)
        self.node_tensors = nodes_tensors
        self.subtree_dict = subtree_dict
        self.cache_id_trafo = cache_id_trafo
        if neighbour_order is None:
            neighbour_order = self.node_tensors[0][0].neighbouring_nodes()
        self.neighbour_order = neighbour_order
        self.ignored_leg = ignored_leg
        if id_trafos is None:
            id_trafos = [lambda x: x for _ in nodes_tensors]
        else:
            if len(id_trafos) != len(nodes_tensors):
                errstr = f"Need as many `id_trafo` as tensors, ({len(id_trafos)},{len(nodes_tensors)})!"
                raise IndexError(errstr)
        self.id_trafos = id_trafos
        self.connection_index = connection_index
        self.current_tensor = CurrentTensor()
        self.contraction_order = list(range(len(self.node_tensors)))

    def no_ignored_legs(self) -> bool:
        """
        Returns whether there are no ignored legs.
        """
        return self.ignored_leg == ""

    def num_ignored(self) -> int:
        """
        Returns the number of ignored legs.
        """
        if self.no_ignored_legs():
            return 0
        return 1

    def is_ignored(self, neigh_id: str, node_index: int) -> bool:
        """
        Returns whether a given neighbour leg is ignored.
        """
        if self.no_ignored_legs():
            return False
        # We must transform the ignored leg to the current node's
        # identifier system.
        transformed_ignored = self.id_trafos[node_index](self.ignored_leg)
        return neigh_id == transformed_ignored

    def subtree_degree(self) -> int:
        """
        Returns the degree of the subtree tensors.

        The degree is the number of legs on the subtree tensors.
        """
        if len(self.subtree_dict) == 0:
            errstr = "No subtree tensors in the subtree dict!\n"
            errstr += "Likely this node is a leaf and needs special treatment."
            raise ValueError(errstr)
        return next(iter(self.subtree_dict.values())).ndim

    def create_reverse_trafo(self,
                             index: int
                             ) -> Callable:
        """
        Creates the mapping from neighbour identifiers of a node to the first
        node's neighbour identifiers.

        Args:
            index (int): The index of the node and tensor in the list.
        """
        node = self.node_tensors[index][0]
        id_trafo = self.id_trafos[index]
        mapping = {node_neigh: init_neigh
                   for node_neigh, init_neigh in product(node.neighbouring_nodes(),
                                                         self.neighbour_order)
                   if node_neigh == id_trafo(init_neigh)}
        return lambda x: mapping[x]

    def contract_first_tensor(self):
        """
        Contracts the tensor at index 0 in the contraction order.
        """
        contr_index = self.contraction_order[0]
        node, tensor = self.node_tensors[contr_index]
        subtree_leg = self.connection_index + contr_index
        ignored_passed = 0
        for neigh_id in node.neighbouring_nodes():
            if not self.is_ignored(neigh_id, contr_index):
                subtree_tensor = self.subtree_dict.get_entry(neigh_id,
                                                             node.identifier)
                # Since we contract in the order of this node's neighbours,
                # the leg to be contracted is always at position 0 in the tensor
                # unless we have passed ignored leg, then it is at position 1.
                tensor_leg = ignored_passed
                tensor = np.tensordot(tensor, subtree_tensor,
                                      axes=(tensor_leg, subtree_leg))
            else:
                ignored_passed += 1
        self.current_tensor.value = tensor
        nopen = node.nopen_legs()
        max_dim = self.subtree_degree() # The number of open legs on a cached subtree tensor.
        if subtree_leg == 0:
            tensor_kind = TensorKind.KET
        elif subtree_leg == max_dim - 1:
            tensor_kind = TensorKind.BRA
        else:
            tensor_kind = TensorKind.OPERATOR
        self.current_tensor.set_first_tensor_open_legs(ignored_passed,
                                                       nopen,
                                                       tensor_kind)
        self.current_tensor.ignored_legs = list(range(ignored_passed))
        # Now we need to set the neighbour legs.
        rev_traf = self.create_reverse_trafo(contr_index)
        ignored_passed = 0
        # After the above contractions, the legs corresponding to the
        # neighbouring nodes are at the end of the tensor
        legs_before_neighs = nopen + self.num_ignored()
        for index, neigh_id in enumerate(node.neighbouring_nodes()):
            if not self.is_ignored(neigh_id,
                               contr_index):
                leg_index = index - ignored_passed
                range_start = legs_before_neighs + leg_index * max_dim
                range_end = legs_before_neighs + (leg_index + 1) * max_dim
                neigh_legs: list[int | None] = list(range(range_start, range_end))
                neigh_legs[contr_index + self.connection_index] = None
                self.current_tensor.neighbour_legs[rev_traf(neigh_id)] = neigh_legs
            else:
                ignored_passed += 1

    def contract_tensor(self,
                        contr_index: int):
        """
        Contracts the local tensor given by the contraction order.

        Args:
            contr_index (int): The index of the contraction order for which
                the tensor should be contracted.
        """
        # Instead of the contraction order index we use the index in the
        # list of nodes and tensor to work with.
        prev_contr_index = self.contraction_order[contr_index - 1]
        contr_index = self.contraction_order[contr_index]
        node, tensor = self.node_tensors[contr_index]
        contr_ten_legs = []
        curr_ten_legs = []
        # Get the open legs to be contracted
        if contr_index < prev_contr_index:
            # This means we are now contracting a lower tensor in the stack.
            # Thus we return the in legs.
            curr_ten_open = self.current_tensor.in_legs
        else:
            curr_ten_open = self.current_tensor.out_legs
        ten_open = node.open_legs
        if len(curr_ten_open) == len(curr_ten_open):
            # This means a state tensor is contracted
            contr_ten_open = ten_open
        elif len(curr_ten_open) // 2 == len(curr_ten_open):
            # This menas an operator tensor is contracted
            if contr_index < prev_contr_index:
                contr_ten_open = ten_open[-(len(curr_ten_open) // 2):]
            else:
                contr_ten_open = ten_open[:(len(curr_ten_open) // 2)]
        else:
            errstr = "Open leg degree does not fit!"
            raise ValueError(errstr)
        contr_ten_legs.extend(contr_ten_open)
        curr_ten_legs.extend(curr_ten_open)
        # Get the virtual legs of the node that are to be contracted
        id_trafo = self.id_trafos[contr_index]
        subtree_tensor_index = self.connection_index + contr_index
        for neigh_id in node.neighbouring_nodes():
            if not self.is_ignored(neigh_id, contr_index):
                contr_ten_legs.append(node.neighbour_index(neigh_id))
                base_neigh_id = id_trafo(neigh_id)
                curr_leg = self.current_tensor.get_neighbour_leg(base_neigh_id,
                                                                 subtree_tensor_index)
                curr_ten_legs.append(curr_leg)
                self.current_tensor.set_neighbour_leg_none(base_neigh_id,
                                                           contr_index)
        curr_tensor = np.tensordot(self.current_tensor.value,
                                   tensor,
                                   axes=(curr_ten_legs,contr_ten_legs))
        # Now all the legs that were already in the tensor would move forward
        # We have to adjust the other legs
        self.current_tensor.adjust_legs(curr_ten_legs)
        # Now we add the new legs to the current tensor
        # They will be at the end of the tensor
        num_rem_legs = self.current_tensor.value.ndim - len(curr_ten_legs)
        # The ignored leg will be the first, as it was a virtual leg
        ignored_legs = list(range(num_rem_legs,
                                  num_rem_legs + self.num_ignored()))
        self.current_tensor.ignored_legs.extend(ignored_legs)
        # Finally, we add the new open legs. They will be the very last ones
        new_open_legs = list(range(num_rem_legs + self.num_ignored(),
                                   curr_tensor.ndim))
        self.current_tensor.set_open_legs_after_contr(new_open_legs)
        self.current_tensor.value = curr_tensor

class CurrentTensor:
    """
    A class keeping track of the current numerical result of a contraction.
    """

    def __init__(self) -> None:
        """
        Initialises an empty CurrentTensor.
        """
        self.value = np.ndarray([1])
        self.in_legs: list[int | None] = []
        self.out_legs: list[int | None] = []
        # The order of these legs is exactly as the order of node,tensor pairs
        self.neighbour_legs: dict[str,list[int | None]] = {}
        self.ignored_legs: list[int] = []

    def get_neighbour_leg(self,
                          neigh_id: str,
                          contr_index: int
                          ) -> int:
        """
        Returns the leg index of this tensor corresponding to the original
        specified subtree tensor leg.

        The subtree tensor is specified by the original neighbour identifier
        and the desired leg of that subtree tensor as an index.

        Args:
            neigh_id (str): The identifier of this neighbour. This is the base
                identifier, i.e. the identifier specified in the neighbour
                order.
            contr_index (int): The index of the leg on the subtree tensor.

        Returns:
            int: The leg index of this tensor corresponding to the original
                specified subtree tensor leg.
        """
        out = self.neighbour_legs[neigh_id][contr_index]
        if out is None:
            errstr = f"The leg {contr_index} of subtree to neighbour {neigh_id}"
            errstr += "was already contracted!"
            raise IndexError(errstr)
        return out

    def set_neighbour_leg_none(self,
                               neigh_id: str,
                               contr_index: int):
        """
        Sets the specified subtree leg to None.

        The subtree tensor is specified by the original neighbour identifier
        and the desired leg of that subtree tensor as an index.

        Args:
            neigh_id (str): The identifier of this neighbour. This is the base
                identifier, i.e. the identifier specified in the neighbour
                order.
            contr_index (int): The index of the leg on the subtree tensor.
        """
        self.neighbour_legs[neigh_id][contr_index] = None

    def adjust_legs(self,
                    contracted_legs: list[int]):
        """
        Reduces the leg indices by the number of legs that wetre removed due
        to contraction.

        Args:
            contracted_legs (list[int]): The leg indices that disappeared due
                to a contraction.
        """
        contracted_legs.sort()
        self.in_legs = [_new_leg_val(leg_val, contracted_legs)
                        for leg_val in self.in_legs]
        self.out_legs = [_new_leg_val(leg_val, contracted_legs)
                         for leg_val in self.out_legs]
        self.neighbour_legs = {neigh_id: [_new_leg_val(leg_val, contracted_legs)
                                     for leg_val in neigh_legs]
                               for neigh_id, neigh_legs in self.neighbour_legs.items()}
        new_ign_legs = []
        for ignored_leg in self.ignored_legs:
            new_val = _new_leg_val(ignored_leg,
                                   contracted_legs)
            if new_val is None:
                errstr = "Ignored leg shouldn't be contracted!"
                raise ValueError(errstr)
            new_ign_legs.append(new_val)
        self.ignored_legs = new_ign_legs

    def set_first_tensor_open_legs(self,
                                   num_ignored: int,
                                   num_open: int,
                                   tensor_kind: TensorKind):
        """
        Sets the tensor's physical legs when the first tensor is contracted.

        Args:
            num_ignored (int): The number of legs towards neighbours that were
                ignored during the contraction.
            num_open (int): The number of open/physical legs of the tensor
                that was contracted.
            tensor_kind (TensorKind): The kind of tensor that was contracted
                first; Ket, Bra or Operator. This decides about the
                distribution of in and out legs.
        """
        if tensor_kind is TensorKind.KET:
            # In this case we have the ket |> tensor contracted first.
            self.out_legs = list(range(num_ignored,
                                        num_ignored + num_open))
            # No open in legs in this case.
        elif tensor_kind is TensorKind.BRA:
            # In this case it is the bra <| tensor contracted first.
            self.in_legs = list(range(num_ignored,
                                        num_ignored + num_open))
            # No open out legs in this case.
        elif tensor_kind is TensorKind.OPERATOR:
            # In this case it is a middle, i.e. operator tensor contracted
            # We have the same convention as matrices, i.e. out legs first.
            assert num_open % 2 == 0, "Operator tensors must have even number of open legs!"
            half = num_open // 2
            self.out_legs = list(range(num_ignored,
                                        num_ignored + half))
            self.in_legs = list(range(num_ignored + half,
                                        num_ignored + num_open))
        else:
            raise ValueError("Invalid TensorKind!")

    def set_open_legs_after_contr(self,
                                  open_legs: list[int]):
        """
        Sets the open legs after a contraction.

        Args:
            open_legs (list[int]): The index values of the new open legs.
        """
        if len(self.in_legs) == 0 and len(self.out_legs) == 0:
            if len(open_legs) != 0:
                errstr = "Invalid number of open legs!"
                raise IndexError(errstr)
        elif len(self.in_legs) != 0 and self.in_legs[0] is None:
            self.in_legs = open_legs
        elif len(self.out_legs) != 0 and self.out_legs[0] is None:
            self.out_legs = open_legs
        elif len(self.in_legs) == 0:
            self.in_legs = open_legs
        elif len(self.out_legs) == 0:
            self.out_legs = open_legs
        else:
            errstr = "Invalid contraction of open legs!"
            raise ValueError(errstr)

def _new_leg_val(leg_value: int | None,
                   sorted_rem_legs: list[int]
                   ) -> int | None:
    """
    Returns the amount new value of the given leg.

    Args:
        leg_value (int | None): The current value of the leg.
        sorted_rem_legs (list[int]): A sorted list of the
            legs that were contracted.
    """
    if leg_value is None:
        # The leg is already gone and stays gone.
        return None
    for i, rem_val in enumerate(sorted_rem_legs[:-1]):
        if leg_value == rem_val:
            # This leg was contracted
            return None
        if leg_value > rem_val and leg_value < sorted_rem_legs[i+1]:
            # This means i many legs that come before this one were
            # contracted.
            return leg_value - i
    # Treat the final one special
    if leg_value == sorted_rem_legs[-1]:
        # This leg was contracted
        return None
    if leg_value > sorted_rem_legs[-1]:
        # Larger than all contracted legs
        return leg_value - len(sorted_rem_legs)
    raise ValueError(f"Invalid leg value {leg_value}!")
