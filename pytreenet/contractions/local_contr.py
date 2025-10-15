"""
This module implements the LocalContraction class, which abstracts the
contraction of tensors on one node with their subtree tensors.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable
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

    def state_tensor(self) -> bool:
        """
        Returns whether the tensor is a state tensor (ket or bra).
        """
        return self in (TensorKind.KET, TensorKind.BRA)

class OpenLegKind(Enum):
    """
    An enumeration of the different kinds of open legs.
    """
    IN = 0
    OUT = 1

    def opposite(self) -> OpenLegKind:
        """
        Returns the opposite kind of open leg.
        """
        if self is OpenLegKind.IN:
            return OpenLegKind.OUT
        elif self is OpenLegKind.OUT:
            return OpenLegKind.IN
        else:
            errstr = f"Invalid `OpenLegKind` {self}!"
            raise ValueError(errstr)

def valid_contraction_order(order: list[int]) -> bool:
    """
    Checks whether the given contraction order is valid.

    A contraction order is valid if for every tensor in the order,
    it is either the highest or lowest in the stack of already contracted
    tensors.

    Args:
        order (list[int]): The contraction order to check.

    Returns:
        bool: True if the contraction order is valid, False otherwise.
    """
    if len(order) == 0:
        return False
    current_min = order[0]
    current_max = order[0]
    for index in order[1:]:
        if index < current_min:
            if index != current_min - 1:
                return False
            current_min = index
        elif index > current_max:
            if index != current_max + 1:
                return False
            current_max = index
        else:
            return False
    return True

class FinalTransposition(Enum):
    """
    An enumeration of the different ways to transpose the final tensor.
    
    STANDARD: The final tensor is brought into the conventional order of legs.
        That is, all legs pointing to the neighbours in the order specified by
        `neighbour_order`, followed by all out legs and then all in legs.
    NONE: The final tensor is not transposed.
    """
    STANDARD = 0
    NONE = 1
    IGNOREDFIRST = 3

class LocalContraction:

    def __init__(self,
                 nodes_tensors: list[tuple[Node,npt.NDArray]],
                 subtree_dict: PartialTreeCachDict,
                 node_identifier: str | None = None,
                 neighbour_order: list[str] | None = None,
                 ignored_leg: str = "",
                 id_trafos: None | list[Callable] = None,
                 connection_index: int = 0,
                 contraction_order: list[int] | None = None,
                 highest_tensor: TensorKind = TensorKind.OPERATOR
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
        node_identifier (str | None): The identifier of this node as used
            in the subtree cache. If None, will be the identifier of the
            first node in `node_tensors`. Defaults to None.
        neighbour_order (list[str] | None): The order in which the neighbour
            legs should be in the end. These are the identifiers used in the
            subtree cache. If not supplied, the order of the first tensor is
            used.
        ignored_leg (str): A leg that is not to be contracted. Should
            correspond to the first node in the node list. If `""` no leg is
            ignored.
        id_trafos (None | list[Callable]): A list mapping the neighbour
            identifiers in `neighbour_order` to the neighbours identifiers of
            the node in this position. Also maps the identifier used for the
            subtrees to the identifier of the node in this position. If None,
            defaults to the identity.
        connection_index(int): The index of the subtree tensors to which the
            first node in the node list should be connected.
        contraction_order (list[int] | None): The order in which the tensors
            should be contracted. If None, the order of the list is used.
        highest_tensor (TensorKind): The kind of the highest tensor in the
            stack. This is needed to determine the number of open legs in a
            few special cases, if there is only a single tensor in the stack
            and no subtree tensors. In this case, if the last tensor is a bra
            tensor, this must be set to `TensorKind.BRA`, otherwise to
            `TensorKind.OPERATOR`.

        """
        if len(nodes_tensors) == 0:
            errstr = "No tensors given to contract!"
            raise ValueError(errstr)
        self.node_tensors = nodes_tensors
        self.subtree_dict = subtree_dict
        if node_identifier is None:
            node_identifier = self.node_tensors[0][0].identifier
        self.node_identifier = node_identifier
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
        if contraction_order is None:
            contraction_order = list(range(len(self.node_tensors)))
        elif not valid_contraction_order(contraction_order):
            errstr = f"Invalid contraction order {contraction_order}!"
            raise ValueError(errstr)
        self.contraction_order = contraction_order
        self.nphys = self._determine_nphys(highest_tensor)

    def _determine_nphys(self,
                         highest_tensor: TensorKind) -> int:
        """
        Determines the number of physical legs of the current contraction.

        Args:
            highest_tensor (TensorKind): The kind of the highest tensor in the
                stack. This is needed to determine the number of open legs in a
                few special cases, if there is only a single tensor in the stack
                and no subtree tensors. In this case, if the last tensor is a bra
                tensor, this must be set to `TensorKind.BRA`, otherwise to
                `TensorKind.OPERATOR`.
        """
        node, _ = self.node_tensors[0]
        if self.connection_index == 0:
            # In this case the first node is a ket node.
            return node.nopen_legs()
        elif len(self.node_tensors) > 1:
            last_node, _ = self.node_tensors[-1]
            last_node_open = last_node.nopen_legs()
            second_last_node, _ = self.node_tensors[-2]
            second_last_node_open = second_last_node.nopen_legs()
            if last_node_open < second_last_node_open:
                # In this case the last node is a bra tensor and all others
                # are operator tensors.
                return last_node_open
            if last_node_open == second_last_node_open:
                # In this case the last node is an operator tensor and all
                # others are operator tensors.
                assert last_node_open % 2 == 0, "Operator tensors must have even number of open legs!"
                return last_node_open // 2
            errstr = "Invalid open leg configuration!"
            raise ValueError(errstr)
        else:
            # This is a very tricky case, where we cannot know if the tensor is
            # a bra or operator tensor. In most cases it will be an operator
            # tensor
            if highest_tensor is TensorKind.BRA:
                return node.nopen_legs()
            elif highest_tensor is TensorKind.OPERATOR:
                assert node.nopen_legs() % 2 == 0, "Operator tensors must have even number of open legs!"
                return node.nopen_legs() // 2
            else:
                errstr = "Invalid highest tensor type!"
                raise ValueError(errstr)

    def determine_tensor_kind(self,
                            index: int
                            ) -> TensorKind:
        """
        Determines the kind of tensor at the given index.

        Args:
            index (int): The index of the tensor in the `node_tensors` list.

        Returns:
            TensorKind: The kind of tensor at the given index.
        """
        node, _ = self.node_tensors[index]
        if index == 0:
            if self.connection_index == 0:
                return TensorKind.KET
            if node.nopen_legs() == self.nphys:
                return TensorKind.BRA
        if index == len(self.node_tensors) - 1:
            if node.nopen_legs() == self.nphys:
                return TensorKind.BRA
        return TensorKind.OPERATOR

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

    def get_phy_legs(self,
                     index: int,
                     open_leg_kind: OpenLegKind
                     ) -> list[int]:
        """
        Returns the physical legs of the tensor at the given index.

        Args:
            index (int): The index of the tensor in the `node_tensors` list.
            open_leg_kind (OpenLegKind): The kind of physical legs to return.
                IN or OUT.
        """
        node, _ = self.node_tensors[index]
        tensor_kind = self.determine_tensor_kind(index)
        open_legs = node.open_legs
        nopen = node.nopen_legs()
        if tensor_kind is TensorKind.OPERATOR:
            if open_leg_kind is OpenLegKind.IN:
                # In are the last half of the open legs
                return open_legs[-(nopen // 2):]
            elif open_leg_kind is OpenLegKind.OUT:
                # Out are the first half of the open legs
                return open_legs[:(nopen // 2)]
            else:
                errstr = f"Invalid `OpenLegKind` {open_leg_kind}!"
                raise ValueError(errstr)
        elif tensor_kind.state_tensor():
            return open_legs
        else:
            errstr = f"Invalid `TensorKind` {tensor_kind}!"
            raise ValueError(errstr)

    def create_reverse_trafo(self,
                             index: int
                             ) -> Callable:
        """
        Creates the mapping from neighbour identifiers of a node to the desired
        neighbour order.

        Args:
            index (int): The index of the node and tensor in the list.
        """
        node = self.node_tensors[index][0]
        id_trafo = self.id_trafos[index]
        mapping = {node_neigh: init_neigh
                   for node_neigh, init_neigh in product(node.neighbouring_nodes(),
                                                         self.neighbour_order)
                   if node_neigh == id_trafo(init_neigh)}
        mapping[node.identifier] = self.node_identifier
        return lambda x: mapping[x]

    def no_subtree_tensors(self) -> bool:
        """
        Returns whether there are no subtree tensors to contract with.
        """
        try:
            node, _ = self.node_tensors[0]
            if node.nneighbours() == self.num_ignored():
                return True
            return False
        except IndexError as exc:
            errstr = "No tensors in the node_tensors list!"
            raise IndexError(errstr) from exc

    def contract_first_tensor(self):
        """
        Contracts the tensor at index 0 in the contraction order.
        """
        contr_index = self.contraction_order[0]
        node, tensor = self.node_tensors[contr_index]
        subtree_leg = self.connection_index + contr_index
        ignored_passed = 0
        rev_trafo = self.create_reverse_trafo(contr_index)
        for neigh_id in node.neighbouring_nodes():
            if not self.is_ignored(neigh_id, contr_index):
                cache_node_id = rev_trafo(node.identifier)
                cache_neigh_id = rev_trafo(neigh_id)
                subtree_tensor = self.subtree_dict.get_entry(cache_neigh_id,
                                                             cache_node_id)
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
        tensor_kind = self.determine_tensor_kind(contr_index)
        self.current_tensor.set_first_tensor_open_legs(ignored_passed,
                                                       nopen,
                                                       tensor_kind)
        # Initialise ignored leg list
        self.current_tensor.ignored_legs = [None] * len(self.node_tensors)
        if self.num_ignored() > 0:
            self.current_tensor.ignored_legs[contr_index] = ignored_passed - 1
        if not self.no_subtree_tensors():
            # Now we need to set the neighbour legs.
            rev_traf = self.create_reverse_trafo(contr_index)
            ignored_passed = 0
            contracted_passed = 0
            # After the above contractions, the legs corresponding to the
            # neighbouring nodes are at the end of the tensor
            max_dim = self.subtree_degree() # The number of open legs on a cached subtree tensor.
            legs_before_neighs = nopen + self.num_ignored()
            for index, neigh_id in enumerate(node.neighbouring_nodes()):
                if not self.is_ignored(neigh_id,
                                contr_index):
                    leg_index = index - ignored_passed
                    offset = legs_before_neighs - contracted_passed
                    range_start = offset + leg_index * max_dim
                    range_end = offset + (leg_index + 1) * max_dim
                    neigh_legs: list[int | None] = list(range(range_start, range_end))
                    contr_leg = contr_index + self.connection_index
                    adjusted_legs = [ind - 1
                                     for ind in neigh_legs[contr_leg+1:]]
                    adjusted_legs = neigh_legs[:contr_leg] + [None] + adjusted_legs
                    self.current_tensor.neighbour_legs[rev_traf(neigh_id)] = adjusted_legs
                    contracted_passed += 1
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
            open_leg_kind = OpenLegKind.IN
        else:
            open_leg_kind = OpenLegKind.OUT
        curr_ten_open = self.current_tensor.open_leg_by_kind(open_leg_kind)
        contr_ten_open = self.get_phy_legs(contr_index, open_leg_kind.opposite())
        contr_ten_legs.extend(contr_ten_open)
        curr_ten_legs.extend(curr_ten_open)
        # Get the virtual legs of the node that are to be contracted
        id_trafo = self.id_trafos[contr_index]
        subtree_tensor_index = self.connection_index + contr_index
        for neigh_id in self.neighbour_order:
            if not self.is_ignored(neigh_id, 0):
                node_neigh_id = id_trafo(neigh_id)
                contr_ten_legs.append(node.neighbour_index(node_neigh_id))
                curr_leg = self.current_tensor.get_neighbour_leg(neigh_id,
                                                                    subtree_tensor_index)
                curr_ten_legs.append(curr_leg)
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
        assert len(ignored_legs) == self.num_ignored()
        if self.num_ignored() > 0:
            self.current_tensor.ignored_legs[contr_index] = ignored_legs[0]
        elif self.num_ignored() > 1:
            errstr = "More than one ignored leg not supported!"
            raise NotImplementedError(errstr)
        # Finally, we add the new open legs. They will be the very last ones
        new_open_legs = list(range(num_rem_legs + self.num_ignored(),
                                   curr_tensor.ndim))
        self.current_tensor.set_open_legs_by_kind(new_open_legs, open_leg_kind)
        self.current_tensor.value = curr_tensor

    def contract_all(self,
                     transpose_option: FinalTransposition | Callable = FinalTransposition.STANDARD
                     ) -> npt.NDArray:
        """
        Contracts all tensors in the contraction order.

        Args:
            transpose_option (FinalTransposition | Callable): The way in which
                to tranpose the final tensor. Can be a callable that takes a
                `CurrentTensor` and outputs the transposed tensor. See
                `FinalTransposition` class for details on the existing
                options.
        
        Returns:
            npt.NDArray: The final contracted tensor.
        """
        self.contract_first_tensor()
        for i in range(1, len(self.contraction_order)):
            self.contract_tensor(i)
        if transpose_option is FinalTransposition.STANDARD:
            # We want the final tensor to be in the order of
            # [neighbour legs..., out legs..., in legs...]
            perm = []
            # First the neighbour legs in the order specified by neighbour_order
            for neigh_id in self.neighbour_order:
                if neigh_id in self.current_tensor.neighbour_legs:
                    perm.extend(self.current_tensor.cleared_neigh_legs(neigh_id))
                elif self.is_ignored(neigh_id, 0):
                    # This means the ignored leg is in the final tensor
                    perm.extend(self.current_tensor.cleared_ignored_legs())
            # Then the out legs
            perm.extend(self.current_tensor.cleared_open_legs(OpenLegKind.OUT))
            # Finally the in legs
            perm.extend(self.current_tensor.cleared_open_legs(OpenLegKind.IN))
            final_tensor = np.transpose(self.current_tensor.value, axes=perm)
            return final_tensor
        if transpose_option is FinalTransposition.NONE:
            return self.current_tensor.value
        errstr = f"Invalid `FinalTransposition` {transpose_option}!"
        if transpose_option is FinalTransposition.IGNOREDFIRST:
            if self.num_ignored() == 0:
                errstr = "No ignored leg, cannot use `IGNOREDFIRST` option!"
                raise ValueError(errstr)
            # We want the final tensor to be in the order of
            # [ignored leg, neighbour legs..., out legs..., in legs...]
            perm = []
            # First the ignored legs
            perm.extend(self.current_tensor.cleared_ignored_legs())
            # Then the neighbour legs in the order specified by neighbour_order
            for neigh_id in self.neighbour_order:
                if neigh_id in self.current_tensor.neighbour_legs:
                    perm.extend(self.current_tensor.cleared_neigh_legs(neigh_id))
            # Then the out legs
            perm.extend(self.current_tensor.cleared_open_legs(OpenLegKind.OUT))
            # Finally the in legs
            perm.extend(self.current_tensor.cleared_open_legs(OpenLegKind.IN))
            final_tensor = np.transpose(self.current_tensor.value, axes=perm)
            return final_tensor
        raise ValueError(errstr)

    def contract_to_scalar(self) -> complex:
        """
        Contracts all tensors in this contraction to a scalar.

        Returns:
            complex: The final contracted scalar.
        """
        final_tensor = self.contract_all(transpose_option=FinalTransposition.NONE)
        if final_tensor.ndim != 0:
            errstr = "Final tensor is not a scalar!"
            raise ValueError(errstr)
        return final_tensor.item()

    def contract_into_cache(self,
                            **kwargs
                            ) -> None:
        """
        Contracts all tensors in this contraction and stores the result in the
        subtree cache.

        Args:
            **kwargs: Additional arguments to pass to `contract_all`.

        Raises:
            ValueError: If there is no ignored leg. In this case the tensor
                cannot be stored in the subtree cache.
        """
        if self.no_ignored_legs():
            errstr = "No ignored leg, cannot store in subtree cache!"
            raise ValueError(errstr)
        final_tensor = self.contract_all(**kwargs)
        node_id = self.node_identifier
        next_node_id = self.ignored_leg
        self.subtree_dict.add_entry(node_id,
                                    next_node_id,
                                    final_tensor)

    def __call__(self,
                 transpose_option: FinalTransposition | Callable = FinalTransposition.STANDARD
                 ) -> npt.NDArray:
        """
        Contracts all tensors in this contraction.

        Args:
            transpose_option (FinalTransposition | Callable): The way in which
                to tranpose the final tensor. Can be a callable that takes a
                `CurrentTensor` and outputs the transposed tensor. See
                `FinalTransposition` class for details on the existing
                options.
        
        Returns:
            npt.NDArray: The final contracted tensor.
        """
        return self.contract_all(transpose_option=transpose_option)

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
        # Same here, but for ignored legs
        self.ignored_legs: list[int | None] = []

    def __str__(self) -> str:
        """
        Returns a string representation of the CurrentTensor.
        """
        out = "CurrentTensor:\n"
        out += f"Value shape: {self.value.shape}\n"
        out += f"In legs: {self.in_legs}\n"
        out += f"Out legs: {self.out_legs}\n"
        out += f"Neighbour legs: {self.neighbour_legs}\n"
        out += f"Ignored legs: {self.ignored_legs}\n"
        return out

    def open_leg_by_kind(self,
                         kind: OpenLegKind
                            ) -> list[int | None]:
        """
        Returns the open legs of the given kind.

        Args:
            kind (OpenLegKind): The kind of open legs to return.
        """
        if kind is OpenLegKind.IN:
            return self.in_legs
        elif kind is OpenLegKind.OUT:
            return self.out_legs
        else:
            errstr = f"Invalid `OpenLegKind` {kind}!"
            raise ValueError(errstr)

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

    def cleared_neigh_legs(self, neigh_id: str) -> list[int]:
        """
        Returns a list of all neighbour legs of the specified neighbour
        that are not None.

        Args:
            neigh_id (str): The identifier of the neighbour.
        
        Returns:
            list[int]: A list of all leg indices that are not None.
        """
        return [leg for leg in self.neighbour_legs[neigh_id]
                if leg is not None]

    def cleared_ignored_legs(self) -> list[int]:
        """
        Returns a list of all ignored legs that are not None.

        Returns:
            list[int]: A list of all leg indices that are not None.
        """
        return [leg for leg in self.ignored_legs
                if leg is not None]

    def cleared_open_legs(self,
                          kind: OpenLegKind
                          ) -> list[int]:
        """
        Returns a list of all specified open legs that are not None.

        Args:
            kind (OpenLegKind): The kind of open legs to return.
        
        Returns:
            list[int]: A list of all leg indices that are not None.
        """
        if kind is OpenLegKind.IN:
            legs = self.in_legs
        elif kind is OpenLegKind.OUT:
            legs = self.out_legs
        else:
            errstr = f"Invalid `OpenLegKind` {kind}!"
            raise ValueError(errstr)
        # Either all legs are None or none are None
        if len(legs) == 0:
            return []
        if legs[0] is None:
            return []
        return legs

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
            if new_val is None and ignored_leg is not None:
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

    def set_open_legs_by_kind(self,
                              open_legs: list[int],
                              kind: OpenLegKind):
        """
        Sets the open legs of the given kind.

        Args:
            open_legs (list[int]): The leg indices to set.
            kind (OpenLegKind): The kind of open legs to set.
        """
        if kind is OpenLegKind.IN:
            self.in_legs = open_legs
        elif kind is OpenLegKind.OUT:
            self.out_legs = open_legs
        else:
            errstr = f"Invalid `OpenLegKind` {kind}!"
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
    if len(sorted_rem_legs) == 0:
        return leg_value
    if leg_value is None:
        # The leg is already gone and stays gone.
        return None
    if leg_value < sorted_rem_legs[0]:
        # Smaller than all contracted legs
        # No need to adjust
        return leg_value
    for i, rem_val in enumerate(sorted_rem_legs[:-1]):
        if leg_value == rem_val:
            # This leg was contracted
            return None
        if leg_value > rem_val and leg_value < sorted_rem_legs[i+1]:
            # This means i many legs that come before this one were
            # contracted.
            return leg_value - (i+1)
    # Treat the final one special
    if leg_value == sorted_rem_legs[-1]:
        # This leg was contracted
        return None
    if leg_value > sorted_rem_legs[-1]:
        # Larger than all contracted legs
        return leg_value - len(sorted_rem_legs)
    raise ValueError(f"Invalid leg value {leg_value}!")
