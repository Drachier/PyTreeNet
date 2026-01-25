"""
Implements a generation function to generate binary TTNS.
"""
from copy import deepcopy
from typing import Self
from enum import Enum
import re

from numpy import ndarray, zeros

from ..ttns import TreeTensorNetworkState
from ..core.node import Node
from ..util.ttn_exceptions import positivity_check
from .special_nodes import constant_bd_trivial_node

__all__ = ["generate_binary_ttns"]

class HelperNode:
    """
    A small helper class that stores a node, as well as its position in the tree.
    """

    def __init__(self, node: Node, level: int, position: int):
        self.node = node
        self.level = level
        self.position = position

    def children_level(self) -> int:
        """
        Returns the level of the children of the node.
        """
        return self.level + 1

    def children_position(self) -> tuple[int, int]:
        """
        Returns the horizontal position of the children of the node.
        """
        return 2*self.position, 2*self.position + 1

    def parent_legs(self, ttns: TreeTensorNetworkState
                    ) -> tuple[int, int]:
        """
        Returns the legs of the current node that serve as parent leg for the
        children.
        """
        if self.node.identifier == ttns.root_id:
            return 0, 1
        return 1, 2

    def generate_children_nodes(self,
                                virtual_prefix: str
                                ) -> tuple[Self, Self]:
        """
        Generates the children of the current node.
        """
        current_level = self.children_level()
        left_child_position, right_child_position = self.children_position()
        left_child_id = create_virtual_node_id(current_level,
                                               left_child_position,
                                               virtual_prefix)
        left_child_node = Node(identifier=left_child_id)
        right_child_id = create_virtual_node_id(current_level,
                                                right_child_position,
                                                virtual_prefix)
        right_child_node = Node(identifier=right_child_id)
        left_helper = HelperNode(left_child_node,
                                 current_level,
                                 left_child_position)
        right_helper = HelperNode(right_child_node,
                                  current_level,
                                  right_child_position)
        return left_helper, right_helper

    def generate_children_tensors(self,
                                  bond_dim: int
                                  ) -> tuple[ndarray, ndarray]:
        """
        Generates the tensors of the children of the current node.
        """
        l_tensor = create_non_root_virt_tensor(bond_dim)
        r_tensor = create_non_root_virt_tensor(bond_dim)
        return l_tensor, r_tensor

def create_virtual_node_id(level: int, position: int, prefix: str) -> str:
    """
    Creates a virtual node identifier.
    """
    return prefix + str(level) + "_" + str(position)

def create_non_root_virt_tensor(bond_dim: int) -> ndarray:
    """
    Creates a virtual tensor for a non-root node.
    """
    virt_tensor = constant_bd_trivial_node(bond_dim, 3)
    return virt_tensor

def add_all_nodes(num_phys: int,
                  bond_dim: int,
                  virtual_prefix: str
                  ) -> tuple[list[Node], TreeTensorNetworkState]:
    """
    Adds all nodes as virtual nodes to the tree tensor network state.

    Args:
        num_phys (int): The number of physical nodes that should be added.
        bond_dim (int): The bond dimension of the tree tensor network state.
        virtual_prefix (str): The prefix for the virtual nodes.

    Returns:
        list[Node]: A list of all nodes that should be physical nodes.
        TreeTensorNetworkState: The tree tensor network state with all nodes.
    
    """
    ttns = TreeTensorNetworkState()
    root_node = Node(identifier=virtual_prefix + "0_0")
    root_tensor = constant_bd_trivial_node(bond_dim, 2)
    ttns.add_root(root_node, root_tensor)
    phys_nodes = [HelperNode(root_node, 0, 0)]
    while len(phys_nodes) != num_phys:
        hnode = phys_nodes.pop(0)
        if len(phys_nodes) == num_phys:
            break
        l_helper, r_helper = hnode.generate_children_nodes(virtual_prefix)
        l_tensor, r_tensor = hnode.generate_children_tensors(bond_dim)
        parent_legs = hnode.parent_legs(ttns)
        parent_id = hnode.node.identifier
        ttns.add_child_to_parent(l_helper.node,
                                l_tensor,
                                0,
                                parent_id,
                                parent_legs[0])
        ttns.add_child_to_parent(r_helper.node,
                                r_tensor,
                                0,
                                parent_id,
                                parent_legs[1])
        # This modification is intended
        phys_nodes.append(l_helper)
        phys_nodes.append(r_helper)
    phys_nodes = [hnode.node for hnode in phys_nodes]
    return phys_nodes, ttns

def transform_phys_nodes(ttns: TreeTensorNetworkState,
                         phys_nodes: list[Node],
                         phys_tensor: ndarray,
                         phys_prefix: str
                         ) -> TreeTensorNetworkState:
    """
    Transforms the physical nodes to physical nodes in the tree tensor network state.

    The nodes are changed in place.

    Args:
        ttns (TreeTensorNetworkState): The tree tensor network state.
        phys_nodes (list[Node]): The physical nodes.
        phys_tensor (ndarray): The tensor for the physical sites.
        phys_prefix (str): The prefix for the physical nodes.
    
    Returns:
        TreeTensorNetworkState: The tree tensor network state with the physical nodes.

    """
    for i, node in enumerate(phys_nodes):
        new_id = phys_prefix + str(i)
        ttns.replace_node(new_id,
                          node.identifier,
                          deepcopy(phys_tensor))
    return ttns

def generate_binary_ttns(num_phys: int,
                         bond_dim: int,
                         phys_tensor: ndarray,
                         phys_prefix: str = "site",
                         virtual_prefix: str = "node"
                         ) -> TreeTensorNetworkState:
    """
    Generates a binary tree tensor network state.

    The TTNS will have constant bond dimension and all physical sites will have
    the same tensor.
    
    Args:
        num_phys (int): The number of physical sites.
        bond_dim (int): The bond dimension of the tree tensor network state.
        phys_tensor (ndarray): The tensor for the physical sites.
        phys_prefix (str): The prefix for the physical nodes.
        virtual_prefix (str): The prefix for the virtual nodes.
    
    Returns:
        TreeTensorNetworkState: The generated tree tensor network state.

    """
    positivity_check(num_phys, "number of physical sites")
    positivity_check(bond_dim, "bond dimension")
    phys_nodes, ttns = add_all_nodes(num_phys, bond_dim, virtual_prefix)
    ttns = transform_phys_nodes(ttns, phys_nodes, phys_tensor, phys_prefix)
    return ttns

class Direction(Enum):
    X = 1
    Y = 2

    def opposite(self) -> Self:
        """
        Returns the opposite direction.
        """
        if self == Direction.X:
            return Direction.Y
        else:
            return Direction.X

    def virt_id_appendix(self,
                         along_index: int,
                         perp_index: int,
                         ) -> str:
        """
        Returns the appendix for a virtual node identifier in the given direction.
        """
        if self == Direction.X:
            return str(perp_index) + "_" + str(along_index)
        else:
            return str(along_index) + "_" + str(perp_index)

class BTTNLevel:
    """
    Helper class to store information about a level in the optimised 2D BTTN.
    """

    def __init__(self,
                 level: list[list[tuple[str,str] | tuple[str]]],
                 own_ids: list[list[str]],
                 pairing_direction: Direction = Direction.X
                 ) -> None:
        self.level = level
        self.own_ids = own_ids
        self.pairing_direction = pairing_direction
        self._y_size_value = len(level)
        if self._y_size_value > 0:
            self._x_size_value = len(level[0])
        else:
            self._x_size_value = 0

    def x_size(self) -> int:
        """
        Returns the size of the level in the x direction.
        """
        return self._x_size_value

    def y_size(self) -> int:
        """
        Returns the size of the level in the y direction.
        """
        return self._y_size_value

    def size_by_direction(self,
                          direction: Direction
                          ) -> int:
        """
        Returns the size of the level in the given direction.
        """
        if direction == Direction.X:
            return self.x_size()
        else:
            return self.y_size()

    def append_to_x_direction(self,
                           element: tuple[str,str] | tuple[str],
                           own_id: str,
                           new_x_row: bool = False
                           ) -> None:
        """
        Appends an element to the last x direction.

        Args:
            element (tuple[str,str] | tuple[str]): The element to append.
            own_id (str): The own identifier of the element.
            new_x_row (bool): Whether to start a new x row.

        Raises:
            ValueError: If the previous row is not complete when starting
                        a new x row, or if the current row is full when
                        appending to the current x row.
        """
        if new_x_row:
            if len(self.level) > 0 and len(self.level[-1]) != self.x_size():
                raise ValueError("Cannot start new x row, previous row not complete!")
            self.level.append([element])
            self.own_ids.append([own_id])
            self._y_size_value += 1
        else:
            if self.x_size() != 1 and len(self.level[-1]) >= self.x_size():
                errstr = "Cannot append to x direction, row full!\n"
                errstr += "Add new_x_row=True to start a new row."
                raise ValueError(errstr)
            self.level[-1].append(element)
            self.own_ids[-1].append(own_id)
        if self.y_size() == 1:
            self._x_size_value += 1

    def append_to_y_direction(self,
                              element: tuple[str,str] | tuple[str],
                              own_id: str,
                              new_y_column: bool = False
                              ) -> None:
        """
        Appends an element to the y direction at the given x index.

        Args:
            element (tuple[str,str] | tuple[str]): The element to append.
            own_id (str): The own identifier of the element.
            new_y_column (bool): Whether to start a new y column.
        """
        if new_y_column:
            if self.y_size() == 0:
                self.level.append([])
                self.own_ids.append([])
            if len(self.level[0]) > 0 and len(self.level) != self.y_size():
                raise ValueError("Cannot start new y column, previous column not complete!")
            self.level[0].append(element)
            self.own_ids[0].append(own_id)
            self._x_size_value += 1
        else:
            if self.x_size() == 1:
                self.level.append([element])
                self.own_ids.append([own_id])
            elif self.x_size() != 1 and len(self.level[-1]) == self.x_size():
                raise ValueError("Cannot append to y direction, column full!")
            else:
                for i, row in enumerate(self.level):
                    if len(row) < self.x_size():
                        row.append(element)
                        self.own_ids[i].append(own_id)
                        break
        if self.x_size() == 1:
            self._y_size_value += 1

    def append_element(self,
                       element: tuple[str,str] | tuple[str],
                       own_id: str,
                       direction: Direction,
                       new_line: bool = False
                       ) -> None:
        """
        Appends an element to the level in the given direction.

        Args:
            element (tuple[str,str] | tuple[str]): The element to append.
            own_id (str): The own identifier of the element.
            direction (Direction): The direction to append in.
            new_line (bool): Whether to start a new line in the given direction.
        """
        if direction == Direction.X:
            self.append_to_x_direction(element, own_id,
                                       new_x_row=new_line)
        else:
            self.append_to_y_direction(element, own_id,
                                       new_y_column=new_line)

    def get_nn_by_direction(self,
                            index_along_dir: int,
                            index_perp_dir: int,
                            direction: Direction
                            ) -> tuple[str,str] | tuple[str]:
        """
        Returns the nearest neighbours in the given direction at the given index.

        Args:
            index_along_dir (int): The index in the given direction.
            index_perp_dir (int): The index perpendicular to the given direction.
            direction (Direction): The direction to get the nearest neighbours in.

        Returns:
            list[tuple[str,str] | tuple[str]]: The nearest neighbours or a
                single site with the level's own identifiers.
        """
        is_last_in_dir = index_along_dir == self.size_by_direction(direction) - 1
        if direction == Direction.X:
            n1 = self.own_ids[index_perp_dir][index_along_dir]
            if not is_last_in_dir:
                n2 = self.own_ids[index_perp_dir][index_along_dir + 1]
                return (n1, n2)
        else:
            n1 = self.own_ids[index_along_dir][index_perp_dir]
            if not is_last_in_dir:
                n2 = self.own_ids[index_along_dir + 1][index_perp_dir]
                return (n1, n2)
        return (n1, )

    @classmethod
    def from_starting_params(cls,
                             lattice_size: int,
                             phys_prefix: str,
                             virtual_prefix: str
                             ) -> Self:
        """
        Creates the bottom level of the BTTN from the lattice size.
        """
        # We combine the sites along the x direction first
        level: list[list[tuple[str] | tuple[str,str]]] = []
        own_ids: list[list[str]] = []
        for y in range(lattice_size):
            row: list[tuple[str] | tuple[str,str]] = []
            own_ids_row: list[str] = []
            for x in range(lattice_size):
                site_id = phys_prefix + str(y * lattice_size + x)
                if x % 2 == 0 and x != lattice_size - 1:
                    neigh_id = phys_prefix + str(y * lattice_size + (x + 1))
                    row.append((site_id, neigh_id))
                    own_ids_row.append(virtual_prefix + "lev1_" + str(y) + "_" + str(x // 2))
                elif x % 2 == 1:
                    continue
                else:
                    # In this case there is a last unpaired site
                    row.append((site_id,))
                    own_ids_row.append(site_id)
            level.append(row)
            own_ids.append(own_ids_row)
        return cls(level, own_ids=own_ids)

    @classmethod
    def from_previous_level(cls,
                            previous_level: Self,
                            virtual_prefix: str,
                            ) -> Self:
        """
        Creates a new level from the previous level.
        """
        pairing_direction = previous_level.pairing_direction.opposite()
        opposite_direction = pairing_direction.opposite()
        if previous_level.size_by_direction(pairing_direction) == 1:
            if previous_level.size_by_direction(opposite_direction) == 1:
                errstr = "Cannot create new level, previous level has size 1 in both directions!"
                raise ValueError(errstr)
            # This means there is nothing to do in the pairing direction
            # Thus we swap the directions
            pairing_direction, opposite_direction = opposite_direction, pairing_direction
        level = cls([], [], pairing_direction=pairing_direction)
        for i1 in range(previous_level.size_by_direction(opposite_direction)):
            for i2 in range(previous_level.size_by_direction(pairing_direction)):
                if i2 % 2 == 0:
                    print(level.own_ids)
                    nn = previous_level.get_nn_by_direction(i2, i1,
                                                            pairing_direction)
                    if len(nn) == 2:
                        new_id = virtual_prefix + pairing_direction.virt_id_appendix(i1, i2 // 2)
                    else:
                        new_id = nn[0]
                    level.append_element(nn, new_id,
                                         pairing_direction,
                                         new_line=(i2 == 0))
                else:
                    continue
        return level

    def to_dict(self) -> dict:
        """
        Converts the level to a dictionary.

        The keys are the own identifiers and the values are the
        tuples of nearest neighbour identifiers.
        """
        result: dict = {}
        for i, row in enumerate(self.level):
            for j, element in enumerate(row):
                own_id = self.own_ids[i][j]
                result[own_id] = element
        return result

def optimised_2d_binary_ttn(lattice_size: int,
                            initial_bond_dim: int,
                            phys_tensor: ndarray,
                            phys_prefix: str = "site",
                            virtual_prefix: str = "node"
                            ) -> TreeTensorNetworkState:
    """
    Generates an optimised binary tree tensor network state for a 2D lattice.

    Args:
        lattice_size (int): The size of the lattice (assumed square).
        initial_bond_dim (int): The bond dimension of the tree tensor network state.
        phys_tensor (ndarray): The tensor for the physical sites. Not included are
            the virtual bond dimension to the parent leg.
        phys_prefix (str): The prefix for the physical nodes.
        virtual_prefix (str): The prefix for the virtual nodes.
    
    Returns:
        TreeTensorNetworkState: The generated tree tensor network state.
    """
    levels = []
    bottom_level = BTTNLevel.from_starting_params(lattice_size,
                                                  phys_prefix,
                                                  virtual_prefix)
    levels.append(bottom_level)
    current_level = bottom_level
    level_index = 2
    while (current_level.size_by_direction(Direction.X),
           current_level.size_by_direction(Direction.Y)) != (1, 1):
        level_virt_prefix = virtual_prefix + "lev" + str(level_index) + "_"
        current_level = BTTNLevel.from_previous_level(current_level,
                                                      level_virt_prefix)
        levels.append(current_level)
        level_index += 1
    # Now we can create the TTNS by running from the top level to the bottom
    phys_dim = phys_tensor.shape[0]
    ttns = TreeTensorNetworkState()
    current_level = levels[-1]
    root_id = current_level.own_ids[0][0]
    root_node = Node(identifier=root_id)
    root_tensor = constant_bd_trivial_node(initial_bond_dim, 2)
    ttns.add_root(root_node, root_tensor)
    for level in reversed(levels):
        dictionary = level.to_dict()
        for own_id, nn_ids in dictionary.items():
            own_node = ttns.nodes[own_id]
            for node_id in nn_ids:
                if not node_id in ttns.nodes:
                    node = Node(identifier=node_id)
                    if re.match(rf"^{phys_prefix}\d+$", node_id):
                        # This is a physical node
                        tensor = zeros((initial_bond_dim, phys_dim),
                                       dtype=phys_tensor.dtype)
                        tensor[0, :] = phys_tensor
                    else:
                        tensor = create_non_root_virt_tensor(initial_bond_dim)
                    ttns.add_child_to_parent(node, tensor, 0,
                                             own_id,
                                             own_node.lowest_open_leg())
    return ttns
