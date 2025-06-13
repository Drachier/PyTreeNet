"""
This module implements a class to update the tensors of a TTNO that was
constructed from a symbolic expression.
"""
from __future__ import annotations
from typing import List, Iterable
from copy import deepcopy

from numpy import ndarray

from .ttno_class import TreeTensorNetworkOperator
from .hyperedge import HyperEdge
from ..util.std_utils import int_to_slice

class TimeDependentTTNO(TreeTensorNetworkOperator):
    """
    A Tree Tensor Network Operator that can be updated in time.

    This class is a subclass of the TreeTensorNetworkOperator class.

    Attributes:
        updatables: A list of objects that have some methods to update the
            tensors of the TTNO.
    
    """

    def __init__(self,
                 updatables: Iterable,
                 ttno: TreeTensorNetworkOperator = None):
        """
        Initializes a TimeDependentTTNO object.

        Args:
            updatables: An iterable of objects that have some methods to update the
                tensors of the TTNO.
            ttno: A TreeTensorNetworkOperator object that contains the nodes
                and tensors of the TTNO. Optional, if not provided, a new
                TTNO is created.

        """
        if ttno is None:
            ttno = TreeTensorNetworkOperator()
        super().__init__()
        self._copy_ttno_attributes(ttno)
        self.updatable_check(updatables)
        # An updatable and an orig_value corresponding to each other will have
        # the same position in the list.
        self.updatables = tuple(updatables)
        self.orig_values = self._get_orig_values(updatables)
        # We want to have the initial values of the tensors in the TTNO
        # They may not be zero.
        self.fill_tensors()

    def _get_orig_values(self, updatables: list) -> tuple[OriginalValues]:
        """
        Returns the original values of the tensors of the TTNO.

        Args:
            updatables: A list of objects that have some methods to update the
                tensors of the TTNO.

        Returns:
            A tuple with the original values of the tensors of the TTNO.

        """
        orig_values = []
        for updatable in updatables:
            node_id = updatable.node_id
            indices = updatable.indices
            values = self._tensors[node_id][indices]
            orig_values.append(OriginalValues(deepcopy(values), node_id, indices))
        return tuple(orig_values)

    def updatable_check(self, updatables: List):
        """
        Checks if the updatables are valid.

        Args:
            updatables: A list of objects that have some methods to update the
                tensors of the TTNO.

        Raises:
            ValueError: If the updatables are not valid.

        """
        for updatable in updatables:
            if updatable.node_id not in self._nodes:
                raise ValueError(f"Node id {updatable.node_id} not found in "
                                 "the TTNO.")
            shape = self._nodes[updatable.node_id].shape
            indices = updatable.indices
            if len(shape) != len(indices):
                raise ValueError(f"Shape of node {updatable.node_id} does not",
                                 "match the shape of the updatable.")
            for i, index in enumerate(indices[:-2]): # Leaving out the physical indices
                if index.stop > shape[i] or index.start < 0:
                    raise ValueError(f"Index {index} is out of bounds for "
                                     f"node {updatable.node_id}.")

    def _copy_ttno_attributes(self, ttno: TreeTensorNetworkOperator):
        self._nodes = deepcopy(ttno.nodes)
        self._tensors = deepcopy(ttno.tensors)
        self.orthogonality_center_id = ttno.orthogonality_center_id
        self._root_id = ttno.root_id

    def update(self, time_step_size: float):
        """
        Updates the tensors of the TTNO.

        Args:
            time_step_size: The size of the time step.

        """
        self.reset_tensors()
        for updatable in self.updatables:
            updatable.update(time_step_size)
        self.fill_tensors()

    def reset_tensors(self):
        """
        Resets the tensors of the TTNO.

        This will reset subparts of the tensors to the values the tensor would
        have, if the updatable didn't appear. If two updateables have the same
        node_id and indices, the last one will overwrite the previous one.

        """
        for orig_val in self.orig_values:
            zero_time_values = orig_val.values
            node_id = orig_val.node_id
            indices = orig_val.indices
            self._tensors[node_id][indices] = zero_time_values

    def fill_tensors(self):
        """
        Fills the tensors with the current values of the updateables.

        This will add the current values of the updateables to the tensors.
        If two updateables have the same node_id and indices, the new values
        will simply both be added to the tensor.

        """
        for updateable in self.updatables:
            current_values = updateable.current_values
            node_id = updateable.node_id
            indices = updateable.indices
            self._tensors[node_id][indices] += current_values

class OriginalValues:
    """
    A simple class to store the original values of a tensor.

    Attributes:
        values (ndarray): The original values of parts of a tensor.
        node_id (str): The id of the node that the tensor belongs to.
        indices (tuple[int | slice]): The indices of the tensor, that this
            operator fits into.
    """

    def __init__(self, values: ndarray,
                 node_id: str,
                 indices: tuple[int | slice]):
        """
        Initializes an OriginalValues object.

        Args:
            values: The original values of parts of a tensor.
            node_id: The id of the node that the tensor belongs to.
            indices: The indices of the tensor, that this operator fits into.

        """
        self.values = values
        self.node_id = node_id
        assert len(values.shape) == len(indices), \
            "The number of indices must match the number of dimensions of the tensor!"
        slice_indices = [int_to_slice(i) if isinstance(i, int) else i
                         for i in indices]
        self.indices = tuple(slice_indices)

class FactorUpdatable:
    """
    An updatable object whose values are updated merely by updating a scalar
    factor.

    For example

    .. math::

        A_{ij}(t+dt) = f(t+dt) A_{ij}

    
    """

    def __init__(self,
                 node_id: str,
                 indices: tuple[int | slice, ...],
                 initial_values: ndarray,
                 factor_function: callable):
        """
        Initializes a FactorUpdatable object.

        Args:
            node_id: The id of the node that the tensor belongs to.
            indices: The indices of the tensor, that this operator fits into.
            original_values: The values of the TTNO tensor without this
                operator.
            initial_values: The values of the tensor at the initial time.
            factor_function: A function that takes a time as input and returns
                a scalar factor.
        
        """
        self.node_id = node_id
        slice_indices = [int_to_slice(i) if isinstance(i, int) else i
                         for i in indices]
        self._indices = tuple(slice_indices)
        assert len(self._indices) == initial_values.ndim, \
            "The number of indices must match the number of dimensions of the tensor!"
        self.initial_values = initial_values
        self.current_values = factor_function(0) * initial_values
        self.factor_function = factor_function
        self.current_time = 0

    @property
    def indices(self) -> tuple:
        """
        Returns the indices of the tensor.

        Returns:
            The indices of the tensor.

        """
        return self._indices

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the tensor.

        Returns:
            The shape of the tensor.

        """
        return self.initial_values.shape

    def update_current_time(self, time_step_size: float):
        """
        Updates the current time.

        Args:
            time_step_size: The size of the time step.

        """
        self.current_time += time_step_size

    def update(self, time_step_size: float):
        """
        Updates the values of the tensor.

        Args:
            time_step_size: The size of the time step.

        """
        self.update_current_time(time_step_size)
        factor = self.factor_function(self.current_time)
        self.current_values = factor * self.initial_values

    def get_zero_time_values(self):
        """
        Returns the values of the tensor at time t=0.

        Returns:
            The values of the tensor at time t=0.

        """
        return self.factor_function(0) * self.initial_values


def updatable_from_hyperedge(hyperedge: HyperEdge,
                             ttno: TreeTensorNetworkOperator):
    """
    Obtain all the information needed to create a updatable object from
    a HyperEdge object.
    """
    node_id = hyperedge.corr_node_id
    indices = hyperedge.find_tensor_position()
    original_values = ttno.tensors[node_id][indices]
    return node_id, indices, original_values
