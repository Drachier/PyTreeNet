"""
This module implements a class to update the tensors of a TTNO that was
constructed from a symbolic expression.
"""

from typing import List
from copy import deepcopy

from numpy import ndarray

from .ttno_class import TreeTensorNetworkOperator
from .hyperedge import HyperEdge

class TimeDependentTTNO(TreeTensorNetworkOperator):
    """
    A Tree Tensor Network Operator that can be updated in time.

    This class is a subclass of the TreeTensorNetworkOperator class.

    Attributes:
        updatables: A list of objects that have some methods to update the
            tensors of the TTNO.
    
    """

    def __init__(self,
                 updatables: List,
                 ttno: TreeTensorNetworkOperator):
        """
        Initializes a TimeDependentTTNO object.

        Args:
            updatables: A list of objects that have some methods to update the
                tensors of the TTNO.
            ttno: A TreeTensorNetworkOperator object that contains the nodes
                and tensors of the TTNO.

        """
        super().__init__()
        self._copy_ttno_attributes(ttno)
        self.updatables = updatables

    def _copy_ttno_attributes(self, ttno: TreeTensorNetworkOperator):
        self._nodes = deepcopy(ttno.nodes)
        self._tensors = deepcopy(ttno.tensors)
        self.orthogonality_center_id = ttno.orthogonality_center_id
        self.root_id = ttno.root_id

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
        for updateable in self.updatables:
            zero_time_values = updateable.get_zero_time_values()
            node_id = updateable.node_id
            indices = updateable.get_indices()
            self._tensors[node_id][indices] = zero_time_values

    def fill_tensors(self):
        """
        Fills the tensors with the current values of the updateables.

        This will add the current values of the updateables to the tensors.
        If two updateables have the same node_id and indices, the new values
        will simply both be added to the tensor.

        """
        for updateable in self.updatables:
            current_values = updateable.get_current_values()
            node_id = updateable.node_id
            indices = updateable.get_indices()
            self._tensors[node_id][indices] += current_values

class FactorUpdatable:
    """
    An updatable object whose values are updated merely by updating a scalar
    factor.

    For example

    .. math::

        A_{ij}(t+dt) = f(t+dt) A_{ij}(t)

    
    """

    def __init__(self,
                 node_id: int,
                 indices: tuple,
                 original_values: ndarray,
                 inital_values: ndarray,
                 factor_function: callable):
        """
        Initializes a FactorUpdatable object.

        Args:
            node_id: The id of the node that the tensor belongs to.
            indices: The indices of the tensor, that this operator fits into.
            original_values: The values of the TTNO tensor without this
                operator.
            inital_values: The values of the tensor at the initial time.
            factor_function: A function that takes a time as input and returns
                a scalar factor.
        
        """
        self.node_id = node_id
        self.indices = indices
        self.original_values = original_values
        self.current_values = inital_values
        self.factor_function = factor_function
        self.current_time = 0

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
        self.current_values = factor * self.original_values

    def get_zero_time_values(self):
        """
        Returns the values of the tensor at time t=0.

        Returns:
            The values of the tensor at time t=0.

        """
        return self.original_values

    def get_current_values(self):
        """
        Returns the current values of the tensor.

        Returns:
            The current values of the tensor.

        """
        return self.current_values

    def get_indices(self):
        """
        Returns the indices of the tensor.

        Returns:
            The indices of the tensor.

        """
        return self.indices

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
