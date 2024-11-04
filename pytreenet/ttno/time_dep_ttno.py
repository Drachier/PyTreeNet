"""
This module implements a class to update the tensors of a TTNO that was
constructed from a symbolic expression.
"""

from typing import List
from copy import deepcopy

from .ttno_class import TreeTensorNetworkOperator

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

        This will reset subparts of the tensors to the original values saved
        in the updatables. If two updateables have the same node_id and
        indices, the last one will overwrite the previous one.

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
