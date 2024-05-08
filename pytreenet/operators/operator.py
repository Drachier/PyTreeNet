"""
Provides an operator class.

This class can be used to store the numeric value of an operator in conjuction
with the identifiers of the node it is applied. It also contains utility
functions that allow treatment of a high dimensional array as one would a
matrix. For example by testing the unitarity or exponentiating the operator.
"""
from __future__ import annotations
from typing import List, Union
from copy import copy, deepcopy

import numpy as np

from ..core.ttn import TreeTensorNetwork
from ..util.ttn_exceptions import positivity_check

class NumericOperator():
    """
    An operator that hold the operator and identifiers it is applied to.

    Atrributes:
        operator (np.ndarray): The numeric value of this operator. If this is
            matrix valued, the convention of the leg order is 
            ``(output_leg, input_leg)``. If it is instead a higher dimensional
            tensor the leg order is defined as

            ``(output_leg1, ..., output_legn, input_leg1, input_leg2, ..., input_legn)``

        identifiers (List[str]): The identifiers of the nodes this operator is
            applied to. Should be in the same order as the legs of the operator.
    """

    def __init__(self, operator: np.ndarray, node_identifiers: Union[List[str],str]):
        """
        Creates an instance of a NumericOperator.

        Args:
            operator (np.ndarray): The numeric value of this operator. If this 
                is matrix valued, the convention of the leg order is
                `(output_leg, input_leg)`. If it is instead a higher
                dimensional tensor the leg order is defined as
                `(output_leg1, ..., output_legn, input_leg1, input_leg2, ..., input_legn)`
            node_identifiers (Union[List[str],str]): The identifiers of the
                nodes this operator is applied to. Should be in the same order
                as the legs of the operator. Can also be a single identifier.
        """
        assert operator.ndim % 2 == 0
        self.operator = operator
        if isinstance(node_identifiers, str):
            self.node_identifiers = [node_identifiers]
        else:
            self.node_identifiers = node_identifiers

    def to_matrix(self) -> NumericOperator:
        """
        Turns an operator into a matrix.

        Returns:
            NumericOperator: A new NumericOperator whose operator is the
                current operator but in matrix form, i.e. all input legs are
                combined and all output legs are combined.
        """
        half_dim = int(self.operator.ndim / 2)
        if half_dim == 1:
            return deepcopy(self)
        new_dim = np.prod(self.operator.shape[0:half_dim])
        matrix = self.operator.reshape((new_dim, new_dim))
        return NumericOperator(matrix, copy(self.node_identifiers))

    def to_tensor(self, dim : Union[int, None] = None,
                  ttn : Union[TreeTensorNetwork, None] = None) -> NumericOperator:
        """
        Turns an operator in matrix form into a high dimensional tensor.

        At least one of `dim` or `ttn` has to be given.

        Args:
            dim (Union[int, None], optional): If all nodes have the same open
                dimension it can be given here. Defaults to None.
            ttn (Union[TreeTensorNetwork, None], optional): If not all nodes
                have the same open dimension a TTN is needed as a reference.
                Defaults to None.

        Returns:
            NumericOperator: A NumericOperator equivalent to the original
                matrix valued operator but in tensor form.
        """
        if dim is None and ttn is None:
            errstr = "`dim` and `ttn` cannot both be `None`!"
            raise ValueError(errstr)
        if not dim is None:
            positivity_check(dim, "dimension")
            shape = [dim for _ in range(0, len(self.node_identifiers))]
        else:
            shape = [ttn.nodes[node_id].open_dimension()
                     for node_id in self.node_identifiers]
        tensor = self.operator.reshape(shape * 2)
        return NumericOperator(tensor, copy(self.node_identifiers))

    def is_unitary(self) -> bool:
        """
        Returns whether this operator is unitary or not.
        """
        if self.operator.ndim == 2:
            identity = np.eye(self.operator.shape[0])
            return np.allclose(identity, self.operator @ self.operator.conj().T)
        return self.to_matrix().is_unitary()
