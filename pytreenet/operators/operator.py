from __future__ import annotations
from typing import List, Union, Dict
from copy import copy

import numpy as np

class Operator:
    """
    An operator hold the information what operation to apply to which node in a TTN.
    """

    def __init__(self, operator: Union[str, np.ndarray], node_identifiers: List[str]):
        self.operator = operator
        self.node_identifiers = node_identifiers

class NumericOperator(Operator):
    """
    An operator that holds the operator associated with it directly as an array.
     If the associated array is not a matrix, we assume the following leg order:
      `(input_node1, input_node2, ..., input_noden, output_node1, ... output_noden)`
    """

    def __init__(self, operator: np.ndarray, node_identifiers: List[str]):
        assert operator.ndim % 2 == 0
        super().__init__(operator, node_identifiers)

    def to_matrix(self) -> NumericOperator:
        """
        Returns a new NumericOperator whose operator is the current operator
         but in matrix form, i.e. all input legs are combined and all output
         legs are combined
        """
        half_dim = int(self.operator.ndim / 2)
        new_dim = np.prod(self.operator.shape[0:half_dim])
        matrix = self.operator.reshape((new_dim, new_dim))
        return NumericOperator(matrix, copy(self.node_identifiers))

    def to_tensor(self, dim : Union[int, None] = None,
                  ttn : Union[TreeTensorNetwork, None]=None) -> NumericOperator:
        """
        Returns a NumericOperator whose operator is the current operator but in tensor form,
         i.e. has one leg for each identifier in node_identifier. 

        Args:
            dim (Union[int, None], optional): If all nodes have the same open dimension
             it can be given here. Defaults to None.
            ttn (Union[TreeTensorNetwork, None], optional): If not all nodes habe the same open
             dimension a TTN is needed as a reference. Defaults to None.

        Raises:
            ValueError: If all inputs are None

        Returns:
            NumericOperator: A NumericOperator whose operator is the current operator but in tensor
             form, i.e. has one leg for each identifier in node_identifier. 
        """
        if dim is None and ttn is None:
            errstr = "`dim` and `ttn` cannot both be `None`!"
            raise ValueError(errstr)
        if not dim is None:
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
        identity = np.eye(self.operator.shape[0])
        return np.allclose(identity, self.operator @ self.operator.conj().T)

class SymbolicOperator(Operator):
    """
    An operator that holds the operator associated with it only as a symbolic value.
    That operator has to be converted before actual use.
    """

    def __init__(self, operator: str, node_identifiers: List[str]):
        super().__init__(operator, node_identifiers)

    def to_numeric(self, conversion_dict: Dict[str, np.ndarray]) -> NumericOperator:
        """
        Converts a symbolic operator into an equivalent numeric operator.

        Args:
            conversion_dict (Dict[str, np.ndarray]): The numeric values in the form of
             an array for the symbol.

        Returns:
            NumericOperator: The converted operator.
        """
        return NumericOperator(conversion_dict[self.operator],
                               self.node_identifiers)
