"""
To utilise quatum numbers, we need to define special nodes.
"""

from numpy import ndarray

from ..node import Node
from ...util.std_utils import permute_iterator

class QNNode(Node):
    """
    A node that considers quantum numbers.
    """

    def __init__(self,
                 tensor: ndarray | None = None,
                 identifier: str | None = None,
                 qn: list[ndarray] | None = None
                    ) -> None:
        """
        Initialise a QNNode object.

        Args:
            tensor (ndarray | None): The tensor associated with the node.
            identifier (str | None): The identifier for the node.
            qn (list[QNumbers] | None): The quantum numbers associated with the node.
        """
        super().__init__(tensor, identifier)
        if qn is not None and tensor is not None:
            assert len(qn) == tensor.ndim, \
                "Quantum numbers must match tensor dimensions!"
        self._qn = qn

    @property
    def qn(self) -> list[ndarray] | None:
        """
        Return the quantum numbers as they are for the transposed tensor.

        Returns:
            list[ndarray] | None: The quantum numbers of the node.
        """
        if self._qn is None:
            return None
        return permute_iterator(self._qn, self._leg_permutation)

    def __eq__(self, other):
        """
        A quantum number node also needs to compare quantum numbers.
        """
        if not super().__eq__(other):
            return False
        if self.qn is None and other.qn is None:
            return True
        if self.qn is None or other.qn is None:
            return False
        if len(self.qn) != len(other.qn):
            return False
        for qn1, qn2 in zip(self.qn, other.qn):
            if not (qn1 == qn2).all():
                return False
        return True

    def _reset_permutation(self):
        # We also need to reset the quantum numbers
        self._qn = self.qn
        super()._reset_permutation()

    def __str__(self):
        string = super().__str__()
        string += f"Quantum Numbers: {self.qn}\n"
        return string

    def link_tensor(self,
                    tensor: tuple[ndarray, list[ndarray]]
                    ) -> None:
        """
        Link the tensor and quantum numbers to the node.

        Args:
            tensor_and_qn (tuple[ndarray, list[ndarray]]): A tuple containing
                the tensor and quantum numbers.
        """
        tensor, qn = tensor
        assert len(qn) == tensor.ndim, \
            "Quantum numbers must match tensor dimensions."
        super().link_tensor(tensor)
        self._qn = qn

    def qn_valid(self) -> bool:
        """
        The quantum numbers are valid, if they are not None and
        their dimension matches the tensor dimensions.
        """
        if self.qn is None:
            return False
        if len(self.qn) != len(self.shape):
            return False
        for dim, qn in zip(self.shape, self.qn):
            if dim != len(qn):
                return False
        return True

    def get_neighbour_qn(self,
                         neighbour_id: str
                         ) -> list[ndarray] | None:
        """
        Get the quantum numbers of the bond pointing to a fiven neighbour.

        Args:
            neighbour_id (str): The identifier of the neighbour.

        Returns:
            list[ndarray] | None: The quantum numbers of the neighbour bond.
        """
        if self.qn is None:
            return None
        index = self.neighbour_index(neighbour_id)
        return self.qn[index]
