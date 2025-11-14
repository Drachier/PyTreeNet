"""
This module implements the measurement class
"""
from __future__ import annotations
from collections import UserDict
from typing import Self

class Measurement(UserDict):
    """
    A class to represent a measurement on a Tree Tensor Network.
    """

    def __init__(self,
                 measures: dict[str,int] | None = None,
                 renormalize: bool = True
                 ) -> None:
        super().__init__(measures or {})
        self.renormalize = renormalize

    @classmethod
    def create_reset_measurement(cls,
                                 nodes: list[str],
                                 **kwargs
                                 ) -> Self:
        """
        Creates a reset measurement.

        Args:
            nodes (list[str]): The list of node IDs to reset.
        
        Returns:
            Self: The reset measurement.
        """
        measures = {node_id: 0 for node_id in nodes}
        return cls(measures)

    @classmethod
    def from_dict(cls,
                  measures: dict[str,int],
                  **kwargs
                  ) -> Self:
        """
        Creates a Measurement from a dictionary.

        Args:
            measures (dict[str,int]): A dictionary specifying the measurements.
        
        Returns:
            Self: The created Measurement instance.
        """
        return cls(measures)

    @classmethod
    def empty(cls,
              **kwargs) -> Self:
        """
        Creates an empty Measurement.

        Returns:
            Self: An empty Measurement instance.
        """
        return cls()

    def is_empty(self) -> bool:
        """
        Checks if the Measurement is empty.

        Returns:
            bool: True if the Measurement is empty, False otherwise.
        """
        return len(self.data) == 0
