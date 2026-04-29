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
                 renormalize: bool = True,
                 reset: bool = False
                 ) -> None:
        """
        Initializes a Measurement instance.

        Args:
            measures (dict[str,int], optional): A dictionary specifying the
                measurements. The keys are node IDs and the values are the
                measurement outcomes (0 or 1). Defaults to None, which creates
                an empty Measurement.
            renormalize (bool, optional): Whether to renormalize the state after
                applying the measurement. Defaults to True.
            reset (bool, optional): Wether to force a reset on the desired
                measurement outcome. This means, if the state does not have any
                overlap with the desired measurement outcome, the next higher
                operation will be applied, followed by an appropriate manipulation
                of the state to cause the reset.
        """
        super().__init__(measures or {})
        self.renormalize = renormalize
        self.reset = reset

    def node_ids(self) -> list[str]:
        """
        Returns the list of node IDs involved in the Measurement.

        Returns:
            list[str]: The list of node IDs.
        """
        return list(self.data.keys())

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
        return cls(measures, reset=True)

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
        return cls(measures, **kwargs)

    @classmethod
    def empty(cls,
              **kwargs) -> Self:
        """
        Creates an empty Measurement.

        Returns:
            Self: An empty Measurement instance.
        """
        return cls(**kwargs)

    def is_empty(self) -> bool:
        """
        Checks if the Measurement is empty.

        Returns:
            bool: True if the Measurement is empty, False otherwise.
        """
        return len(self.data) == 0

    def otimes(self,
               other: Measurement
               ) -> Self:
        """
        Combines two Measurements using the outer tensor product.

        Args:
            other (Measurement): The other Measurement to combine with.
        
        Returns:
            Measurement: A new Measurement representing the combined measurements.
        
        Raises:
            ValueError: If there are overlapping node IDs in the two Measurements or if
                        the two measurements don't agree on renormalization.
        """
        overlapping_keys = set(self.data.keys()).intersection(set(other.data.keys()))
        if overlapping_keys:
            errstr = f"Cannot combine Measurements with overlapping node IDs: {overlapping_keys}!"
            raise ValueError(errstr)
        if self.renormalize != other.renormalize:
            errstr = "Cannot combine Measurements with different renormalization settings!"
            raise ValueError(errstr)
        if self.reset != other.reset:
            errstr = "Cannot combine a reset Measurement with a non-reset Measurement!"
            raise ValueError(errstr)
        new = self.__class__(renormalize=self.renormalize,
                             reset=self.reset)
        new.data = {**self.data, **other.data}
        return new

    def system_size(self) -> int:
        """
        Returns the number of nodes involved in the Measurement.

        Returns:
            int: The number of nodes.
        """
        return len(self.data)

    def __eq__(self,
               other: object
               ) -> bool:
        if not isinstance(other, Measurement):
            return False
        return self.renormalize == other.renormalize and self.reset == other.reset and self.data == other.data
