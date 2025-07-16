"""
This module implements a simulation parameter parent class that can be
subclassed to extend the simulation parameters for different experiments.
"""
from __future__ import annotations
from typing import Any, Self
from dataclasses import dataclass
from enum import Enum
import json
from hashlib import sha256

from h5py import File

@dataclass
class SimulationParameters:
    """
    A parent class for simulation parameters.

    This class should be extended by specific simulation parameter classes.
    """

    def to_json_dict(self) -> dict[str, Any]:
        """
        Converts the simulation parameters to a dictionary that can be 
        converted to JSON.

        Returns:
            dict[str, Any]: A dictionary representation of the simulation
                parameters.
        """
        out = {}
        for name, value in self.__dict__.items():
            if isinstance(value, Enum):
                out[name] = value.value
            else:
                out[name] = value
        return out

    def save_to_h5(self, file: File) -> File:
        """
        Saves the simulation parameters to an HDF5 file.

        Args:
            file (File): The HDF5 file to save the parameters to.

        Returns:
            File: The HDF5 file with the saved parameters.
        """
        for name, value in self.__dict__.items():
            if isinstance(value, Enum):
                value = value.value
            file.attrs[name] = value
        return file

    @classmethod
    def from_dict(cls, param_dict: dict[str, Any]) -> Self:
        """
        Initializes the simulation parameters from a dictionary.

        If a subclass has enums, it should extend this method to convert
        the string values back to enum types.

        Args:
            param_dict (dict[str, Any]): A dictionary containing the simulation
                parameters.
        """
        return cls(**param_dict)

    def get_hash(self) -> str:
        """
        Returns a hash of the simulation parameters.

        Returns:
            str: A string representation of the hash.
        """
        json_dict = self.to_json_dict()
        param_str = json.dumps(json_dict, sort_keys=True)
        return sha256(param_str.encode()).hexdigest()
