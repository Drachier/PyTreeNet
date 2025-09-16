"""
This module implements a simulation parameter parent class that can be
subclassed to extend the simulation parameters for different experiments.
"""
from __future__ import annotations
from typing import Any, Self, get_type_hints
from dataclasses import dataclass, fields
from enum import Enum
import json
from hashlib import sha256

from h5py import File
import numpy as np

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
            elif isinstance(value, (np.generic,)):       # numpy scalar (e.g. np.float64)
                out[name] = value.item()
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

        Args:
            param_dict (dict[str, Any]): A dictionary containing the simulation
                parameters.
        """
        # We have to handle Enums manually
        # because they may not be supported directly
        type_hints = get_type_hints(cls)
        kwargs = {}
        for field in fields(cls):
            if field.name in param_dict:
                value = param_dict[field.name]
                field_type = type_hints.get(field.name, field.type)
                if isinstance(field_type, type) and issubclass(field_type, Enum) and not isinstance(value, Enum):
                    value = field_type(value)
                kwargs[field.name] = value
        return cls(**kwargs)

    @classmethod
    def load_from_json(cls,
                       filepath: str
                       ) -> Self:
        """
        Loads the simulation data from a json file.

        Args:
            filepath (str): The path to the json file.
        """
        with open(filepath, "r") as file:
            data = json.load(file)
        return cls.from_dict(data)

    def get_hash(self) -> str:
        """
        Returns a hash of the simulation parameters.

        Returns:
            str: A string representation of the hash.
        """
        json_dict = self.to_json_dict()
        param_str = json.dumps(json_dict, sort_keys=True)
        return sha256(param_str.encode()).hexdigest()
