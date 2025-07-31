"""
This module provides utilities for handling metadata files in experiments.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
import os
import json
from collections import UserDict

from h5py import File

from ...time_evolution.results import Results

if TYPE_CHECKING:
    from .sim_params import SimulationParameters

METADATAFILE_STANDARD_NAME = "metadata.json"

def standard_result_file_name(hash_val: str) -> str:
    """
    The standard way to create a result file name from a hash value.

    Args:
        hash_val (str): The hash value to create the file name from.

    Returns:
        str: The standard result file name.
    """
    return f"{hash_val}.h5"

class MetadataFilter(UserDict):
    """
    A class to filter metadata based on keys and values.
    """

    def __init__(self, data: dict[str, Any] = None):
        """
        Initialize the MetadataFilter with a dictionary.

        Args:
            data (dict[str, Any]): A dictionary containing key-value pairs
                to filter metadata. Defaults to an empty dictionary.
        """
        super().__init__(data if data is not None else {})
        self._md_file_path = ""
        self._md_dict = {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetadataFilter:
        """
        Creates a MetadataFilter instance from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary to create the filter from.

        Returns:
            MetadataFilter: An instance of MetadataFilter.
        """
        return cls(data)

    def change_criterium(self,
                         key: str,
                         value: Any
                         ) -> None:
        """
        Changes the validation criteria of the filter.

        Args:
            key (str): The key to change or add in the filter.
            value (Any): The value to associate with the key.
        """
        self.data[key] = value

    def change_criteria(self,
                        criteria: dict[str, Any]
                        ) -> None:
        """
        Changes the validation criteria of the filter.

        Args:
            criteria (dict[str, Any]): A dictionary of key-value pairs to set
                as the validation criteria.
        """
        self.data.update(criteria)

    def remove_criterium(self,
                         key: str
                         ) -> None:
        """
        Removes a key from the validation criteria of the filter.

        Args:
            key (str): The key to remove from the filter.
        """
        if key in self.data:
            del self.data[key]

    def dict_valid(self,
                   dictionary: dict[str, Any]
                   ) -> bool:
        """
        Checks if the given dictionary passes the validation criteria.

        A dictionary is considered valid if all keys in this filter have the
        same value in the dictionary being validated.

        Args:
            dictionary (dict[str, Any]): The dictionary to validate.

        Returns:
            bool: True if the dictionary is valid, False otherwise.
        """
        for key, value in self.data.items():
            if key not in dictionary:
                return False
            if dictionary[key] != value:
                return False
        return True

    def _load_metadata_file(self,
                          directory: str,
                          metadatafile_name: str = METADATAFILE_STANDARD_NAME
                          ) -> None:
        """
        Loads the metadata file from the specified directory.

        Args:
            directory (str): The directory where the metadata file is located.
            metadatafile_name (str): The filename of the metadata file.
                Defaults to `"metadata.json"`.
        """
        metadata_file_path = os.path.join(directory, metadatafile_name)
        if metadata_file_path == self._md_file_path:
            # This avoids reloading the same file if it has not changed.
            return
        if not os.path.exists(metadata_file_path):
            raise FileNotFoundError(f"Metadata file {metadata_file_path} does not exist!")
        with open(metadata_file_path, 'r') as f:
            metadata_index = json.load(f)
        self._md_file_path = metadata_file_path
        self._md_dict = metadata_index

    def filter_hashes(self,
                      directory: str,
                      metadatafile_name: str = METADATAFILE_STANDARD_NAME
                      ) -> set[str]:
        """
        Filters the metadata file in the given directory.

        Args:
            directory (str): The directory where the metadata file is located.
            metadatafile_name (str): The filename of the metadata file.
                Defaults to `"metadata.json"`.
        
        Returns:
            set[str]: A set of hashes from the metadata file.
        """
        self._load_metadata_file(directory, metadatafile_name)
        metadata_index = self._md_dict
        out = set()
        for hash_val, dictionary in metadata_index.items():
            if self.dict_valid(dictionary):
                out.add(hash_val)
        return out

    def load_valid_results(self,
                           directory: str,
                           results_class = Results,
                           file_name_creation: Callable = standard_result_file_name,
                           metadatafile_name: str = METADATAFILE_STANDARD_NAME
                           ) -> list:
        """
        Loads results from the metadata file in the given directory that match

        Args:
            directory (str): The directory where the metadata file and the
                simulation results files are located.
            results_class: The class of the results to load.
                Defaults to `Results`.
            file_name_creation (Callable): A function to create the file name
                from the hash value. Defaults to `standard_result_file_name`.
            metadatafile_name (str): The filename of the metadata file.
                Defaults to `"metadata.json"`.
            
        Returns:
            list: A list of results objects that match the filter criteria.
        """
        hashes = self.filter_hashes(directory, metadatafile_name)
        results = []
        for hash_val in hashes:
            file_name = file_name_creation(hash_val)
            file_path = os.path.join(directory, file_name)
            with File(file_path, 'r') as file:
                result = results_class.load_from_h5(file)
                results.append(result)
        return results

    def load_valid_results_and_parameters(self,
                                          directory: str,
                                            parameter_class: type[SimulationParameters],
                                            results_class = Results,
                                            file_name_creation: Callable = standard_result_file_name,
                                            metadatafile_name: str = METADATAFILE_STANDARD_NAME
                                            ) -> list[tuple[SimulationParameters, Results]]:
        """
        Loads results and their corresponding parameters from the metadata file
        in the given directory that match the filter criteria.

        Args:
            directory (str): The directory where the metadata file and the
                simulation results files are located.
            parameter_class (type[SimulationParameters]): The class of the
                parameters to load.
            results_class: The class of the results to load.
                Defaults to `Results`.
            file_name_creation (Callable): A function to create the file name
                from the hash value. Defaults to `standard_result_file_name`.
            metadatafile_name (str): The filename of the metadata file.
                Defaults to `"metadata.json"`.

        Returns:
            list[tuple[SimulationParameters, Results]]: A list of tuples,
                where each tuple contains a SimulationParameters object and a
                Results object that match the filter criteria.
        """
        hashes = self.filter_hashes(directory, metadatafile_name)
        results = []
        for hash_val in hashes:
            file_name = file_name_creation(hash_val)
            file_path = os.path.join(directory, file_name)
            with File(file_path, 'r') as file:
                result = results_class.load_from_h5(file)
                parameters = parameter_class.from_dict(file.attrs)
                results.append((parameters, result))
        return results

def add_new_key_to_metadata_file(save_directory: str,
                                  key: str,
                                  value: str,
                                  parameter_class: type[SimulationParameters],
                                  metadatafile_name: str = METADATAFILE_STANDARD_NAME
                                  ) -> None:
    """
    Adds a new key-value pair to the metadata file in the save directory.

    This will also adapt the hashes of the existing parameter sets to include
    the new key. The intended use case is to add a new key to the metadata
    file that is was not yet present in the parameter sets.

    Args:
        save_directory (str): The directory where the metadata file is located.
        key (str): The key to add to the metadata file.
        value (str): The value to associate with the key.
        parameter_class (type[SimulationParameters]): The class of the
        metadatafile_name (str): The filename of the metadata file.
            Defaults to `"metadata.json"`.
    """
    metadata_file_path = os.path.join(save_directory, metadatafile_name)
    if not os.path.exists(metadata_file_path):
        raise FileNotFoundError(f"Metadata file {metadata_file_path} does not exist!")
    out = {}
    with open(metadata_file_path, 'r') as f:
        metadata_index = json.load(f)
    for parameters in metadata_index.values():
        parameters[key] = value
        parameters_obj = parameter_class.from_dict(parameters)
        out[parameters_obj.get_hash()] = parameters_obj.to_json_dict()
    with open(metadata_file_path, 'w') as f:
        json.dump(out, f, indent=4)
