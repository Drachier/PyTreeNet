"""
This module provides utilities for handling metadata files in experiments.
"""
from __future__ import annotations
from warnings import warn
from typing import TYPE_CHECKING, Any, Callable
import os
import json
from collections import UserDict
from copy import deepcopy
import shutil

from h5py import File

from ...time_evolution.results import Results
from .sim_params import SimulationParameters
from .status_enum import Status

METADATAFILE_STANDARD_NAME = "metadata.json"
STANDARD_REFERENCE_FLAG = "exact"

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

        ## These are purely internal variables to avoid reloading the
        ## metadata file if it has not changed. Irrelevant when not dealing
        ## with files.
        self._md_file_path = ""
        self._md_dict = {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetadataFilter:
        """
        Creates a MetadataFilter instance from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary to create the filter from.
                Can contain lists as values to allow for multiple values to be
                checked for a key.

        Returns:
            MetadataFilter: An instance of MetadataFilter.
        """
        return cls(data)

    def filters_parameter(self,
                         parameter: str
                         ) -> bool:
        """
        Checks if the filter contains a given parameter.

        Args:
            parameter (str): The parameter to check for.

        Returns:
            bool: True if the parameter is in the filter, False otherwise.
        """
        return parameter in self.data

    def get_criterium(self,
                      key: str
                      ) -> list[Any]:
        """
        Gets the value(s) associated with a key in the filter.

        Args:
            key (str): The key to get the value(s) for.

        Returns:
            list[Any]: A list of values associated with the key. If the key
                does not exist, an empty list is returned. If the value is not
                a list, it is returned as a single-element list.
        """
        if not self.filters_parameter(key):
            return []
        value = self.data[key]
        if not isinstance(value, list):
            value = [value]
        return value

    def change_criterium(self,
                         key: str,
                         value: Any | list[Any]
                         ) -> None:
        """
        Changes the validation criteria of the filter.

        Args:
            key (str): The key to change or add in the filter.
            value (Any | list[Any]): The value to associate with the key.
        """
        if not isinstance(value, list):
            value = [value]
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

    def add_to_criterium(self,
                         key: str,
                         value: Any | list[Any]
                         ) -> None:
        """
        Adds a value to the list of values for a given key in the filter.

        Args:
            key (str): The key to add the value to.
            value (Any | list[Any]): The value or values to add to the key.
        """
        if key not in self.data:
            self.data[key] = value
            return
        if not isinstance(value, list):
            value = [value]
        if isinstance(self.data[key], list):
            for v in value:
                if v not in self.data[key]:
                    self.data[key].append(v)
        else:
            self.data[key] = [self.data[key]]
            self.data[key].extend(value)

    def remove_from_criterium(self,
                                key: str,
                                value: Any
                                ) -> None:
        """
        Removes a value from the list of values for a given key in the filter.

        Args:
            key (str): The key to remove the value from.
            value (Any): The value to remove from the key.
        """
        if key in self.data:
            if isinstance(self.data[key], list):
                if value in self.data[key]:
                    self.data[key].remove(value)
            else:
                if self.data[key] == value:
                    self.remove_criterium(key)

    def remove_value(self,
                     value: Any):
        """
        Removes a value from all keys in the filter.

        Args:
            value (Any): The value to remove from all keys.
        """
        for key in list(self.data.keys()):
            self.remove_from_criterium(key, value)

    def remove_all_but_value(self,
                             value: Any):
        """
        From entries with lists, removes all values except the given one.

        Args:
            value (Any): The value to keep in all keys.
        """
        for key in list(self.data.keys()):
            if isinstance(self.data[key], list):
                if value in self.data[key]:
                    self.data[key] = [value]

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
            if isinstance(value, list):
                if dictionary[key] not in value:
                    return False
            else:
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
        the filter criteria.

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

    def load_unique_results(self,
                           directory: str,
                           results_class = Results,
                           file_name_creation: Callable = standard_result_file_name,
                           metadatafile_name: str = METADATAFILE_STANDARD_NAME
                           ) -> Results:
        """
        Loads unique results from the metadata file in the given directory.

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
            Results: A single results object that matches the filter criteria.

        Raises:
            ValueError: If the number of results found is not exactly one.
        """
        results = self.load_valid_results(directory, results_class,
                                          file_name_creation, metadatafile_name)
        if len(results) != 1:
            raise ValueError(f"Expected exactly one result, found: {len(results)}!")
        return results[0]

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

    def load_reference_data(self,
                            directory: str,
                            parameter_class: type[SimulationParameters],
                            results_class = Results,
                            file_name_creation: Callable = standard_result_file_name,
                            metadatafile_name: str = METADATAFILE_STANDARD_NAME,
                            reference_flag: str = STANDARD_REFERENCE_FLAG
                            ) -> list[tuple[SimulationParameters, Results]]:
        """
        Loads only the the reference data.

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
            reference_flag (str): The value of the key that indicates reference
                data. Defaults to `"exact"`.

        Returns:
            list[tuple[SimulationParameters, Results]]: A list of tuples,
                where each tuple contains a SimulationParameters object and a
                Results object that are marked as reference data.
        """
        ref_filter = deepcopy(self)
        ref_filter.remove_all_but_value(reference_flag)
        return ref_filter.load_valid_results_and_parameters(
            directory,
            parameter_class,
            results_class,
            file_name_creation,
            metadatafile_name
        )

    def load_simulation_data(self,
                            directory: str,
                            parameter_class: type[SimulationParameters],
                            results_class = Results,
                            file_name_creation: Callable = standard_result_file_name,
                            metadatafile_name: str = METADATAFILE_STANDARD_NAME,
                            reference_flag: str = STANDARD_REFERENCE_FLAG
                            ) -> list[tuple[SimulationParameters, Results]]:
        """
        Loads only the simulation data (i.e., non-reference data).

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
            reference_flag (str): The value of the key that indicates reference
                data. Defaults to `"exact"`.

        Returns:
            list[tuple[SimulationParameters, Results]]: A list of tuples,
                where each tuple contains a SimulationParameters object and a
                Results object that are not marked as reference data.
        """
        sim_filter = deepcopy(self)
        sim_filter.remove_value(reference_flag)
        return sim_filter.load_valid_results_and_parameters(
            directory,
            parameter_class,
            results_class,
            file_name_creation,
            metadatafile_name
        )

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

def generate_metadata_file(
    save_directory: str,
    metadatafile_name: str = METADATAFILE_STANDARD_NAME,
    parameter_class: type[SimulationParameters] = SimulationParameters):
    """
    Generates a metadata file, using the .h5 files in the save directory.

    Args:
        save_directory (str): The directory where the metadata file should be
            created.
        metadatafile_name (str): The filename of the metadata file.
            Defaults to `"metadata.json"`.
        parameter_class (type[SimulationParameters]): The class of the
            parameters to load. Defaults to `SimulationParameters`.
        
    Raises:
        FileNotFoundError: If no .h5 files are found in the save directory.
    """
    metadata_file_path = os.path.join(save_directory, metadatafile_name)
    if not os.path.exists(save_directory):
        raise FileNotFoundError(f"Save directory {save_directory} does not exist!")
    out = {}
    for file_name in os.listdir(save_directory):
        if file_name.endswith('.h5'):
            file_path = os.path.join(save_directory, file_name)
            with File(file_path, 'r') as file:
                parameters = dict(file.attrs)
                parameters_obj = parameter_class.from_dict(parameters)
                hash = parameters_obj.get_hash()
                out[hash] = parameters_obj.to_json_dict()
    if len(out) == 0:
        raise FileNotFoundError(f"No .h5 files found in {save_directory}!")
    with open(metadata_file_path, 'w') as f:
        json.dump(out, f, indent=4)

def combine_metadata_files(
        source_directories: list[str],
        save_directory: str,
        metadatafile_name: str = METADATAFILE_STANDARD_NAME
        ) -> None:
    """
    Combines metadata files from multiple source directories into a single
    metadata file in the save directory.

    Args:
        source_directories (list[str]): A list of directories where the
            source metadata files are located.
        save_directory (str): The directory where the combined metadata file
            should be saved.
        metadatafile_name (str): The filename of the metadata files.
            Defaults to `"metadata.json"`.
    
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    elif save_directory in source_directories:
        del source_directories[source_directories.index(save_directory)]
    if metadatafile_name in os.listdir(save_directory):
        filepath = os.path.join(save_directory, metadatafile_name)
        with open(filepath, 'r') as f:
            combined_metadata = json.load(f)
    else:
        combined_metadata = {}
    for directory in source_directories:
        metadata_file_path = os.path.join(directory, metadatafile_name)
        if not os.path.exists(metadata_file_path):
            warn(f"Metadata file {metadata_file_path} does not exist! Skipping...")
            continue
        with open(metadata_file_path, 'r') as f:
            metadata_index = json.load(f)
        for hash_val, parameters in metadata_index.items():
            if hash_val in combined_metadata:
                existing = combined_metadata[hash_val]
                existing_status = Status(existing.get("status", "unknown"))
                new_status = Status(parameters.get("status", "unknown"))
                if new_status > existing_status:
                    combined_metadata[hash_val] = parameters
            else:
                combined_metadata[hash_val] = parameters
    combined_metadata_file_path = os.path.join(save_directory, metadatafile_name)
    with open(combined_metadata_file_path, 'w') as f:
        json.dump(combined_metadata, f, indent=4)

def combine_data_directories(
        source_directories: list[str],
        save_directory: str,
        metadatafile_name: str = METADATAFILE_STANDARD_NAME
        ) -> None:
    """
    Combines data directories from multiple source directories into a single
    save directory, merging their metadata files.

    Args:
        source_directories (list[str]): A list of directories where the
            source data directories are located.
        save_directory (str): The directory where the combined data should be
            saved.
        metadatafile_name (str): The filename of the metadata files.
            Defaults to `"metadata.json"`.
    
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    elif save_directory in source_directories:
        del source_directories[source_directories.index(save_directory)]
    combine_metadata_files(source_directories,
                           save_directory,
                           metadatafile_name=metadatafile_name)
    copied_hashes = set()
    for directory in source_directories:
        for file_name in os.listdir(directory):
            if file_name.endswith('.h5'):
                source_path = os.path.join(directory, file_name)
                dest_path = os.path.join(save_directory, file_name)
                if os.path.exists(dest_path):
                    warn(f"File {dest_path} already exists! Skipping...")
                    continue
                shutil.copy2(source_path, dest_path)
                copied_hashes.add(file_name)
    print(f"Copied {len(copied_hashes)} files to {save_directory}:")
    for hash_val in copied_hashes:
        print(f" - {hash_val}")
