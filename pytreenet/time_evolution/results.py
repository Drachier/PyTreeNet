"""
This module implements a class to store the simulation results of a time
evolution algorithm in PyTreeNet.
"""
from __future__ import annotations
from typing import Hashable, Any, Self
import re

from numpy.typing import NDArray, DTypeLike
import numpy as np
from h5py import File

from ..util.std_utils import average_data

TIMES_ID = "times"

class Results:
    """
    Class to store the simulation results of a time evolution algorithm.

    Attributes:
        results (dict[Hashable, NDArray]): A dictionary where keys are operator
            names and values are their corresponding results as NumPy arrays.
        attributes (dict[str, tuple[str, Any]]): A dictionary to store additional
            attributes for each operator. The keys are operator names and the
            values are lists of tuples, where each tuple contains an attribute
            name and its corresponding value.
    """

    def __init__(self,
                 metadata: dict[str,Any] | None = None
                 ) -> None:
        """
        Initializes the Results object with the number of time steps.

        Args:
            metadata (dict[str, Any]): Metadata containing other information
                about the simulation, not directly relevant to the results.
        """
        self.results: dict[Hashable, NDArray] = {}
        self.attributes: dict[str, list[tuple[str,Any]]] = {}
        self.metadata = metadata if metadata is not None else {}

    def close_to(self, other: Self) -> bool:
        """
        Checks if the results object is close to another results object.

        Args:
            other (Results): Another Results object to compare with.

        Returns:
            bool: True if the results are close, False otherwise.
        """
        if len(self.results) != len(other.results):
            return False
        for key, val in self.results.items():
            if key not in other.results:
                return False
            if not np.allclose(val, other.results[key]):
                return False
        if self.metadata != other.metadata:
            return False
        if len(self.attributes) != len(other.attributes):
            return False
        for key, val in self.attributes.items():
            if key not in other.attributes:
                return False
            if len(val) != len(other.attributes[key]):
                return False
            for attr in val:
                if attr not in other.attributes[key]:
                    return False
        return True

    def num_results(self) -> int:
        """
        Returns the number of results stored in the results object.

        Returns:
            int: The number of results.
        """
        return len(self.results)

    def results_length(self) -> int:
        """
        Returns the length of the results arrays.

        Returns:
            int: The length of the results arrays.
        """
        self.not_initialized_error()
        return len(self.results[TIMES_ID])

    def shape(self) -> tuple[int, int]:
        """
        Returns the shape of the result.

        Returns:
            tuple[int, int]: A tuple containing the number of operators and the
                length of the results arrays. This includes the time array and
                the result at time zero.
        """
        return (self.num_results(),
                self.results_length())

    def is_initialized(self) -> bool:
        """
        Checks if the results object is initialized.

        Returns:
            bool: True if the results object is initialized, False otherwise.
        """
        return bool(self.results)

    def not_initialized_error(self) -> None:
        """
        Raises an error if the results object is not initialized.

        Raises:
            ValueError: If the results object is not initialized.
        """
        if not self.is_initialized():
            raise ValueError("Results object is not initialized!")

    def initialize(self,
                   operators: dict[Hashable, DTypeLike],
                   num_time_steps: int,
                   with_time: bool = True
                   ) -> None:
        """
        Initializes the results object with the given operators and number of
         time steps.

        Args:
            operators (dict[Hashable, DTypeLike]): A dictionary where keys are
                operator names and values are their corresponding data types.
                If the data type is not specified, it defaults to complex128.
            num_time_steps (int): The number of time steps in the simulation.
            with_time (bool): If True, includes a time array in the results.
                Defaults to True.

        Raises:
            ValueError: If any operator name is an integer.
        """
        if self.is_initialized():
            raise ValueError("Results object is already initialized!")
        if with_time:
            self.results[TIMES_ID] = np.zeros((num_time_steps + 1,),
                                                      dtype=np.float64)
        for operator, dtype in operators.items():
            if isinstance(operator, int):
                raise ValueError("Operator names mustn't be integers!")
            if dtype is None:
                dtype = np.complex128
            self.results[operator] = np.zeros((num_time_steps + 1,),
                                                      dtype=dtype)

    def set_attribute(self,
                      operator_id: str,
                      attribute_name: str,
                      value: Any) -> None:
        """
        Sets an attribute in the results object.

        Args:
            operator_id (str): The name of the attribute.
            value (Any): The value of the attribute.
        """
        if operator_id not in self.results:
            self.attributes[operator_id] = []
        self.attributes[operator_id].append((attribute_name,value))

    def set_element(self,
                     operator: Hashable,
                     index: int,
                     value: Any
                     ) -> None:
        """
        Sets the value of a specific operator at a given index.

        Args:
            operator (Hashable): The name of the operator.
            index (int): The index at which to set the value.
        
        """
        self.results[operator][index] = value

    def get_element(self,
                     operator: Hashable,
                     index: int
                     ) -> Any:
        """
        Gets the value of a specific operator at a given index.

        Args:
            operator (Hashable): The name of the operator.
            index (int): The index from which to get the value.

        Returns:
            Any: The value of the operator at the specified index.
        """
        return self.results[operator][index]

    def set_operator_result(self,
                            operator: Hashable,
                            values: NDArray) -> None:
        """
        Sets the result of a specific operator.

        Args:
            operator (Hashable): The name of the operator.
            values (NDArray): An array containing the results of the operator.
        """
        self.results[operator] = values

    def set_time(self,
                  index: int,
                  time: float) -> None:
        """
        Sets the time at a specific index.

        Args:
            index (int): The index at which to set the time.
            time (float): The time value to set.
        """
        self.results[TIMES_ID][index] = time

    def get_time(self, index: int) -> float:
        """
        Gets the time at a specific index.

        Args:
            index (int): The index from which to get the time.

        Returns:
            float: The time value at the specified index.
        """
        return self.results[TIMES_ID][index]

    def times(self) -> NDArray:
        """
        Returns the time values stored in the results.

        Returns:
            NDArray: An array of time values.
        """
        self.not_initialized_error()
        return self.results[TIMES_ID]

    def result_real(self,
                    operator: Hashable) -> bool:
        """
        Checks if the result of a specific operator is real.

        Args:
            operator (Hashable): The name of the operator.

        Returns:
            bool: True if the result is real, False otherwise.
        """
        op_results = self.results.get(operator)
        if np.isrealobj(op_results):
            return True
        return np.allclose(np.imag(op_results),
                           np.zeros_like(op_results))

    def results_real(self) -> bool:
        """
        Checks if all results in the results object are real.

        Returns:
            bool: True if all results are real, False otherwise.
        """
        for operator in self.results:
            if not self.result_real(operator):
                return False
        return True

    def operator_result(self,
                        operator: Hashable,
                        realise: bool = False
                        ) -> NDArray:
        """
        Returns the result of a specific operator.

        Args:
            operator (Hashable): The name of the operator.
            realise (bool): If True, returns the real part of the result.
        
        Returns:
            NDArray: The result of the operator, either real or complex.
        """
        self.not_initialized_error()
        op_results = self.results[operator]
        if realise and not np.isrealobj(op_results):
            return np.real(op_results)
        return op_results

    def _operator_key_desired(self,
                              operator_key: Hashable,
                              operators: list[Hashable] | re.Pattern | None
                              ) -> bool:
        """
        Checks if the operator key matches the desired operators.

        Args:
            operator_key (Hashable): The key of the operator to check.
            operators (list[Hashable] | re.Pattern | None): A list of operator
                names or a regex pattern to filter operators. If None, all
                operators except for the time are considered.
        
        Returns:
            bool: True if the operator key matches the desired operators,
                False otherwise.
        """
        if operators is None:
            return operator_key != TIMES_ID
        if isinstance(operators, list):
            return operator_key in operators
        if isinstance(operators, re.Pattern):
            if not isinstance(operator_key, str):
                errstr = "Operators must be strings to match against a regex pattern!"
                raise TypeError(errstr)
            return bool(operators.match(operator_key))
        errstr = "Operators must be a list of Hashable or a regex pattern!"
        raise TypeError(errstr)

    def get_results(self,
                operators: list[Hashable] | str | re.Pattern | None = None,
                realise: bool = False
                ) -> dict[Hashable, NDArray]:
        """
        Returns the results of the specified operators.

        Args:
            operators (list[Hashable] | re.Pattern | None): A list of operator
                names or a regex pattern to filter operators. If it is a
                string, all operators starting with that string are returned.
                If None, all operators except for the time are returned.
            realise (bool): If True, returns the real parts of the results.
        
        Returns:
            dict[Hashable, NDArray]: A dictionary containing the results of
                the specified operators.
        """
        self.not_initialized_error()
        if isinstance(operators, str):
            operators = re.compile(r"^" + operators + r"\d+$")
        return {
            operator_key: self.operator_result(operator_key, realise=realise)
            for operator_key in self.results
            if self._operator_key_desired(operator_key, operators)
        }

    def operator_results(self,
                         operators: list[Hashable] | None = None,
                         realise: bool = False
                         ) -> NDArray:
        """
        Returns all the results of the specified operators.

        Args:
            operators (list[Hashable]): A list of operator names. If None
             all operators except for the time are returned.
            realise (bool): If True, returns the real parts of the results.

        Returns:
            NDArray: An array containing the results of the specified operators.

        """
        self.not_initialized_error()
        if operators is None:
            operators = list(self.results.keys())
        out = np.zeros_like(self.results[operators[0]],
                            shape=(len(operators),
                                     len(self.results[operators[0]])),
                            dtype=complex)
        for i, operator in enumerate(operators):
            if operator != TIMES_ID:
                out[i, :] = self.operator_result(operator, realise)
        if realise:
            out = np.real(out)
        return out

    def average_results(self,
                        operators: list[Hashable] | str | re.Pattern | None = None,
                        realise: bool = False
                        ) -> NDArray:
        """
        Averages the results of the specified operators.

        Args:
            operators (list[Hashable] | None): A list of operator names or a
                regex pattern to filter operators. If None, all operators
                except for the time are averaged. If it is a string, all
                operators starting with that string are averaged.
            realise (bool): If True, averages the real parts of the results.
        
        Returns:
            NDArray: An array containing the averaged results of the specified
                operators.
        """
        results = self.get_results(operators=operators, realise=realise)
        data = list(results.values())
        return average_data(data)

    def save_to_h5(self,
                   file: str | File) -> None:
        """
        Saves the results to an HDF5 file.

        Args:
            file (str | File): The path to the HDF5 file or an open h5py File
             object. If a string is provided, the file will be opened in write
             mode.

        Raises:
            ValueError: If the results object is not initialized.
        """
        self.not_initialized_error()
        if isinstance(file, str):
            with File(file, "w") as h5file:
                self.save_to_h5(h5file)
        else:
            for key, value in self.results.items():
                if not isinstance(key, str):
                    key_str = str(key)
                else:
                    key_str = key
                dset = file.create_dataset(key_str, data=value)
                if key in self.attributes:
                    for attr in self.attributes[key]:
                        attr_key, attr_value = attr
                        dset.attrs[attr_key] = attr_value
            if TIMES_ID in self.results:
                file.attrs[TIMES_ID] = self.results[TIMES_ID]
                file.attrs["num_time_steps"] = len(self.results[TIMES_ID]) - 1
            else:
                file.attrs["num_measurements"] = len(next(iter(self.results.values())))

    @classmethod
    def load_from_h5(cls,
                     file: str | File,
                     loaded_ops: list[str] | None = None
                     ) -> Self:
        """
        Loads the results from an HDF5 file.

        Args:
            file (str | File): The path to the HDF5 file or an open h5py File
             object. If a string is provided, the file will be opened in read
             mode.
            loaded_ops (list[str] | None): A list of operator names to load into
                the results object. If None, all operators are loaded.
        
        Returns:
            Results: An instance of the Results class containing the loaded
             data.
        
        """
        if isinstance(file, str):
            with File(file, "r") as h5file:
                return cls.load_from_h5(h5file,
                                        loaded_ops=loaded_ops)
        results = cls(metadata=dict(file.attrs))
        if loaded_ops is None:
            iterator = file.keys()
        else:
            iterator = loaded_ops
        for i, key in enumerate(iterator):
            dset = file[key]
            attrs = dset.attrs
            for attr_key, attr_value in attrs.items():
                results.set_attribute(key, attr_key, attr_value)
            loaded_data = dset[:]
            if i == 0:
                len_data = len(loaded_data)
            else:
                if len(loaded_data) != len_data:
                    raise ValueError("All datasets must have the same length!")
            results.results[key] = loaded_data
        return results
