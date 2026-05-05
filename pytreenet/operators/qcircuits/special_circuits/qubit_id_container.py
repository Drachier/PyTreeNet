"""
Qubit ID Container class for ease of use with multi-part quantum circuits.
"""
from __future__ import annotations
from typing import Callable, Self

QUBIT_ID = "qubit"
QUBIT_PREFIX = QUBIT_ID+"_"

def gen_qubit_id(i: int) -> str:
    """
    Generates a qubit ID string based on the given index.

    Args:
        i (int): The index of the qubit.

    Returns:
        str: The generated qubit ID string.
    """
    return f"{QUBIT_ID}_{i}"

class QubitIDContainer:
    """
    A class to easily distuingish the different types of qubits in the circuit.

    Mainly used for readability, as the qubits are just identified by strings and the
    different types of qubits are interspersed in the circuit.

    It is assumed that the qubits are ordered consecutively for the same part
    of the system.
    """

    def __init__(self,
                 gen_qubit_id_func: Callable[[int], str],
                 tot_num_qub_per_log: int,
                 num_log_qubits: int = 1
                 ) -> None:
        self.tot_num_qub_per_log = tot_num_qub_per_log
        self.num_log_qubits = num_log_qubits
        self.tot_num_qubits = self.num_log_qubits*self.tot_num_qub_per_log
        self.qubit_ids = [gen_qubit_id_func(i) for i in range(self.tot_num_qubits)]
        self._node_prefix: str | None = None

    @property
    def node_prefix(self) -> str:
        """
        Get the node prefix for the qubits. This is used for the TTN state generation.
        """
        if self._node_prefix is None:
            errstr = "No node prefix is set for the QubitIDContainer!"
            raise ValueError(errstr)
        return self._node_prefix
    
    def main_index_valid(self, index: int) -> bool:
        """
        Check if the given index is valid for the main qubits.
        """
        return index >= 0 and index < self.num_log_qubits

    def _main_index_check(self, index: int) -> None:
        """
        Check if the given index is valid for the main qubits and raise an error if not.
        """
        if not self.main_index_valid(index):
            errstr = f"Invalid index {index} for main qubits! Must be between 0 and {self.num_log_qubits-1}."
            raise ValueError(errstr)
        
    def _main_index_to_actual_index(self, index: int) -> int:
        """
        Convert the given index for the main qubits to the actual index in the qubit ID list.
        """
        self._main_index_check(index)
        return index*self.tot_num_qub_per_log
    
    def main_qubit(self, index: int) -> str:
        """
        Get the ID of the main qubit number `index`.
        """
        return self.qubit_ids[self._main_index_to_actual_index(index)]
    
    def main_qubits(self) -> list[str]:
        """
        Get the IDs of all main qubits.
        """
        return [self.main_qubit(i) for i in range(self.num_log_qubits)]
    
    def all_qubits_per_logical(self, index: int) -> list[str]:
        """
        Get the IDs of all qubits for the logical qubit number `index`.
        """
        self._main_index_check(index)
        start_index = self._main_index_to_actual_index(index)
        end_index = self._main_index_to_actual_index(index+1)
        return self.qubit_ids[start_index:end_index]

    @classmethod
    def with_standard_gen_func(cls,
                               tot_num_qub_per_log: int,
                               num_log_qubits: int = 1) -> Self:
        """
        Create a QubitIDContainer with the standard qubit ID generation function.
        """
        out = cls(gen_qubit_id,
                  tot_num_qub_per_log,
                  num_log_qubits=num_log_qubits)
        out._node_prefix = QUBIT_PREFIX
        return out
