"""
Circuits for three qubits.
"""
from __future__ import annotations
from enum import Enum
from typing import Callable

import numpy as np

from ..qcircuit import QCircuit

class ThreeQubitState(Enum):
    """
    Enumeration of three qubit states.
    """
    GHZ = "ghz"
    W = "w"

    def preparation_function(self
                              ) -> Callable[[QCircuit, list[str], int], int]:
        """
        Returns the function that constructs adds the gates to the circuit to
        prepare the corresponding three qubit state.

        Returns:
            Callable[[QCircuit, list[str], int], int]: A function that takes a
                quantum circuit, a list of qubits, and a level index, and adds
                the gates to prepare the corresponding three qubit state.
        """
        if self == ThreeQubitState.GHZ:
            return prepare_ghz
        if self == ThreeQubitState.W:
            return prepare_w
        raise ValueError(f"Unknown three qubit state: {self}!")

def prepare_ghz(circuit: QCircuit,
                qubits: list[str],
                level_index: int
                ) -> int:
    """
    Add the preparation circuit for the GHZ state to the circuit.

    Note, this is not the encoded GHZ state.

    Args:
        circuit (QCircuit): The quantum circuit to which the state preparation will
            be applied.
        qubits (list[str]): A list of qubit IDs for the GHZ state preparation.
        level_index (int): The index of the level in the circuit where the state
            preparation will be applied.

    Returns:
        int: The next level index after applying the GHZ state preparation.
    
    Raises:
        ValueError: If the number of qubits provided is not equal to 3, which is
            required for the GHZ state preparation.
    """
    if len(qubits) != 3:
        errstr = f"Expected 3 qubits for the GHZ state preparation, but got {len(qubits)}!"
        raise ValueError(errstr)
    circuit.add_hadamard(qubits[0],
                         level_index=level_index)
    level_index += 1
    # circuit.add_cnot(qubits[0],
    #                     qubits[1],
    #                     level_index=level_index)
    # level_index += 1
    # circuit.add_cnot(qubits[0],
    #                     qubits[2],
    #                     level_index=level_index)
    circuit.add_mx([qubits[0]], [],
                    qubits[1:],
                    level_index=level_index)
    return level_index + 1

def prepare_w(circuit: QCircuit,
              qubits: list[str],
              level_index: int
              ) -> int:
    """
    Add the preparation circuit for the W state to the circuit.

    Args:
        circuit (QCircuit): The quantum circuit to which the state preparation will
            be applied.
        qubits (list[str]): A list of qubit IDs for the W state preparation.
        level_index (int): The index of the level in the circuit where the state
            preparation will be applied.

    Returns:
        int: The next level index after applying the W state preparation.
    """
    if len(qubits) != 3:
        errstr = f"Expected 3 qubits for the W state preparation, but got {len(qubits)}!"
        raise ValueError(errstr)
    angle = 2*np.arccos(1/np.sqrt(3)) / np.pi # In multiples of pi
    circuit.add_ry(qubits[0],
                   angle,
                   level_index=level_index)
    level_index += 1
    circuit.add_ch(qubits[0],
                   qubits[1],
                   level_index=level_index)
    level_index += 1
    circuit.add_cnot(qubits[1],
                     qubits[2],
                     level_index=level_index)
    level_index += 1
    circuit.add_cnot(qubits[0],
                     qubits[1],
                     level_index=level_index)
    level_index += 1
    circuit.add_x(qubits[0],
                  level_index=level_index)
    return level_index + 1