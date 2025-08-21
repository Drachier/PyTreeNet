"""
This file implements some frequently used quantum circuits.
"""
from __future__ import annotations
from typing import Self
from enum import Enum

from .qcircuit import QCircuit
from .qgate import QGate

class BellState(Enum):
    """
    Enum for Bell states.

    The Bell states are defined as follows:
    - B00: (|00⟩ + |11⟩) / √2
    - B01: (|01⟩ + |10⟩) / √2
    - B10: (|00⟩ - |11⟩) / √2
    - B11: (|01⟩ - |10⟩) / √2
    """
    B00 = 0
    B01 = 1
    B10 = 2
    B11 = 3

    def get_circuit(self, id_0: str, id_1: str) -> QCircuit:
        """
        Get the quantum circuit for the Bell state.

        Args:
            id_0 (str): Identifier for the first qubit.
            id_1 (str): Identifier for the second qubit.

        Returns:
            QCircuit: The quantum circuit that prepares the Bell state.
        """
        if self == BellState.B00:
            return b00_circuit(id_0, id_1)
        if self == BellState.B01:
            return b01_circuit(id_0, id_1)
        if self == BellState.B10:
            return b10_circuit(id_0, id_1)
        if self == BellState.B11:
            return b11_circuit(id_0, id_1)
        raise ValueError("Unknown Bell state!")

    @classmethod
    def from_indices(cls,
                     index_0: int,
                     index_1: int
                     ) -> Self:
        """
        Get the Bell state from indices.

        Args:
            index_0 (int): Index of the first qubit, i.e. the number of Z in 
                its matricised version.
            index_1 (int): Index of the second qubit, i.e. the number of X in 
                its matricised version.
        
        Returns:
            Self: The corresponding BellState enum value.
        """
        if index_0 == 0 and index_1 == 0:
            return cls.B00
        if index_0 == 0 and index_1 == 1:
            return cls.B01
        if index_0 == 1 and index_1 == 0:
            return cls.B10
        if index_0 == 1 and index_1 == 1:
            return cls.B11
        raise ValueError("Invalid indices for Bell state!")

    @classmethod
    def from_gate(cls,
                  gate: QGate
                  ) -> Self:
        """
        Get the Bell state from a quantum gate.

        Args:
            gate (QGate): The quantum state that is the matricised version of
                the Bell state.

        Returns:
            Self: The corresponding BellState enum value.
        """
        if gate == QGate.IDENTITY:
            return cls.B00
        if gate == QGate.PAULI_Z:
            return cls.B10
        if gate == QGate.PAULI_X:
            return cls.B01
        if gate == QGate.PAULI_Y:
            return cls.B11
        raise ValueError("Invalid gate for Bell state!")
    
def get_bell_state_circuit(bell_state: BellState | QGate | tuple[int, int],
                           id_0: str = "q0",
                            id_1: str = "q1"
                            ) -> QCircuit:
    """
    Get the quantum circuit for a Bell state.

    Args:
        bell_state (BellState | QGate | tuple[int, int]): The Bell state
            to prepare. Can be a BellState enum, a QGate enum, or a tuple of
            indices (index_0, index_1).
        id_0 (str): Identifier for the first qubit.
        id_1 (str): Identifier for the second qubit.

    Returns:
        QCircuit: The quantum circuit that prepares the Bell state.
    """
    if isinstance(bell_state, QGate):
        bell_state = BellState.from_gate(bell_state)
    elif isinstance(bell_state, tuple):
        bell_state = BellState.from_indices(bell_state[0],
                                            bell_state[1])
    elif not isinstance(bell_state, BellState):
        raise ValueError("Invalid Bell state!")
    return bell_state.get_circuit(id_0, id_1)

def b00_circuit(id_0: str,
                id_1: str
                ) -> QCircuit:
    """
    Create a quantum circuit that prepares the Bell state
    |B00⟩ = (|00⟩ + |11⟩) / √2.
    
    Returns:
        QCircuit: The quantum circuit that prepares the Bell state |B00⟩.
    """
    qc = QCircuit()
    qc.add_hadamard(id_0)
    qc.add_cnot(id_0, id_1, level_index=1)
    return qc

def b01_circuit(id_0: str,
                id_1: str
                ) -> QCircuit:
    """
    Create a quantum circuit that prepares the Bell state
    |B01⟩ = (|01⟩ + |10⟩) / √2.
    
    Returns:
        QCircuit: The quantum circuit that prepares the Bell state |B01⟩.
    """
    qc = QCircuit()
    qc.add_hadamard(id_0)
    qc.add_cnot(id_1, id_0, level_index=1)
    qc.add_x(id_0, level_index=2)
    return qc

def b10_circuit(id_0: str,
                id_1: str
                ) -> QCircuit:
    """
    Create a quantum circuit that prepares the Bell state
    |B10⟩ = (|00⟩ - |11⟩) / √2.
    
    Returns:
        QCircuit: The quantum circuit that prepares the Bell state |B10⟩.
    """
    qc = QCircuit()
    qc.add_x(id_0, level_index=0)
    qc.add_hadamard(id_0, level_index=1)
    qc.add_cnot(id_0, id_1, level_index=2)
    return qc

def b11_circuit(id_0: str,
                id_1: str
                ) -> QCircuit:
    """
    Create a quantum circuit that prepares the Bell state
    |B11⟩ = (|01⟩ - |10⟩) / √2.

    Returns:
        QCircuit: The quantum circuit that prepares the Bell state |B11⟩.
    """
    qc = QCircuit()
    qc.add_x(id_0, level_index=0)
    qc.add_hadamard(id_0, level_index=1)
    qc.add_cnot(id_0, id_1, level_index=2)
    qc.add_x(id_1, level_index=3)
    return qc
