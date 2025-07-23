"""
This module implements the actual quantum circuit.
"""
from __future__ import annotations
from collections import UserList

from ..hamiltonian import Hamiltonian
from .qgate import QuantumGate

class QCLevel:
    """
    Class representing a quantum circuit level.
    
    A quantum circuit level is a collection of quantum gates that act on
    sites.
    """

    def __init__(self) -> None:
        """
        Initialize a quantum circuit level.
        """
        self.gates: list[QuantumGate] = []
        self.qubit_ids: set[str] = set()

    def add_gate(self, gate: QuantumGate):
        """
        Add a quantum gate to this level.

        Args:
            gate (QuantumGate): The quantum gate to add.
        """
        self.gates.append(gate)
        for qubit_id in gate.qubit_ids:
            if qubit_id in self.qubit_ids:
                errstr = f"Qubit ID {qubit_id} already exists in this level!"
                raise ValueError(errstr)
        self.qubit_ids.update(gate.qubit_ids)

    @classmethod
    def from_gates(cls, gates: list[QuantumGate]) -> QCLevel:
        """
        Create a QCLevel from a list of quantum gates.

        Args:
            gates (list[QuantumGate]): List of quantum gates to include in
                the level.

        Returns:
            QCLevel: A new QCLevel instance containing the provided gates.
        """
        level = cls()
        for gate in gates:
            level.add_gate(gate)
        return level

    def compile(self) -> Hamiltonian:
        """
        Compile the quantum circuit level into a Hamiltonian.

        Returns:
            Hamiltonian: The compiled Hamiltonian representing the level.
        """
        hamiltonian = Hamiltonian()
        for gate in self.gates:
            hamiltonian.add_hamiltonian(gate.get_generator())
        return hamiltonian

class QCircuit:
    """
    Class representing a quantum circuit.
    
    A quantum circuit is a collection of quantum circuit levels.
    """

    def __init__(self) -> None:
        """
        Initialize a quantum circuit.
        """
        self.levels: list[QCLevel] = []

    def add_level(self,
                  level: QCLevel | None = None):
        """
        Add a quantum circuit level to the circuit.

        Args:
            level (QCLevel): The quantum circuit level to add. If None,
                a new level will be created. Defaults to None.
        """
        if level is None:
            level = QCLevel()
        self.levels.append(level)

    def add_gate(self,
                  gate: QuantumGate,
                  level_index: int = -1):
        """
        Add a quantum gate to the circuit.

        Args:
            gate (QuantumGate): The quantum gate to add.
            level_index (int): The index of the level to add the gate to.
                If -1, the gate will be added to the last level. If it is
                one larger than the number of levels, a new level will be
                created.
        
        Raises:
            IndexError: If the level index is out of bounds.
        """
        if level_index == -1:
            level_index = len(self.levels) - 1
        if level_index == len(self.levels) + 1:
            self.add_level()
        if level_index < 0 or level_index >= len(self.levels):
            errstr = f"Level index {level_index} out of bounds for circuit with {len(self.levels)} levels!"
            raise IndexError(errstr)
        self.levels[level_index].add_gate(gate)

    def compile(self) -> CompiledQuantumCircuit:
        """
        Compile the quantum circuit into a compiled quantum circuit.

        Returns:
            CompiledQuantumCircuit: The compiled quantum circuit.
        """
        compiled_circuit = CompiledQuantumCircuit()
        for level in self.levels:
            compiled_level = level.compile()
            compiled_circuit.add_level(compiled_level)
        return compiled_circuit

class CompiledQuantumCircuit(UserList):
    """
    Class representing a compiled quantum circuit.
    
    A compiled quantum circuit turned all gates into their generator
    Hamiltonian.
    """

    def __init__(self) -> None:
        """
        Initialize a compiled quantum circuit.
        """
        super().__init__()
        self.qubit_ids: set[str] = set()

    def add_level(self, level: Hamiltonian):
        """
        Add a level to the compiled quantum circuit.

        Args:
            level (Hamiltonian): The Hamiltonian representing the level.
        """
        self.data.append(level)
        self.qubit_ids.update(level.node_ids())
