"""
This module implements the actual quantum circuit.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Self, Callable
from copy import copy

import numpy as np
import numpy.random as npr

from ...core.ttn import TreeTensorNetwork
from ...ttno.ttno_class import TreeTensorNetworkOperator
from ...ttns.ttns import TreeTensorNetworkState
from ...ttno.time_dep_ttno import DiscreetTimeTTNO
from ...ttno.state_diagram import TTNOFinder
from ..hamiltonian import Hamiltonian
from .qgate import (QuantumGate,
                    QuantumOperation,
                    ProjectionOperation,
                    InvolutarySingleSiteGate,
                    CNOTGate,
                    QGate,
                    SWAPGate,
                    ToffoliGate,
                    PhaseGate,
                    MultiControlledGate,
                    RotationGate,
                    pauli_gates)
from ..measurment import Measurement


class AbstractLevel(ABC):
    """
    Abstract class for different levels in a quantum circuit.

    This is one level made of uquantum operations
    """

    def __init__(self) -> None:
        """
        Initialize an abstract level.
        """
        self.operations: list[QuantumOperation] = []
        self.qubit_ids: set[str] = set()

    def __str__(self) -> str:
        """
        Return a string representation of the abstract level.
        """
        return ", ".join([str(op) for op in self.operations])

    def num_operations(self) -> int:
        """
        Return the number of operations.
        """
        return len(self.operations)

    def width(self) -> int:
        """
        Return the width, i.e. number of different sites of this level.
        """
        return len(self.qubit_ids)

    def acts_on(self, qubit_id: str | list[str]) -> bool:
        """
        Check if the level acts on a specific qubit or list of qubits.

        Args:
            qubit_id (str | list[str]): The ID(s) of the qubit(s) to check.

        Returns:
            bool: True if the qubit(s) is/are in this level, False otherwise.
        """
        if isinstance(qubit_id, str):
            return qubit_id in self.qubit_ids
        for qid in qubit_id:
            if qid not in self.qubit_ids:
                return False
        return True

    def add_operation(self, operation: QuantumOperation):
        """
        Add a quantum operation to this level.

        Args:
            operation (QuantumOperation): The quantum operation to add.
        """
        self.operations.append(operation)
        for qubit_id in operation.qubit_ids:
            if qubit_id in self.qubit_ids:
                errstr = f"Qubit ID {qubit_id} already exists in this level!"
                raise ValueError(errstr)
        self.qubit_ids.update(operation.qubit_ids)

    @abstractmethod
    def otimes_level(self, 
                     other: AbstractLevel,
                     inplace: bool = False
                     ) -> Self:
        """
        Perform the kronecker product of this level with another level.

        Args:
            other (AbstractLevel): The other level to kronecker product with.
            inplace (bool): If True, perform the operation in place and
                return self. If False, return a new AbstractLevel instance.
        
        Returns:
            AbstractLevel: A new AbstractLevel instance containing the
                kronecker product of the two levels, or self if inplace is True.
        """
        pass

    @abstractmethod
    def compile(self) -> Hamiltonian | Measurement:
        """
        Compile the level into a Hamiltonian or Measurement.

        Returns:
            Hamiltonian | Measurement: The compiled Hamiltonian or Measurement
                representing the level.
        """
        pass

class QCLevel(AbstractLevel):
    """
    Class representing a quantum circuit level.

    A quantum circuit level is a collection of quantum gates that act on
    sites.
    """

    def __init__(self) -> None:
        """
        Initialize a quantum circuit level.
        """
        super().__init__()
        self.operations: list[QuantumGate]

    # Legacy Naming
    @property
    def gates(self) -> list[QuantumGate]:
        """
        Get the list of quantum gates in this level.

        Returns:
            list[QuantumGate]: The list of quantum gates in this level.
        """
        return self.operations

    def num_gates(self) -> int:
        """
        Return the number of gates.
        """
        return self.num_operations()

    def contains_gate_type(self, symbol: str | QGate) -> bool:
        """
        Check if the level contains a specific gate by its symbol.

        Args:
            symbol (str | QGate): The symbol of the gate to check.

        Returns:
            bool: True if the gate is in this level, False otherwise.
        """
        if isinstance(symbol, str):
            symbol = QGate(symbol)
        return any(gate.symbol == symbol for gate in self.gates)

    def contains_gate(self,
                      gate: QuantumGate | str | QGate
                      ) -> bool:
        """
        Check if the level contains a specific gate.

        Args:
            gate (QuantumGate): The quantum gate to check.

        Returns:
            bool: True if the gate is in this level, False otherwise.
        """
        if not isinstance(gate, QuantumGate):
            return self.contains_gate_type(gate)
        return gate in self.gates

    def add_gate(self, gate: QuantumGate):
        """
        Add a quantum gate to this level.

        Args:
            gate (QuantumGate): The quantum gate to add.
        """
        self.add_operation(gate)

    def otimes_level(self,
                     other: QCLevel,
                     inplace: bool = False
                     ) -> QCLevel:
        """
        Perform the kronecker product of this level with another level.

        Args:
            other (QCLevel): The other quantum circuit level to kronecker
                product with.
            inplace (bool): If True, perform the operation in place and
                return self. If False, return a new QCLevel instance.

        Returns:
            QCLevel: A new QCLevel instance containing the kronecker
                product of the two levels, or self if inplace is True.
        """
        if inplace:
            for gate in other.gates:
                self.add_gate(gate)
            return self
        new_level = QCLevel()
        new_level.otimes_level(self, inplace=True)
        new_level.otimes_level(other, inplace=True)
        return new_level

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

    def as_circuit_ttno(self,
                        ref_tree: TreeTensorNetwork,
                        method: TTNOFinder = TTNOFinder.SGE
                        ) -> TreeTensorNetworkOperator:
        """
        Convert the quantum circuit level to a TTNO.

        Args:
            ref_tree (TreeTensorNetwork): The reference tree tensor network
                to use for the TTNO.
            method (TTNOFinder): The method to use for finding the
                tree tensor network operator in the state diagram.
                Defaults to `TTNOFinder.SGE`.

        Returns:
            TreeTensorNetworkOperator: The TTNO representing the quantum
                circuit level as quantum gates. This means the TTNO is
                applied directly to the qubits.
        """
        ham = Hamiltonian()
        for gate in self.gates:
            ham = ham.otimes(gate.as_sum_of_products())
        ham.include_identities()
        return TreeTensorNetworkOperator.from_hamiltonian(ham,
                                                          ref_tree,
                                                          method=method)

    def invert(self) -> Self:
        """
        Get the inverse of the quantum circuit level.

        Returns:
            QCLevel: The inverse of the quantum circuit level.
        """
        inverted_level = self.__class__()
        for gate in self.gates:
            inverted_gate = gate.invert()
            inverted_level.add_gate(inverted_gate)
        inverted_level.qubit_ids = copy(self.qubit_ids)
        return inverted_level

class ProjectionLevel(AbstractLevel):
    """
    Class representing a projection level in a quantum circuit.

    A projection level is a collection of projections that act on sites.
    """

    def __init__(self) -> None:
        """
        Initialize a projection level.
        """
        super().__init__()
        self.operations: list[ProjectionOperation]

    def add_projection(self,
                       projection: ProjectionOperation):
        """
        Add a projection operation to this level.

        Args:
            projection (ProjectionOperation): The projection operation to add.
        """
        self.add_operation(projection)

    def num_projections(self) -> int:
        """
        Return the number of projections.
        """
        return self.num_operations()

    def otimes_level(self,
                     other: ProjectionLevel,
                     inplace: bool = False
                     ) -> ProjectionLevel:
        """
        Perform the kronecker product of this level with another projection
        level.

        Args:
            other (ProjectionLevel): The other projection level to kronecker
                product with.
            inplace (bool): If True, perform the operation in place and
                return self. If False, return a new ProjectionLevel instance.

        Returns:
            ProjectionLevel: A new ProjectionLevel instance containing the
                kronecker product of the two levels, or self if inplace is True.
        """
        if inplace:
            for projection in other.operations:
                self.add_projection(projection)
            return self
        new_level = ProjectionLevel()
        new_level.otimes_level(self, inplace=True)
        new_level.otimes_level(other, inplace=True)
        return new_level

    def compile(self) -> Measurement:
        """
        Compile the projection level into a Measurement.

        Returns:
            Measurement: The compiled Measurement representing the level.
        """
        measurement = self.operations[0].to_measurement()
        for projection in self.operations[1:]:
            measurement = measurement.otimes(projection.to_measurement())
        return measurement


class AbstractQCircuit(ABC):
    """
    Abstract class for quantum circuits.
    """

    def __init__(self) -> None:
        """
        Initialize a quantum circuit.
        """
        self.levels = []

    def __str__(self) -> str:
        """
        Return a string representation of the quantum circuit.
        """
        return "\n".join([f"Level {i}: {str(level)}"
                          for i, level in enumerate(self.levels)])

    def depth(self) -> int:
        """
        Get the depth of the quantum circuit including projections.

        Returns:
            int: The number of levels in the quantum circuit.
        """
        return len(self.levels)

    def gate_depth(self) -> int:
        """
        Get the depth of the quantum circuit excluding projections.

        Returns:
            int: The number of quantum circuit levels in the quantum circuit.
        """
        gate_levels = [level for level in self.levels
                       if isinstance(level, QCLevel)]
        return len(gate_levels)

    @abstractmethod
    def add_level(self, level: Any | None = None, level_kind: type[AbstractLevel] | None = None):
        """
        Add a level to the Qcircuit.
        """
        errstr = "This method must be implemented by a subclass!"
        raise NotImplementedError(errstr)

    def get_level(self, level_index: int) -> Any:
        """
        Get a specified level.
        """
        return self.levels[level_index]

    def _index_level_check(self, level_index: int):
        """
        Check if the level index is valid.

        Args:
            level_index (int): The index of the level to check.

        Raises:
            IndexError: If the level index is out of bounds.
        """
        if level_index < 0 or level_index > self.depth():
            errstr = f"Level index {level_index} out of bounds for circuit with {self.depth()} levels!"
            raise IndexError(errstr)

    def _gate_addition_preparation(self,
                                   level_index: int,
                                   level_kind: type[AbstractLevel] | None = None) -> int:
        """
        Prepares the Circuit for adding a gate.

        Args:
            level_index: The index of the level to add the gate to.
                If -1, the gate will be added to the last level. If it is
                one larger than the number of levels, a new level will be
                created.

        Returns:
            int: The actual level index as an integer.
        """
        if level_index == -1:
            if self.depth() > 0:
                level_index = self.depth() - 1
            else:
                level_index = 0
        if level_index == self.depth():
            self.add_level(level_kind=level_kind)
        self._index_level_check(level_index)
        return level_index


class QCircuit(AbstractQCircuit):
    """
    Class representing a quantum circuit.

    A quantum circuit is a collection of quantum circuit and projection levels.
    """

    def __init__(self) -> None:
        """
        Initialize a quantum circuit.
        """
        super().__init__()
        self.levels: list[AbstractLevel]

    def __str__(self) -> str:
        """
        Return a string representation of the quantum circuit.
        """
        return "\n".join([f"Level {i}: {str(level)}"
                          for i, level in enumerate(self.levels)])

    def support(self) -> set[str]:
        """
        Get the support of the quantum circuit, i.e. the set of qubits it acts on.

        Returns:
            set[str]: The set of qubits the quantum circuit acts on.
        """
        return self.qubit_ids()

    def width(self) -> int:
        """
        Get the width of the quantum circuit.

        Returns:
            int: The number of different qubits in the circuit.
        """
        return len(self.qubit_ids())

    def contains_gate(self,
                      gate: QuantumGate,
                      level_index: int | None = None
                      ) -> bool:
        """
        Check if the quantum circuit contains a specific quantum gate.

        Args:
            gate (QuantumGate): The quantum gate to check for.
            level_index (int | None): The index of the level to check.
                Defaults to None, which checks all levels.

        Returns:
            bool: True if the quantum circuit contains the specified quantum
                gate, False otherwise.
        """
        if level_index is None:
            for qlevel in self.levels:
                if isinstance(qlevel, QCLevel) and qlevel.contains_gate(gate):
                    return True
            return False
        self._index_level_check(level_index)
        level = self.levels[level_index]
        if isinstance(level, QCLevel):
            return level.contains_gate(gate)
        return False

    def qubit_ids(self) -> set[str]:
        """
        Get the set of qubit IDs in the quantum circuit.

        Returns:
            set[str]: The set of qubit IDs in the quantum circuit.
        """
        qubit_ids = set()
        for level in self.levels:
            qubit_ids.update(level.qubit_ids)
        return qubit_ids

    def add_level(self,
                  level: AbstractLevel | None = None,
                  level_kind: type[AbstractLevel] = QCLevel):
        """
        Add a quantum circuit level to the circuit.

        Args:
            level (AbstractLevel | None): The quantum circuit level to add. If None,
                a new level will be created. Defaults to None.
            level_kind (type[AbstractLevel]): The kind of level to create if
                level is None. Defaults to QCLevel.
        """
        if level is None:
            level = level_kind()
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
            ValueError: If the level at level_index is not a QCLevel.
        """
        level_index = self._gate_addition_preparation(level_index, QCLevel)
        level = self.levels[level_index]
        if not isinstance(level, QCLevel):
            errstr = f"Level {level_index} is not a QCLevel!"
            raise TypeError(errstr)
        level.add_gate(gate)

    def _add_projection(self,
                       projection: ProjectionOperation,
                       level_index: int = -1):
        """
        Add a projection operation to the circuit.

        Args:
            projection (ProjectionOperation): The projection operation to add.
            level_index (int): The index of the level to add the projection to.
                If -1, the projection will be added to the last level. If it is
                one larger than the number of levels, a new level will be
                created.
        
        Raises:
            IndexError: If the level index is out of bounds.
            ValueError: If the level at level_index is not a ProjectionLevel.
        """
        level_index = self._gate_addition_preparation(level_index, ProjectionLevel)
        level = self.levels[level_index]
        if not isinstance(level, ProjectionLevel):
            errstr = f"Level {level_index} is not a ProjectionLevel!"
            raise TypeError(errstr)
        level.add_projection(projection)

    def add_qcircuit(self,
                     circuit: QCircuit,
                     level_index: int | None = None):
        """
        Add a quantum circuit to the current circuit.

        Args:
            circuit (QCircuit): The quantum circuit to add.
            level_index (int): The index of the level to add the circuit to.
                If None, the circuit will be added to a new level and onwards.
                Otherwise, it will be added from the specified level onwards.

        Raises:
            IndexError: If the level index is out of bounds.
        """
        if level_index is None:
            level_index = self.depth()
        else:
            self._index_level_check(level_index)
        for i, level in enumerate(circuit.levels):
            current_level_index = level_index + i
            if current_level_index >= self.depth():
                self.add_level(level)
            else:
                self.levels[current_level_index].otimes_level(level,
                                                              inplace=True)

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

    def as_circuit_ttno(self,
                        ref_tree: TreeTensorNetwork,
                        method: TTNOFinder = TTNOFinder.SGE
                        ) -> list[TreeTensorNetworkOperator | Measurement]:
        """
        Convert the quantum circuit to a list of TTNOs.

        Args:
            ref_tree (TreeTensorNetwork): The reference tree tensor network
                to use for the TTNOs.
            method (TTNOFinder): The method to use for finding the
                tree tensor network operators in the state diagrams.
                Defaults to `TTNOFinder.SGE`.

        Returns:
            list[TreeTensorNetworkOperator | Measurement]:  A list of TTNOs
                representing the quantum circuit and measurements.

        """
        ttnos = []
        for level in self.levels:
            if isinstance(level, QCLevel):
                ttnos.append(level.as_circuit_ttno(ref_tree, method=method))
            elif isinstance(level, ProjectionLevel):
                compiled_level = level.compile()
                ttnos.append(compiled_level)
        return ttnos

    def invert(self) -> Self:
        """
        Get the inverse of the quantum circuit.

        Returns:
            AbstractQCircuit: The inverse of the quantum circuit.
        """
        inverted_circuit = self.__class__()
        for level in reversed(self.levels):
            if not hasattr(level, "invert"):
                errstr = "All levels must support inversion!"
                raise TypeError(errstr)
            inverted_level = level.invert()
            inverted_circuit.add_level(inverted_level)
        return inverted_circuit

    def apply_to_state(self,
                       state: TreeTensorNetworkState,
                       application_function: Callable[[TreeTensorNetworkState, TreeTensorNetworkOperator], TreeTensorNetworkState],
                       inter_level_function: Callable[[int, TreeTensorNetworkState, TreeTensorNetworkOperator | Measurement]] | None = None,
                       generation_method: TTNOFinder = TTNOFinder.SGE,
                       measurement_kwargs: dict[str, Any] | None = None
                       ) -> TreeTensorNetworkState:
        """
        Apply the quantum circuit to a given state.

        Args:
            state (TreeTensorNetworkState): The state to apply the circuit to.
            application_function (Callable[[TreeTensorNetworkState, TreeTensorNetworkOperator], TreeTensorNetworkState]):
                A function that takes a state and a TTNO and returns the state
                after applying the TTNO to it.
            inter_level_function (Callable[[int, TreeTensorNetworkState, TreeTensorNetworkOperator | Measurement]] | None):
                An optional function that is called between levels. It takes the
                level index, the current state, and the TTNO or measurement of
                the level as arguments. Defaults to None.
                May be used to record intermediate states or data such as bond
                dimensions.
            generation_method (TTNOFinder): The method to use for finding the
                tree tensor network operators in the state diagrams. Defaults to
                `TTNOFinder.SGE`.
            measurement_kwargs (dict[str, Any] | None): Optional keyword arguments
                to pass to the measurement projection function. Defaults to None.
        
        Returns:
            TreeTensorNetworkState: The state after applying the quantum circuit.
        """
        current_state = state
        ttnos_and_measurements = self.as_circuit_ttno(state,
                                                      method=generation_method)
        for i, operation in enumerate(ttnos_and_measurements):
            if isinstance(operation, TreeTensorNetworkOperator):
                current_state = application_function(current_state,
                                                     operation)
            else:
                operation.apply(current_state,
                                **(measurement_kwargs or {}))
            if inter_level_function is not None:
                inter_level_function(i, current_state, operation)
        return current_state

    def add_pauli_error_gates(self,
                              num_width: dict[int, float] | float,
                              probs: dict[QGate, float] | None = None,
                              levels: list[int] | None = None,
                              seed: int | None | npr.Generator = None):
        """
        Add random Pauli error gates to the circuit.

        Args:
            num_width (dict[int, float] | float): A dictionary mapping
                the number of qubits on which an error occurs to a probability.
                If a single float is provided, it will be used as the
                probability of a single error occuring.
            probs (dict[QGate, float]): A dictionary mapping Pauli gates to
                their probabilities of being added if an error occurs at all.
                If None, all Pauli gates will have equal probability.
                Defaults to None.
            levels (list[int] | None): A list of level indices that an error
                can be added to. If None, errors can be added to all levels.
                Defaults to None.
            seed (int | None | npr.Generator): An optional random seed or random
                generator to use for reproducibility. Defaults to None.
        """
        rng = npr.default_rng(seed)
        if isinstance(num_width, float) or isinstance(num_width, int):
            num_width = {1: num_width}
        elif len(num_width) == 0:
            return
        if probs is None:
            probs = {gate: 1/3
                     for gate in pauli_gates()}
        if levels is None:
            levels = list(range(self.depth()))
        num_width[0] = 1 - sum(num_width.values())
        num_gates = rng.choice(list(num_width.keys()),
                               p=list(num_width.values()))
        gates = rng.choice(list(probs.keys()),
                            p=list(probs.values()),
                            size=num_gates)
        qubit_ids = rng.choice(list(self.qubit_ids()),
                                size=num_gates,
                                replace=False)
        level = rng.choice(levels)
        self.levels.insert(level, QCLevel())
        for gate, qubit_id in zip(gates, qubit_ids):
            self.add_gate(InvolutarySingleSiteGate.from_enum(gate, qubit_id),
                          level_index=level)

    # Utility methods for adding gates
    def add_x(self, qubit_id: str, level_index: int = -1):
        """
        Add a Pauli-X gate to the circuit.

        Args:
            qubit_id (str): The ID of the qubit to apply the gate to.
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, qubit_id)
        self.add_gate(gate, level_index)

    def add_y(self, qubit_id: str, level_index: int = -1):
        """
        Add a Pauli-Y gate to the circuit.

        Args:
            qubit_id (str): The ID of the qubit to apply the gate to.
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = InvolutarySingleSiteGate.from_enum(QGate.PAULI_Y, qubit_id)
        self.add_gate(gate, level_index)

    def add_z(self, qubit_id: str, level_index: int = -1):
        """
        Add a Pauli-Z gate to the circuit.

        Args:
            qubit_id (str): The ID of the qubit to apply the gate to.
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = InvolutarySingleSiteGate.from_enum(QGate.PAULI_Z, qubit_id)
        self.add_gate(gate, level_index)

    def add_hadamard(self, qubit_id: str, level_index: int = -1):
        """
        Add a Hadamard gate to the circuit.

        Args:
            qubit_id (str): The ID of the qubit to apply the gate to.
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = InvolutarySingleSiteGate.from_enum(QGate.HADAMARD, qubit_id)
        self.add_gate(gate, level_index)

    def add_identity(self, qubit_id: str, level_index: int = -1):
        """
        Add an Identity gate to the circuit.

        Args:
            qubit_id (str): The ID of the qubit to apply the gate to.
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = InvolutarySingleSiteGate.from_enum(QGate.IDENTITY, qubit_id)
        self.add_gate(gate, level_index)

    def add_cnot(self,
                 control_id: str,
                 target_id: str,
                 level_index: int = -1):
        """
        Add a CNOT gate to the circuit.

        Args:
            control_id (str): The ID of the control qubit.
            target_id (str): The ID of the target qubit.
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = CNOTGate(control_id, target_id)
        self.add_gate(gate, level_index)

    def add_ch(self,
               control_id: str,
               target_id: str,
               level_index: int = -1):
        """
        Add a controlled Hadamard gate to the circuit.

        Args:
            control_id (str): The ID of the control qubit.
            target_id (str): The ID of the target qubit.
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = MultiControlledGate([control_id],
                                   [],
                                   [target_id],
                                   QGate.HADAMARD,
                                   "CH")
        self.add_gate(gate, level_index)

    def add_swap(self,
                 qubit_id1: str,
                 qubit_id2: str,
                 level_index: int = -1):
        """
        Add a SWAP gate to the circuit.

        Args:
            qubit_id1 (str): The ID of the first qubit.
            qubit_id2 (str): The ID of the second qubit.
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = SWAPGate(qubit_id1, qubit_id2)
        self.add_gate(gate, level_index)

    def add_toffoli(self,
                    control_id1: str,
                    control_id2: str,
                    target_id: str,
                    level_index: int = -1):
        """
        Add a Toffoli gate to the circuit.

        Args:
            control_id1 (str): The ID of the first control qubit.
            control_id2 (str): The ID of the second control qubit.
            target_id (str): The ID of the target qubit.
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = ToffoliGate(control_id1, control_id2, target_id)
        self.add_gate(gate, level_index)

    def add_phase(self,
                  qubit_id: str,
                  phase: float,
                  level_index: int = -1):
        """
        Add a Phase gate to the circuit.

        Args:
            qubit_id (str): The ID of the qubit to apply the gate to.
            phase (float): The phase to apply.
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = PhaseGate(phase, qubit_id)
        self.add_gate(gate, level_index)

    def add_projection(self,
                       qubit_ids: str | list[str],
                       level_index: int = -1,
                       **kwargs):
        """
        Add a projection operation to the circuit.

        Args:
            qubit_ids (str | list[str]): The ID(s) of the qubit(s) to project.
            level_index (int): The index of the level to add the projection to.
                Defaults to -1 (last level).
            **kwargs: Additional keyword arguments to pass to the measurement.
        """
        if isinstance(qubit_ids, str):
            qubit_ids = [qubit_ids]
        projection = ProjectionOperation(f"M", qubit_ids, **kwargs)
        self._add_projection(projection, level_index)

    def add_mx(self,
               control_ids: list[str],
               inverse_controls: list[str],
               target_ids: str | list[str],
               level_index: int = -1):
        """
        Add a multi-controlled X gate to the circuit.

        Args:
            control_ids (list[str]): The IDs of the control qubits.
            inverse_controls (list[str]): The IDs of the inverse control qubits.
            target_id (str | list[str]): The ID(s) of the target qubit(s).
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = MultiControlledGate(control_ids,
                                   inverse_controls,
                                   target_ids,
                                   QGate.PAULI_X,
                                   "MCX")
        self.add_gate(gate, level_index)

    def add_mz(self,
               control_ids: list[str],
               inverse_controls: list[str],
               target_ids: str | list[str],
               level_index: int = -1):
        """
        Add a multi-controlled Z gate to the circuit.

        Args:
            control_ids (list[str]): The IDs of the control qubits.
            inverse_controls (list[str]): The IDs of the inverse control qubits.
            target_ids (str | list[str]): The ID(s) of the target qubit(s).
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = MultiControlledGate(control_ids,
                                   inverse_controls,
                                   target_ids,
                                   QGate.PAULI_Z,
                                   "MCZ")
        self.add_gate(gate, level_index)

    def add_my(self,
               control_ids: list[str],
               inverse_controls: list[str],
               target_ids: str | list[str],
               level_index: int = -1):
        """
        Add a multi-controlled Y gate to the circuit.

        Args:
            control_ids (list[str]): The IDs of the control qubits.
            inverse_controls (list[str]): The IDs of the inverse control qubits.
            target_ids (str | list[str]): The ID(s) of the target qubit(s).
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = MultiControlledGate(control_ids,
                                   inverse_controls,
                                   target_ids,
                                   QGate.PAULI_Y,
                                   "MCY")
        self.add_gate(gate, level_index)

    def add_rx(self,
               qubit_id: str,
               angle: float,
               level_index: int = -1):
        """
        Add a rotation around the X axis to the circuit.

        Args:
            qubit_id (str): The ID of the qubit to apply the gate to.
            angle (float): The angle to rotate by.
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = RotationGate(qubit_id,
                            QGate.PAULI_X,
                            angle)
        self.add_gate(gate, level_index)

    def add_ry(self,
               qubit_id: str,
               angle: float,
               level_index: int = -1):
        """
        Add a rotation around the Y axis to the circuit.

        Args:
            qubit_id (str): The ID of the qubit to apply the gate to.
            angle (float): The angle to rotate by.
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = RotationGate(qubit_id,
                            QGate.PAULI_Y,
                            angle)
        self.add_gate(gate, level_index)

    def add_rz(self,
               qubit_id: str,
               angle: float,
               level_index: int = -1):
        """
        Add a rotation around the Z axis to the circuit.

        Args:
            qubit_id (str): The ID of the qubit to apply the gate to.
            angle (float): The angle to rotate by.
            level_index (int): The index of the level to add the gate to.
                Defaults to -1 (last level).
        """
        gate = RotationGate(qubit_id,
                            QGate.PAULI_Z,
                            angle)
        self.add_gate(gate, level_index)

class CompiledQuantumCircuit(AbstractQCircuit):
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
        self.levels: list[Hamiltonian | Measurement]
        self.qubit_ids: set[str] = set()

    def width(self,
              level_index: int | None = None
              ) -> int:
        """
        The width of the quantum circuit.

        Args:
            level_index (int | None): The index of the level, whose width to
                return. If None, the width of the whole circuit is returned.
                Defaults to None.
        """
        if level_index is None:
            return len(self.qubit_ids)
        self._index_level_check(level_index)
        return self.levels[level_index].system_size()

    def close_to(self,
                 other: CompiledQuantumCircuit
                 ) -> bool:
        """
        Whether the two compiled quantum circuits are close to each other.
        """
        if self.depth() != other.depth():
            return False
        if self.qubit_ids != other.qubit_ids:
            return False
        for i, level in enumerate(self.levels):
            other_level = other.levels[i]
            if isinstance(level, Measurement) and isinstance(other_level, Measurement):
                if not level == other_level:
                    return False
            elif isinstance(level, Hamiltonian) and isinstance(other_level, Hamiltonian):
                if level != other_level:
                    return False
                if not level.compare_dicts(other_level):
                    return False
            else:
                return False
        return True

    def add_level(self, level: Hamiltonian | Measurement | None = None):
        """
        Add a level to the compiled quantum circuit.

        Args:
            level (Hamiltonian | None): The Hamiltonian representing the
                level. If None, an empty level will be added.
                Defaults to None.
        """
        if level is None:
            level = Hamiltonian()
        self.levels.append(level)
        self.qubit_ids.update(level.node_ids)

    def add_hamiltonian(self,
                        hamiltonian: Hamiltonian,
                        level_index: int = -1):
        """
        Add a Hamiltonian to the compiled quantum circuit.

        Args:
            hamiltonian (Hamiltonian): The Hamiltonian to add.
            level_index (int): The index of the level to add the Hamiltonian
                to. If -1, the Hamiltonian will be added to the last level.
                If it is one larger than the number of levels, a new level
                will be created.

        Raises:
            IndexError: If the level index is out of bounds.
        """
        level_index = self._gate_addition_preparation(level_index)
        if level_index < self.depth():
            level = self.levels[level_index]
            if isinstance(level, Hamiltonian):
                level.add_hamiltonian(hamiltonian)
            else:
                errstr = f"Level {level_index} is not a Hamiltonian!"
                raise TypeError(errstr)

    def add_measurement(self,
                        measurement: Measurement,
                        level_index: int = -1):
        """
        Add a Measurement to the compiled quantum circuit.

        Args:
            measurement (Measurement): The Measurement to add.
            level_index (int): The index of the level to add the Measurement
                to. If -1, the Measurement will be added to the last level.
                If it is one larger than the number of levels, a new level
                will be created.

        Raises:
            IndexError: If the level index is out of bounds.
        """
        level_index = self._gate_addition_preparation(level_index)
        if level_index < self.depth():
            level = self.levels[level_index]
            if isinstance(level, Measurement):
                level = level.otimes(measurement)
                self.levels[level_index] = level
            else:
                errstr = f"Level {level_index} is not a Measurement!"
                raise TypeError(errstr)

    def add_constant_hamiltonian(self,
                                 hamiltonian: Hamiltonian):
        """
        Add a constant Hamiltonian to the compiled quantum circuit.

        That is a Hamiltonian that appears in every level of the
        compiled quantum circuit.

        Args:
            hamiltonian (Hamiltonian): The constant Hamiltonian to add.
        """
        for i in range(self.depth()):
            self.add_hamiltonian(hamiltonian, level_index=i)

    def to_ttnos(self,
                 ref_tree: TreeTensorNetwork,
                 method: TTNOFinder = TTNOFinder.SGE
                 ) -> list[TreeTensorNetworkOperator | Measurement]:
        """
        Convert the compiled quantum circuit to a list of TTNOs.

        Args:
            ref_tree (TreeTensorNetwork): The reference tree tensor network
                to use for the TTNOs.
            method (TTNOFinder): The method to use for finding the
                tree tensor network operators in the state diagrams.
                Defaults to `TTNOFinder.SGE`.

        Returns:
            list[TreeTensorNetworkOperator | Measurement]: A list of TTNOs and measurements
                representing the compiled quantum circuit.
        """
        out = []
        for level in self.levels:
            if isinstance(level, Hamiltonian):
                level.include_identities()
                level.combine_equivalent_identities()
                ttno = TreeTensorNetworkOperator.from_hamiltonian(level,
                                                                  ref_tree,
                                                                  method=method)
                out.append(ttno)
            else:
                out.append(level)
        return out

    def to_time_dep_ttno(self,
                         ref_tree: TreeTensorNetwork,
                         dt: float = 1.0,
                         method: TTNOFinder = TTNOFinder.SGE
                         ) -> DiscreetTimeTTNO:
        """
        Convert the compiled quantum circuit to a time-dependent TTNO.

        Args:
            ref_tree (TreeTensorNetwork): The reference tree tensor network
                to use for the TTNO.
            dt (float): The time to pass between each switch to the next
                TreeTensorNetworkOperator. Default is 1.0.
            method (TTNOFinder): The method to use for finding the
                tree tensor network operators in the state diagrams.
                Defaults to `TTNOFinder.SGE`.

        Returns:
            DiscreetTimeTTNO: A time-dependent TTNO representing the
                compiled quantum circuit.
        """
        found_ttnos = self.to_ttnos(ref_tree, method=method)
        length = len(found_ttnos)
        ttnos: list[TreeTensorNetworkOperator] = []
        meas: list[Measurement] = []
        for index, item in enumerate(found_ttnos):
            if isinstance(item, TreeTensorNetworkOperator):
                ttnos.append(item)
                if index < length - 1 and not isinstance(found_ttnos[index + 1], Measurement):
                    meas.append(Measurement.empty())
            elif isinstance(item, Measurement):
                meas.append(item)
                if index < length - 1 and isinstance(found_ttnos[index + 1], Measurement):
                    errstr = "Two consecutive Measurements found in compiled quantum circuit!"
                    raise ValueError(errstr)
        return DiscreetTimeTTNO(ttnos, dt=dt, measurements=meas)
