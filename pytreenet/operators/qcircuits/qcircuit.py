"""
This module implements the actual quantum circuit.
"""
from __future__ import annotations

from ...core.ttn import TreeTensorNetwork
from ...ttno.ttno_class import TreeTensorNetworkOperator
from ...ttno.time_dep_ttno import DiscreetTimeTTNO
from ...ttno.state_diagram import StateDiagram, TTNOFinder
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

class AbstractQCircuit:
    """
    Abstract class for quantum circuits.
    """

    def __init__(self) -> None:
        """
        Initialize a quantum circuit.
        """
        self.levels = []

    def depth(self) -> int:
        """
        Get the depth of the quantum circuit.

        Returns:
            int: The number of levels in the quantum circuit.
        """
        return len(self.levels)

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

class QCircuit(AbstractQCircuit):
    """
    Class representing a quantum circuit.
    
    A quantum circuit is a collection of quantum circuit levels.
    """

    def __init__(self) -> None:
        """
        Initialize a quantum circuit.
        """
        super().__init__()
        self.levels: list[QCLevel]

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
            level_index = self.depth() - 1
        if level_index == self.depth():
            self.add_level()
        self._index_level_check(level_index)
        self.levels[level_index].add_gate(gate)

    def add_qircuit(self,
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
                self.levels[current_level_index].otimes_level(level)

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
        self.levels: list[Hamiltonian]
        self.qubit_ids: set[str] = set()

    def add_level(self, level: Hamiltonian):
        """
        Add a level to the compiled quantum circuit.

        Args:
            level (Hamiltonian): The Hamiltonian representing the level.
        """
        self.levels.append(level)
        self.qubit_ids.update(level.node_ids())

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
        if level_index == -1:
            level_index = len(self.levels) - 1
        if level_index == len(self.levels):
            self.levels.append(Hamiltonian())
        self._index_level_check(level_index)
        if level_index < len(self.levels):
            self.levels[level_index].add_hamiltonian(hamiltonian)

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

    def to_state_diagrams(self,
                          ref_tree: TreeTensorNetwork,
                          method = TTNOFinder.SGE
                          ) -> list[StateDiagram]:
        """
        Convert the compiled quantum circuit to a list of state diagrams.

        Args:
            ref_tree (TreeTensorNetwork): The reference tree tensor network
                to use for the state diagrams.
            method (TTNOFinder): The method to use for finding the
                tree tensor network operators in the state diagrams.
                Defaults to `TTNOFinder.SGE`.

        Returns:
            list[StateDiagram]: A list of state diagrams representing the
                compiled quantum circuit.
        """
        hams = [ham.pad_with_identities(ref_tree) for ham in self.levels]
        return [StateDiagram.from_hamiltonian(ham, ref_tree, method=method)
                for ham in hams]

    def to_ttnos(self,
                 ref_tree: TreeTensorNetwork,
                 method: TTNOFinder = TTNOFinder.SGE
                 ) -> list[TreeTensorNetworkOperator]:
        """
        Convert the compiled quantum circuit to a list of TTNOs.

        Args:
            ref_tree (TreeTensorNetwork): The reference tree tensor network
                to use for the TTNOs.
            method (TTNOFinder): The method to use for finding the
                tree tensor network operators in the state diagrams.
                Defaults to `TTNOFinder.SGE`.

        Returns:
            list[TreeTensorNetworkOperator]: A list of TTNOs representing
                the compiled quantum circuit.
        """
        return [TreeTensorNetworkOperator.from_hamiltonian(ham, ref_tree, method=method)
                for ham in self.levels]

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
        ttnos = self.to_ttnos(ref_tree, method=method)
        return DiscreetTimeTTNO(ttnos, dt=dt)
