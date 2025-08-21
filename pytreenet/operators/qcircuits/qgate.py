"""
Implement quantum gates for use with PyTreeNet.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Self
from abc import ABC, abstractmethod
from fractions import Fraction
from enum import Enum

import numpy as np

from ..common_operators import (pauli_matrices,
                                hadamard,
                                swap_gate,
                                toffoli_gate)
from ..hamiltonian import Hamiltonian
from ..tensorproduct import TensorProduct

if TYPE_CHECKING:
    import numpy.typing as npt

PI_SYMBOL = "pi"

class QGate(Enum):
    """
    Enumerator for the different types of quantum gates.
    """
    IDENTITY = "I"
    PAULI_X = "X"
    PAULI_Y = "Y"
    PAULI_Z = "Z"
    HADAMARD = "H"
    CNOT = "CNOT"
    SWAP = "SWAP"
    TOFFOLI = "TOFFOLI"
    PHASE = "PHASE"

class QuantumGate(ABC):
    """
    Abstract base class for quantum gates.
    """

    def __init__(self,
                 symbol: str,
                 qubit_ids: list[str]
                 ) -> None:
        """
        Initialize a quantum gate.

        Args:
            symbol (str): The symbol representing the quantum gate.
            qubit_ids (list[str]): List of qubit IDs that the gate acts on.
        """
        self.symbol = symbol
        self.qubit_ids = qubit_ids

    def __eq__(self, other: QuantumGate) -> bool:
        """
        Check equality of two quantum gates based on their symbol and qubit IDs.

        Args:
            other (object): The other object to compare with.

        Returns:
            bool: True if both gates have the same symbol and qubit IDs, False otherwise.
        """
        return (self.symbol == other.symbol and
                self.qubit_ids == other.qubit_ids)

    @abstractmethod
    def get_generator(self) -> Hamiltonian:
        """
        Get the generator of the quantum gate, i.e.

        ..math::
            G = e^{i H}

        where :math:`H` is the generator and :math:`G` is the matrix
        representation of the gate.

        Returns:
            Hamiltonian: The generator of the quantum gate.
        """
        raise NotImplementedError("Subclasses must implement this method!")

    def acts_on(self, qubit_ids: str | list[str]) -> bool:
        """
        Check if the quantum gate acts on the given qubit IDs.

        Args:
            qubit_ids (str | list[str]): The ID or list of IDs of the qubits
                to check.

        Returns:
            bool: True if the gate acts on the given qubit IDs, False
                otherwise.
        """
        if isinstance(qubit_ids, str):
            return qubit_ids in self.qubit_ids
        return all(qid in self.qubit_ids for qid in qubit_ids)

    @abstractmethod
    def matrix(self) -> npt.NDArray[np.complex64]:
        """
        Get the matrix representation of the quantum gate.

        Returns:
            npt.NDArray[np.complex64]: The matrix representation of the
                quantum gate.
        """
        raise NotImplementedError("Subclasses must implement this method!")

class SingleQubitGate(QuantumGate):
    """
    Abstract base class for single-qubit gates.
    """

    def __init__(self,
                 symbol: str,
                 qubit_id: str
                 ) -> None:
        """
        Initialize a single-qubit gate.
        """
        super().__init__(symbol, [qubit_id])
        self.qubit_id = qubit_id

class InvolutarySingleSiteGate(SingleQubitGate):
    """
    Class for involutary single-site gates, which are gates that
    satisfy the property :math:`G^2 = I`, where :math:`I` is the identity
    """

    def __init__(self,
                 symbol: str,
                 gate_matrix: npt.NDArray[np.complex64],
                 qubit_id: str
                 ) -> None:
        """
        Initialize a Pauli gate.
        """
        super().__init__(symbol, qubit_id)
        self.gate_matrix = gate_matrix
        self.gen_factor = (Fraction(1,2),PI_SYMBOL,complex(np.pi))

    def get_generator(self) -> Hamiltonian:
        """
        Get the generator of the Pauli gate.

        Returns:
            Hamiltonian: The generator of the Pauli gate.
        """
        identity = np.eye(2, dtype=np.complex64)
        id_symbol = "I2"
        ham = Hamiltonian()
        tp1 = TensorProduct()
        tp1.add_operator(self.qubit_id, self.symbol)
        ham.add_term((self.gen_factor[0], self.gen_factor[1], tp1))
        tp2 = TensorProduct()
        tp2.add_operator(self.qubit_id, id_symbol)
        ham.add_term((Fraction(-1, 2), PI_SYMBOL, tp2))
        conv_dict = {self.symbol: self.gate_matrix,
                     id_symbol: identity}
        coeffs_map = {self.gen_factor[1]: self.gen_factor[2]}
        ham.update_mappings(conv_dict, coeffs_map)
        return ham

    def matrix(self) -> npt.NDArray[np.complex64]:
        """
        Get the matrix representation of the Pauli gate.

        Returns:
            npt.NDArray[np.complex64]: The matrix representation of the
                Pauli gate.
        """
        return self.gate_matrix

    @classmethod
    def from_enum(cls,
                  gate_enum: QGate | str,
                  qubit_id: str
                  ) -> Self:
        """
        Create a Pauli gate from an enum or string representation.

        Args:
            gate_enum (QGate | str): The enum or string representation of the
                Pauli gate.
            qubit_id (str): The ID of the qubit the gate acts on.

        Returns:
            PauliGate: The corresponding Pauli gate instance.
        """
        if isinstance(gate_enum, str):
            gate_enum = QGate(gate_enum)
        if gate_enum == QGate.PAULI_X:
            return cls(QGate.PAULI_X.value,
                        pauli_matrices()[0],
                        qubit_id)
        elif gate_enum == QGate.PAULI_Y:
            return cls(QGate.PAULI_Y.value,
                       pauli_matrices()[1],
                       qubit_id)
        elif gate_enum == QGate.PAULI_Z:
            return cls(QGate.PAULI_Z.value,
                       pauli_matrices()[2],
                       qubit_id)
        elif gate_enum == QGate.HADAMARD:
            return cls(QGate.HADAMARD.value,
                       hadamard(),
                       qubit_id)
        errstr = f"Invalid Enum for InvolutarySingleSiteGate: {gate_enum}!"
        raise ValueError(errstr)

class PhaseGate(SingleQubitGate):
    """
    Class for the Phase gate.
    """

    def __init__(self,
                 phase: float,
                 qubit_id: str
                 ) -> None:
        """
        Initialize a Phase gate.

        Args:
            phase (float): The phase angle for the gate in radians in
                multiples of pi.
            qubit_id (str): The ID of the qubit the gate acts on.
        """
        super().__init__(QGate.PHASE.value + f"{phase}", qubit_id)
        self.phase = phase

    def __eq__(self,
               other: QuantumGate
               ) -> bool:
        """
        Check equality of two quantum gates based on their symbol and qubit IDs.

        Args:
            other (QuantumGate): The other quantum gate to compare with.

        Returns:
            bool: True if both gates have the same symbol and qubit IDs, False otherwise.
        """
        if not isinstance(other, PhaseGate):
            return False
        return super().__eq__(other) and self.phase == other.phase

    def get_generator(self) -> Hamiltonian:
        """
        Get the generator of the Phase gate.

        Returns:
            Hamiltonian: The generator of the Phase gate.
        """
        gate = np.diag([0,self.phase])
        ham = Hamiltonian()
        tp = TensorProduct()
        tp.add_operator(self.qubit_id, gate)
        ham.add_term((Fraction(-1), PI_SYMBOL, tp))
        conversion_dict = {
            self.symbol: gate,
            "I2": np.eye(2, dtype=np.complex64)
        }
        coeffs_map = {
            PI_SYMBOL: complex(np.pi)
        }
        ham.update_mappings(conversion_dict, coeffs_map)
        return ham

    def matrix(self) -> npt.NDArray[np.complex64]:
        """
        Get the matrix representation of the Phase gate.

        Returns:
            npt.NDArray[np.complex64]: The matrix representation of the
                Phase gate.
        """
        return np.array([[1, 0],
                         [0, np.exp(1j * self.phase * np.pi)]],
                         dtype=np.complex64)

    @classmethod
    def from_enum(cls,
                  gate_enum: QGate | str,
                  phase: float,
                  qubit_id: str
                  ) -> Self:
        """
        Create a Phase gate from its qubit ID.

        Args:
            gate_enum (QGate | str): The enum or string representation of the
                Phase gate.
            phase (float): The phase angle for the gate in radians in
                multiples of pi.
            qubit_id (str): The ID of the qubit the gate acts on.

        Returns:
            PhaseGate: The corresponding Phase gate instance.
        """
        if isinstance(gate_enum, str):
            gate_enum = QGate(gate_enum)
        if gate_enum != QGate.PHASE:
            errstr = f"Invalid Enum for PhaseGate: {gate_enum}!"
            raise ValueError(errstr)
        return cls(phase, qubit_id)

class CNOTGate(QuantumGate):
    """
    Class for the CNOT gate.
    """

    def __init__(self, control_qubit_id: str, target_qubit_id: str) -> None:
        """
        Initialize a CNOT gate.

        Args:
            control_qubit_id (str): The ID of the control qubit.
            target_qubit_id (str): The ID of the target qubit.
        """
        super().__init__(QGate.CNOT.value, [control_qubit_id, target_qubit_id])
        self.control_qubit_id = control_qubit_id
        self.target_qubit_id = target_qubit_id

    def __eq__(self, other: QuantumGate) -> bool:
        """
        Check equality of two CNOT gates.
        """
        if not isinstance(other, CNOTGate):
            return False
        return (self.control_qubit_id == other.control_qubit_id and
                self.target_qubit_id == other.target_qubit_id and
                super().__eq__(other))

    def get_generator(self) -> Hamiltonian:
        """
        Get the generator of the CNOT gate.

        Returns:
            Hamiltonian: The generator of the CNOT gate.
        """
        identity = "I2"
        pauli_x = QGate.PAULI_X.value
        pauli_z = QGate.PAULI_Z.value
        ham = Hamiltonian()
        tp1 = TensorProduct()
        tp1.add_operator(self.control_qubit_id, identity)
        tp1.add_operator(self.target_qubit_id, identity)
        ham.add_term((Fraction(1, 4), PI_SYMBOL, tp1))
        tp2 = TensorProduct()
        tp2.add_operator(self.control_qubit_id, pauli_z)
        tp2.add_operator(self.target_qubit_id, identity)
        ham.add_term((Fraction(-1, 4), PI_SYMBOL, tp2))
        tp3 = TensorProduct()
        tp3.add_operator(self.control_qubit_id, identity)
        tp3.add_operator(self.target_qubit_id, pauli_x)
        ham.add_term((Fraction(-1, 4), PI_SYMBOL, tp3))
        tp4 = TensorProduct()
        tp4.add_operator(self.control_qubit_id, pauli_z)
        tp4.add_operator(self.target_qubit_id, pauli_x)
        ham.add_term((Fraction(1, 4), PI_SYMBOL, tp4))
        conversion_dict = {
            "I2": np.eye(2, dtype=np.complex64),
            QGate.PAULI_X.value: pauli_matrices()[0],
            QGate.PAULI_Z.value: pauli_matrices()[2]
        }
        coeffs_map = {
            PI_SYMBOL: complex(np.pi)
        }
        ham.update_mappings(conversion_dict, coeffs_map)
        return ham

    def matrix(self) -> npt.NDArray[np.complex64]:
        """
        Get the matrix representation of the CNOT gate.

        Returns:
            npt.NDArray[np.complex64]: The matrix representation of the
                CNOT gate.
        """
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=np.complex64)

    @classmethod
    def from_enum(cls,
                  gate_enum: QGate | str,
                  control_qubit_id: str,
                  target_qubit_id: str
                  ) -> Self:
        """
        Create a CNOT gate from its control and target qubit IDs.

        Args:
            gate_enum (QGate | str): The enum or string representation of the
                CNOT gate.
            control_qubit_id (str): The ID of the control qubit.
            target_qubit_id (str): The ID of the target qubit.

        Returns:
            CNOTGate: The corresponding CNOT gate instance.
        """
        if isinstance(gate_enum, str):
            gate_enum = QGate(gate_enum)
        if gate_enum != QGate.CNOT:
            errstr = f"Invalid Enum for CNOTGate: {gate_enum}!"
            raise ValueError(errstr)
        return cls(control_qubit_id, target_qubit_id)

class SWAPGate(QuantumGate):
    """
    Class for the SWAP gate.
    """

    def __init__(self, qubit_id1: str, qubit_id2: str) -> None:
        """
        Initialize a SWAP gate.

        Args:
            qubit_id1 (str): The ID of the first qubit.
            qubit_id2 (str): The ID of the second qubit.
        """
        super().__init__(QGate.SWAP.value, [qubit_id1, qubit_id2])
        self.qubit_id1 = qubit_id1
        self.qubit_id2 = qubit_id2

    def get_generator(self) -> Hamiltonian:
        """
        Get the generator of the SWAP gate.

        Returns:
            Hamiltonian: The generator of the SWAP gate.
        """
        ham = Hamiltonian()
        identity = "I2"
        pauli_x = QGate.PAULI_X.value
        pauli_y = QGate.PAULI_Y.value
        pauli_z = QGate.PAULI_Z.value
        tp1 = TensorProduct()
        tp1.add_operator(self.qubit_id1, identity)
        tp1.add_operator(self.qubit_id2, identity)
        ham.add_term((Fraction(7, 4), PI_SYMBOL, tp1))
        tp2 = TensorProduct()
        tp2.add_operator(self.qubit_id1, pauli_x)
        tp2.add_operator(self.qubit_id2, pauli_x)
        ham.add_term((Fraction(1, 4), PI_SYMBOL, tp2))
        tp3 = TensorProduct()
        tp3.add_operator(self.qubit_id1, pauli_y)
        tp3.add_operator(self.qubit_id2, pauli_y)
        ham.add_term((Fraction(1, 4), PI_SYMBOL, tp3))
        tp4 = TensorProduct()
        tp4.add_operator(self.qubit_id1, pauli_z)
        tp4.add_operator(self.qubit_id2, pauli_z)
        ham.add_term((Fraction(1, 4), PI_SYMBOL, tp4))
        conversion_dict = {
            "I2": np.eye(2, dtype=np.complex64),
            QGate.PAULI_X.value: pauli_matrices()[0],
            QGate.PAULI_Y.value: pauli_matrices()[1],
            QGate.PAULI_Z.value: pauli_matrices()[2]
        }
        coeffs_map = {
            PI_SYMBOL: complex(np.pi)
        }
        ham.update_mappings(conversion_dict, coeffs_map)
        return ham

    def matrix(self) -> npt.NDArray[np.complex64]:
        """
        Get the matrix representation of the SWAP gate.

        Returns:
            npt.NDArray[np.complex64]: The matrix representation of the
                SWAP gate.
        """
        return swap_gate()

    @classmethod
    def from_enum(cls,
                  gate_enum: QGate | str,
                  qubit_id1: str,
                  qubit_id2: str
                  ) -> Self:
        """
        Create a SWAP gate from its qubit IDs.

        Args:
            gate_enum (QGate | str): The enum or string representation of the
                SWAP gate.
            qubit_id1 (str): The ID of the first qubit.
            qubit_id2 (str): The ID of the second qubit.

        Returns:
            SWAPGate: The corresponding SWAP gate instance.
        """
        if isinstance(gate_enum, str):
            gate_enum = QGate(gate_enum)
        if gate_enum != QGate.SWAP:
            errstr = f"Invalid Enum for SWAPGate: {gate_enum}!"
            raise ValueError(errstr)
        return cls(qubit_id1, qubit_id2)

class ToffoliGate(QuantumGate):
    """
    Class for the Toffoli gate.
    """

    def __init__(self,
                control_qubit_id1: str,
                control_qubit_id2: str,
                target_qubit_id: str) -> None:
        """
        Initialize a Toffoli gate.

        Args:
            control_qubit_id1 (str): The ID of the first control qubit.
            control_qubit_id2 (str): The ID of the second control qubit.
            target_qubit_id (str): The ID of the target qubit.
        """
        super().__init__(QGate.TOFFOLI.value,
                            [control_qubit_id1,
                            control_qubit_id2,
                            target_qubit_id])
        self.control_qubit_id1 = control_qubit_id1
        self.control_qubit_id2 = control_qubit_id2
        self.target_qubit_id = target_qubit_id

    def __eq__(self, other: QuantumGate) -> bool:
        """
        Check equality of two Toffoli gates.
        """
        if not isinstance(other, ToffoliGate):
            return False
        contr_eq = {self.control_qubit_id1, self.control_qubit_id2} == \
                   {other.control_qubit_id1, other.control_qubit_id2}
        return (contr_eq and
                self.target_qubit_id == other.target_qubit_id and
                super().__eq__(other))

    def get_generator(self) -> Hamiltonian:
        """
        Get the generator of the Toffoli gate.

        Returns:
            Hamiltonian: The generator of the Toffoli gate.
        """
        identity = "I2"
        pauli_x = QGate.PAULI_X.value
        pauli_z = QGate.PAULI_Z.value
        ham = Hamiltonian()
        combs_pos = [(identity, identity, identity),
                     (pauli_z, pauli_z, identity),
                     (identity, pauli_z, pauli_x),
                     (pauli_z, identity, pauli_x)
                     ]
        combs_neg = [(pauli_z, pauli_z, pauli_x),
                     (identity, pauli_z, identity),
                     (pauli_z, identity, identity),
                     (identity, identity, pauli_x)]
        for comb in combs_pos:
            tp = TensorProduct()
            tp.add_operator(self.control_qubit_id1, comb[0])
            tp.add_operator(self.control_qubit_id2, comb[1])
            tp.add_operator(self.target_qubit_id, comb[2])
            ham.add_term((Fraction(1, 8), PI_SYMBOL, tp))
        for comb in combs_neg:
            tp = TensorProduct()
            tp.add_operator(self.control_qubit_id1, comb[0])
            tp.add_operator(self.control_qubit_id2, comb[1])
            tp.add_operator(self.target_qubit_id, comb[2])
            ham.add_term((Fraction(-1, 8), PI_SYMBOL, tp))
        conversion_dict = {
            "I2": np.eye(2, dtype=np.complex64),
            QGate.PAULI_X.value: pauli_matrices()[0],
            QGate.PAULI_Z.value: pauli_matrices()[2]
        }
        coeffs_map = {
            PI_SYMBOL: complex(np.pi)
        }
        ham.update_mappings(conversion_dict, coeffs_map)
        return ham

    def matrix(self) -> npt.NDArray[np.complex64]:
        """
        Get the matrix representation of the Toffoli gate.

        Returns:
            npt.NDArray[np.complex64]: The matrix representation of the
                Toffoli gate.
        """
        return toffoli_gate()
    
    @classmethod
    def from_enum(cls,
                  gate_enum: QGate | str,
                  control_qubit_id1: str,
                  control_qubit_id2: str,
                  target_qubit_id: str) -> Self:
        """
        Create a Toffoli gate from its control and target qubit IDs.

        Args:
            gate_enum (QGate | str): The enum or string representation of the
                Toffoli gate.
            control_qubit_id1 (str): The ID of the first control qubit.
            control_qubit_id2 (str): The ID of the second control qubit.
            target_qubit_id (str): The ID of the target qubit.

        Returns:
            ToffoliGate: The corresponding Toffoli gate instance.
        """
        if isinstance(gate_enum, str):
            gate_enum = QGate(gate_enum)
        if gate_enum != QGate.TOFFOLI:
            errstr = f"Invalid Enum for ToffoliGate: {gate_enum}!"
            raise ValueError(errstr)
        return cls(control_qubit_id1,
                   control_qubit_id2,
                   target_qubit_id)
