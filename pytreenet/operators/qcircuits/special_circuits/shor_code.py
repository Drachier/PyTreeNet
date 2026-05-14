"""
Functions to build the Shor code circuit.
"""
from __future__ import annotations
from typing import Callable

from ..qcircuit import QCircuit
from .qubit_id_container import QubitIDContainer, gen_qubit_id, QUBIT_PREFIX
from ...measurment import (Outcome,
                           TensorProduct)
from ..qgate import QuantumGate
from ...common_operators import pauli_matrices

NUM_QUBITS_PER_LOGICAL_QUBIT = 11

class QubitIDContainerShor(QubitIDContainer):
    """
    A class to easily distuingish the different types of qubits in the circuit.

    The Shor code contains three different parts per logical qubit, each
    containing 3 qubits. For each of these parts there is a main qubit,
    we call it the main logical qubit, one of those is the actual main qubit.
    Additionally, there are two ancilla qubits per part.
    """

    def __init__(self, gen_qubit_id_func: Callable[[int], str],
                 num_log_qubits: int = 1) -> None:
        super().__init__(gen_qubit_id_func,
                         NUM_QUBITS_PER_LOGICAL_QUBIT,
                         num_log_qubits)
        self.qubits_per_logical_part = 3
        self.logical_parts = 3
        self.num_meas_per_logical = 2
        
    @classmethod
    def with_standard_gen_func(cls,
                               num_log_qubits: int = 1
                               ) -> QubitIDContainerShor:
        """
        Create a QubitIDContainer with the standard qubit ID generation
        function.
        """
        out = cls(gen_qubit_id, num_log_qubits=num_log_qubits)
        out._node_prefix = QUBIT_PREFIX
        return out
    
    def logical_index_valid(self, logic_index: int) -> bool:
        """
        Check if the given logical index is valid for the Shor code.

        Args:
            logic_index (int): The logical index to check.
        
        Returns:
            bool: True if the logical index is valid for the Shor code,
                False otherwise.
        """
        return logic_index >= 0 and logic_index < self.qubits_per_logical_part
    
    def _logical_index_check(self, logic_index: int) -> None:
        """
        Check if the given logical index is valid for the Shor code and raise an error if not.

        Args:
            logic_index (int): The logical index to check.
        """
        if not self.logical_index_valid(logic_index):
            errstr = f"Invalid logical index {logic_index} for the Shor code!\n"
            errstr += f"Valid logical indices are from 0 to {self.qubits_per_logical_part-1}."
            raise ValueError(errstr)
        
    def _main_and_logical_index_to_actual_index(self,
                                              main_index: int,
                                              logic_index: int
                                              ) -> int:
        """
        Convert the given main and logical indices to the actual index in the
        qubit ID list.

        Args:
            main_index (int): The main index to convert.
            logic_index (int): The logical index to convert.

        Returns:
            int: The actual index in the qubit ID list corresponding to the
            given main and logical indices.
        """
        self._main_index_check(main_index)
        self._logical_index_check(logic_index)
        actual_main_index = self._main_index_to_actual_index(main_index)
        qubit_index = actual_main_index + logic_index*self.qubits_per_logical_part
        return qubit_index
    
    def main_logical_qubit(self,
                           main_index: int,
                           logic_index: int
                           ) -> str:
        """
        Return the ID of the main logical qubit for the given indices.

        Args:
            main_index (int): The index of the main logical qubit.
            logic_index (int): The part index inside of the logical qubit.

        
        Returns:
            str: The ID of the main logical qubit for the given indices.
        """
        qubit_index = self._main_and_logical_index_to_actual_index(main_index,
                                                                logic_index)
        return self.qubit_ids[qubit_index]
    
    def main_logical_qubits(self, main_index: int) -> list[str]:
        """
        Return the IDs of the main logical qubits for the given main index.

        Args:
            main_index (int): The index of the main logical qubits.
        
        Returns:
            list[str]: The IDs of the main logical qubits for the given main index.
        """
        return [self.main_logical_qubit(main_index, logic_index)
                for logic_index in range(self.logical_parts)]
    
    def main_logical_qubits_without_actual_main(self,
                                                main_index: int
                                                ) -> list[str]:
        """
        Return the IDs of the main logical qubits for the given main index,
        without the actual main logical qubit.
        """
        return self.main_logical_qubits(main_index)[1:]
    
    def ancilla_qubits_for_logical_part(self,
                                        main_index: int,
                                        logic_index: int
                                        ) -> list[str]:
        """
        Return the IDs of the ancilla qubits for the given indices.

        Args:
            main_index (int): The index of the main logical qubit.
            logic_index (int): The part index inside of the logical qubit.

        Returns:
            list[str]: The IDs of the ancilla qubits for the given indices.
        """
        qubit_index = self._main_and_logical_index_to_actual_index(main_index,
                                                                logic_index)
        return [self.qubit_ids[qubit_index+1],
                self.qubit_ids[qubit_index+2]]
    
    def logical_qubits_of_part(self,
                               main_index: int,
                               logic_index: int
                               ) -> list[str]:
        """
        Return the IDs of the logical qubits of the given part.

        Args:
            main_index (int): The index of the main logical qubit.
            logic_index (int): The part index inside of the logical qubit.
        
        Returns:
            list[str]: The IDs of the qubits that are part of the given
                logical part.
        """
        qubit_index = self._main_and_logical_index_to_actual_index(main_index,
                                                                logic_index)
        return self.qubit_ids[qubit_index:qubit_index+self.qubits_per_logical_part]
    
    def all_but_measurement_qubits_of_part(self,
                                            main_index: int
                                            ) -> list[str]:
        """
        Return the IDs of all qubits of the given logical part except for the
        measurement qubits.

        Args:
            main_index (int): The index of the main logical qubit.

        Returns:
            list[str]: The IDs of the qubits that are part of the given
                logical part, excluding the measurement qubits.
        """
        out = []
        for log_index in range(self.logical_parts):
            out.extend(self.logical_qubits_of_part(main_index, log_index))
        return out

    def measurement_qubits(self,
                          main_index: int
                          ) -> list[str]:
        """
        Return the IDs of the measurment qubits for the given main index.

        Args:
            main_index (int): The index of the main logical qubits for which to
                return the measurment qubits.

        Returns:
            list[str]: The IDs of the measurment qubits for the given main index.
        """
        # These are the last two qubits of the logical qubit.
        main_index = self._main_index_to_actual_index(main_index)
        first_meas_index = main_index + self.logical_parts*self.qubits_per_logical_part
        return [self.qubit_ids[first_meas_index + i] for i in (0,1)]

def encoder(circuit: QCircuit,
            idcontainer: QubitIDContainerShor,
            logical_qubit_index: int = 0,
            level_index: int = 0
            ) -> int:
    """
    Adds the encoding circuit for the Shor code to the given circuit.

    Args:
        circuit (QCircuit): The circuit to which the encoding circuit should
            be added.
        idcontainer (QubitIDContainerShor): The container for the qubit IDs.
        logical_qubit_index (int, optional): Which logical qubit as defined in
            the idcontainer should be encoded. Defaults to 0.
        level_index (int, optional): The index of the level at which to add the
            encoding circuit. Defaults to 0.

    Returns:
        int: The index of the level after adding the encoding circuit. This
            can be used to add further operations at the correct level after
            the encoding circuit.
    """
    circuit.add_mx([idcontainer.main_qubit(logical_qubit_index)], [],
                   idcontainer.main_logical_qubits_without_actual_main(logical_qubit_index),
                   level_index=level_index)
    level_index += 1
    for qubit_id in idcontainer.main_logical_qubits(logical_qubit_index):
        circuit.add_hadamard(qubit_id, level_index=level_index)
    level_index += 1
    for log_index in range(idcontainer.logical_parts):
        circuit.add_mx([idcontainer.main_logical_qubit(logical_qubit_index, log_index)],
                       [],
                       idcontainer.ancilla_qubits_for_logical_part(logical_qubit_index, log_index),
                       level_index=level_index)
    level_index += 1
    return level_index

def local_bit_flip_correction(circuit: QCircuit,
                               logical_qubits: list[str],
                               measurement_qubits: list[str],
                               level_index: int) -> int:
    """
    Perform syndrome measurement and correction for a bit flip code.

    Args:
        circuit: The quantum circuit to which the syndrome measurement and
            correction gates will be added.
        logical_qubits: A list of names of the logical qubits that are part of
            the bit flip code.
        measurement_qubits: A list of names of the qubits used for syndrome
            measurement.
        level_index: The index of the circuit level at which the syndrome
            measurement and correction gates will be added.
        
    Returns:
        The new level index after adding the syndrome measurement and
        correction gates.
    """
    circuit.add_cnot(logical_qubits[0], measurement_qubits[0],
                     level_index=level_index)
    level_index += 1
    circuit.add_mx([logical_qubits[1]], [],
                   measurement_qubits,
                   level_index=level_index)
    level_index += 1
    circuit.add_cnot(logical_qubits[2], measurement_qubits[1],
                     level_index=level_index)
    level_index += 1
    pauli_x = pauli_matrices()[0]
    outcome_10 = Outcome(measurement_qubits,[1,0])
    operation_10 = TensorProduct({logical_qubits[0]: pauli_x,
                                  measurement_qubits[0]: pauli_x})
    outcome_11 = Outcome(measurement_qubits,[1,1])
    operation_11 = TensorProduct({logical_qubits[1]: pauli_x,
                                  measurement_qubits[0]: pauli_x,
                                  measurement_qubits[1]: pauli_x})
    outcome_01 = Outcome(measurement_qubits,[0,1])
    operation_01 = TensorProduct({logical_qubits[2]: pauli_x,
                                  measurement_qubits[1]: pauli_x})
    operation = {outcome_10: operation_10,
                 outcome_11: operation_11,
                 outcome_01: operation_01,
                 }
    circuit.add_measurement_controlled_gate(measurement_qubits,
                                            operation,
                                            level_index=level_index)
    return level_index + 1

def global_phase_flip_correction(circuit: QCircuit,
                               logical_qubits: tuple[list[str],list[str],list[str]],
                               measurement_qubits: list[str],
                               level_index: int) -> int:
    """
    Perform syndrome measurement and correction for a phase flip code.

    Args:
        circuit: The quantum circuit to which the syndrome measurement and
            correction gates will be added.
        logical_qubits: A tuple of three lists of names of the logical qubits
            that are part of the phase flip code. Each list corresponds to one
            of the three parts of the logical qubits in the Shor code.
        measurement_qubits: A list of names of the qubits used for syndrome
            measurement.
        level_index: The index of the circuit level at which the syndrome
            measurement and correction gates will be added.
    
    Returns:
        The new level index after adding the syndrome measurement and
        correction gates.
    """
    for qubit in measurement_qubits:
        circuit.add_hadamard(qubit,
                             level_index=level_index)
    level_index += 1
    circuit.add_mx([measurement_qubits[0]],
                   [],
                   logical_qubits[0]+logical_qubits[1],
                   level_index=level_index)
    level_index += 1
    circuit.add_mx([measurement_qubits[1]],
                    [],
                    logical_qubits[1]+logical_qubits[2],
                    level_index=level_index)
    level_index += 1
    for qubit in measurement_qubits:
        circuit.add_hadamard(qubit,
                             level_index=level_index)
    level_index += 1
    pauli_x, _, pauli_z = pauli_matrices()
    outcome_10 = Outcome(measurement_qubits,[1,0])
    operation_10 = TensorProduct({logical_qubits[0][0]: pauli_z,
                                  measurement_qubits[0]: pauli_x})
    outcome_11 = Outcome(measurement_qubits,[1,1])
    operation_11 = TensorProduct({logical_qubits[1][0]: pauli_z,
                                  measurement_qubits[0]: pauli_x,
                                  measurement_qubits[1]: pauli_x})
    outcome_01 = Outcome(measurement_qubits,[0,1])
    operation_01 = TensorProduct({logical_qubits[2][0]: pauli_z,
                                  measurement_qubits[1]: pauli_x})
    operation = {outcome_10: operation_10,
                 outcome_11: operation_11,
                 outcome_01: operation_01,
                }
    circuit.add_measurement_controlled_gate(measurement_qubits,
                                            operation,
                                            level_index=level_index)
    return level_index + 1

def shor_correction(circuit: QCircuit,
                    idcontainer: QubitIDContainerShor,
                    logical_qubit_index: int = 0,
                    level_index: int = 0
                    ) -> int:
    """
    Add the full error correction circuit for the Shor code to the given circuit.

    Args:
        circuit (QCircuit): The circuit to which the error correction circuit
            should be added.
        idcontainer (QubitIDContainerShor): The container for the qubit IDs.
        logical_qubit_index (int, optional): Which logical qubit as defined in
            the idcontainer should be corrected. Defaults to 0.
        level_index (int, optional): The index of the level at which to add the
            error correction circuit. Defaults to 0.
    
    Returns:
        int: The index of the level after adding the error correction circuit. This
            can be used to add further operations at the correct level after
            the error correction circuit.
    """
    logical_qubits = tuple([idcontainer.logical_qubits_of_part(logical_qubit_index,
                                                         log_index)
                        for log_index in range(idcontainer.logical_parts)])
    measurement_qubits = idcontainer.measurement_qubits(logical_qubit_index)
    for logical_part in logical_qubits:
        level_index = local_bit_flip_correction(circuit,
                                               logical_part,
                                               measurement_qubits,
                                               level_index)
    level_index = global_phase_flip_correction(circuit,
                                               logical_qubits,
                                               measurement_qubits,
                                               level_index)
    return level_index

def shor_code_circuit(circuit: QCircuit,
                      idcontainer: QubitIDContainerShor,
                      logical_qubit_index: int = 0,
                      level_index: int = 0,
                      error_gates: list[QuantumGate] | None = None
                      ) -> int:
    """
    Add the full Shor code circuit, including encoding and error correction,
    to the given circuit.

    Args:
        circuit (QCircuit): The circuit to which the Shor code circuit should be
            added.
        idcontainer (QubitIDContainerShor): The container for the qubit IDs.
        logical_qubit_index (int, optional): Which logical qubit as defined in
            the idcontainer should be used for the Shor code. Defaults to 0.
        level_index (int, optional): The index of the level at which to add the
            Shor code circuit. Defaults to 0.
        error_gates (list[QuantumGate] | None, optional): A list of quantum
            gates representing the errors to be applied after encoding and
            before error correction. If None, no errors will be applied. Defaults
            to None.

    Returns:
        int: The index of the level after adding the Shor code circuit. This
            can be used to add further operations at the correct level after
            the Shor code circuit.
    """
    level_index = encoder(circuit,
                          idcontainer,
                          logical_qubit_index=logical_qubit_index,
                          level_index=level_index)
    if error_gates:
        for error_gate in error_gates:
            circuit.add_gate(error_gate,
                             level_index=level_index)
        level_index += 1
    level_index = shor_correction(circuit,
                                  idcontainer,
                                  logical_qubit_index=logical_qubit_index,
                                  level_index=level_index)
    return level_index