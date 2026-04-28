"""
Functions used to build a measurement free Steane code circuit.
"""
from __future__ import annotations
from typing import Callable, Self
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy

import numpy as np

from ..qcircuit import QCircuit
from ....ttns.ttns import TreeTensorNetworkState
from ....core.node import Node
from ....special_ttn.star import StarTreeTensorState
from ....special_ttn.mps import MatrixProductState
from ....operators.common_operators import ket_i
from ....util.experiment_util.sim_params import SimulationParameters
from .three_qubit_circuits import ThreeQubitState


QUBIT_ID = "qubit"

NUM_QUB_IN_LOG = 7
NUM_QUB_ANCILLA = 3
TOT_NUM_QUB_PER_LOG = 2*NUM_QUB_IN_LOG + NUM_QUB_ANCILLA
NUM_LOG_QUB = 3

class Structures(Enum):
    """
    Enumeration of the different tree structures that can suite the Steane code.
    """
    MPS = "mps"
    TSTAR = "tstar"
    DOUBLETSTAR = "doubletstar"

    def zero_state(self,
                   id_container: QubitIDContainer | None = None
                   ) -> TreeTensorNetworkState:
        """
        Returns the zero state of the corresponding structure for the Steane code.

        Args:
            id_container (QubitIDContainer | None, optional): A container for the
                qubit IDs to be used in the state. If None, a default container with
                standard qubit ID generation will be used. Defaults to None.
        
        Returns:
            TreeTensorNetworkState: The zero state of the corresponding structure for
                the Steane code.
        """
        if id_container is None:
            id_container = QubitIDContainer.with_standard_gen_func()
        if self is Structures.MPS:
            return MatrixProductState.constant_product_state(0, 2,
                                                id_container.tot_num_qubits,
                                                node_prefix=id_container.node_prefix,
                                                root_site=id_container.tot_num_qubits//2)
        if self is Structures.TSTAR:
            # Here the main qubits are connected at the center.
            # All other qubits are radiating outwards from it
            num_chains = id_container.num_log_qubits
            ## Shape is three virtual and one trivial physical leg
            center_tensor = np.asarray([1], dtype=complex).reshape((1,1,1,1))
            one_chain_tensor: list[np.ndarray] = [ket_i(0,2).reshape((1,1,2))
                                                for _ in range(NUM_QUB_IN_LOG)]
            # The last one only connected to one other tensor
            one_chain_tensor[-1] = ket_i(0,2).reshape(1,2)
            chain_tensors = [[deepcopy(tensor) for tensor in one_chain_tensor]
                            for _ in range(num_chains)]
            identifiers = [id_container.all_qubits_per_logical(i)
                        for i in range(num_chains)]
            state = StarTreeTensorState.from_tensor_lists(center_tensor,
                                                        chain_tensors,
                                                        identifiers=identifiers)
            return state
        if self is Structures.DOUBLETSTAR:
            # Here the main qubits are connected at the center
            # The remaining logical qubits are attached as a chain to their main qubit.
            # Then another intersection follows separating the ancilla logical and the
            # full ancilla qubits, which are each attached as a chain.
            num_chains = id_container.num_log_qubits*3
            state = TreeTensorNetworkState()
            # The center tensor is the root with 3 virtual and one trivial physical leg
            center_tensor = np.asarray([1], dtype=complex).reshape((1,1,1,1))
            center_id = "center"
            state.add_root(Node(identifier=center_id),
                        center_tensor)
            for i in range(id_container.num_log_qubits):
                # Adding the main chain tensors
                main_tensor = ket_i(0,2).reshape((1,1,2))
                main_tensors = [deepcopy(main_tensor)
                                for _ in range(NUM_QUB_IN_LOG)]
                main_ids = id_container.logical_qubits(i)
                state.add_chain_to_parent(main_ids,
                                        main_tensors,
                                        parent_id=center_id,
                                        parent_leg=i)
                # Adding the next intersection tensor
                inter_tensor = deepcopy(center_tensor)
                inter_id = f"inter_{i}"
                state.add_child_to_parent(Node(identifier=inter_id),
                                        inter_tensor,
                                        parent_id=main_ids[-1],
                                        parent_leg=1,
                                        child_leg=0)
                # Adding the ancilla logical chain
                anc_log_tensors: list[np.ndarray] = [deepcopy(main_tensor)
                                                    for _ in range(NUM_QUB_IN_LOG)]
                anc_log_tensors[-1] = ket_i(0,2).reshape(1,2)
                anc_log_ids = id_container.logical_ancilla_qubits(i)
                ## Note we want the last qubits at the intersection, as they interact
                ## the most with the full ancilla qubits and the main chain
                anc_log_ids.reverse()
                parent_leg = 1
                state.add_chain_to_parent(anc_log_ids,
                                        anc_log_tensors,
                                        inter_id,
                                        parent_leg)
                # Adding the ancilla chain
                anc_tensors: list[np.ndarray] = [deepcopy(main_tensor)
                                                for _ in range(NUM_QUB_ANCILLA)]
                anc_tensors[-1] = ket_i(0,2).reshape(1,2)
                anc_ids = id_container.ancilla_qubits(i)
                parent_leg += 1
                state.add_chain_to_parent(anc_ids,
                                        anc_tensors,
                                        inter_id,
                                        parent_leg)
            return state
        errstr = f"Cannot generate zero state for structure {self}!"
        raise ValueError(errstr)

@dataclass
class STEANCodeParams(SimulationParameters):
    """
    Parameters for the measurement-free STEAN code simulation.
    """
    seed: int = 42
    structure: Structures = Structures.MPS
    considered_state: ThreeQubitState = ThreeQubitState.GHZ

def gen_qubit_id(i: int) -> str:
    """
    Generates a qubit ID string based on the given index.

    Args:
        i (int): The index of the qubit.

    Returns:
        str: The generated qubit ID string.
    """
    return f"{QUBIT_ID}_{i}"

def apply_hadamards(circuit: QCircuit,
                    qubits: list[str],
                    level_index: int
                    ) -> int:
    """
    Applies Hadamard gates to the specified qubits in the given quantum circuit.

    Args:
        circuit (QCircuit): The quantum circuit to which the gates will be
            applied.
        qubits (list[str]): A list of qubit IDs to which the Hadamard gates
            will be applied.
        level_index (int): The index of the level in the circuit where the
            gates will be applied.
    """
    for qubit in qubits:
        circuit.add_hadamard(qubit,
                             level_index=level_index)
    return level_index + 1

def apply_resets(circuit: QCircuit,
                qubits: list[str],
                level_index: int
                ) -> None:
    """
    Adds reset operations to the specified qubits in the given quantum circuit.

    Args:
        circuit (QCircuit): The quantum circuit to which the reset operations will
            be applied.
        qubits (list[str]): A list of qubit IDs to which the reset operations will
            be applied.
        level_index (int): The index of the level in the circuit where the reset
            operations will be applied.
    
    """
    for qubit in qubits:
        circuit.add_reset(qubit,
                        level_index=level_index)

def apply_parallel_cnot(circuit: QCircuit,
                        control_qubits: list[str],
                       target_qubits: list[str],
                       level_index: int
                       ) -> int:
    """
    Applies parallel CNOT gates between the specified control and target qubits
    in the given quantum circuit.

    Args:
        circuit (QCircuit): The quantum circuit to which the gates will be
            applied.
        control_qubits (list[str]): A list of control qubit IDs for the CNOT
            gates.
        target_qubits (list[str]): A list of target qubit IDs for the CNOT
            gates.
        level_index (int): The index of the level in the circuit where the
            gates will be applied.
    """
    for control, target in zip(control_qubits, target_qubits):
        circuit.add_cnot(control,
                         target,
                         level_index=level_index)
    return level_index + 1

def apply_serial_cnot_control(circuit: QCircuit,
                              control_qubits: list[str],
                              target_qubit: str,
                              level_index: int
                              ) -> int:
    """
    Applies serial CNOT gates with the specified control qubits and a single
    target qubit in the given quantum circuit.

    Args:
        circuit (QCircuit): The quantum circuit to which the gates will be
            applied.
        control_qubits (list[str]): A list of control qubit IDs for the CNOT
            gates.
        target_qubit (str): The ID of the target qubit for the CNOT gates.
        level_index (int): The index of the level in the circuit where the
            first CNOT gate will be applied. Subsequent CNOT gates will be
            applied at increasing level indices.

    Returns:
        int: The next level index after applying all the CNOT gates.
    """
    for control in control_qubits:
        circuit.add_cnot(control,
                         target_qubit,
                         level_index=level_index)
        level_index += 1
    return level_index

def apply_serial_cnot_target(circuit: QCircuit,
                             control_qubit: str,
                             target_qubits: list[str],
                             level_index: int
                             ) -> int:
    """
    Applies serial CNOT gates with a single control qubit and the specified
    target qubits in the given quantum circuit.

    Args:
        circuit (QCircuit): The quantum circuit to which the gates will be
            applied.
        control_qubit (str): The ID of the control qubit for the CNOT gates.
        target_qubits (list[str]): A list of target qubit IDs for the CNOT
            gates.
        level_index (int): The index of the level in the circuit where the
            first CNOT gate will be applied. Subsequent CNOT gates will be
            applied at increasing level indices.
    
    Returns:
        int: The next level index after applying all the CNOT gates.
    """
    for target in target_qubits:
        circuit.add_cnot(control_qubit,
                         target,
                         level_index=level_index)
        level_index += 1
    return level_index

def logical_plus(circuit: QCircuit,
                 qubits: list[str],
                 level_index: int
                 ) -> int:
    """
    Encodes the logical plus state of the Steane code on the specified qubits in the
    given quantum circuit.

    Args:
        circuit (QCircuit): The quantum circuit to which the state preparation will
            be applied.
        qubits (list[str]): A list of qubit IDs for the logical plus state
            preparation.
        level_index (int): The index of the level in the circuit where the state
            preparation will be applied. Subsequent operations within the state
            preparation will be applied at increasing level indices.
    
    Returns:
        int: The next level index after applying the logical plus state
            preparation.
    """
    assert len(qubits) == 7
    level_index = encoder(circuit,
                          qubits[0],
                          qubits[1:],
                          level_index=level_index)
    level_index = apply_hadamards(circuit,
                                  qubits,
                                  level_index=level_index)
    return level_index

def encoder(circuit: QCircuit,
            main_qubit: str,
            added_qubits: list[str],
            level_index: int
            ) -> int:
    """
    Adds an encoder circuit to the given quantum circuit.

    It encodes the state of the main qubit into the Steane code.

    Args:
        circuit (QCircuit): The quantum circuit to which the encoder will be
            applied.
        main_qubit (str): The ID of the main qubit whose state will be encoded.
        added_qubits (list[str]): A list of qubit IDs that will be used as
            additional qubits in the encoding process.
        level_index (int): The index of the level in the circuit where the
            encoder will be applied. Subsequent operations within the encoder
            will be applied at increasing level indices.
    """
    if len(added_qubits) != 6:
        errstr = f"Expected 6 added qubits for the encoder, but got {len(added_qubits)}!"
        raise ValueError(errstr)
    apply_hadamards(circuit,
                    added_qubits[:3],
                    level_index)
    qubits = [(main_qubit,added_qubits[3:5]),
              (added_qubits[2],[main_qubit,added_qubits[3],added_qubits[5]]),
              (added_qubits[1],[main_qubit,added_qubits[4],added_qubits[5]]),
              (added_qubits[0],added_qubits[3:])]
    for control, targets in qubits:
        circuit.add_mx([control],[],
                        targets,
                        level_index=level_index)
        level_index += 1
    return level_index

def x_block(circuit: QCircuit,
            logic_q_ids: list[str],
            copy_q_ids: list[str],
            ancilla_q_ids: list[str],
            level_index: int
            ) -> int:
    """
    Applies the X-block of the measurement-free STEAN code to the specified
    qubits in the given quantum circuit.

    Args:
        circuit (QCircuit): The quantum circuit to which the X-block will be
            applied.
        logic_q_ids (list[str]): A list of logical qubit IDs for the X-block.
        copy_q_ids (list[str]): A list of copy qubit IDs for the X-block.
        ancilla_q_ids (list[str]): A list of ancilla qubit IDs for the X-block.
        level_index (int): The index of the level in the circuit where the X-block
            will be applied. Subsequent operations within the X-block will be
            applied at increasing level indices.
    
    Returns:
        int: The next level index after applying the X-block.
    """
    level_index = logical_plus(circuit,
                               copy_q_ids,
                               level_index=level_index)
    level_index = apply_parallel_cnot(circuit,
                                      logic_q_ids,
                                      copy_q_ids,
                                      level_index=level_index)
    level_index = apply_serial_cnot_control(circuit,
                                            copy_q_ids[4:],
                                            ancilla_q_ids[0],
                                            level_index=level_index)
    level_index = apply_serial_cnot_control(circuit,
                                            [copy_q_ids[i] for i in range(0,7,2)],
                                            ancilla_q_ids[1],
                                            level_index=level_index)
    level_index = apply_serial_cnot_control(circuit,
                                            [copy_q_ids[i] for i in [1,2,5,6]],
                                            ancilla_q_ids[2],
                                            level_index=level_index)
    control = [[True, True, True],
               [True, False, True],
               [True, True, False],
               [True, False, False],
                [False, True, True],
                [False, False, True]]
    main_qubits = [copy_q_ids[i] for i in [4,5,6]]
    for i, contr in enumerate(control):
        apply_resets(circuit,
                        main_qubits,
                        level_index)
        level_index += 1
        level_index = apply_parallel_cnot(circuit,
                                            ancilla_q_ids,
                                            [copy_q_ids[i] for i in [4,5,6]],
                                            level_index=level_index)
        cont_qs = [main_qubits[j] for j in range(3) if contr[j]]
        icont_qs = [main_qubits[j] for j in range(3) if not contr[j]]
        circuit.add_mx(cont_qs,
                        icont_qs,
                        logic_q_ids[-1*i],
                        level_index=level_index)
        level_index += 1
    circuit.add_mx([ancilla_q_ids[1]],
                    [ancilla_q_ids[0], ancilla_q_ids[2]],
                    logic_q_ids[0],
                    level_index=level_index-1)
    return level_index

def z_block(circuit: QCircuit,
            logic_q_ids: list[str],
            copy_q_ids: list[str],
            ancilla_q_ids: list[str],
            level_index: int
            ) -> int:
    """
    Applies the Z-block of the measurement-free STEAN code to the specified
    qubits in the given quantum circuit.

    Args:
        circuit (QCircuit): The quantum circuit to which the Z-block will be
            applied.
        logic_q_ids (list[str]): A list of logical qubit IDs for the X-block.
        copy_q_ids (list[str]): A list of copy qubit IDs for the X-block.
        ancilla_q_ids (list[str]): A list of ancilla qubit IDs for the X-block.
        level_index (int): The index of the level in the circuit where the X-block
            will be applied. Subsequent operations within the X-block will be
            applied at increasing level indices.
    
    Returns:
        int: The next level index after applying the X-block.
    """
    apply_hadamards(circuit,
                    ancilla_q_ids,
                    level_index)
    level_index = encoder(circuit,
                          copy_q_ids[0],
                          copy_q_ids[1:],
                          level_index)
    level_index = apply_parallel_cnot(circuit,
                                      copy_q_ids,
                                      logic_q_ids,
                                      level_index=level_index)
    level_index = apply_serial_cnot_target(circuit,
                                           ancilla_q_ids[0],
                                            copy_q_ids[4:],
                                            level_index=level_index)
    level_index = apply_serial_cnot_target(circuit,
                                           ancilla_q_ids[1],
                                            [copy_q_ids[i] for i in range(0,7,2)],
                                            level_index=level_index)
    level_index = apply_serial_cnot_target(circuit,
                                           ancilla_q_ids[2],
                                            [copy_q_ids[i] for i in [1,2,5,6]],
                                            level_index=level_index)
    apply_hadamards(circuit,
                    ancilla_q_ids,
                    level_index)
    level_index += 1
    control = [[True, True, True],
               [True, False, True],
               [True, True, False],
               [True, False, False],
                [False, True, True],
                [False, False, True]]
    main_qubits = [copy_q_ids[i] for i in [4,5,6]]
    for i, contr in enumerate(control):
        apply_resets(circuit,
                        main_qubits,
                        level_index)
        level_index += 1
        level_index = apply_parallel_cnot(circuit,
                                            ancilla_q_ids,
                                            [copy_q_ids[i] for i in [4,5,6]],
                                            level_index=level_index)
        cont_qs = [main_qubits[j] for j in range(3) if contr[j]]
        icont_qs = [main_qubits[j] for j in range(3) if not contr[j]]
        circuit.add_mz(cont_qs,
                        icont_qs,
                        logic_q_ids[-1*i],
                        level_index=level_index)
        level_index += 1
    circuit.add_mz([ancilla_q_ids[1]],
                    [ancilla_q_ids[0], ancilla_q_ids[2]],
                    logic_q_ids[0],
                    level_index=level_index-1)
    return level_index

class QubitIDContainer:
    """
    A class to easily distuingish the different types of qubits in the circuit.

    Mainly used for readability, as the qubits are just identified by strings and the
    different types of qubits are interspersed in the circuit.
    """

    def __init__(self,
                 gen_qubit_id_func: Callable[[int], str]
                 ) -> None:
        self.num_log_qubits = NUM_LOG_QUB
        self.tot_num_qubits = self.num_log_qubits*TOT_NUM_QUB_PER_LOG
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

    def _main_index_to_acutal_index(self, index: int) -> int:
        """
        Convert the given index for the main qubits to the actual index in the qubit ID list.
        """
        self._main_index_check(index)
        return index*TOT_NUM_QUB_PER_LOG

    def main_qubit(self, index: int) -> str:
        """
        Get the ID of the main qubit number `index`.
        """
        return self.qubit_ids[self._main_index_to_acutal_index(index)]

    def main_qubits(self) -> list[str]:
        """
        Get the IDs of all main qubits.
        """
        return [self.main_qubit(i) for i in range(self.num_log_qubits)]

    def logical_qubits(self, index: int) -> list[str]:
        """
        Get the IDs of the logical qubits for logical qubit number `index`.

        Includes the corresponding main qubit as the first one.
        """
        start = self._main_index_to_acutal_index(index)
        return self.qubit_ids[start:start+NUM_QUB_IN_LOG]

    def logical_ancilla_qubits(self, index: int) -> list[str]:
        """
        Get the IDs of the logical ancilla qubits for logical qubit number `index`.
        """
        start = self._main_index_to_acutal_index(index)
        return self.qubit_ids[start+NUM_QUB_IN_LOG:start+2*NUM_QUB_IN_LOG]

    def ancilla_qubits(self, index: int) -> list[str]:
        """
        Get the IDs of the ancilla qubits for logical qubit number `index`.
        """
        start = self._main_index_to_acutal_index(index)
        return self.qubit_ids[start+2*NUM_QUB_IN_LOG:start+TOT_NUM_QUB_PER_LOG]

    def all_qubits_per_logical(self, index: int):
        """
        Get the IDs of all qubits (main, logical, acilla_logical, ancilla) for logical qubit number `index`.
        """
        out = self.logical_qubits(index)
        out.extend(self.logical_ancilla_qubits(index))
        out.extend(self.ancilla_qubits(index))
        return out

    @classmethod
    def with_standard_gen_func(cls) -> Self:
        """
        Create a QubitIDContainer with the standard qubit ID generation function.
        """
        out = cls(gen_qubit_id)
        out._node_prefix = QUBIT_ID+"_"
        return out

def build_circuit(state: ThreeQubitState,
                  idcontainer: QubitIDContainer | None = None
                  ) -> QCircuit:
    """
    Builds the quantum circuit to store the given state in the measurement-free STEAN code.

    Args:
        state (ThreeQubitState): The state to be stored in the code. This
            determines the state preparation part of the circuit.
        idcontainer (QubitIDContainer | None, optional): A container for the
            qubit IDs to be used in the circuit. If None, a default container
            with standard qubit ID generation will be used. Defaults to None.

    Returns:
        QCircuit: The constructed quantum circuit for the measurement-free STEAN
            code.
    """
    if idcontainer is None:
        idcontainer = QubitIDContainer.with_standard_gen_func()
    circuit = QCircuit()
    # First we generate the state on the main qubits
    level_index = 0
    prep_function = state.preparation_function()
    level_index = prep_function(circuit,
                                idcontainer.main_qubits(),
                                level_index)
    # Then we apply the X and Z blocks for each logical qubit
    level_index_after_prep = level_index
    for i in range(idcontainer.num_log_qubits):
        # Since the x_block starts with an encoding of the + state on the
        # copy/ancillary logical qubit, we can start it at the same time as the
        # main encoder. Since it is slightly longer, it will yield the new
        # level index.
        encoder(circuit,
                idcontainer.main_qubit(i),
                idcontainer.logical_qubits(i),
                level_index=level_index)
        level_index = x_block(circuit,
                              idcontainer.logical_qubits(i),
                              idcontainer.logical_ancilla_qubits(i),
                              idcontainer.ancilla_qubits(i),
                              level_index=level_index)
        level_index = z_block(circuit,
                              idcontainer.logical_qubits(i),
                              idcontainer.logical_ancilla_qubits(i),
                              idcontainer.ancilla_qubits(i),
                              level_index=level_index)
        # The error correction on each qubit can happen in parallel
        level_index = level_index_after_prep
    return circuit

