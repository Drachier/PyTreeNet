"""
Simulation script for the quantum circuit simulation experiment.
"""
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
from time import time
import os

from h5py import File

from pytreenet.operators.qcircuits.qcircuit import QCircuit
from pytreenet.operators.qcircuits.qgate import HaarRandomSingleQubitGate
from pytreenet.special_ttn.special_states import (TTNStructure,
                                                  generate_zero_state)
from pytreenet.special_ttn.special_nodes import trivial_virtual_node
from pytreenet.operators.common_operators import ket_i
from pytreenet.core.node import Node
from pytreenet.ttns.ttns import TTNS
from pytreenet.util.experiment_util.sim_params import SimulationParameters
from pytreenet.ttns.ttns_ttno.application import (ApplicationMethod,
                                                  apply_ttno_to_ttns)
from pytreenet.ttno.ttno_class import TTNO
from pytreenet.util.tensor_splitting import SVDParameters
from pytreenet.time_evolution.results import Results
from pytreenet.util.experiment_util.script_util import script_main

@dataclass
class CircuitSimParams(SimulationParameters):
    """
    Parameters for the circuit simulation experiment.
    """
    ttn_structure: TTNStructure = TTNStructure.MPS
    appl_method: ApplicationMethod = ApplicationMethod.DIRECT_TRUNCATE
    num_circuit_repeats: int = 1
    min_bond_dim: int = 2
    max_bond_dim: int = 4
    bond_dim_step: int = 2
    seed: int = 12345

def add_random_two_qubit_gate(qcircuit: QCircuit,
                              level_index: int,
                              qubit_id1: str,
                              qubit_id2: str,
                              seed: int | None = None,
                              ) -> None:
    """
    Add a random two-qubit gate (CNOT) to the quantum circuit at the specified level.

    Args:
        qcircuit (QCircuit): The quantum circuit to which the gate will be added.
        level_index (int): The index of the level in the quantum circuit.
        qubit_id1 (str): The ID of the first qubit.
        qubit_id2 (str): The ID of the second qubit.
        seed (int | None): Seed for random number generation. Defaults to None.
    """
    # We only need to add single-qubit gates before the CNOT
    # Gates after it would just be absorbed into the next gates
    # As we run U U† = I for truncation, we can ignore the final single-qubit gates
    random_sq_gates = [HaarRandomSingleQubitGate(qubit_id, seed=seed)
                       for qubit_id in [qubit_id1, qubit_id2]]
    for gate in random_sq_gates:
        qcircuit.add_gate(gate, level_index=level_index)
    qcircuit.add_cnot(qubit_id1, qubit_id2,
                      level_index=level_index+1)

def build_circuit(cnot_pairs: list[list[tuple[str,str]]],
                  seed: int | None = None,
                  ) -> QCircuit:
    """
    Build a quantum circuit based on the specified CNOT gate pairs.

    Args:
        cnot_pairs (list[list[tuple[str,str]]]): A list of levels, each
            containing a list of CNOT gate pairs.
        seed (int | None): Seed for random number generation.
            Defaults to None.

    Returns:
        QCircuit: The constructed quantum circuit.
    """
    qcircuit = QCircuit()
    for level_index, level_cnot_pairs in enumerate(cnot_pairs):
        actual_level_index = level_index * 2
        for qubit_id1, qubit_id2 in level_cnot_pairs:
            add_random_two_qubit_gate(qcircuit,
                                      level_index=actual_level_index,
                                      qubit_id1=qubit_id1,
                                      qubit_id2=qubit_id2,
                                      seed=seed)
    return qcircuit

def node_id(qubit_index: int) -> str:
    """
    Generate a node ID based on the qubit index.

    Args:
        qubit_index (int): The index of the qubit.

    Returns:
        str: The generated node ID.
    """
    return f"q{qubit_index}"

def generate_cnot_pairs() -> list[list[tuple[str, str]]]:
    """
    Generate the list of CNOT gate pairs for the quantum circuit.

    Returns:
        list[list[tuple[str, str]]]: The generated CNOT gate pairs.
    """
    cnot_pairs = []
    # Level 0
    level = [(node_id(j+i), node_id(j+i+1))
             for i in range(0, 8, 2)
             for j in (0,9,18)]
    cnot_pairs.append(level)
    # Level 1
    level = [(node_id(i+j), node_id(i+j+2))
             for i in (0,1,4,5)
             for j in (0,9,18)]
    cnot_pairs.append(level)
    # Level 2
    level = [(node_id(i+j), node_id(i+j+3))
             for i in (0,4)
             for j in (0,9,18)]
    level.extend([(node_id(i+j), node_id(i+j+1))
                   for i in (1,5)
                   for j in (0,9,18)])
    cnot_pairs.append(level)
    # Level 3
    level = [(node_id(i), node_id(j))
             for i, j in [(3,8), (12,17), (21,26)]]
    cnot_pairs.append(level)
    # Level 4
    level = [(node_id(i), node_id(j))
             for i, j in [(7,8), (12,17), (25,26)]]
    cnot_pairs.append(level)
    # Level 5
    level = [(node_id(8), node_id(17))]
    cnot_pairs.append(level)
    # Level 6
    level = [(node_id(17), node_id(26))]
    cnot_pairs.append(level)
    # Level 7
    level = [(node_id(8), node_id(26))]
    cnot_pairs.append(level)
    return cnot_pairs

def build_simulation_circuit(num_repeats: int,
                             seed: int | None = None
                             ) -> QCircuit:
    """
    Build the quantum circuit for the simulation.

    Args:
        num_repeats (int): Number of times to repeat the circuit.
        seed (int | None): Seed for random number generation.
            Defaults to None.
        
    Returns:
        QCircuit: The constructed quantum circuit.
    """
    cnot_circuits = generate_cnot_pairs()
    circuit_parts: list[QCircuit] = []
    for _ in range(num_repeats):
        circuit_part = build_circuit(cnot_circuits, seed=seed)
        circuit_parts.append(circuit_part)
    inverted_parts = [part.invert()
                      for part in reversed(circuit_parts)]
    full_circuit = QCircuit()
    for part in circuit_parts + inverted_parts:
        full_circuit.add_qcircuit(part)
    return full_circuit

def generate_circuit_ttno(num_repeats: int,
                          ref_tree: TTNS,
                         seed: int | None = None,
                         ) -> list[TTNO]:
    """
    Generate the circuit TTNO for the simulation.

    Args:
        num_repeats (int): Number of times to repeat the circuit.
        ref_tree (TTNS): The reference TTNS structure.
        seed (int | None): Seed for random number generation.
            Defaults to None.

    Returns:
        list[TTNO]: The generated circuit TTNO as a list of TTNOs.
            One TTNO per level in the circuit.
    """
    qcircuit = build_simulation_circuit(num_repeats, seed=seed)
    return qcircuit.as_circuit_ttno(ref_tree)

def initial_state(ttn_structure: TTNStructure) -> TTNS:
    """
    Generate the initial zero state for the quantum circuit simulation.

    Args:
        ttn_structure (TTNStructure): The TTN structure for the state.
    
    Returns:
        TTNS: The generated zero state.
    """
    root_id = "root"
    if ttn_structure is TTNStructure.MPS:
        ttns = generate_zero_state(27, ttn_structure,
                                   node_prefix="q")
        return ttns
    if ttn_structure is TTNStructure.TSTAR:
        ttns = TTNS()
        root_tensor = trivial_virtual_node((1,1,1))
        ttns.add_root(Node(identifier=root_id), root_tensor)
        main_connectors = (8,17,26)
        for i, connector in enumerate(main_connectors):
            qubit_node = Node(identifier=node_id(connector))
            tensor = ket_i(0,2).reshape((1,1,2))
            ttns.add_child_to_parent(qubit_node, tensor, 0,
                                     root_id, i)
            # Add the virtual connector node
            v_node = Node(identifier=f"v{connector}")
            v_tensor = trivial_virtual_node((1,1,1))
            ttns.add_child_to_parent(v_node, v_tensor, 0,
                                     node_id(connector), 1)
            # Add the first sub_mps
            for j in range(i*9+3, i*9-1, -1):
                if j == i*9+3:
                    parent_id = f"v{connector}"
                else:
                    parent_id = node_id(j+1)
                child_id = node_id(j)
                if j == i*9:
                    shape = (1,2)
                else:
                    shape = (1,1,2)
                tensor = ket_i(0,2).reshape(shape)
                ttns.add_child_to_parent(Node(identifier=child_id),
                                         tensor, 0,
                                         parent_id, 1)
            # Add the second sub_mps
            for j in range((i+1)*9-2, i*9+3, -1):
                if j == (i+1)*9-2:
                    parent_id = f"v{connector}"
                    parent_leg = 2
                else:
                    parent_id = node_id(j+1)
                    parent_leg = 1
                child_id = node_id(j)
                if j == i*9+4:
                    shape = (1,2)
                else:
                    shape = (1,1,2)
                tensor = ket_i(0,2).reshape(shape)
                ttns.add_child_to_parent(Node(identifier=child_id),
                                         tensor, 0,
                                         parent_id, parent_leg)
        return ttns
    if ttn_structure is TTNStructure.BINARY:
        ttns = TTNS()
        root_tensor = trivial_virtual_node((1,1,1))
        ttns.add_root(Node(identifier=root_id), root_tensor)
        lqs = [((0,1,2,3),(4,5,6,7)),
               ((9,10,11,12),(13,14,15,16)),
               ((18,19,20,21),(22,23,24,25))]
        for i, upper_q, lower_qs in zip((0,1,2),(8,17,26),lqs):
            upper_v_id = f"v_{i}_0"
            upper_v_tensor = trivial_virtual_node((1,1,1))
            ttns.add_child_to_parent(Node(identifier=upper_v_id),
                                     upper_v_tensor, 0,
                                     root_id, i)
            q_node = Node(identifier=node_id(upper_q))
            q_tensor = ket_i(0,2).reshape((1,2))
            ttns.add_child_to_parent(q_node, q_tensor, 0,
                                     upper_v_id, 1)
            lower_v_id = f"v_{i}_1"
            lower_v_tensor = trivial_virtual_node((1,1,1))
            ttns.add_child_to_parent(Node(identifier=lower_v_id),
                                     lower_v_tensor, 0,
                                     upper_v_id, 2)
            for j in range(2):
                upper_min_v_id = f"v_{i}_{j}_0"
                upper_min_v_tensor = trivial_virtual_node((1,1,1))
                ttns.add_child_to_parent(Node(identifier=upper_min_v_id),
                                         upper_min_v_tensor, 0,
                                         lower_v_id, j+1)
                current_lqs = lower_qs[j]
                upper_q_id = node_id(current_lqs[-1])
                upper_q_tensor = ket_i(0,2).reshape((1,2))
                ttns.add_child_to_parent(Node(identifier=upper_q_id),
                                         upper_q_tensor, 0,
                                         upper_min_v_id, 1)
                lower_min_v_id = f"v_{i}_{j}_1"
                lower_min_v_tensor = trivial_virtual_node((1,1,1,1))
                ttns.add_child_to_parent(Node(identifier=lower_min_v_id),
                                         lower_min_v_tensor, 0,
                                         upper_min_v_id, 2)
                for k in range(len(current_lqs)-1):
                    tensor = ket_i(0,2).reshape((1,2))
                    qubit_id = node_id(current_lqs[k])
                    ttns.add_child_to_parent(Node(identifier=qubit_id),
                                             tensor, 0,
                                             lower_min_v_id, k+1)
        return ttns
    raise ValueError("Unsupported TTN structure!")

def bond_dim_range(params: CircuitSimParams) -> range:
    """
    Generates a range of bond dimensions to be used in the simulation.

    Args:
        params (CircuitSimParams): The parameters for the simulation.

    Returns:
        range: A range of bond dimensions from min to max with the specified
            step.
    """
    return range(params.min_bond_dim,
                 params.max_bond_dim + 1,
                 params.bond_dim_step)


RES_IDS = ("bond_dim", "trunc_error", "run_time", "max_size")

def init_results(params: CircuitSimParams) -> Results:
    """
    Initializes the results object for the simulation.

    Args:
        params (CircuitSimParams): The parameters for the simulation.

    Returns:
        Results: The initialized and empty results object.
    """
    num_res = len(list(bond_dim_range(params))) - 1
    results = Results()
    res_dtypes = (int, float, float, float)
    results.initialize(dict(zip(RES_IDS, res_dtypes)),
                       num_res,
                       with_time=False)
    return results

def run_simulation(params: CircuitSimParams
                   ) -> Results:
    """
    Run the quantum circuit simulation experiment.

    Args:
        params (CircuitSimParams): The parameters for the simulation.

    Returns:
        Results: The results of the simulation.
    """
    init_state = initial_state(params.ttn_structure)
    circuit_ttno = generate_circuit_ttno(params.num_circuit_repeats,
                                         init_state,
                                         seed=params.seed)
    results = init_results(params)
    for idx, bond_dim in enumerate(bond_dim_range(params)):
        max_size = float("-inf")
        ttns = deepcopy(init_state)
        svd_params = SVDParameters(max_bond_dim=bond_dim)
        if params.appl_method == ApplicationMethod.VARIATIONAL:
            kwargs = {"num_sweeps": 2,
                      "svd_params": svd_params}
        elif params.appl_method == ApplicationMethod.ZIPUP_VARIATIONAL:
            kwargs = {"num_sweeps": 2,
                      "var_svd_params": svd_params,
                      "dm_svd_params": svd_params}
        elif params.appl_method == ApplicationMethod.HALF_DENSITY_MATRIX_VARIATIONAL:
            kwargs = {"num_sweeps": 2,
                      "var_svd_params": svd_params,
                      "dm_svd_params": svd_params}
        elif params.appl_method == ApplicationMethod.SRC:
            kwargs = {"desired_dimension": bond_dim}
        else:
            kwargs = {"svd_params": svd_params}
        start = time()
        for ttno in circuit_ttno:
            apply_ttno_to_ttns(ttns, ttno,
                               method=params.appl_method,
                               **kwargs)
            max_size = max(max_size, ttns.size())
        end = time()
        error = init_state.distance(ttns, normalise=True)
        res_values = (bond_dim, error, end - start, max_size)
        for res_id, res_value in zip(RES_IDS, res_values):
            results.set_element(res_id, idx, res_value)
    return results

def save_results(params: CircuitSimParams,
                 results: Results,
                 save_directory: str
                 ) -> None:
    """
    Saves the simulation results to an HDF5 file.

    Args:
        params (CircuitSimParams): The parameters for the simulation.
        results (Results): The results of the simulation.
        save_directory (str): The directory to save the results in.
    """
    filename = params.get_hash() + ".h5"
    filepath = os.path.join(save_directory, filename)
    with File(filepath, "w") as f:
        results.save_to_h5(f)
        params.save_to_h5(f)

def run_and_save(params: CircuitSimParams,
                   save_directory: str
                   ):
    """
    Run the simulation and save the results.

    Args:
        params (CircuitSimParams): The parameters for the simulation.
        save_directory (str): The directory to save the results in.
    """
    results = run_simulation(params)
    save_results(params, results, save_directory)

if __name__ == "__main__":
    script_main(run_and_save,
                CircuitSimParams)
