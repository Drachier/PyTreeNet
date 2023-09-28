import sys
sys.path.append('.')

import pytreenet as ptn
import numpy as np
from scipy.linalg import expm
from copy import deepcopy


state = ptn.models.mps_zero(num_sites=5,
                            virtual_bond_dimension=6,
                            name_prefix="qubit_")
ref_state = deepcopy(state)

circ = ptn.quantum_circuits.circuit.Circuit(name="Open Circuit",
                                             state=state, 
                                             qubit_ids=["qubit_0", "qubit_1", "qubit_2", "qubit_3", "qubit_4"], 
                                             print_actions=True, 
                                             measurement_types=["Z"])
circ.progress_bar = False
circ.print_actions = False


hamiltonian = ptn.models.mps_heisenberg(num_sites=5,
                                        jx=.0009,
                                        h=.009,
                                        name_prefix="qubit_",
                                        add_extra_qubit_dim=True)

ss_dict = dict()
ss_dict["qubit_0"] = ((0,),)
ss_dict["qubit_1"] = ((3,0,),)
ss_dict["qubit_2"] = ((3,0,),)
ss_dict["qubit_3"] = ((3,0,),)
ss_dict["qubit_4"] = ((3,),)

ts_dict = dict()
ts_dict["qubit_0"] = ((2,),)
ts_dict["qubit_1"] = ((2, 2,),)
ts_dict["qubit_2"] = ((2, 2,),)
ts_dict["qubit_3"] = ((2, 2,),)
ts_dict["qubit_4"] = ((2,),)


def encode(circuit):
    # H  H  I  H  I
    circuit.gate("qubit_0", "H", operation_time=.1)
    circuit.gate("qubit_1", "H", operation_time=.1)
    circuit.gate("qubit_3", "H", operation_time=.1)

    # I  C  C  C  Z
    circuit.cu_multi(("qubit_1", "qubit_2", "qubit_3"), "qubit_4", "Z", operation_time=1)

    # I  c  C  c  Z
    circuit.cu_multi(("qubit_1", "qubit_2", "qubit_3"), "qubit_4", "Z", operation_time=1, flip_control={"qubit_1": True, "qubit_2": False, "qubit_3": True})

    # I  I  C  I  X
    circuit.cu("qubit_2", "qubit_4", "X", operation_time=1)

    # C  I  X  I  X
    circuit.cnot("qubit_0", "qubit_4", operation_time=1)
    circuit.cnot("qubit_0", "qubit_2", operation_time=1)

    # I  I  X  C  I
    circuit.cu("qubit_3", "qubit_2", "X", operation_time=1)

    # I  C  I  I  X
    circuit.cu("qubit_1", "qubit_4", "X", operation_time=1)

    # I  I  Z  C  C
    circuit.cu_multi(("qubit_3", "qubit_4"), "qubit_2", "Z", operation_time=1)

    return circuit


def decode(circuit):
    # I  I  Z  C  C
    circuit.cu_multi(("qubit_3", "qubit_4"), "qubit_2", "Z", operation_time=1)

    # I  C  I  I  X
    circuit.cu("qubit_1", "qubit_4", "X", operation_time=1)

    # I  I  X  C  I
    circuit.cu("qubit_3", "qubit_2", "X", operation_time=1)

    # C  I  X  I  X   ????????? what is this gate ???????????
    circuit.cnot("qubit_0", "qubit_4", operation_time=1)
    circuit.cnot("qubit_0", "qubit_2", operation_time=1)

    # I  I  C  I  X
    circuit.cu("qubit_2", "qubit_4", "X", operation_time=1)

    # I  c  C  c  Z
    circuit.cu_multi(("qubit_1", "qubit_2", "qubit_3"), "qubit_4", "Z", operation_time=1, flip_control={"qubit_1": True, "qubit_2": False, "qubit_3": True})

    # I  C  C  C  Z
    circuit.cu_multi(("qubit_1", "qubit_2", "qubit_3"), "qubit_4", "Z", operation_time=1)

    circuit.gate("qubit_0", "H", operation_time=.1)
    circuit.gate("qubit_1", "H", operation_time=.1)
    circuit.gate("qubit_3", "H", operation_time=.1)

    return circuit


circ = encode(circ)

print(circ.state_dict)

circ.provide_hamiltonian(hamiltonian, ss_dict, ts_dict, operation_type="add")
circ.gate("qubit_2", "I", operation_time=10.)
circ.remove_hamiltonian()
circ = decode(circ)   


print("\n", circ.fidelity(ref_state, mode="trace_ratio", limited_qubits=("qubit_2",)))



