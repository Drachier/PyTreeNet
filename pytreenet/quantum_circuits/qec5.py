import numpy as np
from ..base import ttn, tensornode
from ..utils.circuit_utils import *
from .circuit import Circuit


ENCODING_STEPS = 9
def _encoder(circuit: Circuit, num, t):
    if num == 1:
        # H  H  I  H  I
        circuit.gate("qubit_1", "H", operation_time=1/3*t/9)
        circuit.gate("qubit_2", "H", operation_time=1/3*t/9)
        circuit.gate("qubit_4", "H", operation_time=1/3*t/9)
    elif num == 2:
        # I  C  C  C  Z
        circuit.cu_multi(("qubit_2", "qubit_3", "qubit_4"), "qubit_5", "-I", operation_time=t/9)
    elif num == 3:
        # I  c  C  c  Z
        circuit.cu_multi(("qubit_2", "qubit_3", "qubit_4"), "qubit_5", "-I", operation_time=t/9, flip_control={"qubit_2": True, "qubit_3": False, "qubit_4": True})
    elif num == 4:
        # I  I  C  I  X
        circuit.cu("qubit_3", "qubit_5", "X", operation_time=t/9)
    elif num == 5:
        # C  I  I  I  X
        circuit.cnot("qubit_1", "qubit_5", operation_time=t/9)
    elif num == 6:
        # C  I  X  I  I
        circuit.cnot("qubit_1", "qubit_3", operation_time=t/9)
    elif num == 7:
        # I  I  X  C  I
        circuit.cu("qubit_4", "qubit_3", "X", operation_time=t/9)
    elif num == 8:
        # I  C  I  I  X
        circuit.cu("qubit_2", "qubit_5", "X", operation_time=t/9)
    elif num == 9:
        # I  I  Z  C  C
        circuit.cu_multi(("qubit_4", "qubit_5"), "qubit_3", "-I", operation_time=t/9)
    else:
        raise ValueError
    
def calc_num_bosons(circuit):
    # First qubit must be connected to only one other qubit and has a physical leg.
    return circuit.state[circuit._qubit_ids[0]].tensor.ndim-2

def repair(circuit):
    num_bosons = calc_num_bosons(circuit)

    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 1], [0, 0]], dtype=complex)

    myR_1 = np.array([P0, P1])

    myR_2 = np.array([[P0, P1, O, O],
                      [O, O, P0, P1]]) 

    myR_3 = np.array([[I, Z, I, I],
                      [I, Z, X, X],
                      [I, X, Z, X],
                      [Z, X@Z, X, Z]])

    myR_4 = np.array([[P0, O],
                      [O, P0],
                      [P1, O],
                      [O, P1]])

    myR_5 = np.array([P0, P1])

    tto = _qubit_tto(circuit, (myR_1, myR_2, myR_3, myR_4, myR_5))

    circuit.apply_tto(tto, truncate=True)
    circuit.state.normalize()
    return circuit
    

def _qubit_tto(circuit, operator_list_pre):
    t1, t2, t3, t4, t5 = operator_list_pre
    operator_list = []
    for i, t in enumerate(operator_list_pre):
        if i==0 or i==4:
            num_connected_qubits = 1
        else:
            num_connected_qubits = 2
        t = _add_boson_legs(circuit, t, num_connected_qubits)
        operator_list.append(t)
    t1, t2, t3, t4, t5 = operator_list
    
    tto = ttn.QuantumTTOperator()
    tto.add_root(tensornode.QuantumOperatorNode(t1, identifier="qubit_1"))
    tto.add_child_to_parent(tensornode.QuantumOperatorNode(t2, identifier="qubit_2"), 0, "qubit_1", 0)
    tto.add_child_to_parent(tensornode.QuantumOperatorNode(t3, identifier="qubit_3"), 0, "qubit_2", 1)
    tto.add_child_to_parent(tensornode.QuantumOperatorNode(t4, identifier="qubit_4"), 0, "qubit_3", 1)
    tto.add_child_to_parent(tensornode.QuantumOperatorNode(t5, identifier="qubit_5"), 0, "qubit_4", 1)
    return tto


def _add_boson_legs(circuit, tensor, start_index):
    num_bosons = calc_num_bosons(circuit)
    tensor = np.reshape(tensor, list(tensor.shape)+[1]*num_bosons)
    legs = list(range(tensor.ndim))

    # Move phsyical indices behind boson indices
    legs.append(legs.pop(start_index)), legs.append(legs.pop(start_index))
    tensor = tensor.transpose(legs) 

    return tensor


def codeword_tto(circuit, num_bosons=2, reverse=False):
    tto_list = []
    one_outer = np.outer(one, one)
    zero_outer = np.outer(zero, zero)

    t1 = np.array([H])
    t2 = np.array([[H]])
    t3 = np.array([[I]])
    t4 = np.array([[H]])
    t5 = np.array([I])
    tto_list.append(_qubit_tto(circuit, (t1, t2, t3, t4, t5)))

    t1 = np.array([I])
    t2 = np.array([[I, one_outer]])
    t3 = np.array([[I, O],
                   [O, one_outer]])
    t4 = np.array([[I, O],
                   [O, one_outer]])
    t5 = np.array([I, -I-I])
    tto_list.append(_qubit_tto(circuit, (t1, t2, t3, t4, t5)))

    t1 = np.array([I])
    t2 = np.array([[I, zero_outer]])
    t3 = np.array([[I, O],
                   [O, one_outer]])
    t4 = np.array([[I, O],
                   [O, zero_outer]])
    t5 = np.array([I, -I-I])
    tto_list.append(_qubit_tto(circuit, (t1, t2, t3, t4, t5)))

    t1 = np.array([I])
    t2 = np.array([[I]])
    t3 = np.array([[I, one_outer]])
    t4 = np.array([[I, O],
                   [O, I]])
    t5 = np.array([I, X-I])
    tto_list.append(_qubit_tto(circuit, (t1, t2, t3, t4, t5)))

    t1 = np.array([I, one_outer])
    t2 = np.array([[I, O],
                   [O, I]])
    t3 = np.array([[I],
                   [X-I]])
    t4 = np.array([[I]])
    t5 = np.array([I])
    tto_list.append(_qubit_tto(circuit, (t1, t2, t3, t4, t5)))

    t1 = np.array([I, one_outer])
    t2 = np.array([[I, O],
                   [O, I]])
    t3 = np.array([[I, O],
                   [O, I]])
    t4 = np.array([[I, O],
                   [O, I]])
    t5 = np.array([I, X-I])
    tto_list.append(_qubit_tto(circuit, (t1, t2, t3, t4, t5)))

    t1 = np.array([I])
    t2 = np.array([[I]])
    t3 = np.array([[I, X-I]])
    t4 = np.array([[I],
                   [one_outer]])
    t5 = np.array([I])
    tto_list.append(_qubit_tto(circuit, (t1, t2, t3, t4, t5)))

    t1 = np.array([I])
    t2 = np.array([[I, one_outer]])
    t3 = np.array([[I, O],
                   [O, I]])
    t4 = np.array([[I, O],
                   [O, I]])
    t5 = np.array([I, X-I])
    tto_list.append(_qubit_tto(circuit, (t1, t2, t3, t4, t5)))

    t1 = np.array([I])
    t2 = np.array([[I]])
    t3 = np.array([[I, -I-I]])
    t4 = np.array([[I, O],
                   [O, one_outer]])
    t5 = np.array([I, one_outer])
    tto_list.append(_qubit_tto(circuit, (t1, t2, t3, t4, t5)))

    if reverse:
        tto_list = tto_list[::-1]
    for tto in tto_list:
        circuit.apply_tto(tto)
        circuit.state.normalize()

    return circuit


def encode(circuit, t):
    if t>0:
        for num in range(1, ENCODING_STEPS+1):
            print(f"Encoding {num}/{ENCODING_STEPS}", end="\r")
            _encoder(circuit, num, t)
    else:
        circuit = codeword_tto(circuit)
    return circuit


def decode(circuit, t):
    if t>0:
        for num in range(1, ENCODING_STEPS+1):
            print(f"Decoding {num}/{ENCODING_STEPS}", end="\r")
            _encoder(circuit, ENCODING_STEPS-num+1, t)
    else:
        circuit = codeword_tto(circuit, reverse=True)
    return circuit