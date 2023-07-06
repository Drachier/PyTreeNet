from ..ttn import QuantumTTState, QuantumTTOperator
from ..tensornode import QuantumStateNode, QuantumOperatorNode
from ..time_evolution import TDVP
from ..util import state_vector_time_evolution

from .gates import *

from copy import deepcopy


class Circuit:
    def __init__(self, state, qubit_ids, print_actions=False, measurement_types=["Z"]):
        self._qubit_ids = qubit_ids
        self.n_qubits = len(qubit_ids)
        self._state = state
        self._time_elapsed = 0
        self.results = None
        self.print_actions = print_actions
        self.measurement_types = measurement_types

    @property
    def size(self):
        return self.n_qubits
    
    @property
    def state(self):
        return self._state
    
    @property
    def state_dict(self):
        tensor = self._state[self._qubit_ids[0]].tensor
        for i, id in enumerate(self._qubit_ids):
            if i>0:
                tensor = np.tensordot(tensor, self._state[id].tensor, axes=(i-1, 0))
        result = tensor.flatten()
        result_dict = dict()
        for i in range(2**self.size):
            if np.abs(result[i]) > 1e-10:
                string = bin(i)[2:]
                string = "|" + "0" * (self.size - len(string)) + string + ">"
                result_dict[string] = np.round(result[i], 3)
        return result_dict
            
    
    def operators(self, gate_ids):
        all_operators = []
        for node_id in self._qubit_ids:
            for gate_id in gate_ids:
                all_operators.append({node_id: all_gates()[gate_id].U})
        return all_operators
    
    def gate(self, node_id, gate_id, operation_time=1):
        assert node_id in self._qubit_ids
        gate = all_gates()[gate_id]
        hamiltonian = trivial_hamiltonian(self._state, bond_dimension=1,  zeros=[node_id])
        hamiltonian[node_id].add(gate.H / operation_time)

        if self.print_actions:
            print("\nApply Gate " + gate_id + " on " + node_id)
        tdvp_sim = self.run(hamiltonian, operation_time)
        if self.print_actions:
            print(self.state_dict)
        return tdvp_sim
    
    def cnot(self, control_id, not_id, operation_time=1):
        assert control_id in self._qubit_ids
        assert not_id in self._qubit_ids

        I = all_gates()["I"]
        X = all_gates()["X"]
        Z = all_gates()["Z"]

        hamiltonian = trivial_hamiltonian(self._state, bond_dimension=1, zeros=[control_id, not_id])
        control_mat = (I.U - Z.U)
        not_mat = (I.U - X.U) * np.pi / 4 / operation_time

        hamiltonian[control_id].add(control_mat)
        hamiltonian[not_id].add(not_mat)

        if self.print_actions:
            print("\nApply Gate CNOT on " + control_id + " and " + not_id)
        tdvp_sim = self.run(hamiltonian, operation_time)
        if self.print_actions:
            print(self.state_dict)
        return tdvp_sim
    
    def swap(self, id_1, id_2, operation_time=1):
        self.cnot(id_1, id_2, operation_time)
        self.cnot(id_2, id_1, operation_time)
        self.cnot(id_1, id_2, operation_time)
    
    def run(self, hamiltonian, operation_time):
        tdvp_sim = TDVP("SecondOrder,OneSite", self._state, hamiltonian, time_step_size=operation_time/20, final_time=self._time_elapsed+operation_time, 
                        initial_time=self._time_elapsed, operators=self.operators(self.measurement_types))
        tdvp_sim.run(pgbar=False)
        self._time_elapsed += operation_time

        if self.results is None:
            self.results = tdvp_sim.results
        else:
            self.results = np.hstack((self.results, tdvp_sim.results[:,1:]))
        return tdvp_sim
    
    
# TODO find better position for this code, this doesn't really fit here

def trivial_hamiltonian(state, bond_dimension=1, zeros=[]):
    hamiltonian = QuantumTTOperator()
    root_id = state.root_id

    root_tensor = np.zeros([bond_dimension] * (state[root_id].tensor.ndim - 1) + [state[root_id].tensor.shape[-1]] * 2, dtype=complex)
    if root_id not in zeros:
        index = tuple([0] * (state[root_id].tensor.ndim - 1))
        root_tensor[index] += np.eye(root_tensor.shape[-1], dtype=complex)

    hamiltonian.add_root(QuantumOperatorNode(root_tensor, identifier=root_id))
    hamiltonian = add_children(hamiltonian, state, root_id, bond_dimension, zeros)
    return hamiltonian


def add_children(hamiltonian, state, current_node_id, bond_dimension, zeros=[]):
    for node_id in state[current_node_id].children_legs.keys():

        tensor = np.zeros([bond_dimension] * (state[node_id].tensor.ndim - 1) + [state[node_id].tensor.shape[-1]] * 2, dtype=complex)
        if node_id not in zeros:
            index = tuple([0] * (state[node_id].tensor.ndim - 1))
            tensor[index] += np.eye(tensor.shape[-1], dtype=complex)

        parent_leg = state[node_id].parent_leg[1]
        own_leg_in_parent = state[current_node_id].children_legs[node_id]
        
        node = QuantumOperatorNode(tensor, identifier=node_id)
        hamiltonian.add_child_to_parent(node, parent_leg, current_node_id, own_leg_in_parent)
        hamiltonian = add_children(hamiltonian, state, node_id, bond_dimension, zeros)
    return hamiltonian    


def rename(state, id_1, id_2):
    all_nodes = deepcopy(list(state.nodes.keys()))
    for node_id in all_nodes:
        if node_id == id_1:
            state.nodes[id_2] = state.nodes.pop(node_id)
        else:
            children_keys = deepcopy(list(state[node_id].children_legs.keys()))
            if id_1 in children_keys:
                state[node_id].children_legs[id_2] = state[node_id].children_legs.pop(id_1)
            parent_leg = deepcopy(state[node_id].parent_leg)
            if len(parent_leg)>0 and parent_leg[0]==id_1:
                state[node_id].parent_leg[0] = id_2
    return state