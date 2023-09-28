from ..ttn import QuantumTTState, QuantumTTOperator
from ..tensornode import QuantumStateNode, QuantumOperatorNode
from ..time_evolution import TDVP
from ..util import state_vector_time_evolution

from .gates import *

from copy import deepcopy


class Circuit:
    def __init__(self, name, state, qubit_ids, print_actions=False, measurement_types=["Z"]):
        self.name = name
        self._qubit_ids = qubit_ids
        self.n_qubits = len(qubit_ids)
        self._state = state
        self._time_elapsed = 0
        self.results = None
        self.print_actions = print_actions
        self.measurement_types = measurement_types

        self.progress_bar = False

        self.results_cache = []
        self._hamiltonian = None

    @property
    def size(self):
        return self.n_qubits
    
    @property
    def state(self):
        return self._state
    
    @property
    def state_dict(self):
        if len(self._qubit_ids) == len(list(self.state.nodes.keys())):  # i.e. if pure state
            tensor = self._state[self._qubit_ids[0]].tensor
            for i, id in enumerate(self._qubit_ids):
                if i>0:
                    tensor = np.tensordot(tensor, self._state[id].tensor, axes=(i-1, 0))
            result = tensor.flatten()
            self.results_cache.append(result)
            
            result_dict = dict()
            for i in range(2**self.size):
                if np.abs(result[i]) > 1e-3:
                    string = bin(i)[2:]
                    string = "|" + "0" * (self.size - len(string)) + string + f">=|{i}>"
                    result_dict[string] = np.round(result[i], 3)
            return result_dict
        else:
            rho_qubits = rho(self.state, self._qubit_ids)
            return {"tr(rho^2)": np.round(np.trace(rho_qubits @ rho_qubits).real, 8)}
            
    
    def operators(self, gate_ids):
        all_operators = []
        for node_id in self._qubit_ids:
            for gate_id in gate_ids:
                all_operators.append({node_id: all_gates()[gate_id].U})
        return all_operators
    
    def gate(self, node_id, gate_id, operation_time=1):
        assert node_id in self._qubit_ids
        gate = all_gates()[gate_id]
        hamiltonian = self.hamiltonian(targets=[node_id], replacements=[gate.H / operation_time])

        if self.print_actions:
            print(f"\n{self.name}: Apply Gate {gate_id} on {node_id}")
        self.run(hamiltonian, operation_time)
        if self.print_actions:
            print(self.state_dict)
    
    def cnot(self, control_id, not_id, operation_time=1, supress_output=False):
        self.cu(control_id=control_id, u_id=not_id, gate_id="X", operation_time=operation_time, supress_output=supress_output)

    def cu(self, control_id, u_id, gate_id, operation_time=1, supress_output=False):
        self.cu_multi((control_id,), u_id, gate_id, operation_time, supress_output)
    
    def cu_multi(self, control_ids, u_id, gate_id, operation_time=1, supress_output=False, flip_control=None):
        assert False not in [id in self._qubit_ids for id in control_ids]
        assert u_id in self._qubit_ids

        I = all_gates()["I"]
        Z = all_gates()["Z"]
        gate = all_gates()[gate_id]

        c_mat = (I.U - Z.U) / 2
        c_mat_flipped = (I.U + Z.U) / 2
        u_mat = (I.U - gate.U) * np.pi / 2 / operation_time

        replacements = []
        for id in self._qubit_ids:
            if id in control_ids:
                if flip_control is not None:
                    if flip_control[id]:
                        replacements.append(c_mat_flipped)
                    else:
                        replacements.append(c_mat)
                else:
                    replacements.append(c_mat)
            elif id == u_id:
                replacements.append(u_mat)
            else:
                replacements.append(I.U)

        hamiltonian = self.hamiltonian(targets=self._qubit_ids, replacements=replacements)

        self.run(hamiltonian, operation_time)
        if not supress_output and self.print_actions:
            print(f"\n{self.name}: Apply Gate controlled {gate_id} on {control_ids} and {u_id}")
            print(self.state_dict)
    
    def swap(self, id_1, id_2, operation_time=1):
        self.cnot(id_1, id_2, operation_time, supress_output=True)
        self.cnot(id_2, id_1, operation_time, supress_output=True)
        self.cnot(id_1, id_2, operation_time, supress_output=True)
        if self.print_actions:
            print(f"\n{self.name}: Apply Gate SWAP on {id_1} and {id_2}")
            print(self.state_dict)

    def cswap(self, control_id, swap_id_1, swap_id_2, operation_time=1):
        # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.53.2855
        self.cu_multi((swap_id_2,), swap_id_1, "X", operation_time/10)
        self.cu_multi((control_id, swap_id_1), swap_id_2, "X", 8*operation_time/10)
        self.cu_multi((swap_id_2,), swap_id_1, "X", operation_time/10)
    
    def run(self, hamiltonian, operation_time):
        tdvp_sim = TDVP("SecondOrder,OneSite", self._state, hamiltonian, time_step_size=operation_time/20, final_time=self._time_elapsed+operation_time, 
                        initial_time=self._time_elapsed, operators=self.operators(self.measurement_types))
        tdvp_sim.contraction_mode = "by_site"  # TODO fyi this is still here
        tdvp_sim.run(pgbar=self.progress_bar)
        self._time_elapsed += operation_time

        if self.results is None:
            self.results = tdvp_sim.results
        else:
            self.results = np.hstack((self.results, tdvp_sim.results[:,1:]))
    
    def provide_hamiltonian(self, hamiltonian, ss_dict, ts_dict, operation_type="add"):
        self._hamiltonian = hamiltonian
        self._ss_dict = ss_dict
        self._ts_dict = ts_dict
        assert operation_type in ["add", "replace"]
        self._hamiltonian_operation_type = operation_type

    def remove_hamiltonian(self):
        self._hamiltonian = None


    def hamiltonian(self, targets, replacements):
        if self._hamiltonian is None:
            hamiltonian = trivial_hamiltonian(self._state, bond_dimension=1, targets=targets)
            for i, id in enumerate(targets):
                if replacements[i] is not None:
                    hamiltonian[id].add(replacements[i])
        else:
            hamiltonian = deepcopy(self._hamiltonian)
            if len(targets)==1:
                use_dict = self._ss_dict
            else:
                use_dict = self._ts_dict
            for i, id in enumerate(targets):
                if id in use_dict.keys():
                    for index in use_dict[id]:
                        if self._hamiltonian_operation_type == "replace":
                            hamiltonian.nodes[id].tensor[index] = np.zeros((2,2), dtype=complex)
                        if replacements[i] is not None:
                            hamiltonian.nodes[id].tensor[index] = hamiltonian.nodes[id].tensor[index] + replacements[i]  # += doesnt work because of trivial dimensions (e.g. shape 2, 2, 1 + shape 2, 2)
        return hamiltonian
    
    def fidelity(self, ref_state, mode="trace_ratio", limited_qubits=None):
        qubit_list = self._qubit_ids
        if limited_qubits is not None:
            qubit_list = limited_qubits

        rho_ref = rho(ref_state, qubit_list)
        rho_circ = rho(self.state, qubit_list)

        if mode=="trace_ratio":
            return np.real(np.trace(rho_circ @ rho_ref) / np.trace(rho_ref @ rho_ref))
        elif mode=="purities":
            return np.trace(rho_circ @rho_circ).real, np.trace(rho_ref @ rho_ref).real
        else:
            return rho_circ, rho_ref
    
    
# TODO find better position for this code, this doesn't really fit here

def rho(state, qubit_list):
    return state.reduced_density_matrix(qubit_list)


def trivial_hamiltonian(state, bond_dimension=1, targets=[]):
    hamiltonian = QuantumTTOperator()
    root_id = state.root_id

    root_tensor = np.zeros([bond_dimension] * (state[root_id].tensor.ndim - 1) + [state[root_id].tensor.shape[-1]] * 2, dtype=complex)
    if root_id not in targets:
        index = tuple([0] * (state[root_id].tensor.ndim - 1))
        root_tensor[index] += np.eye(root_tensor.shape[-1], dtype=complex)

    hamiltonian.add_root(QuantumOperatorNode(root_tensor, identifier=root_id))
    hamiltonian = add_children(hamiltonian, state, root_id, bond_dimension, targets)
    return hamiltonian


def add_children(hamiltonian, state, current_node_id, bond_dimension, targets=[]):
    for node_id in state[current_node_id].children_legs.keys():

        tensor = np.zeros([bond_dimension] * (state[node_id].tensor.ndim - 1) + [state[node_id].tensor.shape[-1]] * 2, dtype=complex)
        if node_id not in targets:
            index = tuple([0] * (state[node_id].tensor.ndim - 1))
            tensor[index] += np.eye(tensor.shape[-1], dtype=complex)

        parent_leg = state[node_id].parent_leg[1]
        own_leg_in_parent = state[current_node_id].children_legs[node_id]
        
        node = QuantumOperatorNode(tensor, identifier=node_id)
        hamiltonian.add_child_to_parent(node, parent_leg, current_node_id, own_leg_in_parent)
        hamiltonian = add_children(hamiltonian, state, node_id, bond_dimension, targets)
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