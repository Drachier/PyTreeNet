import numpy as np
from ..base.ttn import QuantumTTState, QuantumTTOperator
from ..base.tensornode import QuantumStateNode, QuantumOperatorNode
from ..utils.util import pauli_matrices, zero_state, create_bosonic_operators


X, Y, Z = pauli_matrices()
O = np.zeros((2, 2), dtype=complex)
I = np.identity(2, dtype=complex)


def bosonic_tree_operator(num_spins, num_bosons_per_spin, boson_dimension, J, g, momega, add_extra_qubit_dim=False, spin_name_prefix="qubit_", indexing=0):
    creation_op, annihilation_op, number_op = create_bosonic_operators(dimension=boson_dimension)
    I_b = np.eye(boson_dimension, dtype=complex)
    B = np.zeros((3, boson_dimension, boson_dimension), dtype=complex)
    B[0,:,:] = I_b
    B[1,:,:] = momega/2 * number_op
    B[2,:,:] = annihilation_op + creation_op
    
    if add_extra_qubit_dim:
        # Spin-Spin interaction
        ss = np.array([[I, O, O, O],
                    [J*X, O, O, O],
                    [O, O, O, O],
                    [O, X, O, I]])        
        # Boson without interaction
        bo = np.array([[I, O, O, O],
                    [O, O, O, O],
                    [O, O, O, O],
                    [I, O, O, I]])
        # Spin-Boson interaction
        sb = np.array([[I, O, O, O],
                    [O, O, O, O],
                    [O, O, O, O],
                    [g*Z, O, O, O]])
        
        s_shape = [4, 4, 2, 2] + [3] * num_bosons_per_spin
    else:
        # Spin-Spin interaction
        ss = np.array([[I, O, O],
                    [J*X, O, O],
                    [O, X, I]])
        # Boson without interaction
        bo = np.array([[O, O, O],
                    [O, O, O],
                    [I, O, O]])
        # Spin-Boson interaction
        sb = np.array([[O, O, O],
                    [O, O, O],
                    [g*Z, O, O]])
        
        s_shape = [3, 3, 2, 2] + [3] * num_bosons_per_spin

    S = np.zeros(s_shape, dtype=complex)

    index = [0] * num_bosons_per_spin
    idx = tuple([Ellipsis] + index)
    S[idx] = ss
    for b in range(num_bosons_per_spin):
        index = [0] * num_bosons_per_spin
        index[b] = 1
        idx = tuple([Ellipsis] + index)
        S[idx] = bo
        index[b] = 2
        idx = tuple([Ellipsis] + index)
        S[idx] = sb

    s_trans = list(range(S.ndim))
    s_trans.append(s_trans.pop(2)), s_trans.append(s_trans.pop(2))
    S = S.transpose(s_trans)

    ham = QuantumTTOperator()

    for site in range(num_spins):
        identifier = spin_name_prefix + str(site + indexing)
        if site == 0:
            # Open boundary conditions on first site
            node_ham = QuantumOperatorNode(1*S[-1,:], identifier=identifier)
            ham.add_root(node_ham)
        else:
            if site == num_spins - 1:
                # Open boundary conditions on last site
                node_ham = QuantumOperatorNode(1*S[:,0], identifier=identifier)
            else:
                node_ham = QuantumOperatorNode(1*S, identifier=identifier)
            parent_id = spin_name_prefix + str(site - 1 + indexing)
            if site == 1:
                # Due to boundary condition on first site
                ham.add_child_to_parent(node_ham, 0, parent_id, 0)
            else:
                ham.add_child_to_parent(node_ham, 0, parent_id, 1)
    
    for site in range(num_spins):
        for boson in range(num_bosons_per_spin):
            identifier = "boson_" + str(site + indexing) + "." + str(boson)
            parent_id = spin_name_prefix + str(site + indexing)

            node = ham[parent_id]
            first_free_leg = int(len(node.parent_leg)!=0) + len(node.children_legs)

            boson_node = QuantumOperatorNode(1*B, identifier=identifier)
            ham.add_child_to_parent(boson_node, 0, parent_id, first_free_leg)

    return ham


def bosonic_tree_state(num_spins, virtual_spin_spin_bond_dimension, num_bosons_per_spin, boson_dimension, virtual_spin_boson_bond_dimension, spin_name_prefix="qubit_", indexing=0):    
    s_shape = tuple([virtual_spin_spin_bond_dimension] * 2 + [virtual_spin_boson_bond_dimension] * num_bosons_per_spin + [2])
    S = np.zeros(s_shape, dtype=complex).flatten()
    S[0] = 1
    S = S.reshape(s_shape)
    
    b_shape = tuple([virtual_spin_boson_bond_dimension] + [boson_dimension])
    B = np.zeros(b_shape, dtype=complex).flatten()
    B[0] = 1
    B = B.reshape(b_shape)

    psi = QuantumTTState()

    for site in range(num_spins):
        identifier = spin_name_prefix + str(site + indexing)
        if site == 0:
            # Open boundary conditions on first site
            node_ham = QuantumStateNode(1*S[0,:], identifier=identifier)
            psi.add_root(node_ham)
        elif 0 < site:
            if site == num_spins - 1:
                # Open boundary conditions on last site
                node_ham = QuantumStateNode(1*S[:,0], identifier=identifier)
            else:
                node_ham = QuantumStateNode(1*S, identifier=identifier)
            parent_id = spin_name_prefix + str(site - 1 + indexing)
            if site == 1:
                # Due to boundary condition on first site
                psi.add_child_to_parent(node_ham, 0, parent_id, 0)
            else:
                psi.add_child_to_parent(node_ham, 0, parent_id, 1)
    
    for site in range(num_spins):
        for boson in range(num_bosons_per_spin):
            identifier = "boson_" + str(site + indexing) + "." + str(boson)
            parent_id = spin_name_prefix + str(site + indexing)

            node = psi[parent_id]
            first_free_leg = int(len(node.parent_leg)!=0) + len(node.children_legs)

            boson_node = QuantumStateNode(1*B, identifier=identifier)
            psi.add_child_to_parent(boson_node, 0, parent_id, first_free_leg)

    return psi


def bosonic_tree_state_for_5qec(num_spins, virtual_spin_spin_bond_dimension, num_bosons_per_spin, boson_dimension, virtual_spin_boson_bond_dimension, spin_name_prefix="qubit_", indexing=0):    
    s_shape = tuple([virtual_spin_spin_bond_dimension[0]] + [virtual_spin_boson_bond_dimension] * num_bosons_per_spin + [2])
    S = np.zeros(s_shape, dtype=complex).flatten()
    S[0] = 1
    S_1 = 1*S.reshape(s_shape)
    S_5 = 1*S.reshape(s_shape)

    s_shape = tuple([virtual_spin_spin_bond_dimension[0]] + [virtual_spin_spin_bond_dimension[1]] + [virtual_spin_boson_bond_dimension] * num_bosons_per_spin + [2])
    S = np.zeros(s_shape, dtype=complex).flatten()
    S[0] = 1
    S_2 = 1*S.reshape(s_shape)

    s_shape = tuple([virtual_spin_spin_bond_dimension[1]] + [virtual_spin_spin_bond_dimension[1]] + [virtual_spin_boson_bond_dimension] * num_bosons_per_spin + [2])
    S = np.zeros(s_shape, dtype=complex).flatten()
    S[0] = 1
    S_3 = 1*S.reshape(s_shape)

    s_shape = tuple([virtual_spin_spin_bond_dimension[1]] + [virtual_spin_spin_bond_dimension[0]] + [virtual_spin_boson_bond_dimension] * num_bosons_per_spin + [2])
    S = np.zeros(s_shape, dtype=complex).flatten()
    S[0] = 1
    S_4 = 1*S.reshape(s_shape)

    S = [S_1, S_2, S_3, S_4, S_5]
    
    b_shape = tuple([virtual_spin_boson_bond_dimension] + [boson_dimension])
    B = np.zeros(b_shape, dtype=complex).flatten()
    B[0] = 1
    B = B.reshape(b_shape)

    psi = QuantumTTState()

    for site in range(num_spins):
        identifier = spin_name_prefix + str(site + indexing)
        if site == 0:
            # Open boundary conditions on first site
            node_ham = QuantumStateNode(1*S[site], identifier=identifier)
            psi.add_root(node_ham)
        elif 0 < site:
            if site == num_spins - 1:
                # Open boundary conditions on last site
                node_ham = QuantumStateNode(1*S[site], identifier=identifier)
            else:
                node_ham = QuantumStateNode(1*S[site], identifier=identifier)
            parent_id = spin_name_prefix + str(site - 1 + indexing)
            if site == 1:
                # Due to boundary condition on first site
                psi.add_child_to_parent(node_ham, 0, parent_id, 0)
            else:
                psi.add_child_to_parent(node_ham, 0, parent_id, 1)
    
    for site in range(num_spins):
        for boson in range(num_bosons_per_spin):
            identifier = "boson_" + str(site + indexing) + "." + str(boson)
            parent_id = spin_name_prefix + str(site + indexing)

            node = psi[parent_id]
            first_free_leg = int(len(node.parent_leg)!=0) + len(node.children_legs)

            boson_node = QuantumStateNode(1*B, identifier=identifier)
            psi.add_child_to_parent(boson_node, 0, parent_id, first_free_leg)

    return psi

