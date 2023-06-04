import numpy as np
from ..ttn import QuantumTTState, QuantumTTOperator
from ..tensornode import QuantumStateNode, QuantumOperatorNode
from ..util import pauli_matrices, zero_state, create_bosonic_operators


X, Y, Z = pauli_matrices()
O = np.zeros((2, 2), dtype=complex)
I = np.identity(2, dtype=complex)


def bosonic_tree_operator(num_spins, num_bosons_per_spin, boson_dimension, J, g, momega):
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
    for b in range(num_bosons_per_spin):
        index = [0] * num_bosons_per_spin
        idx = tuple([Ellipsis] + index)
        S[idx] = ss
        index[b] = 1
        idx = tuple([Ellipsis] + index)
        S[idx] = bo
        index[b] = 2
        idx = tuple([Ellipsis] + index)
        S[idx] = sb

    s_trans = list(range(S.ndim))
    s_trans.remove(2)
    s_trans.remove(3)
    s_trans.append(2)
    s_trans.append(3)
    S = S.transpose(s_trans)
    
    creation_op, annihilation_op, number_op = create_bosonic_operators(dimension=boson_dimension)
    I_b = np.eye(boson_dimension, dtype=complex)
    B = np.zeros((3, boson_dimension, boson_dimension), dtype=complex)
    B[0,:,:] = I_b
    B[1,:,:] = momega/2 * number_op
    B[2,:,:] = annihilation_op + creation_op

    ham = QuantumTTOperator()

    for site in range(num_spins):
        identifier = "spin_" + str(site)
        if site == 0:
            # Open boundary conditions on first site
            node_ham = QuantumOperatorNode(S[2,:], identifier=identifier)
            ham.add_root(node_ham)
        elif 0 < site:
            if site == num_spins - 1:
                # Open boundary conditions on last site
                node_ham = QuantumOperatorNode(S[:,0], identifier=identifier)
            else:
                node_ham = QuantumOperatorNode(S, identifier=identifier)
            parent_id = "spin_" + str(site - 1)
            if site == 1:
                # Due to boundary condition on first site
                ham.add_child_to_parent(node_ham, 0, parent_id, 0)
            else:
                ham.add_child_to_parent(node_ham, 0, parent_id, 1)
    
    for site in range(num_spins):
        for boson in range(num_bosons_per_spin):
            identifier = "boson_" + str(site) + "." + str(boson)
            parent_id = "spin_" + str(site)

            node = ham[parent_id]
            first_free_leg = int(len(node.parent_leg)!=0) + len(node.children_legs)

            boson_node = QuantumOperatorNode(B, identifier=identifier)
            ham.add_child_to_parent(boson_node, 0, parent_id, first_free_leg)

    return ham


def bosonic_tree_state(num_spins, virtual_spin_spin_bond_dimension, num_bosons_per_spin, boson_dimension, virtual_spin_boson_bond_dimension):    
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
        identifier = "spin_" + str(site)
        if site == 0:
            # Open boundary conditions on first site
            node_ham = QuantumStateNode(S[0,:], identifier=identifier)
            psi.add_root(node_ham)
        elif 0 < site:
            if site == num_spins - 1:
                # Open boundary conditions on last site
                node_ham = QuantumStateNode(S[:,0], identifier=identifier)
            else:
                node_ham = QuantumStateNode(S, identifier=identifier)
            parent_id = "spin_" + str(site - 1)
            if site == 1:
                # Due to boundary condition on first site
                psi.add_child_to_parent(node_ham, 0, parent_id, 0)
            else:
                psi.add_child_to_parent(node_ham, 0, parent_id, 1)
    
    for site in range(num_spins):
        for boson in range(num_bosons_per_spin):
            identifier = "boson_" + str(site) + "." + str(boson)
            parent_id = "spin_" + str(site)

            node = psi[parent_id]
            first_free_leg = int(len(node.parent_leg)!=0) + len(node.children_legs)

            boson_node = QuantumStateNode(B, identifier=identifier)
            psi.add_child_to_parent(boson_node, 0, parent_id, first_free_leg)

    return psi

