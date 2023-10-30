import numpy as np
from ..base.ttn import QuantumTTState, QuantumTTOperator
from ..base.tensornode import QuantumStateNode, QuantumOperatorNode
from ..utils.util import pauli_matrices, zero_state


X, Y, Z = pauli_matrices()
O = np.zeros((2, 2), dtype=complex)
I = np.identity(2, dtype=complex)


def mps_zero(num_sites, virtual_bond_dimension, name_prefix="site", indexing=0):
    zero = zero_state([virtual_bond_dimension, virtual_bond_dimension, 2])
    mps = QuantumTTState()

    for site in range(num_sites):
        identifier = name_prefix + str(site + indexing)
        if site == 0:
            # Open boundary conditions on first site
            node_mps = QuantumStateNode(zero[0], identifier=identifier)
            mps.add_root(node_mps)
        elif 0 < site:
            if site == num_sites - 1:
                # Open boundary conditions on last site
                node_mps = QuantumStateNode(zero[:,0,:], identifier=identifier)
            else:
                node_mps = QuantumStateNode(zero, identifier=identifier)
            parent_id = name_prefix + str(site - 1 + indexing)
            if site == 1:
                # Due to boundary condition on first site
                mps.add_child_to_parent(node_mps, 0, parent_id, 0)
            else:
                mps.add_child_to_parent(node_mps, 0, parent_id, 1)
    
    return mps


def mps_heisenberg(num_sites, jx=0, jy=0, jz=0, h=0, name_prefix="site", indexing=0, add_extra_qubit_dim=False):
    if add_extra_qubit_dim:
        A = np.array([[I, O, O, O, O, O],
                    [X, O, O, O, O, O],
                    [Y, O, O, O, O, O],
                    [Z, O, O, O, O, O],
                    [O, O, O, O, O, O],
                    [h*Z, jx*X, jy*Y, jz*Z, O, I]])
    else:
        A = np.array([[I, O, O, O, O],
                    [X, O, O, O, O],
                    [Y, O, O, O, O],
                    [Z, O, O, O, O],
                    [h*Z, jx*X, jy*Y, jz*Z, I]])

    A = np.transpose(A, (0, 1, 2, 3))

    remove = [i for i, val in enumerate([1, jx, jy, jz]) if val == 0]
    A = np.delete(A, remove, axis=0)
    A = np.delete(A, remove, axis=1)

    ham = QuantumTTOperator()

    for site in range(num_sites):
        identifier = name_prefix + str(site + indexing)
        if site == 0:
            # Open boundary conditions on first site
            node_ham = QuantumOperatorNode(1*A[-1,:,:,:], identifier=identifier)
            ham.add_root(node_ham)
        elif 0 < site:
            if site == num_sites - 1:
                # Open boundary conditions on last site
                node_ham = QuantumOperatorNode(1*A[:,0,:,:], identifier=identifier)
            else:
                node_ham = QuantumOperatorNode(1*A, identifier=identifier)
            parent_id = name_prefix + str(site - 1 + indexing)
            if site == 1:
                # Due to boundary condition on first site
                ham.add_child_to_parent(node_ham, 0, parent_id, 0)
            else:
                ham.add_child_to_parent(node_ham, 0, parent_id, 1)
    
    return ham