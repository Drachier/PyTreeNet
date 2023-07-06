import numpy as np
from ..ttn import QuantumTTState, QuantumTTOperator
from ..tensornode import QuantumStateNode, QuantumOperatorNode
from ..util import pauli_matrices, zero_state


X, Y, Z = pauli_matrices()
O = np.zeros((2, 2), dtype=complex)
I = np.identity(2, dtype=complex)


def mps_zero(num_sites, virtual_bond_dimension, name_prefix="site"):
    zero = zero_state([virtual_bond_dimension, virtual_bond_dimension, 2])
    mps = QuantumTTState()

    for site in range(num_sites):
        identifier = name_prefix + str(site)
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
            parent_id = name_prefix + str(site - 1)
            if site == 1:
                # Due to boundary condition on first site
                mps.add_child_to_parent(node_mps, 0, parent_id, 0)
            else:
                mps.add_child_to_parent(node_mps, 0, parent_id, 1)
    
    return mps


def mps_heisenberg(num_sites, jx, jy, jz, h, name_prefix="site"):
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
        identifier = name_prefix + str(site)
        if site == 0:
            # Open boundary conditions on first site
            node_ham = QuantumOperatorNode(A[-1,:,:,:], identifier=identifier)
            ham.add_root(node_ham)
        elif 0 < site:
            if site == num_sites - 1:
                # Open boundary conditions on last site
                node_ham = QuantumOperatorNode(A[:,0,:,:], identifier=identifier)
            else:
                node_ham = QuantumOperatorNode(A, identifier=identifier)
            parent_id = name_prefix + str(site - 1)
            if site == 1:
                # Due to boundary condition on first site
                ham.add_child_to_parent(node_ham, 0, parent_id, 0)
            else:
                ham.add_child_to_parent(node_ham, 0, parent_id, 1)
    
    return ham