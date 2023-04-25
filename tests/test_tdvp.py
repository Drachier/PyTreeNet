import pytreenet as ptn
import numpy as np

class TestTDVP:
    def __init__(self):
        self.initialize_ising_ttn()
        self.initialize_ising_hamiltonian()

        self.state = ptn.TreeTensorState(self.ttn)
        self.hamiltonian = ptn.TreeTensorOperator(self.hamiltonian)

        # self.state.plot("State > TreeTensorState")
        # self.hamiltonian.plot("Hamiltonian > TreeTensorOperator")

        self.ttn = ptn.tdvp(self.state, self.hamiltonian, .5, 2, "1site")
    
    def initialize_ising_ttn(self):
        self.ttn = ptn.TreeTensorNetwork()

        site1 = ptn.random_tensor_node((5,2), identifier="site1", zeros=True)
        site2 = ptn.random_tensor_node((5,5,2), identifier="site2", zeros=True)
        site3 = ptn.random_tensor_node((5,5,2), identifier="site3", zeros=True)
        site4 = ptn.random_tensor_node((5,2), identifier="site4", zeros=True)

        self.ttn.add_root(site1)
        self.ttn.add_child_to_parent(site2, 0, "site1", 0)
        self.ttn.add_child_to_parent(site3, 0, "site2", 1)
        self.ttn.add_child_to_parent(site4, 0, "site3", 1)
    
    def initialize_ising_hamiltonian(self):
        self.hamiltonian = ptn.TreeTensorOperator()

        site1 = ptn.random_tensor_node((2,5,2), identifier="site1", zeros=True)
        site2 = ptn.random_tensor_node((2,5,5,2), identifier="site2", zeros=True)
        site3 = ptn.random_tensor_node((2,5,5,2), identifier="site3", zeros=True)
        site4 = ptn.random_tensor_node((2,5,2), identifier="site4", zeros=True)

        self.hamiltonian.add_root(site1)
        self.hamiltonian.add_child_to_parent(site2, 1, "site1", 1)
        self.hamiltonian.add_child_to_parent(site3, 1, "site2", 2)
        self.hamiltonian.add_child_to_parent(site4, 1, "site3", 2)

        self.hamiltonian.add_link("site1", "site1")
        self.hamiltonian.add_link("site2", "site2")
        self.hamiltonian.add_link("site3", "site3")
        self.hamiltonian.add_link("site4", "site4")

"""
def construct_ising_hamiltonian_mpo(J, g, L, pbc=False):
    
    Construct Ising Hamiltonian on a 1D lattice with `L` sites as MPO,
    for interaction parameter `J` and external field parameter `g`.
    
    # Pauli-X and Z matrices
    X = np.array([[0., 1.], [1.,  0.]])
    Z = np.array([[1., 0.], [0., -1.]])
    I = np.identity(2)
    O = np.zeros((2, 2))
    A = np.array([[I, O, O], [Z, O, O], [-g*X, -J*Z, I]])
    # flip the ordering of the virtual bond dimensions and physical dimensions
    A = np.transpose(A, (2, 3, 0, 1))
    if pbc:
        # periodic boundary conditions:
        # add a direct transition b -> a which applies -J Z at the rightmost lattice site
        AL = np.array([[-g*X, -J*Z, I], [Z, O, O]])
        AR = np.array([[I, -J*Z], [Z, O], [-g*X, O]])
        # flip the ordering of the virtual bond dimensions and physical dimensions
        AL = np.transpose(AL, (2, 3, 0, 1))
        AR = np.transpose(AR, (2, 3, 0, 1))
        return MPO([AL if i == 0 else A if i < L-1 else AR for i in range(L)])
    else:
        return MPO([A[:, :, 2:3, :] if i == 0 else A if i < L-1 else A[:, :, :, 0:1] for i in range(L)])



MPS 
 elif fill == 'random complex':
            # random complex entries
            self.A = [crandn(size=(d, D[i], D[i+1])) / np.sqrt(d*D[i]*D[i+1]) for i in range(len(D)-1)]
"""