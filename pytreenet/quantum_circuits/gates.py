import numpy as np


class Gate:
    def __init__(self) -> None:
        pass

    def hamiltonian(self):
        """
        U = e^(-i Ht)
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.U)
        ln_U = eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.conj().T
        return 1j * ln_U
    
    @property
    def H(self):
        return self.hamiltonian()
    
    @property
    def U(self):
        return self.matrix()
    
    
class UGate(Gate):
    def __init__(self, angles) -> None:
        self.angles = np.array(angles) * np.pi

    def matrix(self):
        the, phi, lam = self.angles
        return np.array([[np.cos(the/2), -np.exp(1j*lam)*np.sin(the/2)], 
                         [np.exp(1j*phi)*np.sin(the/2), np.exp(1j*(phi+lam))*np.cos(the/2)]], 
                         dtype=complex)
    
class MatrixGate(Gate):
    def __init__(self, matrix) -> None:
        self._matrix = matrix
    
    def matrix(self):
        return self._matrix


def all_gates():
    d = dict()
    d["X"] = UGate([1, 0, 1])
    d["Y"] = UGate([1, 1/2, 1/2])
    d["Z"] = UGate([0, 0, 1])
    d["I"] = UGate([0, 0, 0])
    d["-I"] = UGate([2, 0, 0])
    d["H"] = UGate([1/2, 0, 1])
    d["S"] = UGate([0, 0, 1/2])
    d["T"] = UGate([0, 0, 1/4])
    d["-"] = MatrixGate(np.array([[1, 0], [0, 0]], dtype=complex))
    d["+"] = MatrixGate(np.array([[0, 0], [0, 1]], dtype=complex))
    return d
