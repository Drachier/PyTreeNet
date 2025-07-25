import numpy as np
from pytreenet.special_ttn import MatrixProductState
from pytreenet.operators import TensorProduct, Hamiltonian
from pytreenet.random.random_matrices import random_hermitian_matrix
from pytreenet.ttno import TreeTensorNetworkOperator


def random_mps_and_mpo(dimension: int, bond_dimensions: int):
    """
    Creates a random MPS and MPO.

    Args:
        dimension (int): The dimension of the local Hilbert space.
        bond_dimensions (int): The bond dimensions of the MPS.

    Returns:
        tuple: A tuple containing the MPS and MPO.
    """
    ttns = MatrixProductState.constant_product_state(1, dimension, 8, bond_dimensions = [bond_dimensions]*7)
    ttns.normalize()
    
    conversion_dict = {chr(i): random_hermitian_matrix(dimension)
                       for i in range(65,70)} # A, B, C, D, E
    conversion_dict["I2"] = np.eye(2)
    conversion_dict["I1"] = np.eye(1)
    terms_ttno= [TensorProduct({"site1": "A", "site2": "B", "site0": "C"}),
            TensorProduct({"site4": "A", "site3": "D", "site7": "C"}),
            # TensorProduct({"site4": "A", "site3": "B", "site1": "A"}),
            # TensorProduct({"site0": "C", "site6": "E", "site4": "C"}),
            # TensorProduct({"site2": "A", "site7": "A", "site0": "D"}),
            TensorProduct({"site6": "A", "site3": "B", "site5": "C"})]
    ham_mps = Hamiltonian(terms_ttno, conversion_dictionary=conversion_dict)
    ham_pad_mps =  ham_mps.pad_with_identities(ttns)
    ttno = TreeTensorNetworkOperator.from_hamiltonian(ham_pad_mps, ttns)
        
    
    return ttns, ttno