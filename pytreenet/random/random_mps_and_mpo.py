import numpy as np
from ..special_ttn.mps import (MatrixProductOperator,
                               MatrixProductState)
from ..operators import TensorProduct, Hamiltonian
from ..random.random_matrices import random_hermitian_matrix
from ..ttno import TreeTensorNetworkOperator
from .random_special_ttno import random_mpo
from .random_special_ttns import random_mps


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

def random_mps_and_mpo_by_dimensions(length: int,
                                     phys_dim: int,
                                     bond_dim_mps: int,
                                     bond_dim_mpo: int,
                                     root_site: int | None = None,
                                     **kwargs
                                     ) -> tuple[MatrixProductState, MatrixProductOperator]:
    """
    Creates a random MPS and MPO by specifying the dimensions.

    Args:
        length (int): The number of sites in the MPS and MPO.
        phys_dim (int): The physical dimension of each site.
        bond_dim_mps (int): The bond dimension of the MPS.
        bond_dim_mpo (int): The bond dimension of the MPO.
        **kwargs: Additional keyword arguments for the random number
            generation.
    
    Returns:
        tuple[MatrixProductState, MatrixProductOperator]: The generated random MPS and MPO.
    """
    mps = random_mps(length, phys_dim, bond_dim_mps, root_site=root_site, **kwargs)
    mpo = random_mpo(length, phys_dim, bond_dim_mpo, root_site=root_site, **kwargs)
    return mps, mpo
