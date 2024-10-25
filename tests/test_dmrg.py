from copy import deepcopy

import numpy as np
from tqdm import tqdm

from pytreenet.core import Node
from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.operators import pauli_matrices, TensorProduct, Hamiltonian
from pytreenet.time_evolution import FirstOrderOneSiteTDVP, ExactTimeEvolution, SecondOrderOneSiteTDVP
from pytreenet.special_ttn import MatrixProductState
from pytreenet.util import SVDParameters
from pytreenet.dmrg import DMRGAlgorithm

from pytreenet.random.random_ttns_and_ttno import small_ttns_and_ttno, big_ttns_and_ttno

X, Y, Z = pauli_matrices()

num_sites = 4
mJ = 0.2
mg = 4

mps_2_site = MatrixProductState.constant_product_state(1,2,num_sites,
                                                           bond_dimensions=[10,10,10])

interaction_term = TensorProduct({"site0": "mJX", "site1": "X"})
single_site_terms = [TensorProduct({"site"+str(i): "mgZ"}) for i in range(num_sites)]
terms = [interaction_term]
terms.extend(single_site_terms)
conversion_dict = {"mJX": mJ * X,
                   "X": X,
                   "mgZ": mg * Z,
                   "I2": np.eye(2)}

hamiltonian_2_site = Hamiltonian(terms, conversion_dict)
hamiltonian_2_site = hamiltonian_2_site.pad_with_identities(mps_2_site)

ham_ttno_2_site = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian_2_site, mps_2_site)

svd_params = SVDParameters(max_bond_dim=10,
                               rel_tol=1e-5,
                               total_tol=1e-5)

dmrg = DMRGAlgorithm(mps_2_site, ham_ttno_2_site, num_sweeps=10, max_iter=10, site='two-site', svd_params=svd_params)
es = dmrg.run()
dmrg_e = es[-1]
ed_e = np.min(np.linalg.eigvalsh(ham_ttno_2_site.as_matrix()[0]))
print(f"The ground state energy from DMRG is {dmrg_e}, and the exact diagnolization energy is {ed_e}")

# Test with random TTNS and TTNO
ttns, ttno = big_ttns_and_ttno()
dmrg = DMRGAlgorithm(ttns, ttno, num_sweeps=10, max_iter=10, site='two-site', svd_params=svd_params)
es = dmrg.run()
dmrg_e = es[-1]
ed_e = np.min(np.linalg.eigvalsh(ttno.as_matrix()[0]))
print(f"The ground state energy from DMRG is {dmrg_e}, and the exact diagnolization energy is {ed_e}")

