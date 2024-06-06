import unittest

import numpy as np

import pytreenet as ptn
from pytreenet.random import (random_hermitian_matrix,
                              random_small_ttns,
                              random_big_ttns_two_root_children,
                              random_hamiltonian_compatible,
                              crandn)

class TestSecondOrderOneSiteTDVPInitSimple(unittest.TestCase):
    def setUp(self) -> None:
        self.conversion_dict = {"root_op1": random_hermitian_matrix(),
                                "root_op2": random_hermitian_matrix(),
                                "I2": np.eye(2),
                                "c1_op": random_hermitian_matrix(size=3),
                                "I3": np.eye(3),
                                "c2_op": random_hermitian_matrix(size=4),
                                "I4": np.eye(4)}
        self.ref_tree = random_small_ttns()
        tensor_prod = [ptn.TensorProduct({"c1": "I3", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op2", "c2": "c2_op"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "I2", "c2": "c2_op"})
                       ]
        ham = ptn.Hamiltonian(tensor_prod, self.conversion_dict)
        operator = ptn.TensorProduct({"root": crandn((2,2))})
        self.hamiltonian = ptn.TTNO.from_hamiltonian(ham, self.ref_tree)
        self.tdvp = ptn.SecondOrderOneSiteTDVP(self.ref_tree, self.hamiltonian,
                                               0.1, 1, operator)

    def test_init_second_order_update_path(self):
        self.assertEqual(["c2","root","c1"],self.tdvp.backwards_update_path)

    def test_init_second_order_orth_path(self):
        self.assertEqual([["root"],["c1"]],self.tdvp.backwards_orth_path)

class TestSecondOrderOneSiteTDVPUpdatesSimple(unittest.TestCase):
    def setUp(self) -> None:
        self.conversion_dict = {"root_op1": random_hermitian_matrix(),
                                "root_op2": random_hermitian_matrix(),
                                "I2": np.eye(2),
                                "c1_op": random_hermitian_matrix(size=3),
                                "I3": np.eye(3),
                                "c2_op": random_hermitian_matrix(size=4),
                                "I4": np.eye(4)}
        self.ref_tree = random_small_ttns()
        tensor_prod = [ptn.TensorProduct({"c1": "I3", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op1", "c2": "I4"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "root_op2", "c2": "c2_op"}),
                       ptn.TensorProduct({"c1": "c1_op", "root": "I2", "c2": "c2_op"})
                       ]
        ham = ptn.Hamiltonian(tensor_prod, self.conversion_dict)
        operator = ptn.TensorProduct({"root": crandn((2,2))})
        self.hamiltonian = ptn.TTNO.from_hamiltonian(ham, self.ref_tree)
        self.tdvp = ptn.SecondOrderOneSiteTDVP(self.ref_tree, self.hamiltonian,
                                               0.1, 1, operator)

class TestSecondOrderOneSiteTDVPInitComplicated(unittest.TestCase):
    def setUp(self):
        self.ref_tree = random_big_ttns_two_root_children()
        self.hamiltonian = ptn.TTNO.from_hamiltonian(random_hamiltonian_compatible(),
                                                     self.ref_tree)
        self.tdvp = ptn.SecondOrderOneSiteTDVP(self.ref_tree, self.hamiltonian, 0.1,1,
                                      ptn.TensorProduct({"site0": ptn.pauli_matrices()[0]}))

    def test_init_second_order_update_path(self):
        correct_path = ["site7","site6","site0","site1",
                        "site2","site3","site5","site4"]
        self.assertEqual(correct_path,self.tdvp.backwards_update_path)

    def test_init_second_order_orth_path(self):
        correct_path = [["site6"],["site0"],["site1"],
                        ["site2","site1"],["site3"],["site5","site3"],
                        ["site4"]]
        self.assertEqual(correct_path,self.tdvp.backwards_orth_path)

if __name__ == "__main__":
    unittest.main()