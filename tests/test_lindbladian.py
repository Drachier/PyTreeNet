from unittest import TestCase, main as main_unit
from fractions import Fraction

from numpy import asarray

from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.operators.lindbladian import (_find_real_operators,
                                             _find_hermitian_operators,
                                             _find_identity_operators,
                                             _find_symmetric_operators,
                                             _add_hamiltonian_ket_terms,
                                             _add_hamiltonian_bra_terms,
                                             _add_jump_operators,
                                             _add_jump_operator_products)

class TestFindingFunctions(TestCase):
    """
    Tests the different functions that find operators of different types.
    """

    def test_find_real_operators(self):
        """
        Tests the function that finds real operators.
        """
        op_dict = {"real": asarray([[1, 2], [3, 4]]),
                   "non_real": asarray([[1, 2+1j], [3, 4j]])}
        real_ops = _find_real_operators(op_dict)
        self.assertEqual(len(real_ops), 2)
        self.assertTrue(real_ops["real"])
        self.assertFalse(real_ops["non_real"])

    def test_find_hermitian_operators(self):
        """
        Tests the function that finds hermitian operators.
        """
        op_dict = {"herm": asarray([[1, 3+2j], [3-2j, 1]]),
                   "non_herm": asarray([[1, 2j], [3+1j, 4]])}
        herm_ops = _find_hermitian_operators(op_dict)
        self.assertEqual(len(herm_ops), 2)
        self.assertTrue(herm_ops["herm"])
        self.assertFalse(herm_ops["non_herm"])

    def test_find_identity_operators(self):
        """
        Tests the function that finds identity operators.
        """
        op_dict = {"id": asarray([[1, 0], [0, 1]]),
                   "id1": asarray([[1]]), 
                   "non_id": asarray([[1, 2], [3, 4]])}
        id_ops = _find_identity_operators(op_dict)
        self.assertEqual(len(id_ops), 3)
        self.assertTrue(id_ops["id"])
        self.assertFalse(id_ops["non_id"])

    def test_find_symmetric_operators(self):
        """
        Tests the function that finds symmetric operators.
        """
        op_dict = {"sym": asarray([[1, 2], [2, 1]]),
                   "non_sym": asarray([[1, 2], [3, 4]])}
        sym_ops = _find_symmetric_operators(op_dict)
        self.assertEqual(len(sym_ops), 2)
        self.assertTrue(sym_ops["sym"])
        self.assertFalse(sym_ops["non_sym"])

class TestHamiltonianTerms(TestCase):
    """
    Tests the addition of hamiltonian terms to the lindbladian.
    """

    def setUp(self):
        self.operators = (TensorProduct({"node1": "A", "node2": "B"}),
                     TensorProduct({"node2": "C", "node3": "D"}))
        self.factors = (Fraction(1, 2), Fraction(1, 3))
        self.symb_factors = ("g", "j")
        self.terms = tuple(zip(self.factors, self.symb_factors, self.operators))
        self.conv_dict = {"A": asarray([[1, 2], [3, 4]]),
                     "B": asarray([[5, 6], [7, 8]]),
                     "C": asarray([[9, 20], [11, 12]]),
                     "D": asarray([[13, 15], [15, 17]])} # symmetric
        self.coeff_map = {"1": 1, "g": 2j, "j": 5}
        self.hamiltonian = Hamiltonian(self.terms,
                                        conversion_dictionary=self.conv_dict,
                                        coeffs_mapping=self.coeff_map)

    def test_add_hamiltonian_ket_terms(self):
        """
        Tests that the normal hamiltonian terms are properly added to the
        lindbladian.
        """
        lindbladian = Hamiltonian()
        suff = "_ket"
        _add_hamiltonian_ket_terms(lindbladian, self.hamiltonian, suff)
        # Testing
        self.assertEqual(len(lindbladian.terms), 2)
        for i, term in enumerate(lindbladian.terms):
            self.assertEqual(term[0], self.factors[i])
            self.assertEqual(term[1], self.symb_factors[i])
            self.assertEqual(term[2], self.operators[i].add_suffix(suff))
        self.assertEqual(lindbladian.conversion_dictionary, self.conv_dict)
        self.assertEqual(lindbladian.coeffs_mapping, self.coeff_map)

    def test_add_hamiltonian_bra_terms(self):
        """
        Tests that the bra hamiltonian terms are properly added to the
        lindbladian.
        """
        lindbladian = Hamiltonian()
        suff = "_bra"
        _add_hamiltonian_bra_terms(lindbladian, self.hamiltonian, suff)
        # testing
        self.assertEqual(len(lindbladian.terms), 2)
        sym_dict = {"A": False, "B": False, "C": False, "D": True}
        tp_ops = [op.transpose(sym_dict) for op in self.operators]
        for i, term in enumerate(lindbladian.terms):
            self.assertEqual(term[0], -1*self.factors[i])
            self.assertEqual(term[1], self.symb_factors[i])
            self.assertEqual(term[2], tp_ops[i].add_suffix(suff))
        transpose_dict = {"A": self.conv_dict["A"].T,
                          "B": self.conv_dict["B"].T,
                          "C": self.conv_dict["C"].T}
        self.assertEqual(len(lindbladian.conversion_dictionary), 3)
        for key, value in lindbladian.conversion_dictionary.items():
            self.assertTrue((value == transpose_dict[key]).all())

class TestAddJumpOperatorTerms(TestCase):
    """
    Tests the addition of jump operators to the lindbladian.
    """

    def setUp(self):
        self.ops = (TensorProduct({"node1": "A", "node2": "B"}),
                     TensorProduct({"node2": "C", "node3": "D"}))
        self.factors = (Fraction(1, 2), Fraction(1, 3))
        self.symb_factors = ("g", "j")
        self.terms = tuple(zip(self.factors, self.symb_factors, self.ops))
        self.jump_operator_dict = {"A": asarray([[1, 2j], [3, 4]]),
                                    "B": asarray([[1, 0], [0, 1]]), # Id and thus real
                                    "C": asarray([[1, 1j], [1j, 1]]), # Product of CC^dagger is symmetric
                                    "D": asarray([[13, -15j], [15j, 17]])} # hermitian
        self.jump_coeff_mapping = {"1": 1, "g": 2j, "j": 5}
        self.ket_suff = "_ket"
        self.bra_suff = "_bra"

    def test_add_jump_operators(self):
        """
        Tests that the jump operator terms are properly added to the lindbladian.
        """
        lindbladian = Hamiltonian()
        _add_jump_operators(lindbladian, self.terms, self.jump_operator_dict,
                            self.jump_coeff_mapping, self.ket_suff, self.bra_suff)
        # Testing
        self.assertEqual(len(lindbladian.terms), 2)
        for i, _ in enumerate(self.terms):
            l_term = lindbladian.terms[i]
            self.assertEqual(l_term[0], self.factors[i])
            self.assertEqual(l_term[1], self.symb_factors[i])
            self.assertEqual(len(l_term[2]), 4)
        # term1
        op1 = lindbladian.terms[0][2]
        self.assertEqual(op1["node1"+self.ket_suff], "A")
        self.assertEqual(op1["node2"+self.ket_suff], "B")
        self.assertEqual(op1["node1"+self.bra_suff], "A_conj")
        self.assertEqual(op1["node2"+self.bra_suff], "B")
        # term2
        op2 = lindbladian.terms[1][2]
        self.assertEqual(op2["node2"+self.ket_suff], "C")
        self.assertEqual(op2["node3"+self.ket_suff], "D")
        self.assertEqual(op2["node2"+self.bra_suff], "C_conj")
        self.assertEqual(op2["node3"+self.bra_suff], "D_conj")

    def test_add_jump_operator_products(self):
        """
        Tests that the terms made of jump operator products are properly added to
        the lindbladian.
        """
        linbladian = Hamiltonian()
        _add_jump_operator_products(linbladian,
                                    self.terms,
                                    self.jump_operator_dict,
                                    self.ket_suff,
                                    self.bra_suff)
        # Testing
        self.assertEqual(len(linbladian.terms), 4)
        # test prefactors
        self.assertEqual(linbladian.terms[0][0], -1*self.factors[0] / 2)
        self.assertEqual(linbladian.terms[1][0], -1*self.factors[0] / 2)
        self.assertEqual(linbladian.terms[2][0], -1*self.factors[1] / 2)
        self.assertEqual(linbladian.terms[3][0], -1*self.factors[1] / 2)
        self.assertEqual(linbladian.terms[0][1], self.symb_factors[0])
        self.assertEqual(linbladian.terms[1][1], self.symb_factors[0])
        self.assertEqual(linbladian.terms[2][1], self.symb_factors[1])
        self.assertEqual(linbladian.terms[3][1], self.symb_factors[1])
        # test operators
        # term1
        op1 = linbladian.terms[0][2]
        self.assertEqual(op1["node1"+self.ket_suff], "A_H_mult_A")
        self.assertEqual(op1["node2"+self.ket_suff], "B")
        # term2
        op2 = linbladian.terms[1][2]
        self.assertEqual(op2["node1"+self.bra_suff], "A_H_mult_A_T")
        self.assertEqual(op2["node2"+self.bra_suff], "B")
        # term3
        op3 = linbladian.terms[2][2]
        self.assertEqual(op3["node2"+self.ket_suff], "C_H_mult_C")
        self.assertEqual(op3["node3"+self.ket_suff], "D_mult_D")
        # term4
        op4 = linbladian.terms[3][2]
        self.assertEqual(op4["node2"+self.bra_suff], "C_H_mult_C")
        self.assertEqual(op4["node3"+self.bra_suff], "D_mult_D_T")

if __name__ == "__main__":
    main_unit()
