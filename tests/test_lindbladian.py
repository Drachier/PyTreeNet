from unittest import TestCase, main as main_unit
from fractions import Fraction
from copy import copy

from numpy import asarray, zeros, eye, kron, allclose, zeros_like, sqrt

from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.operators.common_operators import ket_i
from pytreenet.special_ttn.binary import generate_binary_ttns
from pytreenet.ttns.ttndo import from_ttns
from pytreenet.random import crandn
from pytreenet.operators.exact_operators import exact_lindbladian
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator
from pytreenet.operators.lindbladian import (_find_real_operators,
                                             _find_hermitian_operators,
                                             _find_identity_operators,
                                             _find_symmetric_operators,
                                             _add_hamiltonian_ket_terms,
                                             _add_hamiltonian_bra_terms,
                                             _add_jump_operators,
                                             _add_jump_operator_products,
                                             generate_lindbladian)

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
        transpose_dict = {"A_T": self.conv_dict["A"].T,
                          "B_T": self.conv_dict["B"].T,
                          "C_T": self.conv_dict["C"].T}
        self.assertEqual(len(lindbladian.conversion_dictionary), 3)
        for key, value in transpose_dict.items():
            self.assertTrue((value == lindbladian.conversion_dictionary[key]).all())

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
            self.assertEqual(l_term[1], self.symb_factors[i] + "*j")
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
        self.assertEqual(linbladian.terms[1][0], self.factors[0] / 2)
        self.assertEqual(linbladian.terms[2][0], -1*self.factors[1] / 2)
        self.assertEqual(linbladian.terms[3][0], self.factors[1] / 2)
        self.assertEqual(linbladian.terms[0][1], self.symb_factors[0] + "*j")
        self.assertEqual(linbladian.terms[1][1], self.symb_factors[0] + "*j")
        self.assertEqual(linbladian.terms[2][1], self.symb_factors[1] + "*j")
        self.assertEqual(linbladian.terms[3][1], self.symb_factors[1] + "*j")
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

class TestLindbladianGeneration(TestCase):
    """
    Tests the generation of a Lindbladian from a Hamiltonian and jump operators.
    """

    def test_generate_lindbladian(self):
        operators = (TensorProduct({"node1": "A", "node2": "B"}),
                     TensorProduct({"node2": "C", "node3": "D"}))
        factors = (Fraction(1, 2), Fraction(1, 3))
        symb_factors = ("g", "j")
        terms = tuple(zip(factors, symb_factors, operators))
        conv_dict = {"A": asarray([[1, 2], [3, 4]]),
                     "B": asarray([[5, 6], [7, 8]]),
                     "C": asarray([[9, 20], [11, 12]]),
                     "D": asarray([[13, 15], [15, 17]])} # symmetric
        coeff_map = {"1": 1, "g": 2j, "j": 5}
        hamiltonian = Hamiltonian(terms,
                                    conversion_dictionary=conv_dict,
                                    coeffs_mapping=coeff_map)
        opsj = (TensorProduct({"node1": "Aj", "node2": "Bj"}),
                     TensorProduct({"node2": "Cj", "node3": "Dj"}))
        factorsj = (Fraction(1, 2), Fraction(1, 3))
        symb_factorsj = ("g", "j")
        termsj = tuple(zip(factorsj, symb_factorsj, opsj))
        jump_operator_dict = {"Aj": asarray([[1, 2j], [3, 4]]),
                                "Bj": asarray([[1, 0], [0, 1]]), # Id and thus real
                                "Cj": asarray([[1, 1j], [1j, 1]]), # Product of CC^dagger is symmetric
                                "Dj": asarray([[13, -15j], [15j, 17]])} # hermitian
        jump_coeff_mapping = {"1": 1, "gj": 2j, "jj": 5}
        ket_suff = "_ket"
        bra_suff = "_bra"
        lindbladian = generate_lindbladian(hamiltonian, termsj,
                                           jump_operator_dict,
                                           jump_coeff_mapping,
                                           ket_suffix=ket_suff,
                                           bra_suffix=bra_suff)
        # Testing
        self.assertEqual(len(lindbladian.terms), 10)
        # Hamiltonian terms
        # ket
        for i, term in enumerate(lindbladian.terms[:2]):
            self.assertEqual(term[0], factors[i])
            self.assertEqual(term[1], symb_factors[i])
        self.assertEqual(lindbladian.terms[0][2]["node1"+ket_suff], "A")
        self.assertEqual(lindbladian.terms[0][2]["node2"+ket_suff], "B")
        self.assertEqual(lindbladian.terms[1][2]["node2"+ket_suff], "C")
        self.assertEqual(lindbladian.terms[1][2]["node3"+ket_suff], "D")
        # bra
        for i, term in enumerate(lindbladian.terms[2:4]):
            self.assertEqual(term[0], -1*factors[i])
            self.assertEqual(term[1], symb_factors[i])
        self.assertEqual(lindbladian.terms[2][2]["node1"+bra_suff], "A_T")
        self.assertEqual(lindbladian.terms[2][2]["node2"+bra_suff], "B_T")
        self.assertEqual(lindbladian.terms[3][2]["node2"+bra_suff], "C_T")
        self.assertEqual(lindbladian.terms[3][2]["node3"+bra_suff], "D")
        # Jump operator terms
        for i, term in enumerate(lindbladian.terms[4:6]):
            self.assertEqual(term[0], factorsj[i])
            self.assertEqual(term[1], symb_factorsj[i] + "*j")
        self.assertEqual(lindbladian.terms[4][2]["node1"+ket_suff], "Aj")
        self.assertEqual(lindbladian.terms[4][2]["node2"+ket_suff], "Bj")
        self.assertEqual(lindbladian.terms[4][2]["node1"+bra_suff], "Aj_conj")
        self.assertEqual(lindbladian.terms[4][2]["node2"+bra_suff], "Bj")
        self.assertEqual(lindbladian.terms[5][2]["node2"+ket_suff], "Cj")
        self.assertEqual(lindbladian.terms[5][2]["node3"+ket_suff], "Dj")
        self.assertEqual(lindbladian.terms[5][2]["node2"+bra_suff], "Cj_conj")
        self.assertEqual(lindbladian.terms[5][2]["node3"+bra_suff], "Dj_conj")
        # Jump operator products
        for i, term in enumerate(lindbladian.terms[6:]):
            if i % 2 == 0:
                self.assertEqual(term[0], -1*factorsj[i//2] / 2)
            else:
                self.assertEqual(term[0], factorsj[i//2] / 2)
            self.assertEqual(term[1], symb_factorsj[i//2] + "*j")
        self.assertEqual(lindbladian.terms[6][2]["node1"+ket_suff], "Aj_H_mult_Aj")
        self.assertEqual(lindbladian.terms[6][2]["node2"+ket_suff], "Bj")
        self.assertEqual(lindbladian.terms[7][2]["node1"+bra_suff], "Aj_H_mult_Aj_T")
        self.assertEqual(lindbladian.terms[7][2]["node2"+bra_suff], "Bj")
        self.assertEqual(lindbladian.terms[8][2]["node2"+ket_suff], "Cj_H_mult_Cj")
        self.assertEqual(lindbladian.terms[8][2]["node3"+ket_suff], "Dj_mult_Dj")
        self.assertEqual(lindbladian.terms[9][2]["node2"+bra_suff], "Cj_H_mult_Cj")
        self.assertEqual(lindbladian.terms[9][2]["node3"+bra_suff], "Dj_mult_Dj_T")
        # Testing Dictionaries
        # conversion dictionary
        self.assertEqual(len(lindbladian.conversion_dictionary), 24)
        corr_keys = ["A", "B", "C", "D", "Aj", "Bj", "Cj", "Dj",
                     "A_T", "B_T", "C_T", "Aj_conj", "Cj_conj", "Dj_conj",
                     "Aj_H_mult_Aj", "Aj_H_mult_Aj_T", "Cj_H_mult_Cj",
                     "Dj_mult_Dj", "Dj_mult_Dj_T"]
        for key in corr_keys:
            self.assertIn(key, lindbladian.conversion_dictionary)
        # coeffs mapping
        boundled_map = copy(coeff_map)
        boundled_map.update({key + "*j": value*1j
                             for key, value in jump_coeff_mapping.items()})
        self.assertEqual(lindbladian.coeffs_mapping, boundled_map)

class TestAgainstExact(TestCase):
    """
    Tests the construction methods against the matrix construction.
    """

    def setUp(self):
        self.num_qubits = 3
        bond_dim = 1
        local_tensor = zeros((bond_dim,2),
                             dtype=complex)
        local_state = ket_i(0,2)
        local_tensor[0,:] = local_state
        ttns = generate_binary_ttns(self.num_qubits,
                                    bond_dim,
                                    local_tensor,
                                    phys_prefix="qubit"
                                    )
        self.ttndo = from_ttns(ttns)

    def _full_tensor(self,
                     lindbladian: Hamiltonian):
        """
        Computes the full tensor of the lindbladian.
        """
        ttno = TreeTensorNetworkOperator.from_hamiltonian(lindbladian,
                                                          self.ttndo
                                                          )
        tensor, order = ttno.as_matrix()
        return tensor, order

    def test_hamiltonian_only(self):
        ham_ops = (TensorProduct({"qubit0": "A", "qubit1": "B"}),
                   TensorProduct({"qubit1": "C", "qubit2": "D"}))
        factors = (Fraction(1, 2), Fraction(1, 5))
        symb_factors = ("g", "j")
        terms = tuple(zip(factors, symb_factors, ham_ops))
        conv_dict = {"A": asarray([[1, 2], [3, 4]], dtype=complex),
                     "B": asarray([[5, 6], [7, 8]], dtype=complex),
                     "C": asarray([[9, 20], [11, 12]], dtype=complex),
                     "D": asarray([[13, 15], [15, 17]], dtype=complex)} # symmetric
        coeff_map = {"1": 1, "g": 2j, "j": 6.0}
        hamiltonian = Hamiltonian(terms,
                                  conversion_dictionary=conv_dict,
                                  coeffs_mapping=coeff_map)
        hamiltonian.include_identities([1,2])
        lindbladian = generate_lindbladian(hamiltonian, [], {}, {})
        found, order = self._full_tensor(lindbladian)
        # Generate exact solution
        # The qubits in the contracted tree have order q1, q2, q0
        term1 = kron(conv_dict["B"], eye(2))
        term1 = kron(term1, conv_dict["A"])
        term1 = factors[0] * coeff_map["g"] * term1
        term2 = kron(conv_dict["C"], conv_dict["D"])
        term2 = kron(term2, eye(2))
        term2 = factors[1] * coeff_map["j"] * term2
        ham = term1 + term2
        exact = exact_lindbladian(ham, [])
        # Testing
        self.assertTrue(allclose(exact,found))

    def test_random_jump_operator(self):
        """
        Test the generation of a lindbladian with a random jump operator only.
        """
        jump_op = (Fraction(1,2),
                   "gjump",
                   TensorProduct({"qubit0": "A",
                                  "qubit1": "B"}))
        jump_dict = {"A": asarray([[1, 2j], [3, 4]], dtype=complex),
                     "B": asarray([[5, 6], [7j, 8j]], dtype=complex)}
        jump_coeff_map = {"gjump": 3.0}
        ham = Hamiltonian()
        ham.include_identities([1,2])
        lindbladian = generate_lindbladian(ham,
                                           [jump_op],
                                           jump_dict,
                                           jump_coeff_map)
        found, order = self._full_tensor(lindbladian)
        # Generate exact solution
        # The qubits in the contracted tree have order q1, q2, q0
        term1 = kron(jump_dict["B"],eye(2))
        term1 = kron(term1,jump_dict["A"])
        term1 = term1
        coeff = jump_op[0] * jump_coeff_map["gjump"]
        exact = exact_lindbladian(zeros_like(term1),
                                  [(coeff,term1)])
        # Testing
        print(order)
        self.assertTrue(allclose(exact,found))



if __name__ == "__main__":
    main_unit()
