import unittest
from copy import deepcopy

import pytreenet as ptn
from pytreenet.random import (random_small_ttns,
                              random_tensor_product,
                              random_tensor_node,
                              crandn)

class TestHamiltonianInitialisation(unittest.TestCase):

    def test_empty_init(self):
        ham = ptn.Hamiltonian()
        self.assertEqual(0, len(ham.terms))
        self.assertEqual(0, len(ham.conversion_dictionary))

    def test_init_with_terms(self):
        terms = [ptn.TensorProduct({"site1": "X", "site2": "Y"}),
                 ptn.TensorProduct({"site2": "Z", "site3": "Y"})]
        ham = ptn.Hamiltonian(terms=terms)
        self.assertEqual(terms[0], ham.terms[0][2])
        self.assertEqual(terms[1], ham.terms[1][2])
        self.assertEqual(0, len(ham.conversion_dictionary))

    def test_init_with_conversion_dict(self):
        conv_dict = {"A": crandn((2,2)), "X": crandn((4,4))}
        ham = ptn.Hamiltonian(conversion_dictionary=conv_dict)
        self.assertEqual(0, len(ham.terms))
        self.assertEqual(conv_dict, ham.conversion_dictionary)

class TestHamiltonianSimpleTree(unittest.TestCase):

    def setUp(self):
        # Numeric Hamiltonian
        self.ref_ttn = random_small_ttns()
        self.terms_num = [random_tensor_product(self.ref_ttn, 2) for _ in range(0,3)]
        self.ham_num = ptn.Hamiltonian(terms=deepcopy(self.terms_num))

        # Symbolic Hamiltonian
        self.terms_symb = [ptn.TensorProduct({"root": "A", "c1": "B"}),
                           ptn.TensorProduct({"root": "A", "c2": "C"})]
        self.conversion_dict = {"A": crandn((2,2)),
                                "B": crandn((3,3)),
                                "C": crandn((4,4))}
        self.ham_symb = ptn.Hamiltonian(terms=deepcopy(self.terms_symb),
                                        conversion_dictionary=self.conversion_dict)

        # Empty Hamiltonian
        self.empty_ham = ptn.Hamiltonian()

    def test_add_term(self):
        term = ptn.TensorProduct({"c2": "C", "c1": "B"})
        self.ham_symb.add_term(deepcopy(term))
        self.assertEqual(3, len(self.ham_symb.terms))
        self.assertEqual(term, self.ham_symb.terms[-1])

    def test_add_terms(self):
        term = ptn.TensorProduct({"c2": "C", "c1": "B"})
        term2 = ptn.TensorProduct({"c2": "F", "c1": "B"})
        self.ham_symb.add_multiple_terms([term, term2])
        self.assertEqual(4, len(self.ham_symb.terms))

    def test_add_hamiltonian(self):
        self.ham_num.add_hamiltonian(self.ham_symb)
        self.assertEqual(5, len(self.ham_num.terms))
        # The other one should not be changed
        self.assertEqual(2, len(self.ham_symb.terms))

    def test_addition_term(self):
        term = ptn.TensorProduct({"c2": "C", "c1": "B"})
        self.ham_symb = self.ham_symb + deepcopy(term)
        self.assertEqual(3, len(self.ham_symb.terms))
        self.assertEqual(term, self.ham_symb.terms[-1])

    def test_addition_hamiltonian(self):
        self.ham_num = self.ham_num + self.ham_symb
        self.assertEqual(5, len(self.ham_num.terms))
        # The other one should not be changed
        self.assertEqual(2, len(self.ham_symb.terms))

    def test_wrong_addition(self):
        self.assertRaises(TypeError, self.ham_num.__add__, 4)

    def test_is_compatible_with_true(self):
        self.assertTrue(self.ham_num.is_compatible_with(self.ref_ttn))

    def test_is_compatible_with_false(self):
        ttn = ptn.TreeTensorNetwork()
        node, tensor = random_tensor_node((2,3,4), identifier="False!")
        ttn.add_root(node, tensor)
        self.assertFalse(self.ham_num.is_compatible_with(ttn))

    def test_pad_with_identities(self):
        padded_ham = self.ham_symb.pad_with_identities(self.ref_ttn)
        # Every term should habe three factors
        for term in padded_ham.terms:
            self.assertEqual(len(self.ref_ttn.nodes), len(term))
        # The correct symbols should be assigned
        self.assertEqual("I4", padded_ham.terms[0]["c2"])
        self.assertEqual("I3", padded_ham.terms[1]["c1"])
        # The old terms, shouldn't be changed
        for term in self.ham_symb.terms:
            self.assertEqual(2, len(term))

if __name__ == "__main__":
    unittest.main()
