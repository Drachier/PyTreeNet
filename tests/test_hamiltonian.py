import unittest
from copy import deepcopy
from fractions import Fraction

from numpy import eye

from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.random import (random_small_ttns,
                              random_tensor_product,
                              random_tensor_node,
                              crandn)

from pytreenet.operators.hamiltonian import (Hamiltonian,
                                             deal_with_term_input)

class TestDealWithTermInput(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual([], deal_with_term_input())

    def test_single_tensor_product(self):
        tp = TensorProduct({"site1": "X", "site2": "Y"})
        found = deal_with_term_input(tp)
        self.assertEqual([(Fraction(1), "1", tp)], found)

    def test_multiple_tensor_products(self):
        tp1 = TensorProduct({"site1": "X", "site2": "Y"})
        tp2 = TensorProduct({"site2": "Z", "site3": "Y"})
        found = deal_with_term_input([tp1,tp2])
        self.assertEqual([(Fraction(1), "1", tp1), (Fraction(1), "1", tp2)],
                         found)

    def test_single_tuple(self):
        tp = TensorProduct({"site1": "X", "site2": "Y"})
        tup = (Fraction(1), "1", tp)
        tup_cor = deepcopy(tup)
        found = deal_with_term_input(tup)
        self.assertEqual([tup_cor], found)

    def test_multiple_tuples(self):
        tp1 = TensorProduct({"site1": "X", "site2": "Y"})
        tp2 = TensorProduct({"site2": "Z", "site3": "Y"})
        tup1 = (Fraction(1), "1", tp1)
        tup2 = (Fraction(1), "1", tp2)
        corr = [deepcopy(tup1), deepcopy(tup2)]
        found = deal_with_term_input([tup1, tup2])
        self.assertEqual(corr, found)

    def test_mixture(self):
        tp1 = TensorProduct({"site1": "X", "site2": "Y"})
        tp2 = TensorProduct({"site2": "Z", "site3": "Y"})
        tup2 = (Fraction(1), "1", tp2)
        found = deal_with_term_input([tp1, tup2])
        self.assertEqual([(Fraction(1), "1", tp1), tup2], found)

class TestHamiltonianInitialisation(unittest.TestCase):

    def test_empty_init(self):
        ham = Hamiltonian()
        self.assertEqual(0, len(ham.terms))
        self.assertEqual(0, len(ham.conversion_dictionary))

    def test_init_with_terms(self):
        terms = [TensorProduct({"site1": "X", "site2": "Y"}),
                 TensorProduct({"site2": "Z", "site3": "Y"})]
        ham = Hamiltonian(terms=deepcopy(terms))
        self.assertEqual((Fraction(1),"1",terms[0]), ham.terms[0])
        self.assertEqual((Fraction(1),"1",terms[1]), ham.terms[1])
        self.assertEqual(0, len(ham.conversion_dictionary))

    def test_init_with_conversion_dict(self):
        conv_dict = {"A": crandn((2,2)), "X": crandn((4,4))}
        ham = Hamiltonian(conversion_dictionary=conv_dict)
        self.assertEqual(0, len(ham.terms))
        self.assertEqual(conv_dict, ham.conversion_dictionary)

class TestHamiltonianSimpleTree(unittest.TestCase):

    def setUp(self):
        # Numeric Hamiltonian
        self.ref_ttn = random_small_ttns()
        self.terms_num = [random_tensor_product(self.ref_ttn, 2) for _ in range(0,3)]
        self.ham_num = Hamiltonian(terms=deepcopy(self.terms_num))

        # Symbolic Hamiltonian
        self.terms_symb = [TensorProduct({"root": "A", "c1": "B"}),
                           TensorProduct({"root": "A", "c2": "C"})]
        self.conversion_dict = {"A": crandn((2,2)),
                                "B": crandn((3,3)),
                                "C": crandn((4,4))}
        self.ham_symb = Hamiltonian(terms=deepcopy(self.terms_symb),
                                        conversion_dictionary=self.conversion_dict)

        # Empty Hamiltonian
        self.empty_ham = Hamiltonian()

    def test_add_term(self):
        term = TensorProduct({"c2": "C", "c1": "B"})
        self.ham_symb.add_term(deepcopy(term))
        self.assertEqual(3, len(self.ham_symb.terms))
        self.assertEqual(term, self.ham_symb.terms[-1][2])

    def test_add_terms(self):
        term = TensorProduct({"c2": "C", "c1": "B"})
        term2 = TensorProduct({"c2": "F", "c1": "B"})
        self.ham_symb.add_multiple_terms([term, term2])
        self.assertEqual(4, len(self.ham_symb.terms))

    def test_add_hamiltonian(self):
        self.ham_num.add_hamiltonian(self.ham_symb)
        self.assertEqual(5, len(self.ham_num.terms))
        # The other one should not be changed
        self.assertEqual(2, len(self.ham_symb.terms))

    def test_addition_term(self):
        term = TensorProduct({"c2": "C", "c1": "B"})
        self.ham_symb = self.ham_symb + deepcopy(term)
        self.assertEqual(3, len(self.ham_symb.terms))
        self.assertEqual(term, self.ham_symb.terms[-1][2])

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
        ttn = TreeTensorNetwork()
        node, tensor = random_tensor_node((2,3,4), identifier="False!")
        ttn.add_root(node, tensor)
        self.assertFalse(self.ham_num.is_compatible_with(ttn))

    def test_pad_with_identities(self):
        padded_ham = self.ham_symb.pad_with_identities(self.ref_ttn)
        # Every term should habe three factors
        for term in padded_ham.terms:
            self.assertEqual(len(self.ref_ttn.nodes), len(term[2]))
        # The correct symbols should be assigned
        self.assertEqual("I4", padded_ham.terms[0][2]["c2"])
        self.assertEqual("I3", padded_ham.terms[1][2]["c1"])
        # The old terms, shouldn't be changed
        for term in self.ham_symb.terms:
            self.assertEqual(2, len(term[2]))

class TestConvDictMethods(unittest.TestCase):

    def test_include_identities_integer(self):
        ham = Hamiltonian()
        ham.include_identities(4)
        self.assertIn("I4",ham.conversion_dictionary)
        self.assertTrue((ham.conversion_dictionary["I4"] == eye(4)).all())

    def test_include_identities_list(self):
        ham = Hamiltonian()
        dims = [1,2]
        ham.include_identities(dims)
        self.assertIn("I1", ham.conversion_dictionary)
        self.assertIn("I2", ham.conversion_dictionary)
        self.assertTrue((ham.conversion_dictionary["I1"] == eye(1)).all())
        self.assertTrue((ham.conversion_dictionary["I2"] == eye(2)).all())

if __name__ == "__main__":
    unittest.main()
