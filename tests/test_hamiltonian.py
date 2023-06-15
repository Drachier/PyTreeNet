import unittest

import pytreenet as ptn


class TestHamiltonian(unittest.TestCase):

    def setUp(self):
        paulis = list(ptn.pauli_matrices())
        num_sites = 8

        self.sites = ["site" + str(integer) for integer in range(0, num_sites)]
        self.terms = ptn.random_terms(4, paulis, self.sites)

    def test_init(self):
        hamiltonian = ptn.Hamiltonian()

        self.assertEqual([], hamiltonian.terms)

        two_terms = self.terms[:2]
        hamiltonian = ptn.Hamiltonian(terms=two_terms)

        self.assertAlmostEqual(two_terms, hamiltonian.terms)

    def test_add_terms(self):

        two_terms = self.terms[:2]

        hamiltonian = ptn.Hamiltonian(terms=two_terms)

        two_more_terms = self.terms[2:4]

        hamiltonian.add_multiple_terms(two_more_terms)

        two_terms.extend(two_more_terms)
        correct_terms = two_terms

        self.assertEqual(correct_terms, hamiltonian.terms)

    def test_addition(self):
        two_terms = self.terms[:2]
        hamiltonian1 = ptn.Hamiltonian(terms=two_terms)

        two_more_terms = self.terms[2:4]
        hamiltonian2 = ptn.Hamiltonian(terms=two_more_terms)

        total_hamiltonian = hamiltonian1 + hamiltonian2

        two_terms.extend(two_more_terms)
        correct_terms = two_terms

        self.assertEqual(correct_terms, total_hamiltonian.terms)

    def test_to_tensor_very_simple(self):
        ttns = ptn.TreeTensorNetwork()
        node1, tensor1 = ptn.random_tensor_node((2, 3), identifier="site1")
        node2, tensor2 = ptn.random_tensor_node((2, 4), identifier="site2")

        ttns.add_root(node1, tensor1)
        ttns.add_child_to_parent(node2, tensor2, 0, "site1", 0)

        term1 = {"site1": "11", "site2": "21"}
        term2 = {"site1": "12", "site2": "22"}
        conversion_dict = {"11": ptn.crandn((3, 3)), "12": ptn.crandn((3, 3)),
                           "21": ptn.crandn((4, 4)), "22": ptn.crandn((4, 4))}

        hamiltonian = ptn.Hamiltonian(terms=[term1, term2],
                                      conversion_dictionary=conversion_dict)

        full_tensor = hamiltonian.to_tensor(ttns)

        self.assertEqual((3, 4, 3, 4), full_tensor.shape)


if __name__ == "__main__":
    unittest.main()
