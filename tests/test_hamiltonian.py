import unittest

import pytreenet as ptn

class TestHamiltonian(unittest.TestCase):
    
    def setUp(self):
        paulis = list(ptn.pauli_matrices())
        num_sites = 8
        
        self.sites = ["site" + str(integer) for integer in range(0,num_sites)]
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

if __name__ == "__main__":
    unittest.main()