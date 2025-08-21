"""
This module contains unit tests for the depreceiated model generation functions.
Use the model classes instead.
"""
from unittest import TestCase, main as unitmain

from pytreenet.operators.models.two_site_model import (BoseHubbardModel,
                                                        FlippedIsingModel,
                                                        IsingModel)
from pytreenet.operators.models import (ising_model,
                                        flipped_ising_model,
                                        ising_model_2D,
                                        flipped_ising_model_2D,
                                        bose_hubbard_model)

class TestIsingModel(TestCase):
    """
    Test the generation of the ising model and associated utility
    functions.
    """

    def test_ising_for_list(self):
        """
        Test the creation of the ising model for a given list of nearest
        neighbours.
        """
        test_list = [("A", "B"), ("B", "C"), ("C", "D")]
        hamiltonian = ising_model(test_list, 3.0, factor=2.0)
        model = IsingModel(factor=2.0,
                           ext_magn=3.0)
        correct = model.generate_hamiltonian(test_list)
        self.assertEqual(hamiltonian.terms, correct.terms)

class TestFlippedIsingModel(TestCase):
    """
    Test the generation of the flipped ising model and associated utility
    functions.
    """

    def test_flipped_ising_for_list(self):
        """
        Test the creation of the flipped ising model for a given list of
        nearest neighbours.
        """
        test_list = [("A", "B"), ("B", "C"), ("C", "D")]
        hamiltonian = flipped_ising_model(test_list, 3.0, factor=2.0)
        model = FlippedIsingModel(factor=2.0,
                                  ext_magn=3.0)
        correct = model.generate_hamiltonian(test_list)
        self.assertEqual(hamiltonian.terms, correct.terms)

class Test2DModels(TestCase):
    """
    Test the generation of 2D ising models and associated utility functions.
    """

    def test_ising_2d(self):
        """
        Tests the generation of a 2D ising model for a small grid.
        """
        factor = 1.3
        ext_magn = 0.5
        is_ham = ising_model_2D(("node",2,3),
                                ext_magn,
                                coupling=factor)
        model = IsingModel(factor=factor,
                           ext_magn=ext_magn)
        correct = model.generate_2d_model(2,3,site_ids="node")
        self.assertEqual(is_ham.terms, correct.terms)

    def test_flipped_ising_2d(self):
        """
        Tests the generation of a 2D ising model for a small grid.
        """
        factor = 1.3
        ext_magn = 0.5
        is_ham = flipped_ising_model_2D(("node",2,3),
                                        ext_magn,
                                        coupling=factor)
        model = FlippedIsingModel(factor=factor,
                                  ext_magn=ext_magn)
        correct = model.generate_2d_model(2,3,site_ids="node")
        self.assertEqual(is_ham.terms, correct.terms)

class TestBoseHubbardModel(TestCase):
    """
    Tests the generation of the Bose-Hubbard model.
    """

    def setUp(self) -> None:
        self.pairs = [("node0_0", "node0_1"),
                      ("node0_0", "node1_0"),
                      ("node1_0", "node1_1"),
                      ("node0_1", "node1_1"),
                      ("node0_1", "node0_2"),
                      ("node1_1", "node1_2"),
                      ("node0_2", "node1_2")]

    def test_all_factors_zero(self):
        """
        Tests the generation of the Bose-Hubbard model with all factors set to
        zero.
        """
        found = bose_hubbard_model(self.pairs,
                                   hopping=0.0,
                                   on_site_int=0.0,
                                   chem_pot=0.0,)
        model = BoseHubbardModel(hopping=0.0,
                                 on_site_int=0.0,
                                    chem_pot=0.0)
        correct = model.generate_hamiltonian(self.pairs)
        self.assertEqual(found.terms, correct.terms)

    def test_only_hopping(self):
        """
        Tests the generation of the Bose-Hubbard model with only the hopping
        factor set to a non-zero value.
        """
        found = bose_hubbard_model(self.pairs,
                                   hopping=1.0,
                                   on_site_int=0.0,
                                   chem_pot=0.0,)
        model = BoseHubbardModel(hopping=1.0,
                                 on_site_int=0.0,
                                    chem_pot=0.0)
        correct = model.generate_hamiltonian(self.pairs)
        self.assertEqual(found.terms, correct.terms)

    def test_only_hopping_all_combs(self):
        """
        Ensures that for the hopping factor, both combinations of
        annihilation and creation operators are generated.
        """
        # Very small example
        node_ids = [("node0","node1")]
        found = bose_hubbard_model(node_ids,
                                   hopping=1.0,
                                   on_site_int=0.0,
                                   chem_pot=0.0)
        model = BoseHubbardModel(hopping=1.0,
                                 on_site_int=0.0,
                                 chem_pot=0.0)
        correct = model.generate_hamiltonian(node_ids)
        self.assertEqual(correct.terms, found.terms)

    def test_only_on_site_int(self):
        """
        Test the generation of the Bose-Hubbard model with only the on-site
        interaction factor set to a non-zero value.
        """
        found = bose_hubbard_model(self.pairs,
                                   hopping=0.0,
                                   on_site_int=1.0,
                                   chem_pot=0.0)
        model = BoseHubbardModel(hopping=0.0,
                                    on_site_int=1.0,
                                    chem_pot=0.0)
        correct = model.generate_hamiltonian(self.pairs)
        self.assertEqual(found.terms, correct.terms)

    def test_only_chem_pot(self):
        """
        Test the generation of the Bose-Hubbard model with only the chemical
        potential factor set to non-zero.
        """
        found = bose_hubbard_model(self.pairs,
                                   hopping=0.0,
                                   on_site_int=0.0,
                                   chem_pot=1.0)
        model = BoseHubbardModel(hopping=0.0,
                                 on_site_int=0.0,
                                 chem_pot=1.0)
        correct = model.generate_hamiltonian(self.pairs)
        self.assertEqual(found.terms, correct.terms)

if __name__ == '__main__':
    unitmain()
