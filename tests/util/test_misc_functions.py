import unittest
import numpy as np
from copy import deepcopy

from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
from pytreenet.util.misc_functions import (linear_combination,
                                           add,
                                           orthogonalise_gram_schmidt,
                                           orthogonalise_cholesky)

np.random.seed(44)
class TestMiscFunctionsSmall(unittest.TestCase):
    def setUp(self):
        self.ttns_1, self.ttno_1 = small_ttns_and_ttno()
        self.ttns_2, _ = small_ttns_and_ttno()
        self.ttns_3, _ = small_ttns_and_ttno()
        self.ttns_1.canonical_form(self.ttns_1.root_id)
        self.ttns_2.canonical_form(self.ttns_2.root_id)
        self.ttns_3.canonical_form(self.ttns_3.root_id)
        self.ttns_1.normalize()
        self.ttns_2.normalize()
        self.ttns_3.normalize()

    def test_add(self):
        c1 = 6.
        c2 = 7.
        # Now perform 1*ttn1+ 2*ttn2
        res = add(deepcopy(self.ttns_1), deepcopy(self.ttns_2), c1, c2)
        res.canonical_form(res.root_id)
        res.normalize()

        ovp = self.ttns_1.scalar_product(self.ttns_2)
        exact_ratio = abs((c1+c2*ovp)/(c2 + c1*ovp))
        result_ratio = abs(res.scalar_product(self.ttns_1)) / \
            abs(res.scalar_product(self.ttns_2))
        self.assertAlmostEqual(exact_ratio, result_ratio)

    def test_linear_combination(self):
        c1 = 6.
        c2 = 7.
        # Now perform 1*ttn1+ 2*ttn2
        res = linear_combination(
            [deepcopy(self.ttns_1), deepcopy(self.ttns_2)], [c1, c2], 10, dtype=np.complex128)
        res.canonical_form(res.root_id)
        res.normalize()

        ovp = self.ttns_1.scalar_product(self.ttns_2)
        exact_ratio = abs((c1+c2*ovp)/(c2 + c1*ovp))
        result_ratio = abs(res.scalar_product(self.ttns_1)) / \
            abs(res.scalar_product(self.ttns_2))
        self.assertAlmostEqual(exact_ratio, result_ratio)

    def test_orthogonalise_gram_schmidt(self):
        ttns_4, _ = small_ttns_and_ttno()
        ttns_4.canonical_form(ttns_4.root_id)
        ttns_4.normalize()
        state_list = [deepcopy(self.ttns_1), deepcopy(
            self.ttns_2), deepcopy(self.ttns_3), deepcopy(ttns_4)]
        res = orthogonalise_gram_schmidt(state_list, 8, 20)

        for i in range(len(res)):
            self.assertAlmostEqual(res[i].scalar_product(res[i]), 1., places=0)
            for j in range(i):
                self.assertAlmostEqual(res[i].scalar_product(res[j]), 0., places=0)
                
    def test_orthogonalise_cholesky(self):
        ttns_4, _ = small_ttns_and_ttno()
        ttns_4.canonical_form(ttns_4.root_id)
        ttns_4.normalize()
        res = orthogonalise_cholesky([deepcopy(self.ttns_1), deepcopy(
            self.ttns_2), deepcopy(self.ttns_3), deepcopy(ttns_4)], 10, 10)

        for i in range(len(res)):
            self.assertAlmostEqual(res[i].scalar_product(res[i]), 1.)
            for j in range(i):
                self.assertAlmostEqual(res[i].scalar_product(res[j]), 0.)


class TestMiscFunctionsBig(unittest.TestCase):
    def setUp(self):
        self.ttns_1, self.ttno_1 = big_ttns_and_ttno()
        self.ttns_2, _ = big_ttns_and_ttno()
        self.ttns_3, _ = big_ttns_and_ttno()

        self.ttns_1.canonical_form(self.ttns_1.root_id)
        self.ttns_2.canonical_form(self.ttns_2.root_id)
        self.ttns_3.canonical_form(self.ttns_3.root_id)
        self.ttns_1.normalize()
        self.ttns_2.normalize()
        self.ttns_3.normalize()

    def test_add(self):
        c1 = 6.
        c2 = 7.
        # Now perform 1*ttn1+ 2*ttn2
        res = add(deepcopy(self.ttns_1), deepcopy(self.ttns_2), c1, c2)
        res.canonical_form(res.root_id)
        res.normalize()

        ovp = self.ttns_1.scalar_product(self.ttns_2)
        exact_ratio = abs((c1+c2*ovp)/(c2 + c1*ovp))
        result_ratio = abs(res.scalar_product(self.ttns_1)) / \
            abs(res.scalar_product(self.ttns_2))
        self.assertAlmostEqual(exact_ratio, result_ratio)

    def test_linear_combination(self):
        c1 = 1.
        c2 = 2.
        c3 = 3.
        # Now perform 1*ttn1+ 2*ttn2
        res = linear_combination([deepcopy(self.ttns_1),
                                  deepcopy(self.ttns_2),
                                  deepcopy(self.ttns_3)],
                                 [c1, c2, c3],
                                 20, dtype=np.complex128)
        res.canonical_form(res.root_id)
        res.normalize()

        res_add = add(deepcopy(self.ttns_1), deepcopy(self.ttns_2), c1, c2)
        res_add = add(res_add, deepcopy(self.ttns_3), 1, c3)
        res_add.canonical_form(res_add.root_id)
        res_add.normalize()
        # since the variational fitting is not exact, we can only get 90% accuracy
        self.assertAlmostEqual(res.scalar_product(res_add), 1., places=0)

    def test_orthogonalise_gram_schmidt(self):
        ttns_4, _ = big_ttns_and_ttno()
        ttns_4.canonical_form(ttns_4.root_id)
        ttns_4.normalize()
        state_list = [deepcopy(self.ttns_1), deepcopy(
            self.ttns_2), deepcopy(self.ttns_3), deepcopy(ttns_4)]
        res = orthogonalise_gram_schmidt(state_list, 8, 20)

        for i in range(len(res)):
            self.assertAlmostEqual(res[i].scalar_product(res[i]), 1.)
            for j in range(i):
                self.assertAlmostEqual(
                    res[i].scalar_product(res[j]), 0., places=0)

if __name__ == "__main__":
    unittest.main()
