from unittest import TestCase, main as unitmain

from numpy import allclose, tensordot, eye

import pytreenet as ptn
from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)

class TestZipupSmall(TestCase):
    def setUp(self):
        # We need a ttns to work with
        self.ttns, self.ttno = small_ttns_and_ttno()
        self.ttns.canonical_form(self.ttns.root_id)
        # Deactivate Truncation
        self.svd_params = ptn.SVDParameters(float("inf"), float("-inf"), float("-inf"))


    def test_zipup(self):
        ttns_zipup = ptn.zipup(self.ttno, self.ttns, self.svd_params)
        vec = (self.ttns.completely_contract_tree(to_copy=True)[0]).flatten()
        exact_ttns = self.ttno.as_matrix()[0] @ vec
        numerical_ttns = (ttns_zipup.completely_contract_tree(to_copy=True)[0]).flatten()
        self.assertTrue(allclose(exact_ttns, numerical_ttns))

class TestZipupBig(TestCase):
    def setUp(self):
        # We need a ttns to work with
        self.ttns, self.ttno = big_ttns_and_ttno()
        self.ttns.canonical_form(self.ttns.root_id)
        # Deactivate Truncation
        self.svd_params = ptn.SVDParameters(float("inf"), float("-inf"), float("-inf"))


    def test_zipup(self):
        ttns_zipup = ptn.zipup(self.ttno, self.ttns, self.svd_params)
        vec = (self.ttns.completely_contract_tree(to_copy=True)[0]).flatten()
        exact_ttns = self.ttno.as_matrix()[0] @ vec
        numerical_ttns = (ttns_zipup.completely_contract_tree(to_copy=True)[0]).flatten()
        self.assertTrue(allclose(exact_ttns, numerical_ttns))

if __name__ == "__main__":
    unitmain()
        