import unittest

from copy import deepcopy

import numpy as np
import numpy.testing as npt

import pytreenet as ptn
from pytreenet.random.random_ttns_and_ttno import (small_ttns_and_ttno,
                                                   big_ttns_and_ttno)
from pytreenet.random.random_mps_and_mpo import random_mps_and_mpo
from pytreenet.random.random_special_ttns import (random_ftps,
                                                  random_binary_state,
                                                  random_mps,
                                                  random_tstar_state)
from pytreenet.random.random_special_ttno import (random_ftpo,
                                                  random_binary_operator,
                                                  random_mpo,
                                                  random_tstar_operator)
from pytreenet.util.tensor_splitting import SVDParameters
from pytreenet.ttns.ttns_ttno.zipup import zipup

class TestVariationalFittingSmall(unittest.TestCase):
    """
    Tests the variational fitting for small TTNS and TTNO.
    """

    def test_one_site_als(self):
        """
        Tests the variational fitting for a single TTNO applied to a TTNS.
        """
        # We need a ttns to work with
        ttns, ttno = small_ttns_and_ttno()
        ttns.canonical_form(ttns.root_id)
        ttns = ttns
        num_sweeps = 10
        max_iter = 50
        svd_params = ptn.SVDParameters(5, 1e-10, 1e-10)
        state_y = deepcopy(ttns)
        state_y.pad_bond_dimensions(6)
        varfit_one_site = ptn.VariationalFitting([ttno],
                                                 [ttns],
                                                 state_y,
                                                 num_sweeps,
                                                 max_iter,
                                                 svd_params,
                                                 "one-site",
                                                 dtype=np.complex128)

        ref_state, state_order = deepcopy(ttns).to_vector()
        ref_op, op_order = deepcopy(ttno).as_matrix()
        ref = ref_op @ ref_state
        _ = varfit_one_site.run()
        state_y = varfit_one_site.y
        found, found_order = state_y.to_vector()
        npt.assert_allclose(found, ref)

    def test_ttns_ttno_sum(self):
        """
        Tests the variational fitting for a sum of TTNOs applied to TTNSs.
        """
        ttns1, ttno1 = small_ttns_and_ttno()
        ttns2, ttno2 = small_ttns_and_ttno()
        ttns1.canonical_form(ttns1.root_id)
        ttns2.canonical_form(ttns2.root_id)
        num_sweeps = 10
        max_iter = 50
        svd_params = ptn.SVDParameters(10, 1e-10, 1e-10)
        state_y = deepcopy(ttns1)
        state_y.pad_bond_dimensions(6)
        varfit_one_site = ptn.VariationalFitting([ttno1, ttno2],
                                                 [ttns1, ttns2],
                                                 state_y,
                                                 num_sweeps,
                                                 max_iter,
                                                 svd_params,
                                                 "one-site",
                                                 dtype=np.complex128)
        ref_state1, state_order1 = deepcopy(ttns1).to_vector()
        ref_op1, op_order1 = deepcopy(ttno1).as_matrix()
        ref_state2, state_order2 = deepcopy(ttns2).to_vector()
        ref_op2, op_order2 = deepcopy(ttno2).as_matrix()
        ref = ref_op1 @ ref_state1 + ref_op2 @ ref_state2
        _ = varfit_one_site.run()
        state_y = varfit_one_site.y
        found, found_order = state_y.to_vector()
        npt.assert_allclose(found, ref)

class TestVariationalFittingMPS(unittest.TestCase):
    """
    Tests the variational fitting for MPS and MPO.
    """

    def test_one_site_als(self):
        """
        Tests the variational fitting for a single TTNO applied to a TTNS.
        """
        # We need a ttns to work with
        ttns = random_mps(5, 2, 5, seed=123)
        ttno = random_mpo(5, 2, 5, seed=456)
        ttns.canonical_form(ttns.root_id)
        ttns = ttns
        num_sweeps = 10
        max_iter = 50
        svd_params = ptn.SVDParameters(10, 1e-10, 1e-10)
        state_y = deepcopy(ttns)
        state_y.pad_bond_dimensions(6)
        varfit_one_site = ptn.VariationalFitting([ttno],
                                                 [ttns],
                                                 state_y,
                                                 num_sweeps,
                                                 max_iter,
                                                 svd_params,
                                                 "one-site",
                                                 dtype=np.complex128)

        ref_state, state_order = deepcopy(ttns).to_vector()
        ref_op, op_order = deepcopy(ttno).as_matrix()
        ref = ref_op @ ref_state
        _ = varfit_one_site.run()
        state_y = varfit_one_site.y
        found, found_order = state_y.to_vector()
        npt.assert_allclose(found, ref)

    def test_ttns_ttno_sum(self):
        """
        Tests the variational fitting for a sum of TTNOs applied to TTNSs.
        """
        ttns1 = random_mps(5, 2, 5, seed=123)
        ttno1 = random_mpo(5, 2, 5, seed=456)
        ttns2 = random_mps(5, 2, 5, seed=123)
        ttno2 = random_mpo(5, 2, 5, seed=456)
        ttns1.canonical_form(ttns1.root_id)
        ttns2.canonical_form(ttns2.root_id)
        num_sweeps = 10
        max_iter = 50
        svd_params = ptn.SVDParameters(10, 1e-10, 1e-10)
        state_y = deepcopy(ttns1)
        state_y.pad_bond_dimensions(6)
        varfit_one_site = ptn.VariationalFitting([ttno1, ttno2],
                                                 [ttns1, ttns2],
                                                 state_y,
                                                 num_sweeps,
                                                 max_iter,
                                                 svd_params,
                                                 "one-site",
                                                 dtype=np.complex128)
        ref_state1, state_order1 = deepcopy(ttns1).to_vector()
        ref_op1, op_order1 = deepcopy(ttno1).as_matrix()
        ref_state2, state_order2 = deepcopy(ttns2).to_vector()
        ref_op2, op_order2 = deepcopy(ttno2).as_matrix()
        ref = ref_op1 @ ref_state1 + ref_op2 @ ref_state2
        _ = varfit_one_site.run()
        state_y = varfit_one_site.y
        found, found_order = state_y.to_vector()
        npt.assert_allclose(found, ref)

class TestVariationalFittingTStar(unittest.TestCase):
    """
    Tests the variational fitting for TStar states and operators.
    """

    def test_one_site_als(self):
        """
        Tests the variational fitting for a single TTNO applied to a TTNS.
        """
        # We need a ttns to work with
        ttns = random_tstar_state(3, 2, 10, seed=123)
        ttno = random_tstar_operator(3, 2, 10, seed=456)
        ttns.canonical_form(ttns.root_id)
        ttns = ttns
        num_sweeps = 10
        max_iter = 50
        svd_params = ptn.SVDParameters(10, 1e-10, 1e-10)
        state_y = deepcopy(ttns)
        state_y.pad_bond_dimensions(12)
        varfit_one_site = ptn.VariationalFitting([ttno],
                                                 [ttns],
                                                 state_y,
                                                 num_sweeps,
                                                 max_iter,
                                                 svd_params,
                                                 "one-site",
                                                 dtype=np.complex128)

        ref_state, state_order = deepcopy(ttns).to_vector()
        ref_op, op_order = deepcopy(ttno).as_matrix()
        ref = ref_op @ ref_state
        _ = varfit_one_site.run()
        state_y = varfit_one_site.y
        found, found_order = state_y.to_vector()
        npt.assert_allclose(found, ref)

    def test_ttns_ttno_sum(self):
        """
        Tests the variational fitting for a sum of TTNOs applied to TTNSs.
        """
        ttns1 = random_tstar_state(3, 2, 10, seed=123)
        ttno1 = random_tstar_operator(3, 2, 10, seed=456)
        ttns2 = random_tstar_state(3, 2, 10, seed=789)
        ttno2 = random_tstar_operator(3, 2, 10, seed=101112)
        ttns1.canonical_form(ttns1.root_id)
        ttns2.canonical_form(ttns2.root_id)
        num_sweeps = 10
        max_iter = 50
        svd_params = ptn.SVDParameters(10, 1e-10, 1e-10)
        state_y = deepcopy(ttns1)
        state_y.pad_bond_dimensions(22)
        varfit_one_site = ptn.VariationalFitting([ttno1, ttno2],
                                                 [ttns1, ttns2],
                                                 state_y,
                                                 num_sweeps,
                                                 max_iter,
                                                 svd_params,
                                                 "one-site",
                                                 dtype=np.complex128)
        ref_state1, state_order1 = deepcopy(ttns1).to_vector()
        ref_op1, op_order1 = deepcopy(ttno1).as_matrix()
        ref_state2, state_order2 = deepcopy(ttns2).to_vector()
        ref_op2, op_order2 = deepcopy(ttno2).as_matrix()
        ref = ref_op1 @ ref_state1 + ref_op2 @ ref_state2
        _ = varfit_one_site.run()
        state_y = varfit_one_site.y
        found, found_order = state_y.to_vector()
        npt.assert_allclose(found, ref)

class TestVariationalFittingBinary(unittest.TestCase):
    """
    Tests the variational fitting for binary tree states and operators.
    """

    def test_one_site_als(self):
        """
        Tests the variational fitting for a single TTNO applied to a TTNS.
        """
        # We need a ttns to work with
        ttns = random_binary_state(3, 2, 4, seed=123)
        ttno = random_binary_operator(3, 2, 4, seed=456)
        ttns.canonical_form(ttns.root_id)
        ttns = ttns
        num_sweeps = 10
        max_iter = 50
        svd_params = ptn.SVDParameters(10, 1e-10, 1e-10)
        state_y = deepcopy(ttns)
        state_y.pad_bond_dimensions(12)
        varfit_one_site = ptn.VariationalFitting([ttno],
                                                 [ttns],
                                                 state_y,
                                                 num_sweeps,
                                                 max_iter,
                                                 svd_params,
                                                 "one-site",
                                                 dtype=np.complex128)

        ref_state, state_order = deepcopy(ttns).to_vector()
        ref_op, op_order = deepcopy(ttno).as_matrix()
        ref = ref_op @ ref_state
        _ = varfit_one_site.run()
        state_y = varfit_one_site.y
        found, found_order = state_y.to_vector()
        npt.assert_allclose(found, ref)

    def test_ttns_ttno_sum(self):
        """
        Tests the variational fitting for a sum of TTNOs applied to TTNSs.
        """
        ttns1 = random_binary_state(3, 2, 4, seed=123)
        ttno1 = random_binary_operator(3, 2, 4, seed=456)
        ttns2 = random_binary_state(3, 2, 4, seed=789)
        ttno2 = random_binary_operator(3, 2, 4, seed=101112)
        ttns1.canonical_form(ttns1.root_id)
        ttns2.canonical_form(ttns2.root_id)
        num_sweeps = 10
        max_iter = 50
        svd_params = ptn.SVDParameters(10, 1e-10, 1e-10)
        state_y = deepcopy(ttns1)
        state_y.pad_bond_dimensions(22)
        varfit_one_site = ptn.VariationalFitting([ttno1, ttno2],
                                                 [ttns1, ttns2],
                                                 state_y,
                                                 num_sweeps,
                                                 max_iter,
                                                 svd_params,
                                                 "one-site",
                                                 dtype=np.complex128)
        ref_state1, state_order1 = deepcopy(ttns1).to_vector()
        ref_op1, op_order1 = deepcopy(ttno1).as_matrix()
        ref_state2, state_order2 = deepcopy(ttns2).to_vector()
        ref_op2, op_order2 = deepcopy(ttno2).as_matrix()
        ref = ref_op1 @ ref_state1 + ref_op2 @ ref_state2
        _ = varfit_one_site.run()
        state_y = varfit_one_site.y
        found, found_order = state_y.to_vector()
        npt.assert_allclose(found, ref)

if __name__ == "__main__":
    unittest.main()
