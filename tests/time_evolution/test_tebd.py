import unittest

from copy import deepcopy

import numpy as np
from scipy.linalg import expm

import pytreenet as ptn
from pytreenet.random import (random_small_ttns,
                              random_tensor_product,
                              crandn)

class TestTEBDinit(unittest.TestCase):

    def setUp(self):
        # Create mock objects for initialization
        self.initial_state = random_small_ttns()
        self.tensor_products = [random_tensor_product(self.initial_state,2),
                                random_tensor_product(self.initial_state,2)]
        self.steps = [ptn.TrotterStep(tp,1) for tp in self.tensor_products]
        self.trotter_splitting = ptn.TrotterSplitting(self.steps)
        self.time_step_size = 0.1
        self.final_time = 1.0
        self.operators = {"operaator": random_tensor_product(self.initial_state, 1)}
        self.svd_parameters = ptn.SVDParameters(100, 1e-10, 1e-15)

    def test_valid_init(self):
        tebd = ptn.TEBD(self.initial_state, self.trotter_splitting,
                        self.time_step_size, self.final_time, self.operators,
                        self.svd_parameters)

        # Test correct instatiation
        self.assertEqual(tebd.initial_state, self.initial_state)
        self.assertEqual(tebd.trotter_splitting, self.trotter_splitting)
        self.assertEqual(tebd.time_step_size, self.time_step_size)
        self.assertEqual(tebd.final_time, self.final_time)
        self.assertEqual(tebd.operators, self.operators)
        self.assertEqual(tebd.svd_parameters,self.svd_parameters)

class TestTEBDsmall(unittest.TestCase):
    def setUp(self):
        # We need a ttns to work with
        self.ttns = random_small_ttns()

        # Time parameters
        self.time_step_size = 0.1
        self.final_time = 1.0

        # Trotter operators
        operator_r = ptn.TensorProduct({"root": crandn((2,2))})
        operator_r_c1 = ptn.TensorProduct({"root": crandn((2,2)),
                                           "c1": crandn((3,3))})
        operator_r_c2 = ptn.TensorProduct({"c2": crandn((4,4)), # Opposite order, to expected order
                                           "root": crandn((2,2))})
        self.tps = [operator_r, operator_r_c1, operator_r_c2]
        self.trotter_factor = 0.2
        self.steps = [ptn.TrotterStep(tp,0.2) for tp in self.tps]
        self.trotter = ptn.TrotterSplitting(self.steps)

        # Operators to measure
        num_ops = 3
        self.meas_operators = [random_tensor_product(self.ttns, 2) for i in range(num_ops)]

        # Deactivate Truncation
        self.svd_params = ptn.SVDParameters(float("inf"), float("-inf"), float("-inf"))

        # Initialise TEBD
        self.tebd = ptn.TEBD(self.ttns, self.trotter, self.time_step_size,
                             self.final_time, self.meas_operators,
                             svd_parameters=self.svd_params)

        # Initialise TEBD No truncation
        self.tebd_no_trunc = ptn.TEBD(self.ttns, self.trotter, self.time_step_size,
                             self.final_time, self.meas_operators,
                             svd_parameters=self.svd_params)

    def test_apply_one_trotter_step_single_site(self):
        reference_state = deepcopy(self.ttns)
        factor = -1j * self.time_step_size * self.trotter_factor
        time_evol_matrix = expm(factor * self.tps[0]["root"])

        # Check equality of both time-evolution operators
        self.assertTrue(np.allclose(time_evol_matrix, self.tebd.exponents[0].operator))

        self.tebd._apply_one_trotter_step_single_site(self.tebd.exponents[0])
        # The initial state should be untouched by this
        self.assertEqual(reference_state, self.tebd.initial_state)

        reference_state = reference_state.completely_contract_tree()[0]
        reference_state = reference_state.reshape(24)
        time_evol_matrix = np.kron(time_evol_matrix, np.eye(12))
        reference_state = time_evol_matrix @ reference_state

        found_state = self.tebd.state.completely_contract_tree(to_copy=True)[0]
        found_state = found_state.reshape(24)
        self.assertTrue(np.allclose(reference_state, found_state))

    def test_apply_one_trotter_step_two_site_order_as_in_nodes(self):
        reference_state = deepcopy(self.ttns)
        factor = -1j * self.time_step_size * self.trotter_factor
        tp = self.tps[1]
        operator = np.kron(tp["root"], tp["c1"])
        time_evol_matrix = expm(factor * operator)

        # Check equality of both time-evolution operators
        internal_op = self.tebd.exponents[1].operator.reshape(6,6)
        self.assertTrue(np.allclose(time_evol_matrix, internal_op))

        self.tebd_no_trunc._apply_one_trotter_step_two_site(self.tebd.exponents[1])
        # The initial state should be untouched by this
        self.assertEqual(reference_state, self.tebd_no_trunc.initial_state)

        reference_state = reference_state.completely_contract_tree()[0]
        reference_state = reference_state.reshape(24)
        time_evol_matrix = np.kron(time_evol_matrix, np.eye(4))
        reference_state = time_evol_matrix @ reference_state

        found_state = self.tebd_no_trunc.state.completely_contract_tree(to_copy=True)[0]
        found_state = found_state.reshape(24)
        self.assertTrue(np.allclose(reference_state, found_state))

    def test_apply_one_trotter_step_two_site_order_not_as_in_nodes(self):
        reference_state = deepcopy(self.ttns)
        factor = -1j * self.time_step_size * self.trotter_factor
        tp = self.tps[2]
        operator = np.kron(tp["c2"], tp["root"])
        time_evol_matrix = expm(factor * operator)

        # Check equality of both time-evolution operators
        internal_op = self.tebd.exponents[2].operator.reshape(8,8)
        self.assertTrue(np.allclose(time_evol_matrix, internal_op))

        self.tebd_no_trunc._apply_one_trotter_step_two_site(self.tebd_no_trunc.exponents[2])
        # The initial state should be untouched by this
        self.assertEqual(reference_state, self.tebd_no_trunc.initial_state)

        reference_state = reference_state.completely_contract_tree()[0]
        reference_state = reference_state.transpose((2,0,1)).reshape(24)
        time_evol_matrix = np.kron(time_evol_matrix, np.eye(3))
        reference_state = time_evol_matrix @ reference_state

        found_state = self.tebd_no_trunc.state.completely_contract_tree(to_copy=True)[0]
        found_state = found_state.transpose(1,0,2).reshape(24)
        self.assertTrue(np.allclose(reference_state, found_state))

    def test_apply_one_trotter_step__one_site(self):
        reference_state = deepcopy(self.ttns)
        factor = -1j * self.time_step_size * self.trotter_factor
        tp = self.tps[1]
        operator = np.kron(tp["root"], tp["c1"])
        time_evol_matrix = expm(factor * operator)

        # Check equality of both time-evolution operators
        internal_op = self.tebd.exponents[1].operator.reshape(6,6)
        self.assertTrue(np.allclose(time_evol_matrix, internal_op))

        self.tebd_no_trunc._apply_one_trotter_step(self.tebd_no_trunc.exponents[1])
        # The initial state should be untouched by this
        self.assertEqual(reference_state, self.tebd_no_trunc.initial_state)

        reference_state = reference_state.completely_contract_tree()[0]
        reference_state = reference_state.reshape(24)
        time_evol_matrix = np.kron(time_evol_matrix, np.eye(4))
        reference_state = time_evol_matrix @ reference_state

        found_state = self.tebd_no_trunc.state.completely_contract_tree(to_copy=True)[0]
        found_state = found_state.reshape(24)
        self.assertTrue(np.allclose(reference_state, found_state))

    def test_apply_one_trotter_step__two_site(self):
        reference_state = deepcopy(self.ttns)
        factor = -1j * self.time_step_size * self.trotter_factor
        tp = self.tps[2]
        operator = np.kron(tp["c2"], tp["root"])
        time_evol_matrix = expm(factor * operator)

        # Check equality of both time-evolution operators
        internal_op = self.tebd.exponents[2].operator.reshape(8,8)
        self.assertTrue(np.allclose(time_evol_matrix, internal_op))

        self.tebd_no_trunc._apply_one_trotter_step(self.tebd_no_trunc.exponents[2])
        # The initial state should be untouched by this
        self.assertEqual(reference_state, self.tebd_no_trunc.initial_state)

        reference_state = reference_state.completely_contract_tree()[0]
        reference_state = reference_state.transpose((2,0,1)).reshape(24)
        time_evol_matrix = np.kron(time_evol_matrix, np.eye(3))
        reference_state = time_evol_matrix @ reference_state

        found_state = self.tebd_no_trunc.state.completely_contract_tree(to_copy=True)[0]
        found_state = found_state.transpose(1,0,2).reshape(24)
        self.assertTrue(np.allclose(reference_state, found_state))

    def test_apply_one_trotter_step__three_sites(self):
        operator = ptn.NumericOperator(crandn((24,24)), ["root", "c1", "c2"])
        self.assertRaises(NotImplementedError, self.tebd_no_trunc._apply_one_trotter_step,
                          operator)

    def test_run_one_time_step(self):
        factor = -1j * self.time_step_size * self.trotter_factor
        operator1 = np.kron(expm(factor * self.tps[0]["root"]),
                            np.eye(3*4))
        operator2 = np.kron(expm(np.kron(factor * self.tps[1]["root"],
                                         self.tps[1]["c1"])),
                            np.eye(4))
        operator3 = expm(factor * np.kron(self.tps[2]["root"],
                                          np.kron(np.eye(3),
                                                  self.tps[2]["c2"])))
        total_operator = operator3 @ operator2 @ operator1
        # Checking the operator is correct
        exponentials1 = np.kron(self.tebd_no_trunc.exponents[0].operator.reshape(2,2),
                                np.eye(12))
        exponentials2 = np.kron(self.tebd_no_trunc.exponents[1].operator.reshape(6,6),
                                np.eye(4))
        exponentials3 = np.kron(self.tebd_no_trunc.exponents[2].operator.reshape(8,8),
                                np.eye(3))
        exponentials3 = exponentials3.reshape((4,2,3,4,2,3)).transpose(1,2,0,4,5,3).reshape(24,24)
        check_operator = exponentials3 @ exponentials2 @ exponentials1
        self.assertTrue(np.allclose(check_operator, total_operator))
        # Run reference computation
        reference_state = deepcopy(self.ttns)
        reference_state = reference_state.completely_contract_tree()[0].reshape(24)
        reference_state = total_operator @ reference_state

        # Run time_step
        self.tebd_no_trunc.run_one_time_step()
        found_state = self.tebd_no_trunc.state.completely_contract_tree()[0]
        found_state = found_state.transpose(0,2,1).reshape(24)
        # Compare the two time evolved states
        self.assertTrue(np.allclose(reference_state, found_state))

if __name__ == "__main__":
    unittest.main()
