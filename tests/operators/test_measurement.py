"""
Test the measurement class
"""
import unittest

import numpy as np
import numpy.testing as npt

from pytreenet.operators.measurment import (Measurement,
                                            MeasurementControlledUnitary,
                                            Outcome)
from pytreenet.special_ttn.mps import MatrixProductState
from pytreenet.operators.common_operators import (hadamard,
                                                  pauli_matrices)
from pytreenet.operators.tensorproduct import TensorProduct

class TestMeasurement(unittest.TestCase):

    def _node_id(self, index: int) -> str:
        return f"node_{index}"

    def setUp(self) -> None:
        self.state = MatrixProductState.constant_product_state(0,2,3,
                                                               node_prefix="node_",
                                                               root_site=1)

    def test_zero_single_node_measurement(self) -> None:
        meas = Measurement([self._node_id(0)])
        outcome, prob = meas.apply(self.state)
        ref_outcome = Outcome.from_iters([self._node_id(0)], [0])
        self.assertEqual(outcome, ref_outcome)
        npt.assert_almost_equal(prob, 1.0)

    def test_zero_two_node_measurement(self) -> None:
        meas = Measurement([self._node_id(0), self._node_id(1)])
        outcome, prob = meas.apply(self.state)
        ref_outcome = Outcome.from_iters([self._node_id(0), self._node_id(1)], [0, 0])
        self.assertEqual(outcome, ref_outcome)
        npt.assert_almost_equal(prob, 1.0)

    def test_zero_three_node_measurement(self) -> None:
        meas = Measurement([self._node_id(0), self._node_id(1), self._node_id(2)])
        outcome, prob = meas.apply(self.state)
        ref_outcome = Outcome.from_iters([self._node_id(0), self._node_id(1), self._node_id(2)], [0, 0, 0])
        self.assertEqual(outcome, ref_outcome)
        npt.assert_almost_equal(prob, 1.0)
    
    def test_superposition_measurement_zero(self) -> None:
        # We apply a Hadamard gate to the first node, which creates a superposition
        # of 0 and 1. 
        self.state.absorb_into_open_legs(self._node_id(0),
                                         hadamard())
        # The seed ensures zero is the outcome.
        meas = Measurement([self._node_id(0)],
                            seed=456245)
        outcome, prob = meas.apply(self.state)
        ref_outcome = Outcome.from_iters([self._node_id(0)], [0])
        self.assertEqual(outcome, ref_outcome)
        npt.assert_almost_equal(prob, 0.5)

    def test_superposition_measurement_one(self) -> None:
        # We apply a Hadamard gate to the first node, which creates a superposition
        # of 0 and 1. 
        self.state.absorb_into_open_legs(self._node_id(0),
                                            hadamard())
        # The seed ensures one is the outcome.
        meas = Measurement([self._node_id(0)],
                           seed=234)
        outcome, prob = meas.apply(self.state)
        ref_outcome = Outcome.from_iters([self._node_id(0)], [1])
        self.assertEqual(outcome, ref_outcome)
        npt.assert_almost_equal(prob, 0.5)

class TestMeasurementControlledUnitary(unittest.TestCase):

    def _node_id(self, index: int) -> str:
        return f"node_{index}"

    def setUp(self) -> None:
        self.num_nodes = 3
        self.state = MatrixProductState.constant_product_state(0,2,self.num_nodes,
                                                               node_prefix="node_",
                                                               root_site=1)

    def test_controlled_x(self) -> None:
        unitary = {Outcome.from_iters([self._node_id(0)], [0]):
                   TensorProduct({self._node_id(0):
                    pauli_matrices()[0]})}
        meas = MeasurementControlledUnitary([self._node_id(0)],
                                            unitary)
        meas.apply(self.state)
        # The state should have changed
        vec = self.state.completely_contract_tree(order=[self._node_id(i)
                                                         for i in range(self.num_nodes)])[0]
        vec = vec.reshape(-1)
        ref = np.zeros(2**self.num_nodes, dtype=complex)
        ref[4] = 1.0
        npt.assert_almost_equal(vec, ref)

    def test_otimes_mcus(self):
        """
        We test the kronecker product of two measurement controlled unitaries.
        """
        unitary_1 = {Outcome.from_iters([self._node_id(0)], [0]):
                   TensorProduct({self._node_id(0):
                    pauli_matrices()[0]})}
        unitary_2 = {Outcome.from_iters([self._node_id(1)], [0]):
                   TensorProduct({self._node_id(1):
                    pauli_matrices()[0]})}
        meas_1 = MeasurementControlledUnitary([self._node_id(0)],
                                            unitary_1)
        meas_2 = MeasurementControlledUnitary([self._node_id(1)],
                                            unitary_2)
        meas = meas_1.otimes(meas_2)
        meas.apply(self.state)
        # The state should have changed
        vec = self.state.completely_contract_tree(order=[self._node_id(i)
                                                         for i in range(self.num_nodes)])[0]
        vec = vec.reshape(-1)
        ref = np.zeros(2**self.num_nodes, dtype=complex)
        ref[6] = 1.0
        npt.assert_almost_equal(vec, ref)

if __name__ == "__main__":
    unittest.main()