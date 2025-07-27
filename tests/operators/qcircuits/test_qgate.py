"""
This module contains unittest for the quantum gates.
"""
import unittest

from scipy.linalg import expm
import numpy as np

from pytreenet.random import random_tensor_node
from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.operators.qcircuits.qgate import (QGate,
                                                 InvolutarySingleSiteGate,
                                                 PhaseGate,
                                                 CNOTGate,
                                                 SWAPGate,
                                                 ToffoliGate)

def close(a, b):
    """
    Helper function to compare two matrices.
    Note that the atol has to be set, as some matrices have zero entries.
    """
    np.testing.assert_allclose(a, b, rtol=1e-15, atol=1e-15)

class TestSingleSiteGates(unittest.TestCase):
    """
    Unit tests for single-site quantum gates.

    We want all gates to perform the gate action after being evolved for a
    time of `1.0`.
    """
    def setUp(self) -> None:
        self.ttns = TreeTensorNetwork()
        self.qubit_id = "q0"
        node, tensor = random_tensor_node((2, ), identifier=self.qubit_id)
        self.ttns.add_root(node, tensor)

    def test_x_gate_abstract(self):
        """
        Test the X gate in an abstract way.
        """
        x_gate = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X,
                                                    self.qubit_id)
        self.assertTrue(x_gate.acts_on(self.qubit_id))
        self.assertEqual(x_gate.symbol, QGate.PAULI_X.value)

    def test_x_gate(self):
        """
        Test the X gate.
        """
        x_gate = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, self.qubit_id)
        generator = x_gate.get_generator()
        matrix = generator.to_matrix(self.ttns)
        evolved_matrix = expm(-1j * matrix.operator)
        expected_matrix = x_gate.matrix()
        close(evolved_matrix, expected_matrix)

    def test_y_gate_abstract(self):
        """
        Test the Y gate in an abstract way.
        """
        y_gate = InvolutarySingleSiteGate.from_enum(QGate.PAULI_Y,
                                                    self.qubit_id)
        self.assertTrue(y_gate.acts_on(self.qubit_id))
        self.assertEqual(y_gate.symbol, QGate.PAULI_Y.value)

    def test_y_gate(self):
        """
        Test the Y gate.
        """
        y_gate = InvolutarySingleSiteGate.from_enum(QGate.PAULI_Y, self.qubit_id)
        generator = y_gate.get_generator()
        matrix = generator.to_matrix(self.ttns)
        evolved_matrix = expm(-1j * matrix.operator)
        expected_matrix = y_gate.matrix()
        close(evolved_matrix, expected_matrix)

    def test_z_gate_abstract(self):
        """
        Test the Z gate in an abstract way.
        """
        z_gate = InvolutarySingleSiteGate.from_enum(QGate.PAULI_Z,
                                                    self.qubit_id)
        self.assertTrue(z_gate.acts_on(self.qubit_id))
        self.assertEqual(z_gate.symbol, QGate.PAULI_Z.value)

    def test_z_gate(self):
        """
        Test the Z gate.
        """
        z_gate = InvolutarySingleSiteGate.from_enum(QGate.PAULI_Z, self.qubit_id)
        generator = z_gate.get_generator()
        matrix = generator.to_matrix(self.ttns)
        evolved_matrix = expm(-1j * matrix.operator)
        expected_matrix = z_gate.matrix()
        close(evolved_matrix, expected_matrix)

    def test_hadamard_gate_abstract(self):
        """
        Test the Hadamard gate in an abstract way.
        """
        h_gate = InvolutarySingleSiteGate.from_enum(QGate.HADAMARD, self.qubit_id)
        self.assertTrue(h_gate.acts_on(self.qubit_id))
        self.assertEqual(h_gate.symbol, QGate.HADAMARD.value)

    def test_hadamard_gate(self):
        """
        Test the Hadamard gate.
        """
        h_gate = InvolutarySingleSiteGate.from_enum(QGate.HADAMARD, self.qubit_id)
        generator = h_gate.get_generator()
        matrix = generator.to_matrix(self.ttns)
        evolved_matrix = expm(-1j * matrix.operator)
        expected_matrix = h_gate.matrix()
        close(evolved_matrix, expected_matrix)

    def test_phase_gate_abstract(self):
        """
        Test the Phase gate in an abstract way.
        """
        p_gate = PhaseGate(1, self.qubit_id)
        self.assertTrue(p_gate.acts_on(self.qubit_id))
        self.assertEqual(p_gate.symbol, QGate.PHASE.value + "1")

    def test_phase_gate_2pi(self):
        """
        Test the Phase gate with a phase of 2 * pi.
        """
        p_gate = PhaseGate(2, self.qubit_id)
        generator = p_gate.get_generator()
        matrix = generator.to_matrix(self.ttns)
        evolved_matrix = expm(-1j * matrix.operator)
        expected_matrix = np.eye(2, dtype=np.complex64)
        close(evolved_matrix, expected_matrix)

    def test_phase_gate_pi(self):
        """
        Test the Phase gate with a phase of pi.
        """
        p_gate = PhaseGate(1, self.qubit_id)
        generator = p_gate.get_generator()
        matrix = generator.to_matrix(self.ttns)
        evolved_matrix = expm(-1j * matrix.operator)
        expected_matrix = np.array([[1, 0], [0, -1]], dtype=np.complex64)
        close(evolved_matrix, expected_matrix)

    def test_phase_gate_half_pi(self):
        """
        Test the Phase gate with a phase of pi / 2.
        """
        p_gate = PhaseGate(0.5, self.qubit_id)
        generator = p_gate.get_generator()
        matrix = generator.to_matrix(self.ttns)
        evolved_matrix = expm(-1j * matrix.operator)
        expected_matrix = np.array([[1, 0], [0, 1j]], dtype=np.complex64)
        close(evolved_matrix, expected_matrix)

class TestTwoQubitGates(unittest.TestCase):
    """
    Unit tests for two-qubit quantum gates.
    """
    def setUp(self) -> None:
        self.ttns = TreeTensorNetwork()
        self.qubit_ids = ["q0", "q1"]
        node, tensor = random_tensor_node((2, 2), identifier=self.qubit_ids[0])
        self.ttns.add_root(node, tensor)
        node, tensor = random_tensor_node((2, 2), identifier=self.qubit_ids[1])
        self.ttns.add_child_to_parent(node, tensor, 0, self.qubit_ids[0], 0)

    def test_cnot_gate_abstract(self):
        """
        Test the CNOT gate in an abstract way.
        """
        cnot_gate = CNOTGate(self.qubit_ids[0], self.qubit_ids[1])
        self.assertTrue(cnot_gate.acts_on(self.qubit_ids))
        self.assertEqual(cnot_gate.symbol, QGate.CNOT.value)

    def test_cnot_gate(self):
        """
        Test the CNOT gate.
        """
        cnot_gate = CNOTGate(self.qubit_ids[0], self.qubit_ids[1])
        generator = cnot_gate.get_generator()
        matrix = generator.to_matrix(self.ttns)
        evolved_matrix = expm(-1j * matrix.operator)
        expected_matrix = cnot_gate.matrix()
        close(evolved_matrix, expected_matrix)

    def test_swap_gate_abstract(self):
        """
        Test the SWAP gate in an abstract way.
        """
        swap_gate = SWAPGate(self.qubit_ids[0], self.qubit_ids[1])
        self.assertTrue(swap_gate.acts_on(self.qubit_ids))
        self.assertEqual(swap_gate.symbol, QGate.SWAP.value)

    def test_swap_gate(self):
        """
        Test the SWAP gate.
        """
        swap_gate = SWAPGate(self.qubit_ids[0], self.qubit_ids[1])
        generator = swap_gate.get_generator()
        matrix = generator.to_matrix(self.ttns)
        evolved_matrix = expm(-1j * matrix.operator)
        expected_matrix = swap_gate.matrix()
        close(evolved_matrix, expected_matrix)

class TestThreeQubitGates(unittest.TestCase):
    """
    Unit tests for three-qubit quantum gates.
    """
    def setUp(self) -> None:
        self.ttns = TreeTensorNetwork()
        self.qubit_ids = ["q0", "q1", "q2"]
        node, tensor = random_tensor_node((2, 2), identifier=self.qubit_ids[0])
        self.ttns.add_root(node, tensor)
        node, tensor = random_tensor_node((2, 2, 2), identifier=self.qubit_ids[1])
        self.ttns.add_child_to_parent(node, tensor, 0, self.qubit_ids[0], 0)
        node, tensor = random_tensor_node((2, 2), identifier=self.qubit_ids[2])
        self.ttns.add_child_to_parent(node, tensor, 0, self.qubit_ids[1], 1)

    def test_toffoli_gate_abstract(self):
        """
        Test the Toffoli gate in an abstract way.
        """
        toffoli_gate = ToffoliGate(self.qubit_ids[0],
                                   self.qubit_ids[1],
                                   self.qubit_ids[2])
        self.assertTrue(toffoli_gate.acts_on(self.qubit_ids))
        self.assertEqual(toffoli_gate.symbol, QGate.TOFFOLI.value)

    def test_toffoli_gate(self):
        """
        Test the Toffoli gate.
        """
        toffoli_gate = ToffoliGate(self.qubit_ids[0],
                                   self.qubit_ids[1],
                                   self.qubit_ids[2])
        generator = toffoli_gate.get_generator()
        matrix = generator.to_matrix(self.ttns)
        evolved_matrix = expm(-1j * matrix.operator)
        expected_matrix = toffoli_gate.matrix()
        close(evolved_matrix, expected_matrix)
