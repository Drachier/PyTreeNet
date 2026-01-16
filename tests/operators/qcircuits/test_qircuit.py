"""
This module contains unittests for Quantum Circuit classes.
"""
import unittest
from copy import deepcopy

import numpy as np

from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.qcircuits.qcircuit import (QCLevel,
                                                    ProjectionLevel,
                                                    QCircuit,
                                                    CompiledQuantumCircuit)
from pytreenet.special_ttn.mps import MatrixProductState
from pytreenet.time_evolution.bug import BUG, BUGConfig
from pytreenet.time_evolution.time_evolution import TimeEvoMode
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.operators.common_operators import (ket_i,
                                                  superposition)
from pytreenet.core.node import Node
from pytreenet.operators.models import (local_magnetisation_from_topology,
                                        Topology)
from pytreenet.operators.qcircuits.qgate import (QGate,
                                                 InvolutarySingleSiteGate,
                                                 CNOTGate,
                                                 ProjectionOperation)
from pytreenet.operators.measurment import Measurement

class TestQCLevel(unittest.TestCase):
    """
    Test the `QCLevel` helper class.
    """

    def setUp(self):
        self.q_ids = ["q0", "q1", "q3"]

    def test_empty_level(self):
        """
        Test the empty QLevel.
        """
        qlevel = QCLevel()
        self.assertEqual(0, qlevel.num_gates())
        self.assertEqual(0, qlevel.width())

    def test_add_gate_sq(self):
        """
        Test adding a single qubit gate.
        """
        qlevel = QCLevel()
        gate = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, self.q_ids[0])
        qlevel.add_gate(gate)
        self.assertEqual(1, qlevel.num_gates())
        self.assertEqual(1, qlevel.width())
        self.assertTrue(qlevel.contains_gate(gate))

    def test_add_gate_tq(self):
        """
        Test adding a two qubit gate.
        """
        qlevel = QCLevel()
        gate = CNOTGate(self.q_ids[0], self.q_ids[1])
        qlevel.add_gate(gate)
        self.assertEqual(1, qlevel.num_gates())
        self.assertEqual(2, qlevel.width())
        self.assertTrue(qlevel.contains_gate(gate))

    def test_add_two_gates(self):
        """
        Tests adding more than one gate.
        """
        qlevel = QCLevel()
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, self.q_ids[0])
        gate2 = CNOTGate(self.q_ids[1], self.q_ids[2])
        qlevel.add_gate(gate1)
        qlevel.add_gate(gate2)
        self.assertEqual(2, qlevel.num_gates())
        self.assertEqual(3, qlevel.width())
        self.assertTrue(qlevel.contains_gate(gate1))
        self.assertTrue(qlevel.contains_gate(gate2))

    def test_add_overlapping_gates(self):
        """
        Tests adding more than one gate that overlap.
        """
        qlevel = QCLevel()
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, self.q_ids[0])
        gate2 = CNOTGate(self.q_ids[0], self.q_ids[2])
        qlevel.add_gate(gate1)
        self.assertRaises(ValueError, qlevel.add_gate, gate2)

    def test_otimes_levels(self):
        """
        Test otimes levels without overlap.
        """
        qlevel1 = QCLevel()
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, self.q_ids[0])
        qlevel1.add_gate(gate1)
        qlevel2 = QCLevel()
        gate2 = CNOTGate(self.q_ids[1], self.q_ids[2])
        qlevel2.add_gate(gate2)
        new_level = qlevel1.otimes_level(qlevel2)
        self.assertEqual(2, new_level.num_gates())
        self.assertEqual(3, new_level.width())
        self.assertTrue(new_level.contains_gate(gate1))
        self.assertTrue(new_level.contains_gate(gate2))

    def test_otimes_levels_inplace(self):
        """
        Test otimes levels without overlap.
        """
        qlevel1 = QCLevel()
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, self.q_ids[0])
        qlevel1.add_gate(gate1)
        qlevel2 = QCLevel()
        gate2 = CNOTGate(self.q_ids[1], self.q_ids[2])
        qlevel2.add_gate(gate2)
        qlevel1.otimes_level(qlevel2, inplace=True)
        self.assertEqual(2, qlevel1.num_gates())
        self.assertEqual(3, qlevel1.width())
        self.assertTrue(qlevel1.contains_gate(gate1))
        self.assertTrue(qlevel1.contains_gate(gate2))

    def test_otimes_levels_overlap(self):
        """
        Test otimes levels with overlap.
        """
        qlevel1 = QCLevel()
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, self.q_ids[0])
        qlevel1.add_gate(gate1)
        qlevel2 = QCLevel()
        gate2 = CNOTGate(self.q_ids[0], self.q_ids[2])
        qlevel2.add_gate(gate2)
        self.assertRaises(ValueError, qlevel1.otimes_level, qlevel2)

    def test_from_gates(self):
        """
        Test getting the QCLevel from a list of quantum gates.
        """
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, self.q_ids[0])
        gate2 = CNOTGate(self.q_ids[1], self.q_ids[2])
        gates = [gate1, gate2]
        qlevel = QCLevel.from_gates(gates)
        self.assertEqual(2, qlevel.num_gates())
        self.assertEqual(3, qlevel.width())
        self.assertTrue(qlevel.contains_gate(gate1))
        self.assertTrue(qlevel.contains_gate(gate2))

    def test_compile(self):
        """
        Test compiling a level to a Hamiltonian.
        """
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X,
                                                   self.q_ids[0])
        gate2 = CNOTGate(self.q_ids[1], self.q_ids[2])
        gates = [gate1, gate2]
        qlevel = QCLevel.from_gates(gates)
        ham = qlevel.compile()
        correct = Hamiltonian()
        correct.add_hamiltonian(gate1.get_generator())
        correct.add_hamiltonian(gate2.get_generator())
        self.assertEqual(ham, correct)
        self.assertTrue(ham.compare_dicts(correct))

class TestProjectionLevel(unittest.TestCase):
    """
    Test the `ProjectionLevel` helper class.
    """

    def setUp(self):
        self.q_ids = ["q0", "q1", "q3"]

    def test_add_projection(self):
        """
        Test adding a projection operation.
        """
        plevel = ProjectionLevel()
        projection = ProjectionOperation("A", self.q_ids[0], 1)
        plevel.add_projection(projection)
        self.assertEqual(1, plevel.num_operations())
        self.assertEqual(1, plevel.width())

    def test_otimes_level(self):
        """
        Test otimes levels without overlap.
        """
        plevel1 = ProjectionLevel()
        projection1 = ProjectionOperation("A", self.q_ids[0], 1)
        plevel1.add_projection(projection1)
        plevel2 = ProjectionLevel()
        projection2 = ProjectionOperation("A", self.q_ids[1], 0)
        plevel2.add_projection(projection2)
        new_level = plevel1.otimes_level(plevel2)
        self.assertEqual(2, new_level.num_operations())
        self.assertEqual(2, new_level.width())

    def test_otimes_level_error(self):
        """
        Test otimes levels with overlap.
        """
        plevel1 = ProjectionLevel()
        projection1 = ProjectionOperation("A", self.q_ids[0], 1)
        plevel1.add_projection(projection1)
        plevel2 = ProjectionLevel()
        projection2 = ProjectionOperation("A", self.q_ids[0], 0)
        plevel2.add_projection(projection2)
        self.assertRaises(ValueError, plevel1.otimes_level, plevel2)

    def test_compile(self):
        """
        Test compiling a level to a Measurement.
        """
        plevel = ProjectionLevel()
        projection1 = ProjectionOperation("A", self.q_ids[0], 1)
        plevel.add_projection(projection1)
        projection2 = ProjectionOperation("B", self.q_ids[1], 0)
        plevel.add_projection(projection2)
        measurement = plevel.compile()
        correct = Measurement.from_dict({self.q_ids[0]: 1,
                                         self.q_ids[1]: 0})
        self.assertEqual(measurement, correct)


class TestQCircuit(unittest.TestCase):
    """
    Test the `QCircuit` class.
    """

    def setUp(self):
        self.q_ids = ["q0", "q1", "q2", "q3"]

    def test_empty_circuit(self):
        """
        Test an empty quantum circuit.
        """
        circuit = QCircuit()
        self.assertEqual(0, circuit.depth())

    def test_add_level(self):
        """
        Test adding a level to the circuit.
        """
        circuit = QCircuit()
        level = QCLevel()
        circuit.add_level(level)
        self.assertEqual(1, circuit.depth())

    def test_add_gate_same_level(self):
        """
        Test adding two gates to the same level.
        """
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, self.q_ids[0])
        gate2 = CNOTGate(self.q_ids[1], self.q_ids[2])
        circuit = QCircuit()
        circuit.add_gate(gate1)
        circuit.add_gate(gate2)
        self.assertEqual(1, circuit.depth())
        self.assertEqual(3, circuit.width())
        self.assertTrue(circuit.contains_gate(gate1))
        self.assertTrue(circuit.contains_gate(gate2))

    def test_add_gate_different_levels(self):
        """
        Test adding gates to different levels.
        """
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X,
                                                   self.q_ids[0])
        gate2 = CNOTGate(self.q_ids[1], self.q_ids[2])
        circuit = QCircuit()
        circuit.add_gate(gate1)
        circuit.add_gate(gate2,
                         level_index=1)
        self.assertEqual(2, circuit.depth())
        self.assertEqual(3, circuit.width())
        self.assertTrue(circuit.contains_gate(gate1))
        self.assertTrue(circuit.contains_gate(gate2,
                                              level_index=1))

    def test_add_gates_different_levels_same_qubit(self):
        """
        Test adding gates to different levels that act on the same qubit.
        """
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X,
                                                   self.q_ids[0])
        gate2 = CNOTGate(self.q_ids[0], self.q_ids[1])
        circuit = QCircuit()
        circuit.add_gate(gate1)
        circuit.add_gate(gate2,
                         level_index=1)
        self.assertEqual(2, circuit.depth())
        self.assertEqual(2, circuit.width())
        self.assertTrue(circuit.contains_gate(gate1))
        self.assertTrue(circuit.contains_gate(gate2,
                                              level_index=1))

    def test_add_gate_invalid_level(self):
        """
        Test adding a gate to an invalid level.
        """
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X,
                                                   self.q_ids[0])
        circuit = QCircuit()
        self.assertRaises(IndexError,
                          circuit.add_gate,
                          gate1,
                          level_index=1)
        self.assertRaises(IndexError,
                          circuit.add_gate,
                          gate1,
                          level_index=-2)

    def test_add_qcircuit_attach(self):
        """
        Test adding a circuit to another circuit at the end of the first
        circuit.
        """
        circuit1 = QCircuit()
        circuit1.add_x(self.q_ids[0])
        circuit1.add_cnot(self.q_ids[0], self.q_ids[1], level_index=1)
        circuit2 = QCircuit()
        circuit2.add_y(self.q_ids[2])
        circuit2.add_cnot(self.q_ids[2], self.q_ids[1], level_index=1)
        circuit1.add_qcircuit(circuit2)
        self.assertEqual(4, circuit1.depth())
        self.assertEqual(3, circuit1.width())
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, self.q_ids[0])
        gate2 = CNOTGate(self.q_ids[0], self.q_ids[1])
        gate3 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_Y, self.q_ids[2])
        gate4 = CNOTGate(self.q_ids[2], self.q_ids[1])
        gates = [(gate1, 0), (gate2, 1), (gate3, 2), (gate4, 3)]
        for gate, level in gates:
            self.assertTrue(circuit1.contains_gate(gate, level_index=level))

    def test_add_circuit_beyond_end(self):
        """
        Test adding a circuit inside some circuit starting at a specific level
        and going beyond the end of the first circuit.
        """
        circuit1 = QCircuit()
        circuit1.add_x(self.q_ids[0])
        circuit1.add_cnot(self.q_ids[0], self.q_ids[1], level_index=1)
        circuit2 = QCircuit()
        circuit2.add_y(self.q_ids[2])
        circuit2.add_level()
        circuit2.add_cnot(self.q_ids[2], self.q_ids[1], level_index=2)
        circuit1.add_qcircuit(circuit2, level_index=0)
        self.assertEqual(3, circuit1.depth())
        self.assertEqual(3, circuit1.width())
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, self.q_ids[0])
        gate2 = CNOTGate(self.q_ids[0], self.q_ids[1])
        gate3 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_Y, self.q_ids[2])
        gate4 = CNOTGate(self.q_ids[2], self.q_ids[1])
        gates = [(gate1, 0), (gate2, 1), (gate3, 0), (gate4, 2)]
        for gate, level in gates:
            self.assertTrue(circuit1.contains_gate(gate,
                                                   level_index=level))

    def test_compile(self):
        """
        Test the compilation method for a QuantumCirctuit.
        """
        circuit1 = QCircuit()
        circuit1.add_x(self.q_ids[0])
        circuit1.add_cnot(self.q_ids[0], self.q_ids[1], level_index=1)
        circuit2 = QCircuit()
        circuit2.add_y(self.q_ids[2])
        circuit2.add_level()
        circuit2.add_cnot(self.q_ids[2], self.q_ids[1], level_index=2)
        circuit1.add_qcircuit(circuit2, level_index=0)
        comp_circuit = circuit1.compile()
        self.assertEqual(3, comp_circuit.depth())
        self.assertEqual(3, comp_circuit.width())
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, self.q_ids[0])
        gate2 = CNOTGate(self.q_ids[0], self.q_ids[1])
        gate3 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_Y, self.q_ids[2])
        gate4 = CNOTGate(self.q_ids[2], self.q_ids[1])
        ham0 = gate1.get_generator()
        ham0.add_hamiltonian(gate3.get_generator())
        ham1 = gate2.get_generator()
        ham2 = gate4.get_generator()
        hams = [ham0, ham1, ham2]
        correct_circuit = CompiledQuantumCircuit()
        for ham in hams:
            correct_circuit.add_level(level=ham)
        self.assertTrue(correct_circuit.close_to(comp_circuit))

    def test_compile_with_proj(self):
        """
        Test the compilation method for a QuantumCirctuit with a projection
        level.
        """
        circuit1 = QCircuit()
        circuit1.add_x(self.q_ids[0])
        circuit1.add_projection(self.q_ids[0], 1, level_index=1)
        circuit1.add_cnot(self.q_ids[0], self.q_ids[1], level_index=2)
        circuit1.add_y(self.q_ids[2], level_index=3)
        circuit1.add_projection(self.q_ids[1], 0, level_index=4)
        gate1 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_X, self.q_ids[0])
        gate2 = CNOTGate(self.q_ids[0], self.q_ids[1])
        gate3 = InvolutarySingleSiteGate.from_enum(QGate.PAULI_Y, self.q_ids[2])
        meas1 = Measurement.from_dict({self.q_ids[0]: 1})
        meas2 = Measurement.from_dict({self.q_ids[1]: 0})
        comp_circuit = circuit1.compile()
        correct_circuit = CompiledQuantumCircuit()
        correct_circuit.add_level(level=gate1.get_generator())
        correct_circuit.add_level(level=meas1)
        correct_circuit.add_level(level=gate2.get_generator())
        correct_circuit.add_level(level=gate3.get_generator())
        correct_circuit.add_level(level=meas2)
        self.assertTrue(correct_circuit.close_to(comp_circuit))


class TestCompiledQCircuit(unittest.TestCase):
    """
    Class to test a compiled quantum circuit.
    """

    def setUp(self):
        self.q_ids = ["q0", "q1", "q2", "q3"]
        circuit1 = QCircuit()
        circuit1.add_x(self.q_ids[0])
        circuit1.add_cnot(self.q_ids[0], self.q_ids[1], level_index=1)
        circuit2 = QCircuit()
        circuit2.add_y(self.q_ids[2])
        circuit2.add_level()
        circuit2.add_cnot(self.q_ids[2], self.q_ids[1], level_index=2)
        circuit1.add_qcircuit(circuit2, level_index=0)
        self.comp_circuit = circuit1.compile()

    def test_add_constant_hamiltonian(self):
        """
        Adds a constant Hamiltonian to every level of the circuit. 
        """
        z_gate = InvolutarySingleSiteGate.from_enum(QGate.PAULI_Z, "q1")
        ref = deepcopy(self.comp_circuit)
        self.comp_circuit.add_constant_hamiltonian(z_gate.get_generator())
        level0 = ref.get_level(0)
        level0.add_hamiltonian(z_gate.get_generator())
        self.assertEqual(level0, self.comp_circuit.get_level(0))
        level1 = ref.get_level(1)
        level1.add_hamiltonian(z_gate.get_generator())
        self.assertEqual(level1, self.comp_circuit.get_level(1))
        level2 = ref.get_level(2)
        level2.add_hamiltonian(z_gate.get_generator())
        self.assertEqual(level2, self.comp_circuit.get_level(2))

def q_name(n: int) -> str:
    q_prefix = "qubit"
    return f"{q_prefix}{n}"

def run_circuit(qc: QCircuit) -> np.ndarray:
    """
    Compiles and runs the time evolution for a single qubit quantum circuit.

    Args:
        qc (QCircuit): The quantum circuit to run.

    Returns:
        np.ndarray: The final state.
    """
    num_qb = qc.width()
    if num_qb == 1:
        mps = TreeTensorNetworkState()
        tensor = ket_i(0, 2)
        mps.add_root(Node(identifier=q_name(0)),
                      tensor)
    elif num_qb == 2:
        mps = MatrixProductState.constant_product_state(0,2,2,
                                                    node_prefix="qubit",
                                                    bond_dimensions=[2])
    comp_qc = qc.compile()
    ttno = comp_qc.to_time_dep_ttno(mps)
    ttno.measurement_renorm_threshold = 1e-10
    ops = local_magnetisation_from_topology(Topology.CHAIN, num_qb,
                                            site_prefix="qubit")
    final_time = qc.depth() * 1
    time_step_size = 0.01
    config = BUGConfig(time_evo_mode=TimeEvoMode.RK45,
                    time_dep=True)
    solver_options = {"rtol": 1e-6, "atol": 1e-6}
    bug = BUG(mps, ttno, time_step_size, final_time, ops,
            config=config,
            solver_options=solver_options)
    bug.run(evaluation_time="inf")
    return bug.state.completely_contract_tree()[0].flatten()

class TestCompiledQCircuitRunning(unittest.TestCase):
    """
    Runs the time evolution for the compiled circuit generator for some simple
    circuits to which we know the solution
    """

    def test_x_gate(self):
        """
        An x-gate should flip the state over time.
        """
        qc = QCircuit()
        qc.add_x(q_name(0))
        found = run_circuit(qc)
        correct = ket_i(1,2)
        np.testing.assert_allclose(found, correct,
                                   atol=1e-10)

    def test_two_x_gate(self):
        """
        Two x-gates, one after the other, should yield the initial state.
        """
        qc = QCircuit()
        qc.add_x(q_name(0))
        qc.add_x(q_name(0),level_index=1)
        found = run_circuit(qc)
        correct = ket_i(0,2)
        np.testing.assert_allclose(found, correct,
                                   atol=1e-10)

    def test_hadamard_gate(self):
        """
        The Hadamard gate should yield the equal superposition state.
        """
        qc = QCircuit()
        qc.add_hadamard(q_name(0))
        found = run_circuit(qc)
        correct = superposition()
        np.testing.assert_allclose(found, correct,
                                   atol=1e-10)

    def test_x_and_hadamard(self):
        """
        Test the circuit with first an x and then a Hadamard gate.
        This should yield the equal superposition with opposite sign.
        """
        qc = QCircuit()
        qc.add_x(q_name(0))
        qc.add_hadamard(q_name(0), level_index=1)
        found = run_circuit(qc)
        correct = superposition(rel_phase=1)
        np.testing.assert_allclose(found, correct,
                                   atol=1e-10)

    def test_hadamard_and_z(self):
        """
        Test the circuit with first a Hadamard and then a Z-gate.
        This should also yield the equal superposition with opposite sign.
        """
        qc = QCircuit()
        qc.add_hadamard(q_name(0))
        qc.add_z(q_name(0), level_index=1)
        found = run_circuit(qc)
        correct = superposition(rel_phase=1)
        np.testing.assert_allclose(found, correct,
                                   atol=1e-10)

    def test_x_and_cnot(self):
        """
        Test the two-qubit circuit where we first apply an X-gate on the first
        qubit and then a CNOT.
        This should yield the |11> state
        """
        qc = QCircuit()
        qc.add_x(q_name(0))
        qc.add_cnot(q_name(0), q_name(1), level_index=1)
        found = run_circuit(qc)
        correct = ket_i(3,4)
        np.testing.assert_allclose(found, correct,
                                   atol=1e-10)

    def test_hadamard_and_cnot(self):
        """
        Test the two-qubit circuit where we first apply a Hadamard on the first
        qubit and then a CNOT.
        This should yield the bell state (|00>+|11>) / sqrt(2).
        """
        qc = QCircuit()
        qc.add_hadamard(q_name(0))
        qc.add_cnot(q_name(0), q_name(1), level_index=1)
        found = run_circuit(qc)
        correct = ket_i(0,4)
        correct[3] = 1
        correct = correct / np.sqrt(2)
        np.testing.assert_allclose(found, correct,
                                   atol=1e-10)

    def test_hadamard_and_0_meas(self):
        """
        Test the circuit with a Hadamard gate followed by a measurement projecting
        onto the 0 state.
        This should yield the 0 state.
        """
        qc = QCircuit()
        qc.add_hadamard(q_name(0))
        qc.add_projection(q_name(0), 0, level_index=1)
        # We need the identity, as final measurements should be done manually usually
        qc.add_identity(q_name(0), level_index=2)
        found = run_circuit(qc)
        correct = ket_i(0,2)
        np.testing.assert_allclose(found, correct,
                                   atol=1e-10)

    def test_hadamard_and_1_meas(self):
        """
        Test the circuit with a Hadamard gate followed by a measurement projecting
        onto the 1 state.
        This should yield the 1 state.
        """
        qc = QCircuit()
        qc.add_x(q_name(0))
        qc.add_projection(q_name(0), 1, level_index=1)
        # We need the identity, as final measurements should be done manually usually
        qc.add_identity(q_name(0), level_index=2)
        found = run_circuit(qc)
        correct = ket_i(1,2)
        np.testing.assert_allclose(found, correct,
                                   atol=1e-10)

    def test_x_and_0_meas(self):
        """
        Test the circuit with a X-gate followed by a measurement projecting
        onto the 0 state.
        This should yield the 0 state.
        """
        qc = QCircuit()
        qc.add_x(q_name(0))
        qc.add_projection(q_name(0), 0, level_index=1)
        # We need the identity, as final measurements should be done manually usually
        qc.add_identity(q_name(0), level_index=2)
        self.assertRaises(ZeroDivisionError, run_circuit, qc)

    def test_x_and_1_meas(self):
        """
        Test the circuit with a X-gate followed by a measurement projecting
        onto the 1 state.
        This should yield the 1 state.
        """
        qc = QCircuit()
        qc.add_x(q_name(0))
        qc.add_projection(q_name(0), 1, level_index=1)
        # We need the identity, as final measurements should be done manually usually
        qc.add_identity(q_name(0), level_index=2)
        found = run_circuit(qc)
        correct = ket_i(1,2)
        np.testing.assert_allclose(found, correct,
                                   atol=1e-10)
        assert False

    def test_hadamard_cnot_and_meas_first_1(self):
        """
        Test the circuit with a Hadamard gate, a CNOT and a measurement
        projecting the first qubit onto the 1 state.
        This should yield the 1 state on both qubits.
        """
        qc = QCircuit()
        qc.add_hadamard(q_name(0))
        qc.add_cnot(q_name(0), q_name(1), level_index=1)
        qc.add_projection(q_name(0), 1, level_index=2)
        # We need the identity, as final measurements should be done manually usually
        qc.add_identity(q_name(0), level_index=3)
        qc.add_identity(q_name(1), level_index=3)
        found = run_circuit(qc)
        correct = ket_i(3,4)
        np.testing.assert_allclose(found, correct,
                                   atol=1e-10)

class TestQCircuitTTNO(unittest.TestCase):
    """
    Tests the turning of a quantum circuit into multiple TTNOs.
    """

    def setUp(self) -> None:
        self.q_ids = ["q0", "q1", "q2", "q3"]

        self.oneq_tree = MatrixProductState.from_tensor_list([ket_i(0,2)],
                                                       node_prefix="q")
        self.twoq_tree = MatrixProductState.from_tensor_list([ket_i(0,2).reshape(1,2)
                                                              for _ in range(2)],
                                                       node_prefix="q")
        tensors = [ket_i(0,2).reshape(1,2)]
        tensors.append(ket_i(0,2).reshape(1,1,2))
        tensors.append(ket_i(0,2).reshape(1,2))
        self.threeq_tree = MatrixProductState.from_tensor_list(tensors,
                                                            node_prefix="q")
        tensors.insert(1, ket_i(0,2).reshape(1,1,2))
        self.fourq_tree = MatrixProductState.from_tensor_list(tensors,
                                                           node_prefix="q")

    def test_circuit_one_gate(self):
        """
        Test a circuit with one gate.
        """
        qc = QCircuit()
        qc.add_x(self.q_ids[0])
        ttno_list = qc.as_circuit_ttno(self.oneq_tree)
        self.assertEqual(1, len(ttno_list))
        matrix, _ = ttno_list[0].as_matrix()
        correct = np.array([[0, 1],
                            [1, 0]], dtype=complex)
        np.testing.assert_allclose(matrix, correct, atol=1e-10)

    def test_circuit_two_gates(self):
        """
        Test a circuit with two gates.
        """
        qc = QCircuit()
        qc.add_x(self.q_ids[0])
        qc.add_hadamard(self.q_ids[1], level_index=0)
        ttno_list = qc.as_circuit_ttno(self.twoq_tree)
        self.assertEqual(1, len(ttno_list))
        matrix, _ = ttno_list[0].as_matrix()
        hadamard = (1/np.sqrt(2)) * np.array([[1, 1],
                                              [1, -1]], dtype=complex)
        pauli_x = np.array([[0, 1],
                            [1, 0]], dtype=complex)
        correct = np.kron(pauli_x, hadamard)
        np.testing.assert_allclose(matrix, correct, atol=1e-10)

    def test_circuit_two_qubit_gate(self):
        """
        Test a circuit with a two-qubit gate.
        """
        qc = QCircuit()
        qc.add_cnot(self.q_ids[0], self.q_ids[1])
        ttno_list = qc.as_circuit_ttno(self.twoq_tree)
        self.assertEqual(1, len(ttno_list))
        matrix, _ = ttno_list[0].as_matrix()
        correct = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]], dtype=complex)
        np.testing.assert_allclose(matrix, correct, atol=1e-10)

    def test_circuit_cnot_and_single(self):
        """
        Test a circuit with a CNOT and a single qubit gate.
        """
        qc = QCircuit()
        qc.add_cnot(self.q_ids[0], self.q_ids[1])
        qc.add_hadamard(self.q_ids[2], level_index=0)
        ttno_list = qc.as_circuit_ttno(self.threeq_tree)
        self.assertEqual(1, len(ttno_list))
        matrix0, _ = ttno_list[0].as_matrix()
        correct0 = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]], dtype=complex)
        correct2 = (1/np.sqrt(2)) * np.array([[1, 1],
                                             [1, -1]], dtype=complex)
        correct = np.kron(correct0, correct2)
        np.testing.assert_allclose(matrix0, correct, atol=1e-10)

    def test_parallel_cnot(self):
        """
        Test a circuit with two CNOTs in parallel.
        """
        qc = QCircuit()
        qc.add_cnot(self.q_ids[0], self.q_ids[1])
        qc.add_cnot(self.q_ids[2], self.q_ids[3], level_index=0)
        ttno_list = qc.as_circuit_ttno(self.fourq_tree)
        self.assertEqual(1, len(ttno_list))
        matrix0, _ = ttno_list[0].as_matrix()
        correct0 = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]], dtype=complex)
        correct1 = correct0
        correct = np.kron(correct0, correct1)
        np.testing.assert_allclose(matrix0, correct, atol=1e-10)
