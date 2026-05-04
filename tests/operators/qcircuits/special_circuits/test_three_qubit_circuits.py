"""
Provide tests for the three qubit circuits.
"""
from __future__ import annotations
import unittest

import numpy as np
import numpy.testing as npt

from pytreenet.special_ttn.mps import MatrixProductState
from pytreenet.time_evolution.bug import BUG, BUGConfig
from pytreenet.time_evolution.time_evolution import TimeEvoMode
from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.operators.common_operators import ket_i
from pytreenet.core.node import Node
from pytreenet.operators.models import (local_magnetisation_from_topology,
                                        Topology)
from pytreenet.operators.qcircuits.qcircuit import QCircuit
from pytreenet.ttns.ttns_ttno.direct_application import direct

from pytreenet.operators.qcircuits.special_circuits.three_qubit_circuits import (ThreeQubitState,
                                                                                 prepare_ghz,
                                                                                 prepare_w)

def q_name(n: int) -> str:
    q_prefix = "qubit"
    return f"{q_prefix}{n}"

def zero_state(num_qb: int) -> MatrixProductState:
    """
    Returns the zero state for a given number of qubits.

    Args:
        num_qb (int): The number of qubits.
    
    Returns:
        MatrixProductState: The zero state as a matrix product state.
    """
    if num_qb == 1:
        mps = TreeTensorNetworkState()
        tensor = ket_i(0, 2)
        mps.add_root(Node(identifier=q_name(0)),
                      tensor)
    elif num_qb > 1:
        mps = MatrixProductState.constant_product_state(0,2,num_qb,
                                                    node_prefix="qubit",
                                                    bond_dimensions=[2 for _ in range(num_qb-1)])
    return mps

def run_circuit(qc: QCircuit) -> np.ndarray:
    """
    Compiles and runs the time evolution for a single qubit quantum circuit.

    Args:
        qc (QCircuit): The quantum circuit to run.

    Returns:
        np.ndarray: The final state.
    """
    num_qb = qc.width()
    mps = zero_state(num_qb)
    comp_qc = qc.compile()
    ttno = comp_qc.to_time_dep_ttno(mps)
    ttno.measurement_renorm_threshold = 1e-10
    ops = local_magnetisation_from_topology(Topology.CHAIN, num_qb,
                                            site_prefix="qubit")
    final_time = qc.gate_depth() * 1
    time_step_size = 0.001
    config = BUGConfig(time_evo_mode=TimeEvoMode.RK45,
                    time_dep=True)
    solver_options = {"rtol": 1e-8, "atol": 1e-8}
    bug = BUG(mps, ttno, time_step_size, final_time, ops,
            config=config,
            solver_options=solver_options)
    bug.run(evaluation_time="inf")
    return bug.state.completely_contract_tree()[0].flatten()

def apply_circuit(qc: QCircuit) -> np.ndarray:
    """
    Applies a quantum circuit directly as a TTNO.

    Args:
        qc (QCircuit): The quantum circuit to apply.
    
    Returns:
        np.ndarray: The final state after applying the circuit.
    """
    mps = zero_state(qc.width())
    res = qc.apply_to_state(mps,
                            direct)
    return res.completely_contract_tree()[0].flatten()

class TestGHZPreparation(unittest.TestCase):
    """
    Test the preparation of the GHZ state.
    """

    def test_via_application(self):
        """
        Test the preparation of the GHZ state via direct application of the
        circuit.
        """
        qc = QCircuit()
        qubit_ids = [q_name(i) for i in range(3)]
        prepare_ghz(qc, qubit_ids, 0)
        found = apply_circuit(qc)
        expected = ThreeQubitState.GHZ.vector()
        npt.assert_allclose(found, expected)

    def test_via_time_evolution(self):
        """
        Test the preparation of the GHZ state via time evolution.
        """
        qc = QCircuit()
        qubit_ids = [q_name(i) for i in range(3)]
        prepare_ghz(qc, qubit_ids, 0)
        found = run_circuit(qc)
        expected = ThreeQubitState.GHZ.vector()
        npt.assert_allclose(found, expected, rtol=1e-5, atol=1e-4)

    def test_call_via_enum(self):
        """
        Test the preparation of the GHZ state via the enumeration.
        """
        qc = QCircuit()
        qubit_ids = [q_name(i) for i in range(3)]
        ThreeQubitState.GHZ.preparation_function()(qc, qubit_ids, 0)
        found = apply_circuit(qc)
        expected = ThreeQubitState.GHZ.vector()
        npt.assert_allclose(found, expected)

class TestWPreparation(unittest.TestCase):
    """
    Test the preparation of the W state.
    """

    def test_via_application(self):
        """
        Test the preparation of the W state via direct application of the
        circuit.
        """
        qc = QCircuit()
        qubit_ids = [q_name(i) for i in range(3)]
        prepare_w(qc, qubit_ids, 0)
        found = apply_circuit(qc)
        expected = ThreeQubitState.W.vector()
        npt.assert_allclose(found, expected)

    def test_via_time_evolution(self):
        """
        Test the preparation of the W state via time evolution.
        """
        qc = QCircuit()
        qubit_ids = [q_name(i) for i in range(3)]
        prepare_w(qc, qubit_ids, 0)
        found = run_circuit(qc)
        expected = ThreeQubitState.W.vector()
        npt.assert_allclose(found, expected, rtol=1e-5, atol=1e-4)

    def test_call_via_enum(self):
        """
        Test the preparation of the W state via the enumeration.
        """
        qc = QCircuit()
        qubit_ids = [q_name(i) for i in range(3)]
        ThreeQubitState.W.preparation_function()(qc, qubit_ids, 0)
        found = apply_circuit(qc)
        expected = ThreeQubitState.W.vector()
        npt.assert_allclose(found, expected)