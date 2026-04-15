import unittest

import numpy as np

import pytreenet as ptn
from pytreenet.random import (random_small_ttns,
                              RandomTTNSMode,
                              random_hermitian_matrix)
from pytreenet.contractions.state_operator_contraction import contract_any
from pytreenet.special_ttn.special_states import (TTNStructure, generate_zero_state,
                                                  STANDARD_NODE_PREFIX,
                                                  Topology)
from pytreenet.operators.models.two_site_model import (IsingParameters, IsingModel)
from pytreenet.ttno.ttno_class import TTNO
from pytreenet.time_evolution.tdvp_algorithms.tdvp_algorithm import TDVPConfig
from pytreenet.time_evolution.tdvp_algorithms.secondorderonesite import SecondOrderOneSiteTDVP
from pytreenet.time_evolution.time_evolution import TimeEvoMode

class TestTwoSiteTDVPSimple(unittest.TestCase):
    def setUp(self) -> None:
        self.state = generate_zero_state(3, TTNStructure.TSTAR,
                                         topology=Topology.TTOPOLOGY,
                                         bond_dim=3)
        model = IsingModel.from_dataclass(IsingParameters(ext_magn=0.5))
        ham = model.generate_t_topology_model(3, site_ids=STANDARD_NODE_PREFIX)
        self.ham = TTNO.from_hamiltonian(ham, self.state)
        self.time_step_size = 0.1
        self.final_time = 1
        self.operators = []
        self.config = TDVPConfig(time_evo_mode=TimeEvoMode.RK45)
        self.tdvp = SecondOrderOneSiteTDVP(self.state, self.ham,
                                           self.time_step_size,
                                           self.final_time,
                                           self.operators,
                                           config=self.config,
                                           solver_options={"rtol": 1e-5, "atol": 1e-8})

    def _node_id(self, index: int | None = None) -> str:
        if index is None:
            return "center"
        return STANDARD_NODE_PREFIX + str(index)

    def test_forward_updatepath(self):
        """
        Checks that the forward update path is correct.
        """
        found = self.tdvp.update_path
        expected = [self._node_id(i) for i in [2,1,0,8,7,6,None,3,4,5]]
        self.assertEqual(found, expected)

    def test_forward_orthpath(self):
        """
        Checks that the forward orthogonalization path is correct.
        """
        found = self.tdvp.orthogonalization_path
        expected = [[self._node_id(i)] for i in [1,0,None,7,6,None,3,4,5]]
        expected[2].extend([self._node_id(i) for i in [6,7,8]])
        self.assertEqual(found, expected)
        for found_path, update_node in zip(found, self.tdvp.update_path[1:]):
            self.assertEqual(found_path[-1], update_node)

    def test_backwards_updatepath(self):
        """
        Checks that the backwards update path is correct.
        """
        found = self.tdvp.backwards_update_path
        expected = [self._node_id(i) for i in [5,4,3,None,6,7,8,0,1,2]]
        self.assertEqual(found, expected)

    def test_backwards_orthpath(self):
        """
        Checks that the backwards orthogonalization path is correct.
        """
        found = self.tdvp.backwards_orth_path
        expected = [[self._node_id(i)] for i in [4,3,None,6,7,8,0,1,2]]
        expected[5].extend([self._node_id(i) for i in [7,6,None]])
        self.assertEqual(found, expected)
