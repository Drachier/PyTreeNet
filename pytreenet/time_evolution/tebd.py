from __future__ import annotations
from typing import Dict, List, Union

from ..ttns import TreeTensorNetworkState
from .time_evolution import TimeEvolution

class TEBD(TimeEvolution):
    """
    Runs the TEBD algorithm on a TTN
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 trotter_splitting: TrotterSplitting, time_step_size: float,
                 final_time: float, operators: Union[List[Dict], Dict],
                 max_bond_dim: int = 100, rel_tol: float =1e-10,
                 total_tol: float = 1e-15):
        """
        A class that can be used to time evolve an initial state in the form
         a tree tensor network state via the TEBD algorithm.

        If no truncation is desired set max_bond_dim = inf, rel_tol = -inf,
         and total_tol = -inf.

        Args:
            initial_state (TreeTensorNetworkState)): The initial state of our
             time-evolution
            trotter_splitting (TrotterSplitting): The Trotter splitting to be
             used for time-evolution.
            time_step_size (float): The time step size to be used.
            final_time (float): The final time until which to run.
            operators (Union[List[Dict], Dict]): Operators for which expectation values
             should be determined
            max_bond_dim (int, optional): The maximum bond dimension allowed between
            nodes. Defaults to 100.
            rel_tol (float, optional): singular values s for which
             ( s / largest singular value) < rel_tol are truncated. Defaults to 1e-10.
            total_tol (float, optional): singular values s for which s < total_tol
             are truncated. Defaults to 1e-15.

        Raises:
            ValueError: If the trotter splitting and TTNS are not compatible.
        """
        super().__init__(initial_state, time_step_size, final_time, operators)
        self._trotter_splitting = trotter_splitting

        self.max_bond_dim = max_bond_dim
        self.rel_tol = rel_tol
        self.total_tol = total_tol

        if not self._trotter_splitting.is_compatible_with_ttn(self.state):
            raise ValueError(
                "State TTN and Trotter Splitting are not compatible!")

        self._exponents = self._trotter_splitting.exponentiate_splitting(self.state,
                                                              self._time_step_size)

    @property
    def exponents(self):
        return self._exponents

    @property
    def trotter_splitting(self):
        return self._trotter_splitting

    def _apply_one_trotter_step_single_site(self, single_site_exponent: Dict):
        """
        Applies a single-site exponential operator of the Trotter splitting.

        Args:
            single_site_exponent (Dict): A dictionary with representing a
            single-site unitary operator. The operator is saved with key
            `"operator"` and the site to which it is applied is saved via
            node identifiers under the key `"site_ids"`
        """
        operator = single_site_exponent["operator"]
        identifier = single_site_exponent["site_ids"][0]
        self.state.absorb_tensor_into_open_legs(identifier, operator)

    def _apply_one_trotter_step_two_site(self, two_site_exponent):
        """
        Applies the two-site exponential operator of the Trotter splitting.

        Parameters
        ----------
        two_site_exponent: dict
            A dictionary with representing a two-site unitary operator.
            The operator is saved with key "operator" and the sites to which it
            is applied are saved via node identifiers in the key "site_ids"

        Returns
        -------
        None.

        """
        operator = two_site_exponent["operator"]
        identifiers = two_site_exponent["site_ids"]

        u_legs, v_legs = self.state.legs_before_combination(identifiers[0],
                                                            identifiers[1])
        self.state.contract_nodes(identifiers[0], identifiers[1],
                           new_identifier="contr")
        self.state.absorb_into_open_legs("contr", operator)
        self.state.split_node_svd("contr", u_legs, v_legs,
                                  u_identifier=identifiers[0],
                                  v_identifier=identifiers[1],
                                  max_bond_dim=self.max_bond_dim,
                                  rel_tol=self.rel_tol,
                                  total_tol=self.total_tol)

    def _apply_one_trotter_step(self, unitary):
        """
        Applies the exponential operator of the Trotter splitting that is
        chosen via index

        Parameters
        ----------
        unitary : dict
            A dictionary representing a time evolution operator (usually a unitary matrix), 
            where the actual operator is saved as an ndarray under the key
            `"operator"` and the sites it is applied to are saved as a list of
            strings/site identifiers under they key `"site_ids"`

        Returns
        -------
        None.

        """
        num_of_sites_acted_upon = len(unitary["site_ids"])

        if num_of_sites_acted_upon == 0:
            pass
        elif num_of_sites_acted_upon == 1:
            self._apply_one_trotter_step_single_site(unitary)
        elif num_of_sites_acted_upon == 2:
            self._apply_one_trotter_step_two_site(unitary)
        else:
            raise NotImplementedError(
                "More than two-site interactions are not yet implemented.")

    def run_one_time_step(self):
        """
        Running one time_step on the TNS according to the exponentials. The
        order in which the trotter splitting is run, is the order in which the
        time-evolution operators are saved in `self.exponents`.
        """
        for unitary in self.exponents:
            self._apply_one_trotter_step(unitary)
