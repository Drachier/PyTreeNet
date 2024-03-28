from __future__ import annotations
from typing import Dict, List, Union

from ..ttns import TreeTensorNetworkState
from .ttn_time_evolution import TTNTimeEvolution
from .trotter import TrotterSplitting
from ..operators.operator import NumericOperator
from ..operators.tensorproduct import TensorProduct

class TEBD(TTNTimeEvolution):
    """
    Runs the TEBD algorithm on a TTN
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 trotter_splitting: TrotterSplitting, time_step_size: float,
                 final_time: float, operators: Union[List[TensorProduct], TensorProduct],
                 max_bond_dim: Union[int, float] = 100, rel_tol: float =1e-16,
                 total_tol: float = 1e-16,
                 **kwargs):
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
            operators (Union[List[TensorProduct], TensorProduct]): Operators in the form of single site
             tensor product for which expectation values should be determined.
            max_bond_dim (int, optional): The maximum bond dimension allowed between
            nodes. Defaults to 100.
            rel_tol (float, optional): singular values s for which
             ( s / largest singular value) < rel_tol are truncated. Defaults to 1e-10.
            total_tol (float, optional): singular values s for which s < total_tol
             are truncated. Defaults to 1e-15.

        Raises:
            ValueError: If the trotter splitting and TTNS are not compatible.
        """
        super().__init__(initial_state,
                         time_step_size, final_time,
                         operators,
                         **kwargs)
        self._trotter_splitting = trotter_splitting

        if max_bond_dim < 1:
            errstr = "The maximum bond dimension must be positive!"
            raise ValueError(errstr)
        if isinstance(max_bond_dim, float) and max_bond_dim != float("inf"):
            errstr = "Maximum bond dimension must be int or inf!"
            raise TypeError(errstr)
        self.max_bond_dim = max_bond_dim
        if rel_tol < 0 and rel_tol != float("-inf"):
            errstr = "The relative tolerance must be non-negativ of -inf!"
            raise ValueError(errstr)
        self.rel_tol = rel_tol
        if total_tol < 0 and rel_tol != float("-inf"):
            errstr = "The total tolerance must be non-negativ of -inf!"
            raise ValueError(errstr)
        self.total_tol = total_tol

        self._exponents = self._trotter_splitting.exponentiate_splitting(self._time_step_size,
                                                                         self.state)

    @property
    def exponents(self) -> List[NumericOperator]:
        """
        Returns the exponentiated Trotter operators.
        """
        return self._exponents

    @property
    def trotter_splitting(self) -> TrotterSplitting:
        """
        Returns the Trotter splitting.
        """
        return self._trotter_splitting

    def _apply_one_trotter_step_single_site(self, single_site_exponent: NumericOperator):
        """
        Applies a single-site exponential operator of the Trotter splitting.

        exp @ |state>

        Args:
            single_site_exponent (NumericOperator): An operator representing a
             single-site unitary operator.
        """
        operator = single_site_exponent.operator.T
        identifier = single_site_exponent.node_identifiers[0]
        self.state.absorb_into_open_legs(identifier, operator)

    def _apply_one_trotter_step_two_site(self, two_site_exponent: NumericOperator):
        """
        Applies the two-site exponential operator of the Trotter splitting.
            exp @ |state>

        Args:
            two_site_exponent (NumericOperator): The exponent which should be
             applied. Contains the numeric value and the identifier of the 
             sites on which application should happen.
        """
        operator = two_site_exponent.operator.transpose([2,3,0,1])
        identifiers = two_site_exponent.node_identifiers

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

    def _apply_one_trotter_step(self, unitary: NumericOperator):
        """
        Applies the exponential operator of the Trotter splitting to the
         current state.
            exp(op) @ |state>

        Args:
            unitary (NumericOperator): The exponent which should be
             applied. Contains the numeric value and the identifier of the 
             sites on which application should happen.

        Raises:
            NotImplementedError: If the operator acts on more than two sites.
        """
        num_of_sites_acted_upon = len(unitary.node_identifiers)

        if num_of_sites_acted_upon == 0:
            pass
        elif num_of_sites_acted_upon == 1:
            self._apply_one_trotter_step_single_site(unitary)
        elif num_of_sites_acted_upon == 2:
            self._apply_one_trotter_step_two_site(unitary)
        else:
            raise NotImplementedError(
                "More than two-site interactions are not implemented.")

    def run_one_time_step(self):
        """
        Running one time_step on the TNS according to the exponentials. The
        order in which the trotter splitting is run, is the order in which the
        time-evolution operators are saved in `self.exponents`.
        """
        for unitary in self.exponents:
            self._apply_one_trotter_step(unitary)
