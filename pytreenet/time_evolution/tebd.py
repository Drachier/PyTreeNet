"""
Module implementing the TEBD time-evolution for TTNS.
"""
from __future__ import annotations
from typing import List, Union, Dict

from ..ttns import TreeTensorNetworkState
from .ttn_time_evolution import TTNTimeEvolution, TTNTimeEvolutionConfig
from .trotter import TrotterSplitting
from ..operators.operator import NumericOperator
from ..operators.tensorproduct import TensorProduct
from ..util.tensor_splitting import SVDParameters
from ..ttno.ttno_class import TreeTensorNetworkOperator as TTNO

class TEBD(TTNTimeEvolution):
    """
    Runs the TEBD algorithm on a TTNS.

    The TEBD algorithm uses a trotterised version of a Hamiltonian time
    evolution. The different Trotter operators are contracted into the system
    one by one. A truncation happens, if it is desired.

    Attributes:
        trotter_splitting (TrotterSplitting): A splitting of a Hamiltonian into
            different few site operators. Currently only one and two site
            operators are considered.
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 trotter_splitting: TrotterSplitting, time_step_size: float,
                 final_time: float,
                 operators: Union[List[Union[TensorProduct, TTNO]],
                                  Dict[str, Union[TensorProduct, TTNO]],
                                  TensorProduct,
                                  TTNO],
                 svd_parameters: Union[SVDParameters,None] = None,
                 config: Union[TTNTimeEvolutionConfig,None] = None):
        """
        Initiates a TEBD object.

        Args:
            initial_state (TreeTensorNetworkState)): The initial state of our
                time-evolution as a TTNS
            trotter_splitting (TrotterSplitting): The Trotter splitting to be
                used for time-evolution.
            time_step_size (float): The time step size to be used.
            final_time (float): The final time until which to run.
            operators (Union[List[TensorProduct], Dict[str,
                Union[TensorProduct, TTNO]], TTNO, TensorProduct]): Operators in
                the form of single site tensor product for which expectation
                values should be determined.
            svd_parameters (Union[SVDParameters,None]): The parameters for the
                SVD to be used in the truncations. The maximum bond dimension,
                relative tolerance, and absolute tolerance can be given. It can
                also be decided, if the singular values should be renormalised
                after truncation. If None is given, the values default to zero.
        """
        super().__init__(initial_state,
                         time_step_size, final_time,
                         operators,
                         config=config)
        self._trotter_splitting = trotter_splitting
        if svd_parameters is None:
            self.svd_parameters = SVDParameters()
        else:
            self.svd_parameters = svd_parameters
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
        operator = single_site_exponent.operator
        identifier = single_site_exponent.node_identifiers[0]
        self.state.absorb_into_open_legs(identifier, operator)

    def _apply_one_trotter_step_two_site(self,
                                         two_site_exponent: NumericOperator):
        """
        Applies the two-site exponential operator of the Trotter splitting.
                exp @ |state>

        Args:
            two_site_exponent (NumericOperator): The exponent which should be
                applied. Contains the numeric value and the identifier of the 
                sites on which application should happen.
        """
        operator = two_site_exponent.operator
        identifiers = two_site_exponent.node_identifiers

        u_legs, v_legs = self.state.legs_before_combination(identifiers[0],
                                                            identifiers[1])
        self.state.contract_nodes(identifiers[0], identifiers[1],
                           new_identifier="contr")
        self.state.absorb_into_open_legs("contr", operator)
        self.state.split_node_svd("contr", u_legs, v_legs,
                                  u_identifier=identifiers[0],
                                  v_identifier=identifiers[1],
                                  svd_params=self.svd_parameters)

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
            errstr = "More than two-site interactions are not implemented."
            raise NotImplementedError(errstr)

    def run_one_time_step(self, **kwargs):
        """
        Runs one TEBD time step.

        This means every Trotter operator is applied once to the TTNS.
        """
        for unitary in self.exponents:
            self._apply_one_trotter_step(unitary)
