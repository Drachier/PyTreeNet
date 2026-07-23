"""
Solve the time-dependent Schrodinger equation using the Magnus expansion.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Self

import numpy as np
from scipy.integrate import (quad,
                             dblquad,
                             tplquad)
from scipy.special import jv

from .ttn_time_evolution import (TTNTimeEvolutionConfig,
                                 TTNTimeEvolution)
from ..ttns.ttns import TTNS
from ..ttno.ttno_class import TTNO
from ..operators.tensorproduct import TensorProduct
from ..core.addition.linear_combination import (LinearCombination,
                                                LinCombParams)

@dataclass
class MagnusConfig(TTNTimeEvolutionConfig):
    """
    Configuration for the Magnus expansion time evolution.
    
    Attributes:
        m_order (int): The order of the Magnus expansion to use. Default is 2.
        c_order (int): The order of the Chebyshev expansion to use for the
            exponential of the Magnus expansion. Default is 2.
    """
    m_order: int = 2
    c_order: int = 2

class Magnus(TTNTimeEvolution):
    """
    Time evolution using the Magnus expansion.
    """
    config_class = MagnusConfig

    def __init__(self,
                 init_state: TTNS,
                 time_independent_hamiltonian: TTNO,
                 time_dependent_hamiltonian: TTNO,
                 control_function: Callable[[float], float],
                 time_step_size: float,
                 final_time: float,
                 operators: list[TensorProduct | TTNO] |
                                  dict[str, TensorProduct | TTNO] |
                                  TensorProduct |
                                  TTNO,
                 lincomb_params: LinCombParams,
                 config: MagnusConfig = MagnusConfig()
                 ) -> None:
        """
        Initialize the Magnus time evolution.

        Args:
            init_state (TTNS): The initial state of the system.
            time_independent_hamiltonian (TTNO): The time-independent part of
                the Hamiltonian.
            time_dependent_hamiltonian (TTNO): The time-dependent part of the
                Hamiltonian.
            control_function (Callable[[float], float]): A function that takes
                time as input and returns the value of the control function at
                that time.
            time_step_size (float): The size of each time step for the evolution.
            final_time (float): The final time up to which to evolve the system.
            operators (list[TensorProduct | TTNO] | dict[str, TensorProduct | TTNO] | TensorProduct | TTNO):
                The operators that will be used in the time evolution. This can
                be a list of TensorProducts or TTNOs, a dictionary mapping
                names to TensorProducts or TTNOs, a single TensorProduct, or a
                single TTNO.
            lincomb_params (LinCombParams): Parameters used to compute the linear
                combination of operators for the time evolution.
            config (MagnusConfig, optional): The configuration for the Magnus
                expansion time evolution. Defaults to MagnusConfig().
        """
        super().__init__(init_state,
                         time_step_size,
                         final_time,
                         operators,
                         config=config)
        if config.m_order < 1:
            raise ValueError("The order of the Magnus expansion must be at least 1!")
        if config.m_order > 3:
            raise NotImplementedError("Only Magnus expansions up to order 3 are implemented!")
        if config.c_order < 1:
            raise ValueError("The order of the Chebyshev expansion must be at least 1!")
        self.time_independent_hamiltonian = time_independent_hamiltonian
        self.time_dependent_hamiltonian = time_dependent_hamiltonian
        self.control_function = control_function
        self.lincomb_params = lincomb_params
        self._current_time_step = 0
        # These are time-independent and can be computed once at the beginning
        self._magnus_expansion_ops = self._magnus_expansion_operators()

    def _integration_limits(self) -> tuple[float, float]:
        """
        Compute the integration limits for the current time step.

        Returns:
            tuple[float, float]: A tuple containing the lower and upper limits
                of integration for the current time step.
        """
        lower_limit = self._current_time_step * self.time_step_size
        upper_limit = lower_limit + self.time_step_size
        return lower_limit, upper_limit

    def _compute_control_function_integrals(self) -> list[complex]:
        """
        Compute the integrals of the control function needed for the Magnus expansion.
 
        Args:
            current_time_step (int): The current time step index.
 
        Returns:
            list[complex]: A list of the computed integrals of the control function. 
        """ 
        lower_limit, upper_limit = self._integration_limits()
        integrals = []
        # The 0th order integral is just proportional to the time step size
        integrals.append(-1j*self.time_step_size)
        if self.config.m_order >= 1:
            # First order integral
            integrand = lambda t: _magnus_integrand_order1(t,
                                                           self.control_function)
            res = quad(integrand,
                       lower_limit,
                       upper_limit)
            int_value = -1j * res[0]
            integrals.append(int_value)
        if self.config.m_order >= 2:
            # Second order integral
            inner_lower_limit = lambda t: lower_limit
            inner_upper_limit = lambda t: t
            integrand = lambda t2, t1: _magnus_integrand_order2(t2, t1,
                                                                self.control_function)
            res = dblquad(integrand,
                          lower_limit,
                          upper_limit,
                          inner_lower_limit,
                          inner_upper_limit)
            int_value = -0.5 * res[0]
            integrals.append(int_value)
        if self.config.m_order >= 3:
            # Third order integral
            lower_limit_t2 = lambda t1: lower_limit
            upper_limit_t2 = lambda t1: t1
            lower_limit_t3 = lambda t1, t2: lower_limit
            upper_limit_t3 = lambda t1, t2: t2
            ## Here we have two integrals
            ## The first
            integrand1 = lambda t3, t2, t1: _magnus_integrand_order3_1(t3, t2, t1,
                                                                       self.control_function)
            res1 = tplquad(integrand1,
                           lower_limit,
                           upper_limit,
                           lower_limit_t2,
                           upper_limit_t2,
                           lower_limit_t3,
                           upper_limit_t3)
            int_value1 = (1j / 6) * res1[0]
            integrals.append(int_value1)
            ## The second
            integrand2 = lambda t3, t2, t1: _magnus_integrand_order3_2(t3, t2, t1,
                                                                       self.control_function)
            res2 = tplquad(integrand2,
                            lower_limit,
                            upper_limit,
                            lower_limit_t2,
                            upper_limit_t2,
                            lower_limit_t3,
                            upper_limit_t3)
            int_value2 = (1j / 6) * res2[0]
            integrals.append(int_value2)
        return integrals
    
    def _magnus_expansion_operators(self) -> list[list[TTNO]]:
        """
        Generates the different terms of the Magnus expansion.

        Every term corresponds to an actual summand in the fully simplified
        expansion and is represented as a list of TTNOs that would be
        multiplied together. In a list the 0th entry is the first operator
        to be multiplied to a state and so on.

        Returns:
            list[list[TTNO]]: A list of the different terms of the Magnus expansion,
                where each term is represented as a list of TTNOs to be multiplied
                together.
        """
        # Prep for readability
        ti = "ti"
        td = "td"
        hams = {ti: self.time_independent_hamiltonian,
                td: self.time_dependent_hamiltonian}
        temp = []
        # First order
        temp.append([ti])
        temp.append([td])
        # Second order
        if self.config.m_order >= 2:
            temp.append([td, ti])
            temp.append([ti, td])
        # Third order
        if self.config.m_order >= 3:
            temp.append([td, ti, ti])
            temp.append([ti, td, ti])
            temp.append([ti, ti, td])
            temp.append([td, ti, td])
            temp.append([td, td, ti])
            temp.append([ti, td, td])
        out = [[hams[op] for op in term] for term in temp]
        return out

    def _magnus_expansion_coefficients(self,
                                       integrals: list[complex]
                                       ) -> list[complex]:
        """
        Yields a list of the coefficients corresponding to the different terms
        of the Magnus expansion.

        These coefficients fit to the operators generated by
        _magnus_expansion_operators.
        """
        out = []
        # First order
        out.append(integrals[0])
        out.append(integrals[1])
        # Second order
        if self.config.m_order >= 2:
            out.append(integrals[2])
            out.append(-1 * integrals[2])
        # Third order
        if self.config.m_order >= 3:
            out.append(integrals[3])
            out.append(-2 * integrals[3])
            out.append(integrals[3])
            out.append(2 * integrals[4])
            out.append(-1 * integrals[4])
            out.append(-1 * integrals[4])
        return out
 
    def _apply_one_magnus_expansion_power(self,
                                          state: TTNS,
                                          integrals: list[complex],
                                          ) -> TTNS:
        """
        Apply one power of the Magnus expansion to the given state.

        Args:
            state (TTNS): The state to which the Magnus expansion will be applied.
            integrals (list[complex]): The list of computed integrals of the control
                function needed for the Magnus expansion.
        
        Returns:
            TTNS: The state obtained after applying one power of the operator
                representing the Magnus expansion.
        """
        coefficients = self._magnus_expansion_coefficients(integrals)
        ttnos = self._magnus_expansion_operators()
        lincomb = LinearCombination(state,
                                    ttnos,
                                    coefficients)
        return lincomb.compute_via_params(self.lincomb_params)
    
    def _full_expansion_application(self) -> TTNS:
        """
        Apply the Magnus expansion to the given state using the computed integrals.

        Returns:
            TTNS: The state obtained after applying the operator representing the
                Magnus expansion.
        """
        # We compute the Chebyshev expansion
        expansion = [ChebyshevMagnus.zeroth_order(),
                      ChebyshevMagnus.first_order()]
        for i in range(2, self.config.c_order+1):
            new_order = ChebyshevMagnus.next_order(expansion[i-1],
                                                   expansion[i-2])
            expansion.append(new_order)
        prefactors: list[complex] = [jv(0,1)]
        prefactors += [2 * jv(k,1)
                       for k in range(1, self.config.c_order+1)]
        for prefactor, order in zip(prefactors, expansion):
            order.prefactors = [prefactor * p for p in order.prefactors]
        # Now we find all the prefactors of operator powers to be applied
        # Here the nth entry is the prefactor of the nth power of Omega
        final_prefactors = [0.0 for _ in range(self.config.c_order+1)]
        for order in expansion:
            for k, prefactor in enumerate(order.prefactors):
                final_prefactors[k] += prefactor
        integrals = self._compute_control_function_integrals()
        summand_states = [self.state]
        for i, _ in enumerate(final_prefactors):
            if i !=0:
                new_state = self._apply_one_magnus_expansion_power(deepcopy(self.state),
                                                                   integrals)
                summand_states.append(new_state)
        lin_comb = LinearCombination(summand_states,
                                     None,
                                     final_prefactors)
        return lin_comb.compute_via_params(self.lincomb_params)
        

    def run_one_time_step(self, **kwargs):
        self.state = self._full_expansion_application()
        self._current_time_step += 1

def _magnus_integrand_order1(t1, control_function):
    return control_function(t1)

def _magnus_integrand_order2(t2, t1, control_function):
    return control_function(t2) - control_function(t1)

def _magnus_integrand_order3_1(t3, t2, t1, control_function):
    return control_function(t3) - 2 * control_function(t2) + control_function(t1)

def _magnus_integrand_order3_2(t3, t2, t1, control_function):
    out = 2 * control_function(t1) * control_function(t3)
    out += -1 * control_function(t1) * control_function(t2)
    out += -1 * control_function(t2) * control_function(t3)
    return out

class ChebyshevMagnus:
    """
    Represents the Cheby
    """

    def __init__(self,
                 order: int,
                 prefactors: list[float]
                 ) -> None:
        """
        Initialize the ChebyshevMagnus object.

        Args:
            order (int): The order of the Chebyshev expansion to use.
            prefactors (list[float]): The prefactors for the Chebyshev expansion.
        """
        if len(prefactors) != order + 1:
            errstr = f"The number of prefactors must be equal to the order + 1!\n"
            errstr += "Got {len(prefactors)} prefactors for order {order}."
            raise ValueError(errstr)
        if order < 0:
            errstr = f"The order of the Chebyshev expansion must be non-negative!\n"
            errstr += f"Got order {order}."
            raise ValueError(errstr)
        self.order = order
        self.prefactors = prefactors

    def __eq__(self, other: object) -> bool:
        if self.order != other.order:
            return False
        prefactors_eq = np.allclose(self.prefactors, other.prefactors)
        return prefactors_eq

    @classmethod
    def zeroth_order(cls):
        return cls(order=0, prefactors=[1.0])
    
    @classmethod
    def first_order(cls):
        return cls(order=1, prefactors=[0.0, 1.0])
    
    @classmethod
    def next_order(cls,
                   order_i: ChebyshevMagnus,
                   order_im1: ChebyshevMagnus
                   ) -> Self:
        """
        Compute the next order of the Chebyshev expansion.

        Args:
            order_i (ChebyshevMagnus): The ChebyshevMagnus object representing the
                current order i.
            order_im1 (ChebyshevMagnus): The ChebyshevMagnus object representing
                the previous order i-1.

        Returns:
            ChebyshevMagnus: The ChebyshevMagnus object representing the next order
                i+1.
        """
        if order_i.order != order_im1.order + 1:
            errstr = f"order_i must be one order higher than order_im1!\n"
            errstr += f"Got order_i of {order_i.order} and order_im1 of {order_im1.order}."
            raise ValueError(errstr)
        new_order = order_i.order + 1
        new_prefactors = []
        for k in range(new_order + 1):
            if k == 0:
                new = order_im1.prefactors[0]
            elif k < order_i.order:
                new = 2 * order_i.prefactors[k-1] + order_im1.prefactors[k]
            else:
                new = 2 * order_i.prefactors[k-1]
            new_prefactors.append(new)
        return cls(order=new_order, prefactors=new_prefactors)
