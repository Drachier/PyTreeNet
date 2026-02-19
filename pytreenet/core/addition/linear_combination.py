"""
Implements the linear combination of tree tensors.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Self
from copy import copy

from ...ttns.ttns_ttno.application import ApplicationMethod
from .addition import (AdditionMethod,
                       add_ttns)
from ...ttns import TTNS
from ...ttno import TTNO

class LinCombParams:
    """
    Parameters for the linear combination of tree tensors.
    """

    def __init__(self,
                 application_method: ApplicationMethod,
                 addition_method: AdditionMethod,
                 args_ap: tuple = (),
                 kwargs_ap: dict | None = None,
                 args_add: tuple = (),
                 kwargs_add: dict | None = None
                 ) -> None:
        """
        Initializes the parameters for the linear combination of tree tensors.

        Args:
            application_method (ApplicationMethod): The method to be used for
                applying the tree tensor network operators to the tree tensor
                networks.
            addition_method (AdditionMethod): The method to be used for adding
                the resulting tree tensor networks together.
            args_ap (tuple, optional): Positional arguments for the application
                method. Defaults to (). 
            kwargs_ap (dict, optional): Keyword arguments for the application
                method. Defaults to None.
            args_add (tuple, optional): Positional arguments for the addition
                method. Defaults to ().
            kwargs_add (dict, optional): Keyword arguments for the addition
                method. Defaults to None.
        """
        self.application_method = application_method
        self.addition_method = addition_method
        self.args_ap = args_ap
        if kwargs_ap is None:
            kwargs_ap = {}
        self.kwargs_ap = kwargs_ap
        self.args_add = args_add
        if kwargs_add is None:
            kwargs_add = {}
        self.kwargs_add = kwargs_add

    @classmethod
    def default(cls) -> Self:
        """
        Returns the default parameters for the linear combination of tree tensors.

        The default application method is the exact application of the TTNO to the
        TTNS. The default addition method is the simple addition of the resulting
        tree tensor networks.

        Returns:
            Self: The default parameters for the linear combination of
                tree tensors.
        """
        return cls(application_method=ApplicationMethod.HALF_DENSITY_MATRIX,
                   addition_method=AdditionMethod.HALF_DENSITY_MATRIX)

class LinearCombination:
    """
    Implements the linear combination of tree tensors.
    """

    def __init__(self,
                 ttnss: list[TTNS] | TTNS,
                 ttnos: list[list[TTNO]] | list[TTNO] | TTNO | None,
                 coefficients: list[complex] | complex | None
                 ) -> None:
        """
        Initialises the linear combination.

        Args:
            ttnss (list[TTNS] | TTNS): The tree tensor networks to be combined.
                If a single tree tensor network is given, it is treated as a
                list of length one.
            ttnos (list[list[TTNO]] | list[TTNO] | TTNO | None): The tree tensor
                network operators to be applied to the tree tensor networks.
                Should have the same structure as ttnss. If a single tree
                tensor network operator is given, it is treated as a list of
                length one, and applied to all tree tensor networks in ttnss.
                If a single TTNO list is given, it is treated depending on the
                number of TTNSs given. If there is only one TTNS, the list of
                TTNOs is applied to that TTNS. If there are multiple TTNSs, the
                list of TTNOs is treated as a list of lists, and the i-th list
                of TTNOs is applied to the i-th TTNS. If None is given, only the
                sum of the TTNS is computed.
                The TTNO appearing first in the list is applied first.
            coefficients (list[complex] | complex | None): The coefficients of
                the linear combination. Should have the same length as ttnss. If a
                single complex number is given, it is treated as a list of
                length one, and applied to all tree tensor networks in ttnss.
                If None is given, all coefficients are set to one.
        """
        if isinstance(ttnss, TTNS):
            ttnss = [ttnss]
        self._ttnss = ttnss
        if isinstance(ttnos, TTNO):
            ttnos = [[ttnos] for _ in ttnss]
        elif isinstance(ttnos, list):
            if all(isinstance(ttno, TTNO) for ttno in ttnos):
                if len(ttnss) == 1:
                    # In this case all of the TTNOs are applied to the single TTNS.
                    ttnos = [ttnos]
                elif len(ttnos) == len(ttnss):
                    # In this case each of the TTNO is applied to the corresponding TTNS i.
                    ttnos = [[ttnos[i]] for i in range(len(ttnss))]
                else:
                    errstr = "If ttnos is a list of TTNOs, it should have the same length as ttnss, or ttnss should have length one!"
                    raise ValueError(errstr)
        elif ttnos is None:
            ttnos = [[] for _ in ttnss]
        self._ttnos: list[list[TTNO]] = ttnos
        if isinstance(coefficients, complex):
            coefficients = [coefficients] * len(ttnss)
        self._coefficients: list[complex | None] = coefficients

    @property
    def ttnss(self) -> list[TTNS]:
        """
        The tree tensor networks to be combined.
        """
        return self._ttnss

    @property
    def ttnos(self) -> list[list[TTNO]]:
        """
        The tree tensor network operators to be applied to the tree tensor
        networks.
        """
        return self._ttnos

    @property
    def coefficients(self) -> list[complex]:
        """
        The coefficients of the linear combination.
        """
        return self._coefficients

    def add_term(self,
                 ttns: TTNS,
                 ttno: list[TTNO] | TTNO,
                 coefficient: complex | None = None
                 ) -> None:
        """
        Adds a term to the linear combination.

        Args:
            ttns (TTNS): The tree tensor network to be added.
            ttno (list[TTNO] | TTNO): The tree tensor network operator to be
                applied to the tree tensor network. If a single TTNO is given,
                it is treated as a list of length one, and applied to the
                TTNS.
            coefficient (complex | None): The coefficient of the term. If None
                is given, the coefficient is set to one.
        """
        if isinstance(ttno, TTNO):
            ttno = [ttno]
        if coefficient is None:
            coefficient = 1.0 + 0.0j
        self._ttnss.append(ttns)
        self._ttnos.append(ttno)
        self._coefficients.append(coefficient)

    def compute_via_params(self, params: LinCombParams) -> TTNS:
        """
        Computes the linear combination via the given parameters.

        Args:
            params (LinCombParams): The parameters for the linear combination.

        Returns:
            TTNS: The result of the linear combination.
        """
        return self.compute(params.application_method,
                            params.addition_method,
                            args_ap=params.args_ap,
                            kwargs_ap=params.kwargs_ap,
                            args_add=params.args_add,
                            kwargs_add=params.kwargs_add)

    def compute(self,
                application_method: ApplicationMethod,
                addition_method: AdditionMethod,
                args_ap: tuple = (),
                kwargs_ap: dict | None = None,
                args_add: tuple = (),
                kwargs_add: dict | None = None
                ) -> TTNS:
        """
        Computes the linear combination.

        Note that some addition methods can perform the application of one
        TTNO and the addition of multiple TTNS resulting from this application
        in one step. In the case, the last application will be performed with
        the given addition method, and the given application method will be
        used for all other applications.

        Args:
            application_method (ApplicationMethod): The method to be used for
                applying the tree tensor network operators to the tree tensor
                networks.
            addition_method (AdditionMethod): The method to be used for adding
                the resulting tree tensor networks together.

        Returns:
            TTNS: The result of the linear combination.
        """
        if kwargs_ap is None:
            kwargs_ap = {}
        if kwargs_add is None:
            kwargs_add = {}
        ttnos = copy(self._ttnos)
        appl_function = application_method.get_function()
        res_ttns: list[TTNS] = []
        for i, ttns_i in enumerate(self._ttnss):
            ttnos_i = ttnos[i]
            while len(ttnos_i) > 1:
                ttno = ttnos_i.pop(0)
                ttns_i = appl_function(ttns_i, ttno, *args_ap, **kwargs_ap)
            if len(ttnos_i) == 1 and not addition_method.lin_comb_possible():
                ttno = ttnos_i.pop(0)
                ttns_i = appl_function(ttns_i, ttno, *args_ap, **kwargs_ap)
            coeff = self._coefficients[i]
            if coeff is not None:
                res_ttns.append(ttns_i.scale(coeff))
        if addition_method.lin_comb_possible():
            ttnos_add = []
            for ttnos_i in ttnos:
                if len(ttnos_i) > 1:
                    errstr = "Something went wrong!"
                    raise ValueError(errstr)
                if len(ttnos_i) > 0:
                    ttnos_add.append(ttnos_i[0])
                else:
                    ttnos_add.append(None)
            lc_function = addition_method.lin_comb_function()
            return lc_function(res_ttns,
                               ttnos_add,
                               *args_add,
                               **kwargs_add)
        ttn =  add_ttns(res_ttns,
                        addition_method,
                        *args_add,
                        **kwargs_add)
        return TTNS.from_ttn(ttn)
