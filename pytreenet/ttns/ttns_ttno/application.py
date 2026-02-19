"""
This module contains the algorithms to apply a TTNO to a TTNS.
"""
from __future__ import annotations
from typing import Callable, TYPE_CHECKING
from enum import Enum

from .direct_application import direct
from .dm_approach import dm_ttns_ttno_application
from .half_dm_approach import half_dm_ttns_ttno_application
from .src import src_ttns_ttno_application
from .zipup import zipup
from .variational import variational_ttns_ttno_application

from ...util.tensor_splitting import SVDParameters
from ...core.truncation.truncating import (TruncationMethod,
                                           truncate_ttns)

if TYPE_CHECKING:
    from ..ttns import TTNS
    from ...ttno.ttno_class import TTNO

class ApplicationMethod(Enum):
    """
    The available application methods.
    """
    DIRECT = "direct"
    DENSITY_MATRIX = "density_matrix"
    HALF_DENSITY_MATRIX = "half_density_matrix"
    SRC = "src"
    ZIPUP = "zipup"
    ZIPUP_CANONICAL = "zipup_canonical"
    VARIATIONAL = "variational"
    ZIPUP_VARIATIONAL = "zipup_variational"
    HALF_DENSITY_MATRIX_VARIATIONAL = "half_density_matrix_variational"
    DIRECT_TRUNCATE = "direct_truncate"
    DIRECT_TRUNCATE_RANDOM = "direct_truncate_random"

    def get_function(self) -> Callable:
        """
        Get the application function corresponding to the application method.
        
        Returns:
            Callable: The application function.
        """
        if self == ApplicationMethod.DIRECT:
            return direct
        if self == ApplicationMethod.DENSITY_MATRIX:
            return dm_ttns_ttno_application
        if self == ApplicationMethod.HALF_DENSITY_MATRIX:
            return half_dm_ttns_ttno_application
        if self == ApplicationMethod.SRC:
            return src_ttns_ttno_application
        if self == ApplicationMethod.ZIPUP:
            # Annoyingly, zipup has a different signature than the other application methods.
            return lambda ttns, ttno, *args, **kwargs: zipup(ttno, ttns, *args, **kwargs)
        if self == ApplicationMethod.ZIPUP_CANONICAL:
            return lambda ttns, ttno, *args, **kwargs: zipup(ttno, ttns, *args,
                                                             canonicalize=True,
                                                             **kwargs)
        if self == ApplicationMethod.VARIATIONAL:
            return variational_ttns_ttno_application
        if self == ApplicationMethod.ZIPUP_VARIATIONAL:
            return variational_zipup
        if self == ApplicationMethod.HALF_DENSITY_MATRIX_VARIATIONAL:
            return half_dm_variational
        if self == ApplicationMethod.DIRECT_TRUNCATE:
            return lambda ttns, ttno, *args, **kwargs: apply_and_truncate(ttns,
                                                                           ttno,
                                                                           ApplicationMethod.DIRECT,
                                                                           TruncationMethod.SVD,
                                                                           trunc_args=args,
                                                                           trunc_kwargs=kwargs)
        if self == ApplicationMethod.DIRECT_TRUNCATE_RANDOM:
            def random_truncation_method(ttns, ttno, *args, **kwargs):
                params = kwargs.get("params", SVDParameters())
                params.random = True
                kwargs["params"] = params
                return apply_and_truncate(ttns,
                                          ttno,
                                          ApplicationMethod.DIRECT,
                                          TruncationMethod.SVD,
                                          trunc_args=args,
                                          trunc_kwargs=kwargs)
            return random_truncation_method
        raise ValueError(f"Unknown application method: {self}")

    def can_sum(self) -> bool:
        """
        Wether the method can be used to directly sum two TTNS together.

        Notably, they should be able to apply a TTNO on each TTNS and sum
        during the application, instead of applying the TTNOs and then sum
        afterwards.
        """
        return self in {ApplicationMethod.DENSITY_MATRIX,
                        ApplicationMethod.HALF_DENSITY_MATRIX,
                        ApplicationMethod.SRC}

def variational_zipup(ttns: TTNS,
                      ttno: TTNO,
                      *args,
                      var_svd_params: SVDParameters | None = None,
                      zipup_svd_params: SVDParameters | None = None,
                      **kwargs
                      ) -> TTNS:
    """
    Apply a TTNO to a TTNS using zipup as an initial guess for variational
    fitting.

    Args:
        ttns (TTNS): The tree tensor network state to which the TTNO is
            applied.
        ttno (TTNO): The tree tensor network operator to apply.
        *args: Additional positional arguments for the variational fitting.
        var_svd_params (SVDParameters | None): SVD parameters for the
            variational fitting. If None, default parameters are used.
        zipup_svd_params (SVDParameters | None): SVD parameters for the zipup
            application. If None, default parameters are used.
        **kwargs: Additional keyword arguments for the variational fitting.

    Returns:
        TTNS: The resulting tree tensor network state after applying the TTNO.
    """
    init_state = zipup(ttno, ttns, svd_params=zipup_svd_params)
    return variational_ttns_ttno_application(ttns, ttno,
                                             *args,
                                             init_state=init_state,
                                             svd_params=var_svd_params,
                                             **kwargs)

def half_dm_variational(ttns: TTNS,
                        ttno: TTNO,
                        *args,
                        dm_svd_params: SVDParameters | None = None,
                        var_svd_params: SVDParameters | None = None,
                        **kwargs
                        ) -> TTNS:
    """
    Apply a TTNO to a TTNS using the half density matrix approach as an
    initial guess for variational fitting.

    Args:
        ttns (TTNS): The tree tensor network state to which the TTNO is
            applied.
        ttno (TTNO): The tree tensor network operator to apply.
        *args: Additional positional arguments for the variational fitting.
        dm_svd_params (SVDParameters | None): SVD parameters for the half
            density matrix application. If None, default parameters are used.
        var_svd_params (SVDParameters | None): SVD parameters for the
            variational fitting. If None, default parameters are used.
        **kwargs: Additional keyword arguments for the variational fitting.

    Returns:
        TTNS: The resulting tree tensor network state after applying the TTNO.
    """
    init_state = half_dm_ttns_ttno_application(ttns, ttno, svd_params=dm_svd_params)
    return variational_ttns_ttno_application(ttns, ttno,
                                             *args,
                                             init_state=init_state,
                                             svd_params=var_svd_params,
                                             **kwargs)

def apply_ttno_to_ttns(ttns: TTNS,
                       ttno: TTNO,
                       method: ApplicationMethod,
                       *args,
                       **kwargs) -> TTNS:
    """
    Apply a TTNO to a TTNS using the specified method.

    Args:
        ttns (TTNS): The tree tensor network state to which the TTNO is
            applied.
        ttno (TTNO): The tree tensor network operator to be applied.
        method (ApplicationMethod): The method to use for the application.
        *args: Additional positional arguments for the application function.
        **kwargs: Additional keyword arguments for the application function.

    Returns:
        TTNS: The resulting tree tensor network state after applying the TTNO.
    """
    app_function = method.get_function()
    return app_function(ttns, ttno, *args, **kwargs)

def apply_and_truncate(ttns: TTNS,
                       ttno: TTNO,
                       app_method: ApplicationMethod,
                       trunc_method: TruncationMethod,
                       app_args: tuple = (),
                       app_kwargs: dict | None = None,
                       trunc_args: tuple = (),
                       trunc_kwargs: dict | None = None
                       ) -> TTNS:
    """
    Apply a TTNO to a TTNS and truncate the result.

    Args:
        ttns (TTNS): The tree tensor network state to which the TTNO is
            applied.
        ttno (TTNO): The tree tensor network operator to be applied.
        app_method (ApplicationMethod): The method to use for the application.
        trunc_method (TruncationMethod): The method to use for the truncation.
        app_args (tuple, optional): Additional positional arguments for the
            application function. Defaults to ().
        app_kwargs (dict, optional): Additional keyword arguments for the
            application function. Defaults to {}.
        trunc_args (tuple, optional): Additional positional arguments for the
            truncation function. Defaults to ().
        trunc_kwargs (dict, optional): Additional keyword arguments for the
            truncation function. Defaults to {}.

    Returns:
        TTNS: The resulting tree tensor network state after applying the TTNO
            and truncating.
    """
    if app_kwargs is None:
        app_kwargs = {}
    if trunc_kwargs is None:
        trunc_kwargs = {}
    # Apply the TTNO to the TTNS
    new_ttns = apply_ttno_to_ttns(ttns, ttno,
                                  app_method,
                                  *app_args, **app_kwargs)
    # Truncate the resulting TTNS
    truncated_ttns = truncate_ttns(new_ttns, trunc_method,
                                   *trunc_args, **trunc_kwargs)
    return truncated_ttns

def power_of_ttno(ttns: TTNS,
                  ttno: TTNO,
                  power: int,
                  app_method: ApplicationMethod,
                  *args,
                  **kwargs) -> TTNS:
    """
    Apply a TTNO to a TTNS multiple times.

    Args:
        ttns (TTNS): The tree tensor network state to which the TTNO is
            applied.
        ttno (TTNO): The tree tensor network operator to be applied.
        power (int): The number of times to apply the TTNO.
        app_method (ApplicationMethod): The method to use for the application.
        *args: Additional positional arguments for the application function.
        **kwargs: Additional keyword arguments for the application function.

    Returns:
        TTNS: The resulting tree tensor network state after applying the TTNO
            multiple times.
    """
    result = ttns
    for _ in range(power):
        result = apply_ttno_to_ttns(result, ttno, app_method, *args, **kwargs)
    return result

def compute_power_series(ttns: TTNS,
                         ttno: TTNO,
                         max_power: int,
                         app_method: ApplicationMethod,
                         *args,
                         **kwargs) -> list[TTNS]:
    """
    Compute the power series of a TTNO applied to a TTNS.

    Args:
        ttns (TTNS): The tree tensor network state to which the TTNO is
            applied.
        ttno (TTNO): The tree tensor network operator to be applied.
        max_power (int): The maximum power to compute.
        app_method (ApplicationMethod): The method to use for the application.
        *args: Additional positional arguments for the application function.
        **kwargs: Additional keyword arguments for the application function.

    Returns:
        list[TTNS]: A list of tree tensor network states corresponding to the
            power series of the TTNO applied to the TTNS.
    """
    series = [ttns]
    current = ttns
    for _ in range(1, max_power + 1):
        series.append(current)
        current = apply_ttno_to_ttns(current, ttno, app_method, *args, **kwargs)
    return series
