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
        raise ValueError(f"Unknown application method: {self}")

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
