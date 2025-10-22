"""
This module implements the variational way of fitting the application of a TTNO to a TTNS.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ...dmrg.variational_fitting import VariationalFitting, SiteUpdateMethod
from ...util.tensor_splitting import SVDParameters
from ...random.random_ttns import random_like

if TYPE_CHECKING:
    from ..ttns import TTNS

def variational_ttns_ttno_application(ttns, ttno,
                                      init_state: TTNS | None = None,
                                      num_sweeps: int = 5,
                                      max_iter: int = 10,
                                      svd_params: SVDParameters | None = None,
                                      residual_rank: int = 0
                                        ) -> TTNS:
    """
    Apply a TTNO to a TTNS using variational fitting.

    Args:
        ttns (TTNS): The tree tensor network state to which the TTNO is
            applied.
        ttno (TTNO): The tree tensor network operator to apply.
        init_state (TTNS | None): An optional initial guess for the resulting
            TTNS. If None, a random TTNS with the same structure as `ttns` is
            used and the bond dimension of the `'svd_params'` is applied.
        num_sweeps (int): The number of sweeps through the network during the
            variational fitting.
        max_iter (int): The number of iterations for the local fitting at each
            node.
        svd_params (SVDParameters | None): Parameters for SVD truncation
            during the fitting. If None, default parameters are used.
        residual_rank (int): The rank of the residual to be added during the
            fitting. Default is 0, meaning no residual is used.
        
    Returns:
        TTNS: The resulting tree tensor network state after applying the TTNO.
    """
    if svd_params is None:
        svd_params = SVDParameters()
    if init_state is None:
        init_state = random_like(ttns,
                                 bond_dim=svd_params.max_bond_dim)
    fitter = VariationalFitting([ttno],
                                [ttns],
                                init_state,
                                num_sweeps,
                                max_iter,
                                svd_params,
                                site = SiteUpdateMethod.ONE_SITE,
                                residual_rank=residual_rank,
                                dtype=np.complex64
                                )
    fitter.run()
    return fitter.get_result_state()
