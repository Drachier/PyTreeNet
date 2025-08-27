"""
This module provides the utility required to plot errors against parameters.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from .convergence import ConvergingResults, ReferenceResults
from ..line_config import LineConfig

if TYPE_CHECKING:
    from typing import Self
    import numpy.typing as npt


class EVPResults:
    """
    This class contains everything nessecarry for one batch of results to be
    plotted.
    """

    def __init__(self,
                 errors: npt.NDArray[np.floating],
                 parameters: npt.NDArray[np.floating],
                 line_config: LineConfig | None = None
                 ) -> None:
        """
        Initialize the EVPResults object with error and parameter data.

        Args:
            errors (npt.NDArray[np.floating]): The error data to be plotted.
            parameters (npt.NDArray[np.floating]): The parameter data to be
                plotted.
            line_config (LineConfig | None, optional): The line configuration
                for the plot. Defaults to None.

        """
        if errors.shape != parameters.shape:
            errstr = "Error and parameter arrays must have the same shape!"
            raise ValueError(errstr)
        self.errors = errors
        self.parameters = parameters
        if line_config is None:
            line_config = LineConfig()
        self.line_config = line_config

    @classmethod
    def from_converging_results_error(cls,
                                      results: ConvergingResults,
                                      reference: ReferenceResults,
                                      equivalent_params: npt.NDArray[np.floating] | None = None
                                      ) -> Self:
        """
        Obtain EVPResults as direct error compared to a reference from
        ConvergingResults.

        Args:
            results (ConvergingResults): The converging results to compare.
            reference (ReferenceResults): The reference results to compare
                against.
            equivalent_params (npt.NDArray[np.floating] | None): The equivalent
                parameters to use for the comparison. If None, the original
                parameters from the results will be used. Defaults to None.

        Returns:
            Self: The constructed EVPResults object.
        """
        equivalent_params = _init_equivalent_params(results,
                                                    equivalent_params)
        errors = results.get_errors(reference).accumulated_results()
        return cls(errors, equivalent_params,
                   line_config=results.line_config)

    @classmethod
    def from_converging_results_convmeas(cls,
                                         results: ConvergingResults,
                                         equivalent_params: npt.NDArray[np.floating] | None = None
                                         ) -> Self:
        """
        Obtain an EVPResults object from the convergence measure of converging
        results.

        Args:
            results (ConvergingResults): The converging results to use.
            equivalent_params (npt.NDArray[np.floating] | None): The equivalent
                parameters to use for the comparison. If None, the original
                parameters from the results will be used. Defaults to None.

        Returns:
            Self: The constructed EVPResults object.
        """
        equivalent_params = _init_equivalent_params(results,
                                                    equivalent_params)
        errors = results.get_convergence().accumulated_results()
        return cls(errors, equivalent_params,
                   line_config=results.line_config)


def _init_equivalent_params(results: ConvergingResults,
                            equivalent_params: npt.NDArray[np.floating] | None
                            ) -> npt.NDArray[np.floating]:
    """
    Deals with the initialization of equivalent parameters.

    Args:
        results (ConvergingResults): The converging results to use.
        equivalent_params (npt.NDArray[np.floating] | None): The equivalent
            parameters to use for the comparison. If None, the original
            parameters from the results will be used.

    Returns:
        npt.NDArray[np.floating]: The initialized equivalent parameters.
    """
    if equivalent_params is None:
        try:
            eparams = results.conv_param_values
            equivalent_params = np.asarray(eparams, dtype=np.float64)
        except Exception as e:
            raise TypeError("Parameters not convertible to float!") from e
    return equivalent_params
