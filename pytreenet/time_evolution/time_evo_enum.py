"""
This moduel implements an enumeration for the time evolution algorithms
implemented in PyTreeNet.
"""
from enum import Enum

from pytreenet.time_evolution.time_evolution import TimeEvolution
from pytreenet.time_evolution.exact_time_evolution import ExactTimeEvolution
from pytreenet.time_evolution.tdvp_algorithms.firstorderonesite import FirstOrderOneSiteTDVP
from pytreenet.time_evolution.tdvp_algorithms.secondorderonesite import SecondOrderOneSiteTDVP
from pytreenet.time_evolution.tdvp_algorithms.secondordertwosite import SecondOrderTwoSiteTDVP
from pytreenet.time_evolution.tebd import TEBD
from pytreenet.time_evolution.fixed_bug import FixedBUG
from pytreenet.time_evolution.bug import BUG

class TimeEvoAlg(Enum):
    """
    Enumeration for the time evolution algorithms implemented in PyTreeNet.
    """

    SITE1ORDER1TDVP = "site1order1tdvp"
    SITE1ORDER2TDVP = "site1order2tdvp"
    SITE2ORDER2TDVP = "site2order2tdvp"
    TEBD = "tebd"
    FIXEDBUG = "fixedbug"
    BUG = "bug"
    EXACT = "exact"

    def ttn_method(self) -> bool:
        """
        Returns True if the algorithm is based on TTN.
        """
        return self is not TimeEvoAlg.EXACT

    def is_tdvp(self) -> bool:
        """
        Returns True if the algorithm is a TDVP algorithm.
        """
        return self in {TimeEvoAlg.SITE1ORDER1TDVP,
                        TimeEvoAlg.SITE1ORDER2TDVP,
                        TimeEvoAlg.SITE2ORDER2TDVP}

    def is_bug(self) -> bool:
        """
        Returns True if the algorithm is a bug.
        """
        return self in {TimeEvoAlg.FIXEDBUG,
                        TimeEvoAlg.BUG}

    def requires_svd(self) -> bool:
        """
        Returns True if the algorithm requires SVD.
        """
        return self in {TimeEvoAlg.SITE2ORDER2TDVP,
                        TimeEvoAlg.BUG}

    def get_class(self) -> type[TimeEvolution]:
        """
        Returns the class of the time evolution algorithm.
        """
        if self is TimeEvoAlg.SITE1ORDER1TDVP:
            return FirstOrderOneSiteTDVP
        if self is TimeEvoAlg.SITE1ORDER2TDVP:
            return SecondOrderOneSiteTDVP
        if self is TimeEvoAlg.SITE2ORDER2TDVP:
            return SecondOrderTwoSiteTDVP
        if self is TimeEvoAlg.TEBD:
            return TEBD
        if self is TimeEvoAlg.FIXEDBUG:
            return FixedBUG
        if self is TimeEvoAlg.BUG:
            return BUG
        if self is TimeEvoAlg.EXACT:
            return ExactTimeEvolution
        raise ValueError(f"Unknown time evolution algorithm: {self}")

    def get_config_class(self) -> type:
        """
        Returns the configuration class for the time evolution algorithm.
        """
        return self.get_class().config_class

    def get_algorithm_instance(self, *args, **kwargs):
        """
        Returns an instance of the time evolution algorithm.
        """
        return self.get_class()(*args, **kwargs)
