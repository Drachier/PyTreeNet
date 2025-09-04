"""
Module containing an enumeration for the status of an experiment.
"""
from enum import Enum

class Status(Enum):
    """
    Enumeration for the status of a simulation.
    """
    SUCCESS = "success"
    FAILED = "failed"
    RUNNING = "running"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

    def __lt__(self, other):
        if not isinstance(other, Status):
            return NotImplemented
        order = [Status.UNKNOWN, Status.RUNNING, Status.FAILED, Status.SUCCESS, Status.TIMEOUT]
        return order.index(self) < order.index(other)

    def __le__(self, other):
        if not isinstance(other, Status):
            return NotImplemented
        order = [Status.UNKNOWN, Status.RUNNING, Status.FAILED, Status.SUCCESS, Status.TIMEOUT]
        return order.index(self) <= order.index(other)

    def __gt__(self, other):
        if not isinstance(other, Status):
            return NotImplemented
        order = [Status.UNKNOWN, Status.RUNNING, Status.FAILED, Status.SUCCESS, Status.TIMEOUT]
        return order.index(self) > order.index(other)

    def __ge__(self, other):
        if not isinstance(other, Status):
            return NotImplemented
        order = [Status.UNKNOWN, Status.RUNNING, Status.FAILED, Status.SUCCESS, Status.TIMEOUT]
        return order.index(self) >= order.index(other)
