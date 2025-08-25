"""
This module implements functions, that help to generate the operators that are
to be evaluated during a time evolution.
"""
from __future__ import annotations
from typing import Callable

from numpy import ndarray, zeros_like

from ..tensorproduct import TensorProduct
from ..sim_operators import (create_single_site_observables)
from ..common_operators import (pauli_matrices,
                                bosonic_operators)
from ...core.ttn import TreeTensorNetwork
from ...util.std_utils import average_data

from .topology import Topology

def _eval_ops_from_topology(topology: Topology,
                            system_size: int,
                            generator_function: Callable,
                            site_prefix: str = "site",
                            **kwargs
                            ) -> dict[str, TensorProduct]:
    """
    Generates the local magnetisation operator for a given topology.

    Args:
        topology (Topology): The topology of the system.
        system_size (int): The characteristic size of the system.
        site_prefix (str): The prefix for the site identifiers.
        Defaults to "site".
        **kwargs: Other keyword arguments of `generator_function`.

    Returns:
        dict[str, TensorProduct]: The local operators to evaluate.

    """
    num_sites = topology.num_sites(system_size)
    structure = [site_prefix + str(i)
                 for i in range(num_sites)]
    return generator_function(structure, **kwargs)

def local_magnetisation_from_topology(topology: Topology,
                                      system_size: int,
                                      site_prefix: str = "site"
                                        ) -> dict[str, TensorProduct]:
    """
    Generates the local magnetisation operator for a given topology.

    Args:
        topology (Topology): The topology of the system.
        system_size (int): The characteristic size of the system.
        site_prefix (str): The prefix for the site identifiers.
            Defaults to "site".

    Returns:
        dict[str, TensorProduct]: The local operators to evaluate.
    
    """
    return _eval_ops_from_topology(topology,
                                   system_size,
                                   local_magnetisation,
                                   site_prefix=site_prefix)

def local_magnetisation(structure: TreeTensorNetwork | list[str]
                        ) -> dict[str, TensorProduct]:
    """
    Generates the local magnetisation operator for a given tree structure.

    Args:
        structure (Union[TreeTensorNetwork,List[str]]): The tree structure for
            which the local magnetisation operator should be generated. Can
            also be a list of node identifiers.
    
    Returns:
        Dict[str,TensorProduct]: The local magnetisation operators.

    """
    sigma_z = pauli_matrices()[2]
    return create_single_site_observables(sigma_z, structure)

def total_magnetisation(local_magnetisations: list[ndarray]
                        ) -> ndarray:
    """
    Computes the total magnetisation from the local magnetisations.
    
    Args:
        local_magnetisations (List[ndarray]): The local magnetisations as a
            list of arrays, where each array contains the local magnetisations
            for one site for different times.

    Returns:
        ndarray: The total magnetisation

        .. math::
            M = 1/L \sum_i^L m_i

    """
    return average_data(local_magnetisations)

def local_number_operator_from_topology(topology: Topology,
                                        system_size: int,
                                        dim: int = 2,
                                        site_prefix: str = "site"
                                        ) -> dict[str,TensorProduct]:
    """
    Generates the local number operators for a given topology.

    Args:
        topology (Topology): The topology of the system.
        system_size (int): The characteristic size of the system.
        dim (int): The dimension of the local bosonic Hilbert space.
        site_prefix (str): The prefix for the site identifiers.
            Defaults to "site".

    Returns:
        dict[str, TensorProduct]: The local operators to evaluate.
    
    """
    return _eval_ops_from_topology(topology,
                                   system_size,
                                   local_number_operator,
                                   site_prefix=site_prefix,
                                   dim=dim)

def local_number_operator(structure: TreeTensorNetwork | list[str],
                          dim: int = 2
                          ) -> dict[str, TensorProduct]:
    """
    Generate the local number operators for a given structure.

    Args:
        structure (TreeTensorNetwork | list[str]): The structure of the
            underlying system.
        dim (int): The dimension of the local bosonic space.

    Returns:
        Dict[str,TensorProduct]: The local number operators.

    """
    number_op = bosonic_operators(dimension=dim)[2]
    return create_single_site_observables(number_op, structure)

def total_particle_number(local_particle_num: list[ndarray]
                          ) -> ndarray:
    """
    Evaluate the total particle number
    
    Args:
        local_particle_num (list[ndarray]): Arrays that contain the determined
            local particle number for all time steps. Each Array corresponds
            to one site.

    Returns:
        ndarray: The total particle number
     
            .. math::
                N = \sum_i^L N_i
    """
    if len(local_particle_num) == 0:
        errstr = "No data given to evaluate from!"
        raise ValueError(errstr)
    out = zeros_like(local_particle_num[0])
    for res in local_particle_num:
        out += res
    return out
