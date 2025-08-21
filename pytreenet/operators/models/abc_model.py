"""
This module implements an abstract parent class for quantum system models.
"""
from abc import ABC, abstractmethod
from typing import Self
import inspect

import numpy as np
import numpy.typing as npt

from .topology import Topology
from ..hamiltonian import Hamiltonian
from ...core.ttn import TreeTensorNetwork
from ...ttno.ttno_class import TreeTensorNetworkOperator
from ...util.ttn_exceptions import non_negativity_check
from ...special_ttn.mps import MatrixProductState

class Model(ABC):
    """
    A model of a quantum system is a very abstract description of a quantum
    systems.

    It is defined only by a set of parameters, that can be used to define
    the Hamiltonian for a given structure of the quantum system.

    """

    @classmethod
    def from_dataclass(cls, dataclass_instance) -> Self:
        """
        Initializes the model from a dataclass instance.

        Args:
            dataclass_instance (Self): An instance of a dataclass containing
                the parameters for the model.

        Returns:
            Self: An instance of the model initialized with the parameters
                from the dataclass.
        """
        param_dict = dataclass_instance.__dict__
        sig = inspect.signature(cls.__init__)
        valid_params = sig.parameters.keys()
        valid_dict = {k: v for k, v in param_dict.items()
                      if k in valid_params}
        return cls(**valid_dict)

    @abstractmethod
    def generate_hamiltonian(self,
                               structure: TreeTensorNetwork | list[tuple[str, ...]]
                               ) -> Hamiltonian:
        """
        Generates the Hamiltonian for the given structure of the quantum system.

        Uses the internal parameters of the model to generate the Hamiltonian.

        Args:
            structure (TreeTensorNetwork | list[tuple[str, ...]]): The structure
                of the quantum system.

        Returns:
            Hamiltonian: The Hamiltonian of the quantum system.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def generate_chain_structure(self,
                           site_ids: list[str]
                           ) -> list[tuple[str, ...]]:
        """
        Generates the chain structure for the quantum system.

        Args:
            site_ids (list[str]): The identifiers for the sites.

        Returns:
            list[str]: A list of site identifiers for the chain structure.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_chain_model(self,
                             num_sites: int,
                             site_ids: list[str] | str = "site"
                             ) -> Hamiltonian:
        """
        Generates the Hamiltonian on a chain.

        Args:
            num_sites (int): The number of sites in the chain.
            site_ids (list[str] | str, optional): The identifiers for the
                sites. If a single string is provided, it will be used as a
                prefix for each site ID. If a list is provided, it must have
                the same length as `num_sites`. Defaults to "site".

        Returns:
            Hamiltonian: The Hamiltonian of the chain model.
        
        Raises:
            ValueError: If `site_ids` is a list and its length does not match
                `num_sites`.
        """
        if isinstance(site_ids, str):
            site_ids = generate_chain_indices(num_sites,
                                              site_ids=site_ids)
        else:
            if len(site_ids) != num_sites:
                errstr = f"Length of site_ids ({len(site_ids)}) does not match num_sites ({num_sites})!"
                raise ValueError(errstr)
        structure = self.generate_chain_structure(site_ids)
        return self.generate_hamiltonian(structure)

    @abstractmethod
    def generate_t_topology_structure(self,
                                      site_ids: tuple[list[str],list[str],list[str]]
                                      ) -> list[tuple[str, ...]]:
        """
        Generates the T-shaped topology structure for the quantum system.

        Args:
            site_ids (tuple[list[str], list[str], list[str]]): The identifiers
                for the sites in the T-topology. Each list corresponds to one
                of the three arms of the T. 

        Returns:
            list[tuple[str, ...]]: The T-shaped topology structure.
        
        Raises:
            ValueError: If `site_ids` is a list and its length does not match
                `chain_length * 3`.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_t_topology_model(self,
                                  chain_length: int,
                                  site_ids: tuple[list[str],list[str],list[str]] | str = "site"
                                  ) -> Hamiltonian:
        """
        Generates the Hamiltonian on a T-shaped topology.

        Args:
            chain_length (int): The length each chain in the T-topology.
            site_ids (tuple[list[str], list[str], list[str]] | str, optional):
                The identifiers for the sites in the T-topology. If a single
                string is provided, it will be used as a prefix for each site ID.
                If a tuple of lists is provided, each list must have the same
                length as `chain_length`. Defaults to "site".
        
        Returns:
            Hamiltonian: The Hamiltonian of the T-topology model.
        
        Raises:
            ValueError: If `site_ids` is a list and its length does not match
                `chain_length * 3`.
        """
        if isinstance(site_ids, str):
            site_ids = generate_t_topology_indices(chain_length,
                                                   site_ids=site_ids)
        else:
            for sites in site_ids:
                if len(sites) != chain_length:
                    errstr = f"Length of site_ids ({len(sites)}) does not match chain_length ({chain_length})!"
                    raise ValueError(errstr)
        structure = self.generate_t_topology_structure(site_ids)
        return self.generate_hamiltonian(structure)

    @abstractmethod
    def generate_2d_structure(self,
                              site_ids: list[list[str]]
                              ) -> list[tuple[str, ...]]:
        """
        Generates a 2D structure for the quantum system.

        Args:
            site_ids (list[list[str]]): A list of lists, where each inner list
                contains the identifiers for the sites in that row.
        
        Returns:
            list[tuple[str, ...]]: The 2D structure represented as a list of
                tuples, where each tuple contains the identifiers of connected
                sites.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_2d_model(self,
                          num_rows: int,
                          num_cols: int | None = None,
                          site_ids: list[list[str]] | str = "site"
                          ) -> Hamiltonian:
        """
        Generates the Hamiltonian on a 2D structure.

        Args:
            num_rows (int): The number of rows in the 2D structure.
            num_cols (int, optional): The number of columns in the 2D structure.
                If not provided, it defaults to `num_rows`. Defaults to None.
            site_ids (list[list[str]] | str, optional): The identifiers for the
                sites. If a single string is provided, it will be used as a
                prefix for each site ID. If a list of lists is provided, each
                inner list must have the same length as `num_cols`. Defaults to "site".

        Returns:
            Hamiltonian: The Hamiltonian of the 2D model.
        
        Raises:
            ValueError: If `site_ids` is a list and has an inconsistent
                structure.
        """
        if isinstance(site_ids, str):
            site_ids = generate_2d_indices(num_rows,
                                           num_cols=num_cols,
                                           site_ids=site_ids)
        else:
            if len(site_ids) != num_rows:
                errstr = f"Length of site_ids ({len(site_ids)}) does not match num_rows ({num_rows})!"
                raise ValueError(errstr)
            if num_cols is None:
                num_cols = num_rows
            for row in site_ids:
                if len(row) != num_cols:
                    errstr = f"Length of row ({len(row)}) does not match num_cols ({num_cols})!"
                    raise ValueError(errstr)
        structure = self.generate_2d_structure(site_ids)
        return self.generate_hamiltonian(structure)

    def generate_by_topology(self,
                            topology: Topology,
                            size_parameter: int,
                            site_id_prefix: str = "site"
                            ) -> Hamiltonian:
        """
        Generates the Hamiltonian based on the topology of the quantum system.

        Args:
            size_parameter (int): The size parameter for the topology.
                For a chain, this is the number of sites; for a T-topology,
                this is the length of each chain; for a 2D structure, this is
                only valid for a square lattice, where it represents the
                number of rows and columns.
            site_id_prefix (str, optional): The prefix for each site identifier.
                Defaults to "site".

        Returns:
            Hamiltonian: The Hamiltonian of the quantum system based on the
                specified topology.
        """
        if topology == Topology.CHAIN:
            return self.generate_chain_model(size_parameter,
                                             site_ids=site_id_prefix)
        if topology == Topology.TTOPOLOGY:
            return self.generate_t_topology_model(size_parameter,
                                                  site_ids=site_id_prefix)
        if topology == Topology.SQUARE:
            return self.generate_2d_model(size_parameter,
                                          site_ids=site_id_prefix)
        errstr = f"Unsupported topology: {topology}."
        errstr += "Cannot generate Hamiltonian!"
        raise ValueError(errstr)

    def generate_matrix(self,
                        topology: Topology,
                        size_parameter: int
                        ) -> npt.NDArray[np.complex64]:
        """
        Generates the Hamiltonian matrix for the specified topology and size.
        """
        if topology not in [Topology.CHAIN, Topology.TTOPOLOGY]:
            errstr = f"Exact solution not implemented for this topology: {topology}!"
            raise NotImplementedError(errstr)
        hamiltonian = self.generate_by_topology(topology,
                                                size_parameter)
        # We need a reference ttns to buid the ttno.
        # We use an MPS, as it can represent all sites by simply being on the chain.
        num_sites = topology.num_sites(size_parameter)
        mps = MatrixProductState.constant_product_state(0,2,num_sites)
        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian,
                                                          mps)
        matrix, _ = ttno.as_matrix()
        return matrix

def generate_chain_indices(num_sites: int,
                           site_ids: str = "site"
                           ) -> list[str]:
    """
    Generates a list of site identifiers for a chain structure.

    Args:
        num_sites (int): The number of sites in the chain.
        site_ids (str, optional): The prefix for each site identifier.
            Defaults to "site".

    Returns:
        list[str]: A list of site identifiers for the chain.
    """
    non_negativity_check(num_sites, "num_sites")
    return [site_ids + str(i) for i in range(num_sites)]

def generate_t_topology_indices(chain_length: int,
                                site_ids: str = "site"
                                ) -> tuple[list[str], list[str], list[str]]:
    """
    Generates lists of site identifiers for a T-shaped topology.

    Args:
        chain_length (int): The length of each chain in the T-topology.
        site_ids (str, optional): The prefix for each site identifier.
            Defaults to "site".

    Returns:
        tuple[list[str], list[str], list[str]]: A tuple containing three lists,
            each representing the identifiers for the sites in the three arms
            of the T-topology.
    """
    non_negativity_check(chain_length, "chain_length")
    sites_1 = [site_ids + str(i) for i in range(chain_length)]
    sites_2 = [site_ids + str(i + chain_length) for i in range(chain_length)]
    sites_3 = [site_ids + str(i + 2 * chain_length) for i in range(chain_length)]
    return (sites_1, sites_2, sites_3)

def generate_2d_indices(num_rows: int,
                        num_cols: int | None = None,
                        site_ids: str = "site"
                        ) -> list[list[str]]:
    """
    Generates a 2D structure of site identifiers.

    Args:
        num_rows (int): The number of rows in the 2D structure.
        num_cols (int, optional): The number of columns in the 2D structure.
            If not provided, it defaults to `num_rows`. Defaults to None.
        site_ids (str, optional): The prefix for each site identifier.
            Defaults to "site".

    Returns:
        list[list[str]]: A list of lists, where each inner list contains the
            identifiers for the sites in that row.
    """
    non_negativity_check(num_rows, "num_rows")
    if num_cols is None:
        num_cols = num_rows
    else:
        non_negativity_check(num_cols, "num_cols")
    return [[site_ids + str(i + j * num_cols)
             for i in range(num_cols)]
            for j in range(num_rows)]
