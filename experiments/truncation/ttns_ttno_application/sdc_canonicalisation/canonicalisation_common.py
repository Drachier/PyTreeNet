"""
Implements the common functionality of the SDC (half density matrix) TTNO-TTNS
application simulations comparing different root nodes and canonicalisation
schemes for MPS.

For every parameter instance, the same random MPS and MPO are used to run six
different instances of the SDC application:
    - the left border node as root without any canonicalisation,
    - the right border node as root without any canonicalisation,
    - the left border node as root with both MPS and MPO orthogonalised to
      this root node,
    - the right border node as root with both MPS and MPO orthogonalised to
      this root node,
    - the left border node as root and the right border node as the
      orthogonality centre of both MPS and MPO,
    - the right border node as root and the left border node as the
      orthogonality centre of both MPS and MPO.

All six instances start from the same MPS and MPO and perform their
modification on a copy of them. The errors and runtimes of all six instances
are saved to the same file.

The random elements are drawn from either a uniform or a Gaussian
distribution, which is switched via the `distribution` field of the
simulation parameters. The distribution parameters are fixed.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from time import time
from copy import deepcopy

from pytreenet.ttns.ttns_ttno.application import (ApplicationMethod,
                                                  apply_ttno_to_ttns)
from pytreenet.random.random_matrices import RandomDistribution
from pytreenet.random.random_special_ttns import random_ttns
from pytreenet.random.random_special_ttno import random_ttno
from pytreenet.special_ttn.mps import MatrixProductTree
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.time_evolution.results import Results
from pytreenet.util.tensor_splitting import SVDParameters

from application_common import (ApplicationParams,
                                bond_dim_range,
                                save_results)

class ElementDistribution(Enum):
    """
    The distribution from which the random MPS and MPO elements are drawn.

    Attributes:
        UNIFORM: Uniform distribution on the interval [-0.5, 1.0].
        GAUSSIAN: Zero mean (complex) Gaussian distribution with standard
            deviation 1.0.
    """
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"

@dataclass
class SDCCanonicalisationParams(ApplicationParams):
    """
    Parameters for the SDC application simulation comparing different root
    nodes and canonicalisation schemes.

    Only the SDC (half density matrix) application method is supported. The
    structure has to be an MPS, since the root and orthogonality centre are
    varied between the border nodes of the chain. The parameters of the
    random element distribution are fixed and only the distribution itself
    can be switched via the `distribution` field.
    """
    appl_method: ApplicationMethod = ApplicationMethod.HALF_DENSITY_MATRIX
    distribution: ElementDistribution = ElementDistribution.UNIFORM

    def random_generation_kwargs(self) -> dict:
        """
        Returns the keyword arguments for the random MPS/MPO generation.

        The parameters of the distribution are fixed, i.e., the uniform
        distribution is drawn from the interval [-0.5, 1.0] and the Gaussian
        distribution has zero mean and standard deviation 1.0.

        Returns:
            dict: Keyword arguments passed on to `crandn`.
        """
        if self.distribution is ElementDistribution.UNIFORM:
            return {"seed": self.seed,
                    "distribution": RandomDistribution.UNIFORM,
                    "low": -0.5,
                    "high": 1.0}
        return {"seed": self.seed,
                "distribution": RandomDistribution.NORMAL,
                "scale": 1.0}

class CanonicalisationMode(Enum):
    """
    The canonicalisation scheme applied to the MPS and MPO of an instance.

    Attributes:
        NONE: No canonicalisation is performed.
        ROOT: Both MPS and MPO are orthogonalised to the root node, i.e., the
            root node is the orthogonality centre.
        OPPOSITE: Both MPS and MPO are orthogonalised to the border node
            opposite to the root, i.e., the other border node is the
            orthogonality centre.
    """
    NONE = "none"
    ROOT = "root"
    OPPOSITE = "opposite"

@dataclass(frozen=True)
class RootConfig:
    """
    The configuration of a single instance of the SDC application.

    Attributes:
        name (str): The identifier of the configuration. Used as a prefix for
            the result identifiers.
        root_is_first_site (bool): Whether the first (leftmost) site is the
            root node. Otherwise the last (rightmost) site is the root node.
        canonicalisation (CanonicalisationMode): The canonicalisation scheme
            applied to the MPS and MPO.
    """
    name: str
    root_is_first_site: bool
    canonicalisation: CanonicalisationMode

ROOT_CONFIGS = (RootConfig("left_root_uncanonicalised", True,
                            CanonicalisationMode.NONE),
                RootConfig("right_root_uncanonicalised", False,
                           CanonicalisationMode.NONE),
                RootConfig("left_root_centred", True,
                           CanonicalisationMode.ROOT),
                RootConfig("right_root_centred", False,
                           CanonicalisationMode.ROOT),
                RootConfig("left_root_right_centre", True,
                           CanonicalisationMode.OPPOSITE),
                RootConfig("right_root_left_centre", False,
                           CanonicalisationMode.OPPOSITE))

BOND_DIM_ID = "bond_dim"

def config_res_ids(config: RootConfig) -> tuple[str, str]:
    """
    Generates the result identifiers for the truncation error and runtime of
    the given configuration.

    Args:
        config (RootConfig): The configuration to generate the identifiers
            for.

    Returns:
        tuple[str, str]: The identifiers of the truncation error and the
            runtime results.
    """
    return (config.name + "_trunc_error",
            config.name + "_run_time")

def init_results(params: ApplicationParams) -> Results:
    """
    Initializes the results object for the simulation.

    Args:
        params (ApplicationParams): The parameters for the simulation.

    Returns:
        Results: The initialized and empty results object.
    """
    num_res = len(list(bond_dim_range(params))) - 1
    res_dtypes = {BOND_DIM_ID: int}
    for config in ROOT_CONFIGS:
        trunc_error_id, run_time_id = config_res_ids(config)
        res_dtypes[trunc_error_id] = float
        res_dtypes[run_time_id] = float
    results = Results()
    results.initialize(res_dtypes,
                       num_res,
                       with_time=False)
    return results

def change_root(matrix_product: MatrixProductTree,
                root_site: int
                ) -> MatrixProductTree:
    """
    Changes the root node of a matrix product tree to the given site.

    The root is moved on a deepcopy of the given matrix product tree, such
    that the represented state or operator is unchanged. Only the node which
    is the root differs.

    Args:
        matrix_product (MatrixProductTree): The matrix product tree, i.e., an
            MPS or MPO. Its nodes must be named "site{index}" following the
            convention of the random MPS/MPO generation.
        root_site (int): The index of the site which should become the root
            node.

    Returns:
        MatrixProductTree: A copy of the given matrix product tree with the
            new root node.
    """
    rerooted = deepcopy(matrix_product)
    rerooted.move_root_to_node(f"site{root_site}")
    return rerooted

def prepare_instance(ttns: MatrixProductTree,
                     ttno: MatrixProductTree,
                     config: RootConfig,
                     sys_size: int
                     ) -> tuple[MatrixProductTree, MatrixProductTree]:
    """
    Prepares the MPS and MPO for a single instance of the experiment.

    Copies of the given MPS and MPO are rooted according to the
    configuration. If the configuration requests a canonicalisation, both
    copies are orthogonalised to the specified orthogonality centre
    afterwards.

    Args:
        ttns (MatrixProductTree): The MPS to prepare.
        ttno (MatrixProductTree): The MPO to prepare.
        config (RootConfig): The configuration of the instance.
        sys_size (int): The number of sites of the MPS and MPO.

    Returns:
        tuple[MatrixProductTree, MatrixProductTree]: The prepared copies of
            the MPS and MPO.
    """
    root_site = 0 if config.root_is_first_site else sys_size - 1
    instance_ttns = change_root(ttns, root_site)
    instance_ttno = change_root(ttno, root_site)
    if config.canonicalisation is CanonicalisationMode.NONE:
        return instance_ttns, instance_ttno
    if config.canonicalisation is CanonicalisationMode.ROOT:
        centre_site = root_site
    else:
        centre_site = sys_size - 1 if config.root_is_first_site else 0
    centre_id = f"site{centre_site}"
    instance_ttns.canonical_form(centre_id)
    instance_ttno.canonical_form(centre_id)
    return instance_ttns, instance_ttno

def run_simulation(params: SDCCanonicalisationParams) -> Results:
    """
    Runs the SDC application simulation comparing the different root nodes
    and canonicalisation schemes.

    Args:
        params (SDCCanonicalisationParams): The parameters for the simulation.

    Returns:
        Results: The results of the simulation.
    """
    if params.structure is not TTNStructure.MPS:
        errstr = "This experiment is only implemented for MPS!"
        raise ValueError(errstr)
    if params.appl_method is not ApplicationMethod.HALF_DENSITY_MATRIX:
        errstr = "This experiment only supports the SDC"
        errstr += " (half density matrix) application method!"
        raise ValueError(errstr)
    random_kwargs = params.random_generation_kwargs()
    ttns = random_ttns(params.structure,
                       params.sys_size,
                       params.phys_dim,
                       params.bond_dim,
                       **random_kwargs)
    ttno = random_ttno(params.structure,
                       params.sys_size,
                       params.phys_dim,
                       params.bond_dim,
                       **random_kwargs)
    # The exact reference states are computed once per configuration from the
    # same starting MPS and MPO. They share the root of their configuration,
    # such that the distances to the approximations can be computed.
    exact_ttnss = {}
    for config in ROOT_CONFIGS:
        instance_ttns, instance_ttno = prepare_instance(ttns,
                                                       ttno,
                                                       config,
                                                       params.sys_size)
        exact_ttnss[config.name] = apply_ttno_to_ttns(instance_ttns,
                                                      instance_ttno,
                                                      ApplicationMethod.DIRECT)
    sdc_function = params.appl_method.get_function()
    results = init_results(params)
    for idx, bond_dim in enumerate(bond_dim_range(params)):
        svd_params = SVDParameters(max_bond_dim=bond_dim)
        results.set_element(BOND_DIM_ID, idx, bond_dim)
        for config in ROOT_CONFIGS:
            start_time = time()
            instance_ttns, instance_ttno = prepare_instance(ttns,
                                                            ttno,
                                                            config,
                                                            params.sys_size)
            approx_ttns = sdc_function(instance_ttns,
                                       instance_ttno,
                                       svd_params=svd_params)
            end_time = time()
            trunc_error = exact_ttnss[config.name].distance(approx_ttns,
                                                            normalise=True)
            trunc_error_id, run_time_id = config_res_ids(config)
            results.set_element(trunc_error_id, idx, trunc_error)
            results.set_element(run_time_id, idx, end_time - start_time)
    return results

def run_and_save(params: SDCCanonicalisationParams,
                 save_directory: str
                 ) -> None:
    """
    Runs the SDC canonicalisation comparison simulation and saves the result.
    """
    results = run_simulation(params)
    save_results(params,
                 results,
                 save_directory)
