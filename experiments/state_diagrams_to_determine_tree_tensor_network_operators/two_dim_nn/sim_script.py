"""
Script for the construction of two dimensional nearest neighbour interaction
TTNOs.
"""
from dataclasses import dataclass
from enum import Enum
import os

from h5py import File

from pytreenet.operators.models.two_site_model import (HeisenbergModel,
                                                       IsingModel)
from pytreenet.ttno.ttno_class import (TTNOFinder,
                                       TTNO)
from pytreenet.util.experiment_util.sim_params import SimulationParameters
from pytreenet.util.experiment_util.script_util import script_main
from pytreenet.special_ttn.special_states import (generate_zero_state,
                                                  Topology,
                                                  TTNStructure,
                                                  STANDARD_NODE_PREFIX)

class ModelKind(Enum):
    ISING = "Ising"
    XXZ = "XXZ"

@dataclass
class TwoDimParams(SimulationParameters):
    model: ModelKind = ModelKind.ISING
    finder: TTNOFinder = TTNOFinder.SGE
    ttn_structure: TTNStructure = TTNStructure.MPS
    sys_size: int = 2

def generate_ising(sys_size: int,
                   ttn_structure: TTNStructure,
                   finder: TTNOFinder) -> TTNO:
    """Generates the TTNO for the 2D nearest neighbour Ising model.

    Args:
        sys_size (int): The size of one side of the 2D grid.
        ttn_structure (TTNStructure): The TTN structure to be used.
        finder (TTNOFinder): The TTNO finding algorithm to be used.
    
    Returns:
        TTNO: The generated TTNO for the 2D Ising model.
    """
    model = IsingModel(factor=1, ext_magn=2)
    ham = model.generate_2d_model(sys_size,
                                  site_ids=STANDARD_NODE_PREFIX)
    ref_tree = generate_zero_state(sys_size,
                                   ttn_structure,
                                   topology=Topology.SQUARE)
    ttno = TTNO.from_hamiltonian(ham, ref_tree,
                                 method=finder)
    return ttno

def generate_xxz(sys_size: int,
                     ttn_structure: TTNStructure,
                     finder: TTNOFinder) -> TTNO:
    """Generates the TTNO for the 2D nearest neighbour XXZ model.

    Args:
        sys_size (int): The size of one side of the 2D grid.
        ttn_structure (TTNStructure): The TTN structure to be used.
        finder (TTNOFinder): The TTNO finding algorithm to be used.
    
    Returns:
        TTNO: The generated TTNO for the 2D XXZ model.
    """
    model = HeisenbergModel(x_factor=1,
                            y_factor=None,
                            z_factor=0.5)
    ham = model.generate_2d_model(sys_size,
                                  site_ids=STANDARD_NODE_PREFIX)
    ref_tree = generate_zero_state(sys_size,
                                   ttn_structure,
                                   topology=Topology.SQUARE)
    ttno = TTNO.from_hamiltonian(ham, ref_tree,
                                 method=finder)
    return ttno

def find_ttno(params: TwoDimParams) -> TTNO:
    """Finds the TTNO for the specified model and parameters.

    Args:
        params (TwoDimParams): The parameters for the TTNO finding.
    
    Returns:
        TTNO: The found TTNO.
    """
    if params.model == ModelKind.ISING:
        ttno = generate_ising(params.sys_size,
                              params.ttn_structure,
                              params.finder)
    elif params.model == ModelKind.XXZ:
        ttno = generate_xxz(params.sys_size,
                             params.ttn_structure,
                             params.finder)
    else:
        raise ValueError(f"Unknown model kind: {params.model}")
    return ttno

def save_bond_dim(dir_path: str,
                  ttno: TTNO,
                  params: TwoDimParams):
    """Saves the bond dimensions of the TTNO to a file.

    Args:
        dir_path (str): The directory path to save the bond dimensions.
        ttno (TTNO): The TTNO whose bond dimensions are to be saved.
        params (TwoDimParams): The parameters used for the TTNO finding.
    """
    param_hash = params.get_hash()
    file_path = os.path.join(dir_path, f"bond_dims_{param_hash}.h5")
    with File(file_path, "w") as f:
        params.save_to_h5(f)
        max_bd = ttno.max_bond_dim()
        avg_bd = ttno.avg_bond_dim()
        f.create_dataset("max_bond_dim", data=max_bd)
        f.create_dataset("avg_bond_dim", data=avg_bd)
        f.create_dataset("sys_size", data=params.sys_size)

def run_and_save(params: TwoDimParams,
                 save_dir: str):
    """Runs the TTNO finding and saves the bond dimensions.

    Args:
        params (TwoDimParams): The parameters for the TTNO finding.
        save_dir (str): The directory to save the bond dimensions.
    """
    ttno = find_ttno(params)
    save_bond_dim(save_dir, ttno, params)

if __name__ == "__main__":
    script_main(run_and_save,
                TwoDimParams)
