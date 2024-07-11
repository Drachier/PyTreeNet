
from typing import Union
from copy import deepcopy
from argparse import ArgumentParser
import re

import numpy as np

from pytreenet.core.node import Node
from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.time_evolution.tebd import TEBD
from pytreenet.time_evolution.trotter import TrotterSplitting, TrotterStep
from pytreenet.time_evolution.ttn_time_evolution import TTNTimeEvolutionConfig
from pytreenet.util import SVDParameters

def choose_state(position: int) -> np.ndarray:
    """
    Choose the local state for the initial state.
    
    They flip depending on the distance to the root.

    Args:
        position (int): The position in the chain.
    """
    state = np.zeros((2, ))
    if position % 2 == 0:
        state[0] = 1
    else:
        state[1] = 1
    return state

def create_init_state(length: int) -> TreeTensorNetworkState:
    """
    Create the initial state for the time evolution as TTN.

    Args:
        length (int): The length of the chains.
    """
    state = np.zeros((2, ))
    zero_state = deepcopy(state)
    zero_state[0] = 1
    one_state = deepcopy(state)
    one_state[1] = 1
    ttns = TreeTensorNetworkState()
    center_node = Node(identifier="0")
    center_tensor = deepcopy(zero_state.reshape(1,1,1,2))
    ttns.add_root(center_node, center_tensor)
    for chain in range(3):
        for chain_position in range(0,length-1):
            chain_node = Node(identifier=f"{chain}{chain_position}")
            chain_tensor = choose_state(chain_position+1).reshape(1,1,2)
            if chain_position == 0:
                ttns.add_child_to_parent(chain_node, chain_tensor,
                                            0,"0",chain)
            else:
                ttns.add_child_to_parent(chain_node, chain_tensor,
                                            0,f"{chain}{chain_position-1}",1)
        end_node = Node(identifier=f"{chain}{length-1}")
        end_tensor = choose_state(length).reshape(1,2)
        ttns.add_child_to_parent(end_node, end_tensor,
                                    0,f"{chain}{length-2}",1)
    return ttns

def create_trotterisation(g: float,
                          ttns: TreeTensorNetworkState) -> TrotterSplitting:
    """
    Generate the transverse-field Ising model as Trotterisation.

    Args:
        g (float): The transverse field strength.
        ttns (TreeTensorNetworkState): The initial state.
    """
    X, _, Z = pauli_matrices()
    gX = -g*X
    steps = []
    for ident, node in ttns.nodes.items():
        tp1 = TensorProduct({ident: gX})
        tp1 = TrotterStep(tp1, 1)
        steps.append(tp1)
        if not node.is_root():
            op = TensorProduct({ident: -1*Z,
                                    node.parent: Z})
            tp = TrotterStep(op, 1)
            steps.append(tp)
    return TrotterSplitting(steps)

def create_magnetisation_operator(ttns: TreeTensorNetworkState) -> TensorProduct:
    """
    Generate the magnetisation operator.

    Args:
        ttns (TreeTensorNetworkState): The initial state.
    """
    Z = pauli_matrices()[2]
    op = TensorProduct({node.identifier: Z for node in ttns.nodes.values()})
    return op

def create_tebd(length: int,
                g: float,
                time_step_size: float,
                final_time: float,
                max_bond_dim: int,
                rel_tol: float,
                total_tol: float) -> TEBD:
    """
    Generate the TEBD for the transverse-field Ising model.

    Args:
        length (int): The length of the chains.
        g (float): The transverse field strength.
        time_step_size (float): The time step size.
        final_time (float): The final time.
        max_bond_dim (int): The maximum bond dimension.
        rel_tol (float): The relative tolerance.
        total_tol (float): The absolute tolerance.
    """
    ttns = create_init_state(length)
    trotter = create_trotterisation(g, ttns)
    magnetisation = create_magnetisation_operator(ttns)
    operators = {"magn": magnetisation}
    svd_params = SVDParameters(max_bond_dim=max_bond_dim,
                               rel_tol=rel_tol,
                               total_tol=total_tol)
    config = TTNTimeEvolutionConfig(record_bond_dim=True)
    tebd = TEBD(ttns, trotter, time_step_size, final_time, operators,
                svd_parameters=svd_params,
                config=config)
    return tebd

def create_file_path(filepath: str, ending: str = ".hdf5") -> str:
    """
    Checks a file path and adds the ending if necessary.
    """
    if not re.match(r".+\."+ ending + r"$", filepath):
        filepath = filepath + ending
    print("Data will be saved in " + filepath)
    return filepath

def input_handling_filepath(parser: Union[ArgumentParser,None] = None) -> ArgumentParser:
    """
    Handle command line arguments.
    """
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("filepath", type=str, nargs=1)
    return parser
