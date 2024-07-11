
from copy import deepcopy
import re

import numpy as np

from pytreenet.ttns.ttns import TreeTensorNetworkState
from pytreenet.core.node import Node
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.time_evolution.tdvp import (FirstOrderOneSiteTDVP,
                                           SecondOrderTwoSiteTDVP,
                                           SecondOrderOneSiteTDVP)
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

def create_init_state(length: int, bond_dim: int = 1) -> TreeTensorNetworkState:
    """
    Create the initial state for the time evolution as TTN.

    Args:
        length (int): The length of the chains.
        bond_dim (int): The bond dimension.
    """
    state = np.zeros((2, ))
    zero_state = deepcopy(state)
    zero_state[0] = 1
    one_state = deepcopy(state)
    one_state[1] = 1
    ttns = TreeTensorNetworkState()
    center_tensor = np.asarray([1,0]).reshape(1,1,1,2)
    center_tensor = np.pad(center_tensor,
                           ((0, bond_dim-1),(0, bond_dim-1),
                            (0, bond_dim-1),(0, 0)))
    center_node = Node(identifier="0")
    ttns.add_root(center_node, center_tensor)
    for chain in range(3):
        for chain_position in range(0,length-1):
            chain_node = Node(identifier=f"{chain}{chain_position}")
            chain_tensor = deepcopy(one_state.reshape(1,1,2))
            bd_away_from_center = min(2**(length-chain_position-1), bond_dim)
            bd_towards_center = min(2**(length-chain_position),bond_dim)
            if bd_away_from_center <= 0:
                bd_away_from_center = bond_dim
            if bd_towards_center <= 0:
                bd_towards_center = bond_dim
            chain_tensor = np.pad(chain_tensor,
                                  ((0, bd_towards_center-1),
                                   (0, bd_away_from_center-1),
                                   (0, 0)))
            if chain_position == 0:
                ttns.add_child_to_parent(chain_node, chain_tensor,
                                            0,"0",chain)
            else:
                ttns.add_child_to_parent(chain_node, chain_tensor,
                                            0,f"{chain}{chain_position-1}",1)
        end_node = Node(identifier=f"{chain}{length-1}")
        end_tensor = deepcopy(zero_state.reshape(1,2))
        bd_towards_center = min(2, bond_dim)
        end_tensor = np.pad(end_tensor, ((0, bd_towards_center-1), (0, 0)))
        ttns.add_child_to_parent(end_node, end_tensor,
                                    0,f"{chain}{length-2}",1)
    return ttns

def create_hamiltonian(g: float,
                       ttns: TreeTensorNetworkState) -> TreeTensorNetworkOperator:
    """
    Find the Hamiltonian for the time evolution.

    The modified TFI Hamiltonian as defined in the user guide.

    Args:
        g (float): The coupling constant.
        ttns (TreeTensorNetworkState): The initial state.

    Returns:
        TreeTensorNetworkOperator: The Hamiltonian in TTNO form.
    """
    X, _, Z = pauli_matrices()
    g = -1*g
    gX = g*X
    con_dict = {"gX": gX, "Z": Z, "-Z": -1*Z,
                        "I1": np.eye(1), "I2": np.eye(2)}
    ham = Hamiltonian(conversion_dictionary=con_dict)
    for ident, node in ttns.nodes.items():
        op = TensorProduct({ident: "gX"})
        ham.add_term(op)
        if not node.is_root():
            op = TensorProduct({ident: "-Z",
                                node.parent: "Z"})
            ham.add_term(op)
    op = TensorProduct({"0": "-Z", "00": "Z", "10": "Z", "20": "Z"})
    ham.add_term(op)
    ham_pad = ham.pad_with_identities(ttns)
    return TreeTensorNetworkOperator.from_hamiltonian(ham_pad,
                                                      ttns)

def create_magnetisation_operator(ttns: TreeTensorNetworkState) -> TensorProduct:
    """
    Generate the magnetisation operator.

    Args:
        ttns (TreeTensorNetworkState): The initial state.
    
    Returns:
        TensorProduct: The magnetisation operator.
    """
    Z = pauli_matrices()[2]
    op = TensorProduct({ide: Z for ide in ttns.nodes})
    return op

def generate_fo_1tdvp(length: int, g: float, max_bd: int,
                      delta_t: float = 0.01,
                      final_time: float = 1) -> FirstOrderOneSiteTDVP:
    """
    Generate the first order one site TDVP.

    Args:
        length (int): The length of the chains.
        g (float): The coupling constant.
        max_bd (int): The maximum bond dimension.
        delta_t (float): The time step.
        final_time (float): The final time.
    
    Returns:
        FirstOrderOneSiteTDVP: The TDVP object.
    """
    ttns = create_init_state(length, max_bd)
    ham = create_hamiltonian(g, ttns)
    magn_op = create_magnetisation_operator(ttns)
    ops = {"magn": magn_op}
    config = TTNTimeEvolutionConfig(record_bond_dim=True)
    tdvp = FirstOrderOneSiteTDVP(ttns, ham,
                                 delta_t, final_time,
                                 ops, config=config)
    return tdvp

def generate_so_1tdvp(length: int, g: float, max_bd: int,
                      delta_t: float = 0.01,
                      final_time: float = 1) -> SecondOrderOneSiteTDVP:
    """
    Generate the first order one site TDVP.

    Args:
        length (int): The length of the chains.
        g (float): The coupling constant.
        max_bd (int): The maximum bond dimension.
        delta_t (float): The time step.
        final_time (float): The final time.
    
    Returns:
        SecondOrderOneSiteTDVP: The TDVP object.
    """
    ttns = create_init_state(length, max_bd)
    ham = create_hamiltonian(g, ttns)
    magn_op = create_magnetisation_operator(ttns)
    ops = {"magn": magn_op}
    config = TTNTimeEvolutionConfig(record_bond_dim=True)
    tdvp = SecondOrderOneSiteTDVP(ttns, ham,
                                  delta_t, final_time,
                                  ops, config=config)
    return tdvp

def generate_so_2tdvp(length: int, g: float,
                      delta_t: float = 0.01, final_time: float = 1,
                      max_bd: int = 1, rel_tol: float = 1e-10,
                      total_tol: float = 1e-10) -> SecondOrderTwoSiteTDVP:
    """
    Generate the second order two site TDVP.

    Args:
        length (int): The length of the chains.
        g (float): The coupling constant.
        delta_t (float): The time step.
        final_time (float): The final time.
        max_bd (int): The maximum bond dimension.
        rel_tol (float): The relative tolerance.
        total_tol (float): The total tolerance.
    """
    ttns = create_init_state(length)
    ham = create_hamiltonian(g, ttns)
    magn_op = create_magnetisation_operator(ttns)
    ops = {"magn": magn_op}
    svd_params = SVDParameters(max_bond_dim=max_bd,
                               rel_tol=rel_tol,
                               total_tol=total_tol)
    config = TTNTimeEvolutionConfig(record_bond_dim=True)
    tdvp = SecondOrderTwoSiteTDVP(ttns, ham,
                                  delta_t, final_time,
                                  ops, svd_params,
                                  config=config)
    return tdvp

def create_file_path(filepath: str, ending: str = ".hdf5") -> str:
    """
    Checks a file path and adds the ending if necessary.
    """
    if not re.match(r".+\."+ ending + r"$", filepath):
        filepath = filepath + ending
    print("Data will be saved in " + filepath)
    return filepath

if __name__ == "__main__":
    create_init_state(5,3)
