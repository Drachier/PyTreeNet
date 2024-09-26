"""
This module provides random TTNS and compatible TTNO for testing purposes.
"""
from typing import Tuple

from numpy import eye

from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator
from pytreenet.ttns.ttns import TreeTensorNetworkState

from pytreenet.random.random_ttns import (random_small_ttns,
                                            random_big_ttns_two_root_children,
                                            RandomTTNSMode)
from pytreenet.random.random_matrices import random_hermitian_matrix
from pytreenet.random.random_hamiltonian import random_hamiltonian_compatible

def small_ttns_and_ttno() -> Tuple[TreeTensorNetworkState,
                                TreeTensorNetworkOperator]:
    """
    Generates a small TreeTensorNetworkState and corresponding TTNO of three nodes:
    The root (`"root"`) and its two children (`"c1"` and `"c2"`). The associated 
    tensors are random, but their dimensions are set.

    Args:
        mode (RandomTTNSMode): The mode of random generation of the TTNS. If mode
            is DIFFVIRT, the virtual bond dimensions are as follows::

                        |2
                        |
                        r
                       / \\
                 3|  5/  6\\   |4
                  |  /     \\  |
                   c1        c2

            Otherwise all virtual bond dimensions default to 2. If the mode is
            SAMEPHYS all phyiscal dimensions will default to 2.
    """
    conversion_dict = {"root_op1": random_hermitian_matrix(),
                            "root_op2": random_hermitian_matrix(),
                            "I2": eye(2),
                            "c1_op": random_hermitian_matrix(size=3),
                            "I3": eye(3),
                            "c2_op": random_hermitian_matrix(size=4),
                            "I4": eye(4)}
    ref_tree = random_small_ttns()
    tensor_prod = [TensorProduct({"c1": "I3", "root": "root_op1", "c2": "I4"}),
                    TensorProduct({"c1": "c1_op", "root": "root_op1", "c2": "I4"}),
                    TensorProduct({"c1": "c1_op", "root": "root_op2", "c2": "c2_op"}),
                    TensorProduct({"c1": "c1_op", "root": "I2", "c2": "c2_op"})
                    ]
    ham = Hamiltonian(tensor_prod, conversion_dict)
    hamiltonian = TreeTensorNetworkOperator.from_hamiltonian(ham, ref_tree)
    return ref_tree, hamiltonian

def big_ttns_and_ttno(mode: RandomTTNSMode = RandomTTNSMode.SAME) -> Tuple[TreeTensorNetworkState,
                                                            TreeTensorNetworkOperator]:
    """
    Returns a random big TTNS and TTNO where the root has only two children.

    For testing it is important to know that the children of 1 will be in a
    different order, if the TTNS is orthogonalised.

    Args:
        mode (RandomTTNSMode): The mode of random generation of the TTNS. If it
            is SAME all legs will be chosen as 2. For DIFFVIRT the virtual
            bond dimensons are all different.
    
    Returns:
        A random TTNS and TTNO with the topology::

                0
               / \\
              /   \\
             1     6
            / \\    \\
           /   \\    \\
          2     3     7
               / \\
              /   \\
             4     5
    
    """
    ref_tree = random_big_ttns_two_root_children(mode=mode)
    ham = random_hamiltonian_compatible()
    hamiltonian = TreeTensorNetworkOperator.from_hamiltonian(ham,
                                                    ref_tree)
    return ref_tree, hamiltonian
