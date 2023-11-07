from __future__ import annotations
from typing import Any, Dict
from argparse import ArgumentParser

from randomised_hamiltonians_to_TTNO import main

import pytreenet as ptn

def tree_root_at_5():
    """
    Generates the desired tree tensor network with root at site 5 used as a
     reference to construct the Hamiltonian.
    """
    ttns = ptn.TreeTensorNetworkState()

    # Physical legs come last
    node1, tensor1 = ptn.random_tensor_node((1, 1, 2), identifier="site1")
    node2, tensor2 = ptn.random_tensor_node((1, 1, 1, 2), identifier="site2")
    node3, tensor3 = ptn.random_tensor_node((1, 2), identifier="site3")
    node4, tensor4 = ptn.random_tensor_node((1, 2), identifier="site4")
    node5, tensor5 = ptn.random_tensor_node((1, 1, 1, 2), identifier="site5")
    node6, tensor6 = ptn.random_tensor_node((1, 2), identifier="site6")
    node7, tensor7 = ptn.random_tensor_node((1, 1, 2), identifier="site7")
    node8, tensor8 = ptn.random_tensor_node((1, 2), identifier="site8")

    ttns.add_root(node5, tensor5)
    ttns.add_child_to_parent(node1, tensor1, 0, "site5", 0)
    ttns.add_child_to_parent(node2, tensor2, 0, "site1", 1)
    ttns.add_child_to_parent(node3, tensor3, 0, "site2", 1)
    ttns.add_child_to_parent(node4, tensor4, 0, "site2", 2)
    ttns.add_child_to_parent(node6, tensor6, 0, "site5", 1)
    ttns.add_child_to_parent(node7, tensor7, 0, "site5", 2)
    ttns.add_child_to_parent(node8, tensor8, 0, "site7", 1)
    return ttns

def tree_root_at_6():
    """
    Generates the desired tree tensor network with root at site 6 used as a
     reference to construct the Hamiltonian.
    """
    ttns = ptn.TreeTensorNetworkState()

    # Physical legs come last
    node1, tensor1 = ptn.random_tensor_node((1, 1, 2), identifier="site1")
    node2, tensor2 = ptn.random_tensor_node((1, 1, 1, 2), identifier="site2")
    node3, tensor3 = ptn.random_tensor_node((1, 2), identifier="site3")
    node4, tensor4 = ptn.random_tensor_node((1, 2), identifier="site4")
    node5, tensor5 = ptn.random_tensor_node((1, 1, 1, 2), identifier="site5")
    node6, tensor6 = ptn.random_tensor_node((1, 2), identifier="site6")
    node7, tensor7 = ptn.random_tensor_node((1, 1, 2), identifier="site7")
    node8, tensor8 = ptn.random_tensor_node((1, 2), identifier="site8")

    ttns.add_root(node6, tensor6)
    ttns.add_child_to_parent(node5, tensor5, 0, "site6", 0)
    ttns.add_child_to_parent(node1, tensor1, 0, "site5", 1)
    ttns.add_child_to_parent(node2, tensor2, 0, "site1", 1)
    ttns.add_child_to_parent(node3, tensor3, 0, "site2", 1)
    ttns.add_child_to_parent(node4, tensor4, 0, "site2", 2)
    ttns.add_child_to_parent(node7, tensor7, 0, "site5", 2)
    ttns.add_child_to_parent(node8, tensor8, 0, "site7", 1)
    return ttns

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filepath", type=str, nargs=1)
    filepath = vars(parser.parse_args())["filepath"][0]
    filepath1 = filepath + "_root_at_5.hdf5"
    print("Data will be saved in " + filepath1)
    main(filepath1, tree_root_at_5())
    filepath2 = filepath + "_root_at_6.hdf5"
    print("Data will be saved in " + filepath2)
    main(filepath2, tree_root_at_6())
